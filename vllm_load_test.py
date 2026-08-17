import time
import statistics
import requests
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============== API ve Model Ayarları ==============
URL = "http://localhost:8000/v1/chat/completions"
VLLM_HOST = "http://localhost:8000"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
CONCURRENT_REQUESTS = 100

PROMPT = """Senden bir Java algoritma sorusunu çözmeni istiyorum. Hem çözümünü açıklayacaksın hem de kodunu yazacaksın.
Sorum şu: Kullanıcıdan bir integer al, bu integer 2D array'in row sayısı olacak. Bir integer daha al, bu da col sayısı olacak.
Sonra bu 2D array'i 5 ile 20 arasındaki random sayılarla doldur."""

# ============== Metrik toplama değişkenleri ==============
gpu_utils = []
vram_used = []
stop_event = threading.Event()


def get_max_model_len():
    """vLLM API'sinden max_model_len bilgisini çeker (varsa)."""
    try:
        response = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                model_info = data["data"][0]
                return model_info.get("max_model_len", "API'de mevcut değil")
    except Exception:
        pass
    return "Erişilemedi"


def monitor_gpu():
    """Test süresince GPU verilerini arka planda saniyelik toplar."""
    while not stop_event.is_set():
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
            util, mem = output.split(", ")
            gpu_utils.append(float(util))
            vram_used.append(float(mem))
        except Exception:
            pass
        stop_event.wait(1)


def send_request(request_id):
    """Tek bir isteği API'ye gönderir; token sayısı, gecikme ve hata bilgisini döner."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.7,
    }
    req_start = time.time()
    try:
        res = requests.post(URL, json=payload, timeout=300)
        latency = time.time() - req_start
        res.raise_for_status()
        data = res.json()
        tokens = data["usage"]["completion_tokens"]
        return {"ok": True, "tokens": tokens, "latency": latency, "error": None}
    except Exception as e:
        latency = time.time() - req_start
        return {
            "ok": False,
            "tokens": 0,
            "latency": latency,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    # Test öncesi sunucu bilgisi
    max_model_len = get_max_model_len()

    # Isınma (warm-up) isteği — CUDA graph / ilk allocation gecikmesini
    # asıl ölçümün dışında tutmak için
    print("Isınma isteği gönderiliyor...")
    warmup = send_request("warmup")
    if not warmup["ok"]:
        print(f"UYARI: Isınma isteği başarısız oldu: {warmup['error']}")
    else:
        print(f"Isınma tamam ({warmup['latency']:.2f} sn)\n")

    # GPU izleme thread'ini başlat
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()

    print(f"{CONCURRENT_REQUESTS} eşzamanlı istek başlatılıyor, lütfen bekleyin...")
    start_time = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(send_request, i) for i in range(CONCURRENT_REQUESTS)]
        for future in as_completed(futures):
            results.append(future.result())

    end_time = time.time()

    # İzlemeyi durdur
    stop_event.set()
    monitor_thread.join(timeout=5)

    # ============== Hesaplamalar ==============
    total_time = end_time - start_time

    successful = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    total_completion_tokens = sum(r["tokens"] for r in successful)
    successful_requests = len(successful)

    latencies = [r["latency"] for r in successful]

    avg_vram = sum(vram_used) / len(vram_used) if vram_used else 0
    max_vram = max(vram_used) if vram_used else 0
    avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
    max_gpu = max(gpu_utils) if gpu_utils else 0
    measure_count = len(vram_used)

    # ============== Formatlı Çıktı ==============
    print(f"\n====== {CONCURRENT_REQUESTS} EŞZAMANLI KULLANICI TESTİ ({MODEL_NAME}) ======")
    print(f"Gönderilen Eşzamanlı İstek Sayısı : {CONCURRENT_REQUESTS}")
    print(f"Max Model Len (API'den)           : {max_model_len}")
    print("-------------------------------------------------------------")
    print(f"Toplam süre                      : {total_time:.2f} saniye")
    print(f"Toplam üretilen token             : {total_completion_tokens}")
    if total_time > 0:
        print(f"Sistem geneli throughput          : {total_completion_tokens / total_time:.1f} token/sn")
    print(f"Başarılı İstek Oranı              : {successful_requests}/{CONCURRENT_REQUESTS}")

    if failed:
        print(f"\nBaşarısız istek sayısı: {len(failed)}")
        error_types = {}
        for r in failed:
            error_types[r["error"]] = error_types.get(r["error"], 0) + 1
        for err, count in error_types.items():
            print(f"  - {count}x {err}")

    if latencies:
        sorted_lat = sorted(latencies)
        p95_idx = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
        print("\n----- İstek Gecikmesi (latency) -----")
        print(f"Ortalama   : {statistics.mean(latencies):.2f} sn")
        print(f"Medyan     : {statistics.median(latencies):.2f} sn")
        print(f"Min / Max  : {min(latencies):.2f} / {max(latencies):.2f} sn")
        print(f"P95        : {sorted_lat[p95_idx]:.2f} sn")

    print("\n----- GPU / VRAM -----")
    print(f"Ortalama VRAM kullanımı          : {int(avg_vram)} MiB")
    print(f"Maksimum VRAM kullanımı          : {int(max_vram)} MiB")
    print(f"Ortalama GPU kullanım oranı      : %{avg_gpu:.1f}")
    print(f"Maksimum GPU kullanım oranı      : %{max_gpu:.1f}")
    print(f"Ölçüm sayısı (saniye)            : {measure_count}")


if __name__ == "__main__":
    main()
