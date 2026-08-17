import time
import requests
import subprocess
import threading

# API ve Model Ayarları
URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

PROMPT = """Senden bir Java algoritma sorusunu çözmeni istiyorum. Hem çözümünü açıklayacaksın hem de kodunu yazacaksın. 
Sorum şu: Kullanıcıdan bir integer al, bu integer 2D array'in row sayısı olacak. Bir integer daha al, bu da column sayısı olacak. 
Sonra bu 2D array'i 5 ile 20 arasındaki random sayılarla doldur."""

# Metrik toplama değişkenleri
gpu_utils = []
vram_used = []
is_running = True

def monitor_gpu():
    """Test süresince GPU verilerini arka planda toplar."""
    global is_running
    while is_running:
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            util, mem = output.split(', ')
            gpu_utils.append(float(util))
            vram_used.append(float(mem))
        except Exception:
            pass
        time.sleep(1)

# 1. GPU İzleme İpliğini (Thread) Başlat
monitor_thread = threading.Thread(target=monitor_gpu)
monitor_thread.start()

# 2. İsteği Gönder ve Süreyi Ölç
payload = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0.0
}

start_time = time.time()
response = requests.post(URL, json=payload).json()
end_time = time.time()

# 3. İzlemeyi Durdur
is_running = False
monitor_thread.join()

# 4. Hesaplamalar
total_time = end_time - start_time
completion_tokens = response['usage']['completion_tokens']
tokens_per_sec = completion_tokens / total_time

avg_vram = sum(vram_used) / len(vram_used) if vram_used else 0
max_vram = max(vram_used) if vram_used else 0
avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
measure_count = len(vram_used)

# 5. Görseldeki Formatla Çıktıyı Ekrana Bas
print(f"\n====== JAVA ALGORİTMA TESTİ SONUÇLARI - v2 (Qwen2.5-7B-AWQ) ======")
print(f"Üretilen token sayısı        : {completion_tokens}")
print(f"Toplam süre                  : {total_time:.3f} saniye")
print(f"Tokens/saniye                : {tokens_per_sec:.1f} token/sn")
print(f"Ortalama VRAM kullanımı      : {int(avg_vram)} MiB")
print(f"Maksimum VRAM kullanımı      : {int(max_vram)} MiB")
print(f"Ortalama GPU kullanım oranı  : %{avg_gpu:.1f}")
print(f"Ölçüm sayısı (saniye)        : {measure_count}")
