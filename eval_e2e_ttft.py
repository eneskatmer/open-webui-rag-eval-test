#!/usr/bin/env python3
"""
=============================================================================
README FOR EVAL_E2E_TTFT
=============================================================================
AÇIKLAMA:
Bu script, uçtan uca (E2E) RAG performansını ölçer. Soru sorulduğu andan itibaren:
1) Retrieval (Arama) süresini,
2) Time-To-First-Token (TTFT - İlk jetonun üretilme süresi) değerini,
3) Toplam yanıt üretme süresini (Generation Latency) milisaniye cinsinden hesaplar.

ÖN GEREKSİNİMLER (Terminalden Çalıştırın):
   pip install requests pandas numpy

KULLANIM ALTERNATİFLERİ (TERMINAL KOMUTLARI):

1) Dense Arama Modunda Uçtan Uca (E2E) Test:
   python3 eval_e2e_ttft.py --mode dense \
       --ground-truth ground_truth_remapped.jsonl \
       --vllm-url http://localhost:8000/v1 \
       --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
       --owui-url http://localhost:8080 \
       --api-key YOUR_API_KEY \
       --collection-name YOUR_COLLECTION_ID \
       --top-k 5 \
       --out-prefix e2e_dense_topk5

2) Hybrid Arama Modunda Uçtan Uca (E2E) Test:
   python3 eval_e2e_ttft.py --mode hybrid \
       --ground-truth ground_truth_remapped.jsonl \
       --vllm-url http://localhost:8000/v1 \
       --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
       --owui-url http://localhost:8080 \
       --api-key YOUR_API_KEY \
       --collection-name YOUR_COLLECTION_ID \
       --top-k 5 \
       --out-prefix e2e_hybrid_topk5

PARAMETRE AÇIKLAMALARI:
   --mode            : Arama modu ('dense' veya 'hybrid')
   --vllm-url        : Yanıt üretecek LLM sunucusunun adresi
   --owui-url        : Retrieval yapacak Open WebUI adresi
   --top-k           : Modele bağlam (context) olarak verilecek doküman sayısı
   --out-prefix      : Sonuç raporlarının kaydedileceği dosya ön ismi
=============================================================================
"""
"""
- Open WebUI üzerinden retrieval (doc query) gecikmesini ölçer.
- vLLM streaming çıktısı üzerinden Time-To-First-Token (TTFT) değerini hesaplar.
- Generation ve toplam E2E gecikme metriklerini CSV olarak kaydeder.
- Arka planda anlık GPU VRAM ve kullanım takibi yapar."
"""
import argparse
import json
import time
import threading
import pandas as pd
import requests
import numpy as np

try:
import pynvml
HAS_NVML = True
except ImportError:
HAS_NVML = False

class GPUMonitor(threading.Thread):
def __init__(self, interval_sec=0.1):
    super().__init__()
    self.interval = interval_sec
    self.stopped = False
    self.samples = []
    self.gpu_ok = False
    if HAS_NVML:
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_ok = True
        except Exception:
            self.gpu_ok = False

def run(self):
    while not self.stopped and self.gpu_ok:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.samples.append({
                "util_pct": util.gpu,
                "mem_used_mb": mem.used / (1024 * 1024)
            })
        except Exception:
            pass
        time.sleep(self.interval)

def stop(self):
    self.stopped = True
    if HAS_NVML and self.gpu_ok:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def main():
parser = argparse.ArgumentParser(description="E2E Latency and TTFT Measurement")
parser.add_argument("--mode", default="hybrid")
parser.add_argument("--ground-truth", required=True)
parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
parser.add_argument("--owui-url", default="http://localhost:8080")
parser.add_argument("--api-key", required=True)
parser.add_argument("--collection-name", required=True)
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--out-prefix", default="e2e_test")
args = parser.parse_args()

questions = []
with open(args.ground_truth, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

headers_owui = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
results = []

gpu_mon = GPUMonitor()
gpu_mon.start()

print(f"\n{'='*60}\n{len(questions)} soru ile E2E test başlıyor: mode={args.mode}\n{'='*60}\n")

for idx, q in enumerate(questions, 1):
    q_text = q["question"]

    # 1. Aşama: Open WebUI'dan Retrieval (Bağlam Getirme)
    t0_ret = time.perf_counter()
    doc_resp = requests.post(
        f"{args.owui_url}/api/v1/rag/queries/doc",
        headers=headers_owui,
        json={"collection_names": [args.collection_name], "query": q_text, "k": args.top_k}
    )
    t1_ret = time.perf_counter()
    retrieval_ms = (t1_ret - t0_ret) * 1000

    retrieved_docs = doc_resp.json().get("documents", []) if doc_resp.status_code == 200 else []
    context_str = "\n".join([str(d) for d in retrieved_docs])

    # 2. Aşama: vLLM Streaming Endpoint'ine İstek Atma (TTFT & Generation Süresi Ölçümü)
    prompt = f"Bağlam:\n{context_str}\n\nSoru: {q_text}\nYanıt:"
    vllm_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.0
    }

    t0_gen = time.perf_counter()
    t_first_token = None
    
    response = requests.post(
        f"{args.vllm_url}/chat/completions",
        json=vllm_payload,
        stream=True
    )

    for chunk in response.iter_lines():
        if chunk:
            if t_first_token is None:
                t_first_token = time.perf_counter() # İlk token ulaştı

    t1_gen = time.perf_counter()

    ttft_ms = (t_first_token - t0_gen) * 1000 if t_first_token else 0
    generation_ms = (t1_gen - t_first_token) * 1000 if t_first_token else (t1_gen - t0_gen) * 1000
    e2e_ms = retrieval_ms + (t1_gen - t0_gen) * 1000

    print(f"[{idx:02d}/{len(questions)}] retrieval={retrieval_ms:6.1f}ms  ttft={ttft_ms:6.1f}ms  gen={generation_ms:7.1f}ms  e2e={e2e_ms:7.1f}ms  {q_text[:30]}")

    results.append({
        "question": q_text,
        "retrieval_ms": retrieval_ms,
        "ttft_ms": ttft_ms,
        "generation_ms": generation_ms,
        "e2e_ms": e2e_ms,
        "context_len_chars": len(context_str)
    })

gpu_mon.stop()
gpu_mon.join()

df = pd.DataFrame(results)
print(f"\n{'='*60}\nÖZET: E2E TTFT - mode={args.mode}\n{'='*60}")
print(f"Retrieval Mean : {df['retrieval_ms'].mean():.1f} ms")
print(f"TTFT Mean      : {df['ttft_ms'].mean():.1f} ms (p95: {df['ttft_ms'].quantile(0.95):.1f} ms)")
print(f"Generation Mean: {df['generation_ms'].mean():.1f} ms")
print(f"E2E Mean       : {df['e2e_ms'].mean():.1f} ms (p95: {df['e2e_ms'].quantile(0.95):.1f} ms)")

df.to_csv(f"{args.out_prefix}_results.csv", index=False)

if __name__ == "__main__":
main()
