#!/usr/bin/env python3
#!/usr/bin/env python3
"""
=============================================================================
README FOR EVAL_RETRIEVAL_METRICS
=============================================================================
Aşağıdaki komutlar Linux / macOS veya Windows terminalinde (Bash / PowerShell)
doğrudan çalıştırılmak üzere tasarlanmıştır.

KULLANIM ALTERNATİFLERİ (TERMINAL KOMUTLARI):

1) Config / Ayar Kontrolü (Arama yapmadan mevcut sunucu ayarlarını gösterir):
   python3 eval_retrieval_metrics.py --verify-config --base-url http://localhost:8080 --api-key YOUR_API_KEY

2) Dense-Only Testi (BM25 Kapalı, Sadece Vektör Araması):
   python3 eval_retrieval_metrics.py --base-url http://localhost:8080 --api-key YOUR_API_KEY \
       --collection-name YOUR_COLLECTION_ID --ground-truth ground_truth_remapped.jsonl \
       --force-hybrid false --label "Dense-only" --out-prefix dense_only_final

3) Hybrid Testi (BM25 + Dense + Reranker Açık):
   python3 eval_retrieval_metrics.py --base-url http://localhost:8080 --api-key YOUR_API_KEY \
       --collection-name YOUR_COLLECTION_ID --ground-truth ground_truth_remapped.jsonl \
       --force-hybrid true --label "Hybrid" --out-prefix hybrid_final
=============================================================================
"""

"""
eval_retrieval_metrics.py
Ground truth setini Open WebUI retrieval API'sine karsi test eder;
Recall@k, MRR'nin yaninda sorgu basina latency ve GPU kullanimi/VRAM
metriklerini de kaydeder. Config'i API uzerinden zorla ayarlayarak
Open WebUI'nin bilinen "hybrid toggle kalici olmuyor" hatasini asar
(bkz. open-webui/open-webui issue #19668).
"""
import argparse
import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import requests


def get_config(base_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(f"{base_url}/api/v1/retrieval/config", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


def set_hybrid(base_url, api_key, enabled: bool):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"ENABLE_RAG_HYBRID_SEARCH": enabled}
    resp = requests.post(f"{base_url}/api/v1/retrieval/config/update",
                          headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def print_hybrid_status(cfg):
    val = cfg.get("ENABLE_RAG_HYBRID_SEARCH") if "ENABLE_RAG_HYBRID_SEARCH" in cfg else \
          cfg.get("rag", {}).get("ENABLE_RAG_HYBRID_SEARCH")
    print(f"Sunucudaki ENABLE_RAG_HYBRID_SEARCH = {val}")
    return val


def load_ground_truth(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return [it for it in items if it.get("source_document")
            and not str(it["source_document"]).startswith("TODO")]


def query_collection(base_url, api_key, collection_names, query, k, timeout=60):
    url = f"{base_url}/api/v1/retrieval/query/collection"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"collection_names": collection_names, "query": query, "k": k}
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    chunks = []
    if isinstance(data, dict) and "metadatas" in data:
        metadatas = data["metadatas"][0] if data["metadatas"] else []
        for meta in metadatas:
            chunks.append(meta.get("source") or meta.get("name") or "")
    return chunks


def source_matches(retrieved_source, expected_source):
    if not retrieved_source or not expected_source:
        return False
    r = retrieved_source.lower().strip()
    e = expected_source.lower().strip()
    return e in r or r in e or Path(r).name == Path(e).name


class GpuSampler:
    def __init__(self, interval=0.3):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._available = self._check()

    def _check(self):
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5, check=True)
            return True
        except Exception:
            return False

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                line = out.stdout.strip().split("\n")[0]
                util, used, total = [float(x.strip()) for x in line.split(",")]
                self.samples.append((util, used, total))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if not self._available:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_summarize(self):
        if not self._available:
            return {"available": False}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"available": True, "avg_util_pct": None, "max_util_pct": None,
                     "avg_mem_mb": None, "max_mem_mb": None}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "available": True,
            "avg_util_pct": round(sum(utils) / len(utils), 1),
            "max_util_pct": round(max(utils), 1),
            "avg_mem_mb": round(sum(mems) / len(mems), 1),
            "max_mem_mb": round(max(mems), 1),
            "n_samples": len(self.samples),
        }


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round((p / 100) * (len(s) - 1)))
    return s[idx]


def run(items, base_url, api_key, collection_names, k_values, label, out_prefix):
    max_k = max(k_values)
    rows = []
    latencies_ms = []

    gpu = GpuSampler(interval=0.3)
    gpu.start()

    print(f"\n{'='*60}\n{len(items)} soru ile test basliyor: {label}\n{'='*60}\n")

    for idx, it in enumerate(items, 1):
        question = it["question"]
        expected = it["source_document"]

        t0 = time.perf_counter()
        try:
            retrieved = query_collection(base_url, api_key, collection_names, question, max_k)
        except Exception as e:
            print(f"[hata] '{question[:40]}...' -> {e}")
            retrieved = []
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        rank = None
        for i, src in enumerate(retrieved):
            if source_matches(src, expected):
                rank = i + 1
                break

        row = {
            "idx": idx,
            "question": question,
            "expected_source": expected,
            "found_rank": rank if rank else "bulunamadi",
            "latency_ms": round(elapsed_ms, 1),
        }
        for k in k_values:
            row[f"hit@{k}"] = int(rank is not None and rank <= k)
        row["reciprocal_rank"] = round(1.0 / rank, 4) if rank else 0.0
        rows.append(row)

        status = f"rank={rank}" if rank else "BULUNAMADI"
        print(f"[{idx:02d}/{len(items)}] [{status:>10}] {elapsed_ms:7.1f}ms  {question[:55]}")

    gpu_summary = gpu.stop_and_summarize()

    n = len(rows)
    summary = {f"Recall@{k}": round(sum(r[f"hit@{k}"] for r in rows) / n, 3) for k in k_values}
    summary["MRR"] = round(sum(r["reciprocal_rank"] for r in rows) / n, 3)
    summary["n_questions"] = n
    summary["latency_mean_ms"] = round(sum(latencies_ms) / n, 1)
    summary["latency_p50_ms"] = round(percentile(latencies_ms, 50), 1)
    summary["latency_p95_ms"] = round(percentile(latencies_ms, 95), 1)
    summary["latency_max_ms"] = round(max(latencies_ms), 1)
    summary.update({f"gpu_{k}": v for k, v in gpu_summary.items()})

    print(f"\n{'='*60}\nÖZET: {label}\n{'='*60}")
    for k, v in summary.items():
        print(f"{k:<20}: {v}")

    csv_path = f"{out_prefix}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = f"{out_prefix}_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {label}\n\n| Metrik | Değer |\n|---|---|\n")
        for k, v in summary.items():
            f.write(f"| {k} | {v} |\n")

    print(f"\nDetay CSV: {csv_path}")
    print(f"Özet: {md_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--collection-name", action="append")
    ap.add_argument("--ground-truth")
    ap.add_argument("--k-values", default="1,3,5,10")
    ap.add_argument("--out-prefix", default="eval")
    ap.add_argument("--label", default=None)
    ap.add_argument("--force-hybrid", choices=["true", "false"], default=None)
    ap.add_argument("--verify-config", action="store_true")
    args = ap.parse_args()

    if args.verify_config:
        cfg = get_config(args.base_url, args.api_key)
        print_hybrid_status(cfg)
        for key in ("HYBRID_BM25_WEIGHT", "TOP_K", "TOP_K_RERANKER", "RAG_RERANKING_MODEL"):
            if key in cfg:
                print(f"{key} = {cfg[key]}")
        return

    if args.force_hybrid is not None:
        want = args.force_hybrid == "true"
        print(f"Sunucu config'i ENABLE_RAG_HYBRID_SEARCH={want} olarak ayarlaniyor...")
        set_hybrid(args.base_url, args.api_key, want)
        time.sleep(1)
        cfg = get_config(args.base_url, args.api_key)
        actual = print_hybrid_status(cfg)
        if bool(actual) != want:
            print(f"[UYARI] Istenen deger {want} ama sunucu hala {actual} gosteriyor!")
            print("Bilinen bir Open WebUI hatasi olabilir (issue #19668).")
            print("Servisi restart etmeyi deneyin: sudo systemctl restart open-webui")
            return
        print("Config dogrulandi, teste devam ediliyor.\n")

    if not args.collection_name or not args.ground_truth:
        print("--collection-name ve --ground-truth gerekli (verify-config disinda).")
        return

    k_values = [int(x) for x in args.k_values.split(",")]
    items = load_ground_truth(args.ground_truth)
    label = args.label or (f"hybrid={args.force_hybrid}" if args.force_hybrid else "mevcut_ayar")
    run(items, args.base_url, args.api_key, args.collection_name, k_values, label, args.out_prefix)


if __name__ == "__main__":
    main()
