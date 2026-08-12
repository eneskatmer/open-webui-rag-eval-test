# Open WebUI Retrieval Evaluator

Open WebUI retrieval API'si için Recall@k, MRR, latency ve GPU/VRAM kullanım metriklerini ölçen test betiği.

## Kurulum

```bash
source ragtest-venv/bin/activate
pip install requests


# 1) Mevcut config'i kontrol edin (opsiyonel ama önerilir)
python3 eval_retrieval_metrics.py --verify-config \
  --base-url http://localhost:8080 --api-key YOUR_API_KEY

# 2) DENSE-ONLY testi (config'i API'den zorla kapatıp çalıştırır)
python3 eval_retrieval_metrics.py \
  --base-url http://localhost:8080 \
  --api-key YOUR_API_KEY \
  --collection-name YOUR_COLLECTION_ID \
  --ground-truth ground_truth_remapped.jsonl \
  --force-hybrid false \
  --label "Dense-only (BM25+Reranker kapalı)" \
  --out-prefix dense_only_final

# 3) HYBRID testi (config'i API'den zorla açıp çalıştırır)
python3 eval_retrieval_metrics.py \
  --base-url http://localhost:8080 \
  --api-key YOUR_API_KEY \
  --collection-name YOUR_COLLECTION_ID \
  --ground-truth ground_truth_remapped.jsonl \
  --force-hybrid true \
  --label "Hybrid (BM25+Dense+Reranker açık)" \
  --out-prefix hybrid_final
