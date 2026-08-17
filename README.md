# Open WebUI Retrieval Evaluator

Open WebUI retrieval API'si için Recall@k, MRR, latency, TTFT ve GPU/VRAM kullanım metriklerini ölçen test betikleri.

## Kurulum

source ragtest-venv/bin/activate
pip install requests pandas numpy

## 1. Retrieval Performans Testi (eval_retrieval_metrics.py)

python3 eval_retrieval_metrics.py --base-url http://localhost:8080 --api-key YOUR_API_KEY --collection-name YOUR_COLLECTION_ID --ground-truth ground_truth_remapped.jsonl --force-hybrid true --top-k 5 --label "Hybrid (TOP_K=5)" --out-prefix hybrid_topk5

## 2. Uçtan Uca (E2E) ve TTFT Performance Testi (eval_e2e_ttft.py)

Open WebUI üzerinden bağlam çekip vLLM sunucusundaki Qwen2.5-7B modeline besleyen; Retrieval Latency, Time-To-First-Token (TTFT), Generation Time ve Total E2E Latency sürelerini ölçen koddur.

python3 eval_e2e_ttft.py --mode hybrid --ground-truth ground_truth_remapped.jsonl --vllm-url http://localhost:8000/v1 --model "Qwen/Qwen2.5-7B-Instruct-AWQ" --owui-url http://localhost:8080 --api-key YOUR_API_KEY --collection-name YOUR_COLLECTION_ID --top-k 5 --out-prefix e2e_hybrid_topk5
