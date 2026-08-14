#!/usr/bin/env python3
"""
generate_ground_truth.py

TOBB ETU RAG projesi icin mevzuat PDF'lerinden otomatik soru-cevap
(ground truth) seti uretir. Her PDF'i chunk'lara boler, her chunk'tan
vLLM (Qwen2.5-7B-Instruct-AWQ) kullanarak 1 soru + cevap uretir.

Cikti: ground_truth_raw.jsonl (elle 30-50 kaliteli soruya suzulmesi gerekir)

Kullanim:
    pip install pypdf requests --break-system-packages
    python3 generate_ground_truth.py --pdf-dir /path/to/mevzuat/pdfs

Notlar:
    - Bu script "extractive" sorular uretir: chunk'in ICINDEKI bilgiyle
      dogrudan cevaplanabilecek sorular. Boylece hangi chunk'in dogru
      kaynak oldugunu KESIN olarak biliyoruz (biz onu verdik).
    - Her dokumandan cok fazla soru uretmemek icin dokuman basina
      max-questions-per-doc ile ornekleme yapilir (tum chunk'lardan degil).
    - Uretilen sorulari KORKORCE guvenmeyin -- model bazen chunk disi
      bilgi katabilir. review_ground_truth.py ile (veya elle) gozden
      gecirin.
"""

import argparse
import json
import random
import re
import sys
import time
import uuid
from pathlib import Path

import requests

try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf bulunamadi. Kurulum: pip install pypdf --break-system-packages", file=sys.stderr)
    sys.exit(1)


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Basit karakter tabanli chunking. Kendi Open WebUI chunk
    parametrelerinizle (chunk size / overlap) esitleyin ki test
    gercek sisteminizi yansitsin."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if len(chunk) > 100:  # cok kisa/bos chunklari atla
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


QA_GENERATION_PROMPT = """Asagida bir universite mevzuat belgesinden alinmis bir metin parcasi var.

METIN:
\"\"\"{chunk}\"\"\"

Gorevin: Bu metnin SADECE bu parcadaki bilgiyle cevaplanabilecek, ogrenci
bakis acisiyla sorulmus, DOGAL ve GERCEKCI bir Turkce soru ve onun kisa
cevabini uretmek. Soru cok genel olmamali (metne ozgu detay icermeli).
Cevap SADECE verilen metinden gelmeli, disaridan bilgi katma.

Sadece asagidaki JSON formatinda cevap ver, baska hicbir metin ekleme:
{{"question": "...", "answer": "..."}}
"""


def generate_qa(vllm_url: str, model: str, chunk: str, timeout: int = 60) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": QA_GENERATION_PROMPT.format(chunk=chunk[:2500])}
        ],
        "max_tokens": 300,
        "temperature": 0.3,
    }
    try:
        resp = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        # Model bazen ```json ... ``` ile sarabilir, temizle
        content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
        data = json.loads(content)
        if "question" in data and "answer" in data:
            return data
    except Exception as e:
        print(f"  [uyari] QA uretilemedi: {e}", file=sys.stderr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True, help="Mevzuat PDF'lerinin bulundugu klasor")
    ap.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM API base URL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Karakter cinsinden (Open WebUI ayarinizla esitleyin)")
    ap.add_argument("--chunk-overlap", type=int, default=200)
    ap.add_argument("--max-questions-per-doc", type=int, default=4, help="Dokuman basina en fazla kac soru uretilsin")
    ap.add_argument("--out", default="ground_truth_raw.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"'{pdf_dir}' icinde PDF bulunamadi.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(pdf_files)} PDF bulundu. Chunking + QA uretimi basliyor...\n")

    results = []
    for pdf_path in pdf_files:
        print(f"-> {pdf_path.name}")
        text = extract_text(pdf_path)
        if len(text.strip()) < 200:
            print("   [atlandi] Metin cok kisa / OCR gerekebilir")
            continue

        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            continue

        # Dokumanin farkli yerlerinden ornek al (basindan/ortasindan/sonundan)
        # tek bir bolgeye yigilmamak icin
        sample_size = min(args.max_questions_per_doc, len(chunks))
        indices = sorted(random.sample(range(len(chunks)), sample_size))

        for idx in indices:
            chunk = chunks[idx]
            qa = generate_qa(args.vllm_url, args.model, chunk)
            if qa is None:
                continue
            results.append({
                "id": str(uuid.uuid4())[:8],
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "source_document": pdf_path.name,
                "chunk_index": idx,
                "chunk_preview": chunk[:200].replace("\n", " ") + "...",
                "difficulty": "auto_generated",
                "reviewed": False,
            })
            print(f"   [{idx}] {qa['question'][:70]}...")
            time.sleep(0.2)  # vLLM'i bogmamak icin hafif limit

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nToplam {len(results)} soru uretildi -> {out_path}")
    print("\nSONRAKI ADIM: Bu dosyayi elle gozden gecirin:")
    print("  1. Yanlis/anlamsiz sorulari silin")
    print("  2. Cevaplarin metne sadik oldugunu dogrulayin")
    print("  3. 'reviewed': true yapin")
    print("  4. En iyi 30-50 taneyi ground_truth.jsonl olarak kaydedin")
    print("  5. ambiguous_questions_template.json'daki cok anlamli sorulari elle doldurup ekleyin")


if __name__ == "__main__":
    main()
