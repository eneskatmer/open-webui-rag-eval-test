""" 1.5B Tek Kullanıcı test betiği
Bu betik, Hugging Face transformers kütüphanesi ve PyTorch kullanarak yerel GPU üzerinde çalışan dil modellerinin (LLM) saf çıkarım (native inference) performansını ölçer.

İşlem öncesi ve sonrası NVML (pynvml) sürücüleri üzerinden VRAM bellek kullanımını takip eder,
belirtilen istem (prompt) doğrultusunda üretim gerçekleştirir ve saniyede üretilen token sayısını
(TPS - Tokens Per Second) hesaplar. Elde edilen tüm metrikler eşzamanlı olarak terminale basılır 
ve zaman damgasıyla birlikte model_performance_log.txt dosyasına kaydedilir.
"""




import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml

# GPU Bellek Ölçümü İçin NVML Başlatma
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_vram_usage():
    """MiB cinsinden anlık kullanılan VRAM'i döndürür."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)

model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Sisli bir sahil kasabasında terk edilmiş bir sahaf dükkanının..."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# --- ÜRETİM VE PERFORMANS ÖLÇÜMÜ ---
start_vram = get_vram_usage()
start_time = time.perf_counter()

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=700,
        temperature=0.7
    )

end_time = time.perf_counter()
end_vram = get_vram_usage()

# Token Hesaplamaları
input_len = model_inputs.input_ids.shape[1]
output_ids = generated_ids[0][input_len:]
generated_tokens_count = len(output_ids)

total_time = end_time - start_time
tps = generated_tokens_count / total_time  # Saniyedeki Token Sayısı

response_text = tokenizer.decode(output_ids, skip_special_tokens=True)

# --- LOG DOSYASINA YAZMA ---
log_content = f"""
========================================
Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}
Model: {model_id}
Üretilen Token Sayısı: {generated_tokens_count}
Toplam Süre: {total_time:.2f} saniye
Hız (TPS): {tps:.2f} token/saniye
Başlangıç VRAM: {start_vram:.2f} MiB
Zirve/Bitiş VRAM: {end_vram:.2f} MiB
========================================
"""

# Konsola Bas
print(log_content)

# Dosyaya Kaydet
with open("model_performance_log.txt", "a", encoding="utf-8") as f:
    f.write(log_content)
