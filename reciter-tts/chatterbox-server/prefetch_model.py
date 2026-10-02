"""Скачивание весов Chatterbox Multilingual в HF-кеш образа при СБОРКЕ.

from_pretrained на CPU при сборке кладёт веса в /opt/hf-cache — на старте
контейнера сети нет. RUAccent греется в тот же кеш (нужен только для /accent).
"""
import os

os.environ.setdefault("HF_HOME", "/opt/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "/opt/hf-cache")

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

ChatterboxMultilingualTTS.from_pretrained(device="cpu")
print("Chatterbox Multilingual weights prefetched into image")

try:
    from ruaccent import RUAccent
    a = RUAccent()
    a.load(omograph_model_size="turbo3.1", use_dictionary=True)
    print("RUAccent warmup into image cache OK")
except Exception as e:
    print(f"RUAccent warmup skipped: {e}")
