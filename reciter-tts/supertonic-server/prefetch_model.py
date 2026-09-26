"""Скачивание ассетов Supertonic в ОБРАЗ при сборке (docker build).

Библиотека supertonic качает ассеты (26 файлов) в кеш внутри контейнера —
том ./cache их не покрывал, и после пересоздания контейнера модель качалась
заново на каждом старте. Прогрев на этапе сборки кладёт всё в слои образа:
на старте контейнера сети нет.

RUAccent прогревается в HF-кеш образа (/opt/hf-cache) — иначе модель
ударений качалась бы на старте.
"""
from supertonic import TTS

TTS(auto_download=True)
print("Supertonic assets prefetched into image")

try:
    from ruaccent import RUAccent
    a = RUAccent()
    a.load(omograph_model_size="turbo3.1", use_dictionary=True)
    print("RUAccent warmup into image cache OK")
except Exception as e:
    print(f"RUAccent warmup skipped: {e}")
