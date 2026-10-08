"""Скачивание весов модели в HF-кеш образа при СБОРКЕ (docker build).

Запускается один раз в Dockerfile, чтобы на старте контейнера не было
сетевых обращений и модель не «качалась каждый раз». Кладёт всё в HF_HOME
(в образе — /opt/hf-cache, путь ВНЕ тома, поэтому bind-mount его не затирает).
"""
import os

from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "openbmb/VoxCPM-0.5B").strip() or "openbmb/VoxCPM-0.5B"
path = snapshot_download(repo_id=model)
print(f"VoxCPM weights prefetched: {model} -> {path}")

# RUAccent прогреваем в ЭТОТ ЖЕ кеш: HF_HOME здесь уже /opt/hf-cache, иначе на
# старте контейнера RUAccent полез бы в сеть за моделью ударений (в базовом
# образе она прогрета в другой каталог, который на рантайме не используется).
try:
    from ruaccent import RUAccent

    a = RUAccent()
    a.load(omograph_model_size="turbo3.1", use_dictionary=True)
    print("RUAccent warmup into image cache OK")
except Exception as e:
    print(f"RUAccent warmup skipped: {e}")
