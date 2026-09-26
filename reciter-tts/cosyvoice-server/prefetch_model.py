"""Скачивание весов Fun-CosyVoice3-0.5B в ОБРАЗ при СБОРКЕ.

Кладём в стабильный путь /opt/models/Fun-CosyVoice3-0.5B (его читает server.py) —
на старте контейнера сети нет. RUAccent греется в HF-кеш (нужен для /accent).
"""
import os

os.environ.setdefault("HF_HOME", "/opt/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "/opt/hf-cache")

from huggingface_hub import snapshot_download

dst = os.environ.get("MODEL_DIR", "/opt/models/Fun-CosyVoice3-0.5B")
repo = os.environ.get("COSY_HF_REPO", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
path = snapshot_download(repo_id=repo, local_dir=dst)
print(f"Fun-CosyVoice3 weights prefetched: {repo} -> {path}")

try:
    from ruaccent import RUAccent
    a = RUAccent()
    a.load(omograph_model_size="turbo3.1", use_dictionary=True)
    print("RUAccent warmup into image cache OK")
except Exception as e:
    print(f"RUAccent warmup skipped: {e}")
