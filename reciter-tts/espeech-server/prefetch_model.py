"""Скачивание весов в HF-кеш образа при СБОРКЕ (docker build) — как у VoxCPM.

В образ запекаются:
  - чекпойнт ESpeech (русский файнтюн F5-TTS);
  - вокодер Vocos (charactr/vocos-mel-24khz) — его качает f5-tts на старте;
  - faster-whisper small — разовая транскрипция референсов без сети;
  - модель ударений RUAccent — в ЭТОТ ЖЕ кеш (HF_HOME=/opt/hf-cache).
"""
import os

from huggingface_hub import snapshot_download

model = os.environ.get("MODEL", "ESpeech/ESpeech-TTS-1_RL-V2").strip()
path = snapshot_download(repo_id=model)
print(f"ESpeech weights prefetched: {model} -> {path}")

voc = snapshot_download(repo_id="charactr/vocos-mel-24khz")
print(f"Vocos vocoder prefetched -> {voc}")

try:
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    print("faster-whisper small prefetched")
except Exception as e:
    print(f"faster-whisper prefetch skipped: {e}")

try:
    from ruaccent import RUAccent
    a = RUAccent()
    a.load(omograph_model_size="turbo3.1", use_dictionary=True)
    print("RUAccent warmup into image cache OK")
except Exception as e:
    print(f"RUAccent warmup skipped: {e}")
