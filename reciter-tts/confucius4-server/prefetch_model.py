# Веса при сборке образа: чекпойнты Confucius4 — в ./checkpoints репозитория
# (так их ищет inference_config.yaml), w2v-BERT и BigVGAN — в кэш HF.
# Кодировщик голоса CAM++ (funasr/campplus) докачивается отдельным слоем в Dockerfile.
import os

from huggingface_hub import snapshot_download

snapshot_download("netease-youdao/Confucius4-TTS", local_dir="/opt/confucius/checkpoints")
for repo in ("facebook/w2v-bert-2.0", "nvidia/bigvgan_v2_22khz_80band_256x"):
    snapshot_download(repo, cache_dir=os.environ.get("HF_HOME"))
print("веса скачаны")
