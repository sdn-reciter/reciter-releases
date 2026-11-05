# ESpeech (русский файнтюн F5-TTS) сервер. Наследуется от ОБЩЕЙ базы
# reciter-tts-base — см. ../base/README.md (база собирается один раз:
# docker build -t reciter-tts-base:cu128 ../base).
#
# Веса ЗАПЕКАЮТСЯ в образ при сборке (PREFETCH_MODEL=1): чекпойнт ESpeech,
# вокодер Vocos, faster-whisper small (транскрипция референсов) и модель
# ударений RUAccent — на старте контейнера сети нет.

FROM reciter-tts-base:cu128

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# f5-tts: torch/torchaudio уже в базе (cu128) — передаются установке как
# ограничения, чтобы pip не заменил их другой CUDA-сборкой.
# faster-whisper — CPU-транскрипция референсов (F5 требует ref_text).
RUN pip3 install --no-cache-dir f5-tts faster-whisper "torch==2.9.*" "torchaudio==2.9.*"

# transformers СТРОГО <5: f5-tts подтягивает 5.x, где токенизатор перестал
# отдавать token_type_ids — ONNX-модель омографов RUAccent их требует и
# падает на текстах с омографами («RUAccent error: Required inputs
# (['token_type_ids']) are missing»).
RUN pip3 install --no-cache-dir "transformers>=4.43,<5"

# Смоук-тест: несовместимость падает здесь, с версиями в логе.
RUN python3 - <<'PYEOF'
import torch, torchaudio
from f5_tts.api import F5TTS
import faster_whisper
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
print("f5-tts import OK; faster-whisper:", faster_whisper.__version__)
PYEOF

# ── Веса модели в ОБРАЗ: качаются ОДИН РАЗ при сборке ────────────────────
ARG MODEL=ESpeech/ESpeech-TTS-1_RL-V2
ARG PREFETCH_MODEL=1
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_CACHE=/opt/hf-cache
COPY prefetch_model.py /app/prefetch_model.py
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        MODEL="$MODEL" python3 /app/prefetch_model.py; \
    else \
        echo "PREFETCH_MODEL=0: веса скачаются при первом старте (см. том в compose)"; \
    fi

RUN mkdir -p /app/voices

COPY server.py /app/server.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
