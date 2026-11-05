# VoxCPM2 сервер. Наследуется от ОБЩЕЙ базы reciter-tts-base — см.
# ../base/README.md (база собирается один раз: docker build -t
# reciter-tts-base:cu128 ../base).
#
# Работает и на CPU (DEVICE=cpu / auto, docker-compose.cpu.yml).
# Веса модели ЗАПЕКАЮТСЯ в образ при сборке (PREFETCH_MODEL=1 по умолчанию):
# качаются один раз и лежат в /opt/hf-cache — на старте контейнера сети нет,
# и это не зависит от тома. Тонкий образ + кеш в томе — PREFETCH_MODEL=0.

FROM reciter-tts-base:cu128

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# VoxCPM. torch/torchaudio уже в базе (cu128) — они передаются установке как
# ограничения, чтобы резолвер pip не заменил их другой CUDA-сборкой. Если
# voxcpm потребует более новый torch — сборка упадёт ЗДЕСЬ с явным
# конфликтом версий (тогда база пересобирается на новую пару, см. base/README).
RUN pip3 install --no-cache-dir voxcpm "torch==2.9.*" "torchaudio==2.9.*"

# transformers СТРОГО <5: с 5.x токенизатор не отдаёт token_type_ids, и
# ONNX-модель омографов RUAccent падает на текстах с омографами
# («Required inputs (['token_type_ids']) are missing»). Совместимость с
# voxcpm проверяет смоук-тест ниже.
RUN pip3 install --no-cache-dir "transformers>=4.43,<5"

# Смоук-тест: ИМПОРТИРУЕМ voxcpm — несовместимость падает здесь, с версиями.
RUN python3 - <<'PYEOF'
import torch, torchaudio
import voxcpm
from voxcpm import VoxCPM
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("voxcpm:", getattr(voxcpm, "__version__", "?"))
print("VoxCPM import OK")
PYEOF

# ── Веса модели в ОБРАЗ: качаются ОДИН РАЗ при сборке ────────────────────
# /opt/hf-cache НЕ монтируется томом, поэтому на старте контейнера нет ни
# скачивания, ни зависимости от bind-mount. Тонкий образ (веса в томе, как
# раньше) — сборка с --build-arg PREFETCH_MODEL=0.
ARG MODEL=openbmb/VoxCPM-0.5B
ARG PREFETCH_MODEL=1
ARG FULL_MODEL=openbmb/VoxCPM2
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_CACHE=/opt/hf-cache
COPY prefetch_model.py /app/prefetch_model.py
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        MODEL="$MODEL" FULL_MODEL="$FULL_MODEL" python3 /app/prefetch_model.py; \
    else \
        echo "PREFETCH_MODEL=0: веса скачаются при первом старте (см. том в compose)"; \
    fi

RUN mkdir -p /app/voices

COPY server.py /app/server.py
COPY healthcheck.py /app/healthcheck.py

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 /app/healthcheck.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
