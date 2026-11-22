# Chatterbox Multilingual (Resemble AI) сервер. Наследуется от ОБЩЕЙ базы
# reciter-tts-base — см. ../base/README.md (база собирается один раз:
# docker build -t reciter-tts-base:cu128 ../base).
#
# Веса ЗАПЕКАЮТСЯ в образ при сборке (PREFETCH_MODEL=1) — на старте сети нет.

FROM reciter-tts-base:cu128

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Chatterbox. Пакет пинит СВОЮ сборку torch — ниже переустанавливаем
# cu128-сборку из базы поверх, иначе на RTX 50xx (sm_120) не заведётся CUDA.
RUN pip3 install --no-cache-dir chatterbox-tts
RUN pip3 install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.9.*" "torchaudio==2.9.*"

# transformers СТРОГО <5: 5.x ломает ONNX-модель омографов RUAccent
# (токенизатор не отдаёт token_type_ids). Chatterbox работает и на 4.x.
RUN pip3 install --no-cache-dir "transformers>=4.43,<5"

# Смоук-тест: несовместимость падает здесь, с версиями в логе.
RUN python3 - <<'PYEOF'
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print("Chatterbox import OK")
PYEOF

# ── Веса модели в ОБРАЗ: качаются ОДИН РАЗ при сборке ────────────────────
ARG PREFETCH_MODEL=1
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_CACHE=/opt/hf-cache
COPY prefetch_model.py /app/prefetch_model.py
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        python3 /app/prefetch_model.py; \
    else \
        echo "PREFETCH_MODEL=0: веса скачаются при первом старте"; \
    fi

RUN mkdir -p /app/voices

COPY server.py /app/server.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
