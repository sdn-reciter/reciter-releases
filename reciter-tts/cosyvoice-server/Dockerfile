# Fun-CosyVoice3-0.5B (Alibaba FunAudioLLM) сервер. Наследуется от ОБЩЕЙ базы
# reciter-tts-base — см. ../base/README.md (база: docker build -t
# reciter-tts-base:cu128 ../base).
#
# ВАЖНО: CosyVoice — самый «тяжёлый» по зависимостям движок. Тянет свой
# репозиторий (+ подмодуль Matcha-TTS) и pynini/WeTextProcessing. Если
# апстрим сменит требования — правьте версии здесь. Веса ЗАПЕКАЮТСЯ в образ.

FROM reciter-tts-base:cu128

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Репозиторий CosyVoice с подмодулем Matcha-TTS (нужен в PYTHONPATH).
RUN git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice /app/CosyVoice

# 2. Нормализация текста: pynini ставится колёсами pip (в базе нет conda).
# WeTextProcessing зависит от pynini.
RUN pip3 install --no-cache-dir pynini==2.1.6.post1 WeTextProcessing \
    || pip3 install --no-cache-dir pynini WeTextProcessing

# 3a. openai-whisper (из requirements CosyVoice) собирается через setup.py,
# который импортирует pkg_resources. setuptools>=81 его выпилил, а build-
# изоляция pip ставит свежий setuptools → «No module named 'pkg_resources'».
# Ставим whisper ЗАРАНЕЕ без изоляции, с setuptools<81 (в нём pkg_resources
# ещё есть). Дальше requirements увидит его как удовлетворённый.
RUN pip3 install --no-cache-dir "setuptools<81" wheel
RUN pip3 install --no-cache-dir --no-build-isolation "openai-whisper==20231117"

# 3. Остальные зависимости CosyVoice. Его requirements.txt может тянуть иную
# сборку torch — ниже переустанавливаем cu128 (sm_120 для RTX 50xx) поверх.
RUN pip3 install --no-cache-dir -r /app/CosyVoice/requirements.txt
RUN pip3 install --no-cache-dir --force-reinstall \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.9.*" "torchaudio==2.9.*"

# transformers СТРОГО <5 (RUAccent-омографы: token_type_ids).
RUN pip3 install --no-cache-dir "transformers>=4.43,<5"

# deepspeed нужен CosyVoice только для ОБУЧЕНИЯ, но transformers на импорте
# сам подхватывает установленный deepspeed, а тот на импорте зовёт `nvcc`,
# которого нет в CUDA-runtime базе → падение на старте
# («No such file or directory: /usr/local/cuda/bin/nvcc»). Инференсу deepspeed
# не нужен — удаляем, и transformers его больше не трогает.
RUN pip3 uninstall -y deepspeed || true

# Смоук-тест импорта (Matcha-TTS в пути).
RUN python3 - <<'PYEOF'
import sys, torch
sys.path.append("/app/CosyVoice/third_party/Matcha-TTS")
sys.path.append("/app/CosyVoice")
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda)
from cosyvoice.cli.cosyvoice import CosyVoice3
print("CosyVoice3 import OK")
PYEOF

# ── Веса модели в ОБРАЗ ─────────────────────────────────────────────────────
ARG PREFETCH_MODEL=1
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_CACHE=/opt/hf-cache
ENV MODEL_DIR=/opt/models/Fun-CosyVoice3-0.5B
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
