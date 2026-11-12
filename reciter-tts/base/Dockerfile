# reciter-tts-base — ОБЩИЙ базовый образ для всех TTS-серверов Reciter
# (XTTS / Supertonic / VoxCPM). Содержит всё тяжёлое и одинаковое:
#   CUDA runtime + Python + ffmpeg/sox
#   torch/torchaudio (cu128, sm_120 = RTX 50xx) + torchcodec (парный)
#   FastAPI-стек + RUAccent с прогретой моделью ударений
#
# Слои базы ОБЩИЕ для всех движковых образов: место на диске не троится,
# пересборка движка не трогает базу и занимает минуты.
#
# Сборка (один раз, и после изменений этого файла):
#   cd base && docker build -t reciter-tts-base:cu128 .
#
# Обновление связки torch (пары см. в комментариях ниже) — через build-args:
#   docker build -t reciter-tts-base:cu130 \
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu130 \
#     --build-arg TORCH_SPEC="torch==2.13.*" \
#     --build-arg TORCHAUDIO_SPEC="torchaudio==2.11.*" \
#     --build-arg TORCHCODEC_SPEC="torchcodec==0.14.*" .

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip wheel

# torch/torchaudio cu128 (sm_120 внутри) + torchcodec.
# Версии строго ПАРНЫЕ: torch 2.9 ↔ torchcodec 0.9 (0.10↔2.10 и т.д.).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_SPEC="torch==2.9.*"
ARG TORCHAUDIO_SPEC="torchaudio==2.9.*"
RUN pip3 install ${TORCH_SPEC} ${TORCHAUDIO_SPEC} --index-url ${TORCH_INDEX}
ARG TORCHCODEC_SPEC="torchcodec==0.9.*"
RUN pip3 install ${TORCHCODEC_SPEC}

# Общий серверный стек.
RUN pip3 install -U fastapi "uvicorn[standard]" python-multipart soundfile pydantic numpy

# RUAccent (ударения, /accent у всех серверов) + прогрев модели в образ.
# transformers СТРОГО <5: с 5.x токенизатор не отдаёт token_type_ids, и
# ONNX-модель омографов RUAccent падает на текстах с омографами
# («Required inputs (['token_type_ids']) are missing»). Дублируется в
# движковых Dockerfile (их pip может перетянуть версию) — но база обязана
# отдавать рабочий RUAccent сама.
RUN pip3 install -U ruaccent "transformers>=4.43,<5"
RUN python3 - <<'PYEOF'
from ruaccent import RUAccent
a = RUAccent()
a.load(omograph_model_size='turbo3.1', use_dictionary=True)
print("RUAccent warmup OK")
PYEOF

# Смоук-тест базы: версии в лог, несовместимость падает здесь.
RUN python3 - <<'PYEOF'
import torch, torchaudio, torchcodec
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("torchaudio:", torchaudio.__version__)
print("torchcodec:", torchcodec.__version__)
print("BASE OK")
PYEOF
