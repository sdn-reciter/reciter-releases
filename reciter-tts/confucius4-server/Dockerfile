# Confucius4-TTS сервер. Наследуется от общей базы reciter-tts-base — см.
# ../base/README.md (база собирается один раз).
#
# Код модели — из репозитория NetEase Youdao, веса — с Hugging Face при сборке
# (запекаются в образ, на старте сети не нужно).

FROM reciter-tts-base:cu128

ENV PYTHONUNBUFFERED=1
WORKDIR /opt

ARG CONFUCIUS_REF=main
RUN git clone --depth 1 --branch ${CONFUCIUS_REF} https://github.com/netease-youdao/Confucius4-TTS.git confucius
WORKDIR /opt/confucius

# В requirements.txt закреплён torch 2.7 — его строки выкидываются: torch/
# torchaudio уже в базе (cu128, нужен для RTX 50xx) и идут ограничением,
# чтобы pip их не заменил.
RUN grep -viE '^(torch|torchaudio|pytorch-lightning|tensorboard|matplotlib|datasets)\b' requirements.txt \
        | sed 's/#.*//' | grep -v '^\s*$' | sort -u > /tmp/req.txt \
    && pip3 install --no-cache-dir -r /tmp/req.txt "torch==2.9.*" "torchaudio==2.9.*"

# Веса — в образ: чекпойнты модели в ./checkpoints, кодировщик речи и
# вокодер — в кэш Hugging Face.
ENV HF_HOME=/opt/hf-cache
COPY prefetch_model.py /opt/prefetch_model.py
RUN python3 /opt/prefetch_model.py

# Код модели тянет и «тренировочные» пакеты (matplotlib в утилитах BigVGAN):
# они ставятся отдельным слоем после весов, чтобы правка списка не качала
# веса заново. Проверка импорта ловит недостающий пакет при сборке, а не на
# старте контейнера.
# Версии — как в их requirements.txt: свежий datasets тянет huggingface_hub 2.x,
# с которым transformers 4.x не импортируется.
RUN pip3 install --no-cache-dir matplotlib==3.10.8 pytorch-lightning==2.5.6 tensorboard==2.20.0 \
        datasets==4.8.4 huggingface_hub==0.36.2 "torch==2.9.*" "torchaudio==2.9.*"
# Веса лежат прямо в /opt/hf-cache (cache_dir при скачивании), а transformers
# по умолчанию ищет в $HF_HOME/hub — без этого на старте «не найдено офлайн».
ENV HF_HUB_CACHE=/opt/hf-cache
# Кодировщик голоса CAM++ модель качает сама при старте — заранее, отдельным
# слоем: на старте сети нет.
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('funasr/campplus', 'campplus_cn_common.bin')"
# Код модели берёт свои чекпойнты через hf_hub_download из кэша HF, а не из
# ./checkpoints. Скачанное уже лежит в checkpoints — кэш собирается из него
# ссылками (refs/main → снимок «local»), второй раз качать не нужно.
RUN c=/opt/hf-cache/models--netease-youdao--Confucius4-TTS \
    && mkdir -p $c/refs $c/snapshots && echo -n local > $c/refs/main \
    && ln -sfn /opt/confucius/checkpoints $c/snapshots/local
ENV PYTHONPATH=/opt/confucius
RUN cd /opt/confucius && python3 -c "from confuciustts.cli.inference import ConfuciusTTS; print('импорт ConfuciusTTS: ок')"
RUN mkdir -p /app/voices
COPY server.py /app/server.py
COPY healthcheck.py /app/healthcheck.py
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python3 /app/healthcheck.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
