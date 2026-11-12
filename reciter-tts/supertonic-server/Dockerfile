# Supertonic-3 сервер. Наследуется от ОБЩЕЙ базы reciter-tts-base — см.
# ../base/README.md (база собирается один раз: docker build -t
# reciter-tts-base:cu128 ../base). torch из базы движку не нужен (ONNX),
# но слои базы общие с XTTS/VoxCPM/ESpeech — лишнего места это не стоит.
#
# Ассеты модели (~404 МБ) ЗАПЕКАЮТСЯ в образ при сборке (PREFETCH_MODEL=1):
# библиотека качает их в кеш ВНУТРИ контейнера (том ./cache их не покрывал),
# и раньше каждое пересоздание контейнера скачивало модель заново.

FROM reciter-tts-base:cu128

WORKDIR /app

# 1. Supertonic.
RUN pip3 install -U supertonic

# 2. GPU-сборка ONNX Runtime вместо CPU-версии, которую тянет supertonic.
# Пакеты дают один и тот же модуль `onnxruntime` — CPU-версию надо удалить.
# При проблемах с GPU сервер сам откатится на CPU (Supertonic и на CPU
# быстрее реального времени).
RUN pip3 uninstall -y onnxruntime && pip3 install -U onnxruntime-gpu

# transformers СТРОГО <5: с 5.x токенизатор не отдаёт token_type_ids, и
# ONNX-модель омографов RUAccent падает на текстах с омографами
# («Required inputs (['token_type_ids']) are missing»).
RUN pip3 install --no-cache-dir "transformers>=4.43,<5"

# 3. Смоук-тест: несовместимость падает здесь, с версиями в логе.
RUN python3 - <<'PYEOF'
import onnxruntime
print("onnxruntime:", onnxruntime.__version__, onnxruntime.get_available_providers())
import supertonic
print("supertonic import OK")
PYEOF

# 4. Ассеты модели + RUAccent в ОБРАЗ (один раз при сборке).
ARG PREFETCH_MODEL=1
ENV HF_HOME=/opt/hf-cache
ENV HF_HUB_CACHE=/opt/hf-cache
COPY prefetch_model.py /app/prefetch_model.py
RUN if [ "$PREFETCH_MODEL" = "1" ]; then \
        python3 /app/prefetch_model.py; \
    else \
        echo "PREFETCH_MODEL=0: ассеты скачаются при первом старте"; \
    fi

RUN mkdir -p /app/styles

COPY server.py /app/server.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
