# XTTS-v2 сервер. Наследуется от ОБЩЕЙ базы reciter-tts-base (CUDA, torch
# cu128 + torchcodec, FastAPI, RUAccent) — см. ../base/README.md, база
# собирается один раз: docker build -t reciter-tts-base:cu128 ../base
#
# Здесь только специфичное для движка: coqui-tts + pin transformers.
# Модель XTTS (~2 ГБ) НЕ запекается в образ — качается при первом старте в
# /cache/tts (том в compose) и переживает пересборки.

FROM reciter-tts-base:cu128

WORKDIR /app

# Coqui TTS — поддерживаемый форк idiap (пакет coqui-tts). Оригинальный
# пакет `tts` заморожен на 0.22.0 и не поддерживается с 2024. API совместим:
# `from TTS.api import TTS` работает без правок server.py.
RUN pip3 install -U coqui-tts
# transformers 5.x удалил isin_mps_friendly, который импортируют слои
# tortoise в coqui-tts — импорт TTS падает. Держимся на последнем 4.x.
RUN pip3 install -U "transformers>=4.43,<5"

# Смоук-тест: несовместимость связки падает здесь, с версиями в логе.
RUN python3 - <<'PYEOF'
import torch, torchcodec, transformers, TTS
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
print("torchcodec:", torchcodec.__version__)
print("transformers:", transformers.__version__)
print("coqui-tts:", TTS.__version__)
from TTS.api import TTS as _TTS
print("TTS import OK")
PYEOF

# Модель качается при первом старте в TTS_HOME (том). Согласие с лицензией
# CPML — переменной окружения + server.py создаёт .coqui_tos в каталоге модели.
ENV TTS_HOME=/cache/tts
ENV COQUI_TOS_AGREED=1

RUN mkdir -p /app/voices

COPY server.py /app/server.py

EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
