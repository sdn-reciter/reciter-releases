# reciter-tts-base — общий базовый образ TTS-серверов

Всё тяжёлое и одинаковое для всех движков собрано в один базовый образ: CUDA
runtime, torch/torchaudio (cu128, RTX 50xx) + torchcodec, FastAPI-стек, RUAccent
с прогретой моделью ударений.

Зачем: слои базы у Docker общие для всех движковых образов — место на диске не
умножается, а пересборка любого движка занимает минуты (база не пересобирается).

## Структура на ПК

Все TTS-серверы живут в одной папке `/mnt/data/reciter-tts` — копия папки
`reciter-tts/`:

```
/mnt/data/reciter-tts/
├── base/                # этот базовый образ
├── xtts-server/         # :8002
├── supertonic-server/   # :8003
├── voxcpm-server/       # :8004
├── espeech-server/      # :8005
├── chatterbox-server/   # :8006
└── cosyvoice-server/    # :8007
```

## Порядок сборки

```bash
# 1. База — один раз (и после изменений base/Dockerfile)
cd /mnt/data/reciter-tts/base
docker build -t reciter-tts-base:cu128 .

# 2. Любой движок — в своей папке, обычным compose (пример: XTTS)
cd ../xtts-server && cp .env.example .env && docker compose up -d --build   # :8002
```

Движковые Dockerfile начинаются с `FROM reciter-tts-base:cu128` — если базы
нет локально, сборка сразу скажет об этом.

## Модели

Веса качаются либо при первом старте в том `./cache` (XTTS/Supertonic/VoxCPM),
либо запекаются в образ при сборке (ESpeech/Chatterbox/CosyVoice) — зависит от
движка, см. его README. В обоих случаях образ пересобирается быстро.

## Обновление torch

Пары версий строгие: torch 2.9 ↔ torchcodec 0.9 (2.10↔0.10, …). Новая связка —
через build-args (пример в шапке `Dockerfile`) и новый тег
(`reciter-tts-base:cu130`), движки переключаются правкой первой строки FROM.
