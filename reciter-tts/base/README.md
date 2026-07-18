# reciter-tts-base — общий базовый образ TTS-серверов

Всё тяжёлое и одинаковое для XTTS / Supertonic / VoxCPM собрано в один
базовый образ: CUDA runtime, torch/torchaudio (cu128, RTX 50xx) + torchcodec,
FastAPI-стек, RUAccent с прогретой моделью ударений.

Зачем: слои базы у Docker общие для всех движковых образов — место на диске
не умножается на три, а пересборка любого движка занимает минуты (база не
пересобирается).

## Структура на ПК

Все TTS-серверы живут в одной папке `/mnt/data/reciter-tts` — зеркало
`docs/reciter-tts/` из репозитория:

```
/mnt/data/reciter-tts/
├── base/                # этот базовый образ
├── xtts-server/         # :8002
├── supertonic-server/   # :8003
└── voxcpm-server/       # :8004
```

## Порядок сборки

```bash
# 1. База — один раз (и после изменений base/Dockerfile)
cd /mnt/data/reciter-tts/base
docker build -t reciter-tts-base:cu128 .

# 2. Движки — каждый в своей папке, обычным compose
cd ../xtts-server       && docker compose build && docker compose up -d   # :8002
cd ../supertonic-server && docker compose build && docker compose up -d   # :8003
cd ../voxcpm-server     && docker compose build && docker compose up -d   # :8004
```

Движковые Dockerfile начинаются с `FROM reciter-tts-base:cu128` — если базы
нет локально, сборка сразу скажет об этом.

## Модели — в томах, не в образах

Веса моделей (XTTS ~2 ГБ, Supertonic ~0.4 ГБ, VoxCPM ~2 ГБ) качаются при
ПЕРВОМ старте контейнера в `./cache` (том) и переживают пересборки образов.
Образы движков остаются тонкими (только pip-зависимости движка).

## Обновление torch

Пары версий строгие: torch 2.9 ↔ torchcodec 0.9 (2.10↔0.10, …). Новая связка —
через build-args (пример в шапке `Dockerfile`) и новый тег
(`reciter-tts-base:cu130`), движки переключаются правкой первой строки FROM.
