# VoxCPM сервер (порт 8004)

TTS на [VoxCPM](https://github.com/OpenBMB/VoxCPM) (OpenBMB), клонирование по
wav. Для русского нужна полная `openbmb/VoxCPM2` (2B, 30 языков, 48 кГц); по
умолчанию для 8-ГБ GPU собирается лёгкая `openbmb/VoxCPM-0.5B` (китайско-
английская, русский с акцентом). Обзор и подключение — [`../README.md`](../README.md),
протокол — [`../server-api.md`](../server-api.md).

## Запуск

```bash
docker build -t reciter-tts-base:cu128 ../base   # база, один раз (см. ../base/README.md)
cp .env.example .env

# GPU:
docker compose build && docker compose up -d
# CPU-only (медленно, только для теста качества):
docker compose -f docker-compose.cpu.yml build && docker compose -f docker-compose.cpu.yml up -d

docker compose logs -f   # веса качаются один раз при сборке (в образ), старт быстрый
```

Проверка:

```bash
curl http://localhost:8004/health
curl -X POST http://localhost:8004/tts \
  -F 'text=Привет! Это тест ВоксЦПМ.' -F 'language=ru' -o test.wav && aplay test.wav
```

## Голоса

Эталоны — wav-файлы (6–20 с чистой речи) в `./voices/`, либо загрузка через
`POST /voices/upload`. Клонирование по wav доступно **только** с полной
`openbmb/VoxCPM2` (0.5B его не поддерживает).

## Русский: нужна VoxCPM2

Для нормального русского соберите полную модель:

```bash
docker compose build --build-arg MODEL=openbmb/VoxCPM2
# и в .env: VOXCPM_MODEL=openbmb/VoxCPM2
```

Защитный откат на 0.5B срабатывает только для GPU меньше `VOXCPM_FULL_MIN_VRAM_GB`
(по умолчанию 7 ГБ).

## Нехватка VRAM (8 ГБ)

VoxCPM2 + денойзер требуют много VRAM; сервер уже смягчает это (обрезка
референса, одна генерация за раз, очистка кэша). Если OOM всё равно ловится — в
`.env`:

```env
VOXCPM_DENOISER=0     # выключить zipenhancer (нужен только для ШУМНЫХ эталонов) — освобождает больше всего
VOXCPM_OPTIMIZE=0     # без CUDA-графов — меньше пиковая память
VOXCPM_REF_MAX_SEC=8  # короче референс
VOXCPM_TIMESTEPS=8    # чуть меньше шаги
```

После правки: `docker compose up -d --force-recreate`.

## Ударения

`/accent` — RUAccent. Метки в модель по умолчанию не идут
(`VOXCPM_SYNTH_STRESS=strip`); поддержка не документирована — можно попробовать
`mark`/`plus` и сравнить.

## Лицензия

Apache-2.0 (коммерческое использование разрешено).
