# VoxCPM2 сервер — тест диффузионной модели рядом с XTTS

Экспериментальный TTS-сервер на [VoxCPM](https://github.com/OpenBMB/VoxCPM)
(OpenBMB). По умолчанию для бытовых 8-ГБ GPU используется
`openbmb/VoxCPM-0.5B`; полная `openbmb/VoxCPM2` (2B, 30 языков,
48 кГц) включается явно через `VOXCPM_MODEL=openbmb/VoxCPM2`. API полностью совпадает с XTTS/Supertonic серверами
(`/tts`, `/accent`, `/voices`, `/health`) — приложение переключается между
движками **сменой адреса сервера**:

| Сервер | Порт | Голоса |
|---|---|---|
| XTTS-v2 | 8002 | клоны по wav |
| Supertonic-3 | 8003 | пресеты + Voice Builder |
| VoxCPM2 | **8004** | клоны по wav (как XTTS) |

## Запуск

```bash
# 0. Один раз: общий базовый образ (CUDA+torch+RUAccent) — см. ../base/README.md
docker build -t reciter-tts-base:cu128 ../base

mkdir -p /mnt/data/reciter-tts/voxcpm-server && cd /mnt/data/reciter-tts/voxcpm-server
# скопируйте сюда файлы этой папки
cp .env.example .env

# GPU (RTX 50xx и т.п.):
docker compose build && docker compose up -d
# CPU-only хост:
docker compose -f docker-compose.cpu.yml build && docker compose -f docker-compose.cpu.yml up -d

docker compose logs -f   # веса качаются ОДИН РАЗ при сборке (в образ), старт быстрый
```

Веса модели скачиваются на этапе `docker compose build` и запекаются в образ
(`/opt/hf-cache`) — на старте контейнера сети нет, том для весов не нужен.
Без переменных собирается low-vram модель `openbmb/VoxCPM-0.5B`; для полной
VoxCPM2 соберите так: `docker compose build --build-arg MODEL=openbmb/VoxCPM2`
и запускайте с `VOXCPM_MODEL=openbmb/VoxCPM2`.
Тонкий образ + кеш в томе (как раньше): собрать с
`--build-arg PREFETCH_MODEL=0`, раскомментировать том `./cache:/opt/hf-cache`
в compose и поставить `VOXCPM_OFFLINE=0` на первый запуск. Обычный запечённый
образ стартует с `HF_HUB_OFFLINE=1`, чтобы контейнер не пытался лезть в сеть
при отсутствии DNS.

Проверка:

```bash
curl http://localhost:8004/health
curl -X POST http://localhost:8004/tts \
  -F 'text=Привет! Это тест модели ВоксЦПМ.' -F 'language=ru' \
  -o test.wav && aplay test.wav
```

## Переключение в приложении

Настройки чтения → «Свой TTS-сервер» → `http://<IP-ПК>:8004`.
Эталоны голосов (.wav, 6–20 сек чистой речи) — те же, что для XTTS: можно
скопировать `voices/*.wav` из папки XTTS-сервера в `./voices` или загрузить
через менеджер голосов приложения.

## Отличия и заметки

- **Клонирование по wav есть** (в отличие от Supertonic) — прямой конкурент
  XTTS по сценарию.
- **CPU поддерживается** (`docker-compose.cpu.yml`) — медленнее реального
  времени, для теста качества; для чтения книг — GPU.
- **Частота 48 кГц** берётся из модели (`tts_model.sample_rate`) — хардкод
  24000 в ранней версии сервера делал голос вдвое ниже и медленнее.
- **Ударения**: `/accent` — RUAccent, как у XTTS. В модель метки по умолчанию
  не идут (`VOXCPM_SYNTH_STRESS=strip`); поддержка не документирована —
  можно попробовать `mark`/`plus` и сравнить.
- Веса ЗАПЕКАЮТСЯ в образ при сборке (`PREFETCH_MODEL=1`, `/opt/hf-cache`):
  качаются один раз, на старте контейнера сети нет. RUAccent прогревается
  туда же. Тонкий образ + кеш в томе — `--build-arg PREFETCH_MODEL=0`.
- Язык определяется моделью по тексту; параметр `language` в `/tts`
  принимается для совместимости и игнорируется.

## Нехватка VRAM (8 ГБ, RTX 5060 и т.п.)

По умолчанию сервер стартует с `openbmb/VoxCPM-0.5B`, потому что она легче.
Но это китайско-английская модель: русский текст она может озвучивать с
китайским акцентом. Для русского нужна полная `openbmb/VoxCPM2`.

Если в `.env` задано `VOXCPM_MODEL=openbmb/VoxCPM2`, `docker compose build`
теперь запекает именно VoxCPM2 в образ и сервер пытается загрузить её на 8-ГБ
карте. Защитный откат на `openbmb/VoxCPM-0.5B` остаётся только для GPU меньше
`VOXCPM_FULL_MIN_VRAM_GB` (по умолчанию 7 ГБ) или при ручном повышении порога.
Если OOM всё равно случится, верните 0.5B через пустой `VOXCPM_MODEL` или
поднимите `VOXCPM_FULL_MIN_VRAM_GB`; это будет быстрее, но не для русского.

VoxCPM2 + встроенный денойзер занимают много VRAM, а кодирование длинного
референса даёт `CUDA out of memory`. Сервер уже смягчает это (обрезка
референса до `VOXCPM_REF_MAX_SEC`, замок на одну генерацию за раз, очистка
кэша VRAM между запросами, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
Если OOM всё равно ловится — в `.env`:

```env
VOXCPM_DENOISER=0        # выключить zipenhancer (нужен только для ШУМНЫХ эталонов)
VOXCPM_OPTIMIZE=0        # без CUDA-графов — меньше пиковая память
VOXCPM_REF_MAX_SEC=8     # короче референс
VOXCPM_TIMESTEPS=8       # чуть меньше шаги
```
`VOXCPM_DENOISER=0` освобождает больше всего — демо-голоса и так чистые.
После правки: `docker compose up -d --force-recreate`.

Важно: `openbmb/VoxCPM-0.5B` не поддерживает `reference_wav_path`, поэтому
сервер не передаёт в неё сохранённые голоса и загруженный `speaker_wav`. Это
устраняет 500 `reference_wav_path is only supported with VoxCPM2 models`, но
не делает 0.5B русской. Клонирование по wav и нормальный русский доступны
только при запуске полной `openbmb/VoxCPM2`.

## Симптомы и причины (было исправлено в vc-2)

- **«Каждый раз разные голоса»** — запрос уходил без референса, а VoxCPM без
  него сэмплирует случайный голос. Теперь: откат на первый доступный голос +
  фиксированный `VOXCPM_SEED`.
- **«Чтение книги не начинается»** — запросы с голосом падали в CUDA OOM
  (500), приложение не могло стартовать. Теперь: обрезка референса + замок
  на GPU (приложение шлёт абзацы параллельно) + очистка VRAM.

## Откат

Эксперимент живёт в ветке `claude/supertonic-test` и отдельном контейнере:
`docker compose down` — XTTS (8002) и Supertonic (8003) не затрагиваются.
