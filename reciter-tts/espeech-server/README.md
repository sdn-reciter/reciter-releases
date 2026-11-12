# ESpeech сервер (порт 8005) — русский файнтюн F5-TTS

TTS на [ESpeech-TTS-1](https://huggingface.co/ESpeech/ESpeech-TTS-1_RL-V2) —
русском файнтюне F5-TTS (4000+ часов русской речи). Русский родной, управляемые
ударения по меткам `+`, клонирование по wav, лёгкая (~0.3B). Обзор и подключение
— [`../README.md`](../README.md), протокол — [`../server-api.md`](../server-api.md).

## Запуск

```bash
docker build -t reciter-tts-base:cu128 ../base   # база, один раз (см. ../base/README.md)
cp .env.example .env
mkdir -p voices                                  # сюда — эталоны (см. «Голоса»)
docker compose build && docker compose up -d     # веса качаются один раз при сборке
docker compose logs -f
```

Проверка:

```bash
curl http://localhost:8005/health
curl -X POST http://localhost:8005/tts \
  -F 'text=Привет! Это тест русского файнтюна Эф пять.' -F 'language=ru' \
  -o test.wav && aplay test.wav
```

## Голоса и референс

Эталоны — wav-файлы в `./voices/`. **Каталог не может быть пуст**: F5 всегда
клонирует по образцу. Лучший эталон — 8–12 с чистой речи одного диктора, без
музыки и рекламных интонаций. Загрузить можно и через `POST /voices/upload`.

F5 требует транскрипт референса; сервер добывает его сам:

- длинный референс обрезается до `ESPEECH_REF_MAX_SEC` (12 с), транскрипт берётся
  от обрезанного куска (кеш в `voices/.refcache`);
- рядом можно положить `<голос>.txt` — для идеальной точности;
- иначе речь один раз распознаётся faster-whisper (CPU) и кешируется.

## Ударения — родные

Модель обучена на метках `+` перед ударной гласной (формат RUAccent). Сервер по
умолчанию сам их проставляет: `ESPEECH_SYNTH_STRESS=plus` — это главный вклад в
качество, без причины не выключайте.

## Настройки (`.env`)

```env
ESPEECH_NFE=32                                  # 16 — почти вдвое быстрее, чуть проще звук
ESPEECH_SYNTH_STRESS=plus                       # родные ударения модели
ESPEECH_MODEL=ESpeech/ESpeech-TTS-1_podcaster   # вариант с «подкастерской» подачей
```

Смена модели — `docker compose build && docker compose up -d`; прочее —
`docker compose up -d --force-recreate`.

## Лицензия

Apache-2.0 (коммерческое использование разрешено).
