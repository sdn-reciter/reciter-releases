# XTTS-сервер — структура и запуск

Готовый комплект для самостоятельного TTS-сервера с клонированием голоса и
ударениями (RUAccent). Подключение приложения, доступ снаружи (Tailscale) и
общий протокол — в [`../README.md`](../README.md) и
[`../server-api.md`](../server-api.md).

## Структура

```
xtts-server/
├── Dockerfile              # образ движка (наследует reciter-tts-base)
├── docker-compose.yml      # сервис, порты, тома, переменные
├── .env.example            # → скопировать в .env (ключ, голос по умолчанию, RUAccent)
├── server.py               # FastAPI: /tts /accent /voices...
└── voices/                 # 🎙 РЕФЕРЕНСЫ голосов (эталоны для клонирования)
    └── default.wav         # голос по умолчанию (положи сам)
```

## Быстрый старт

```bash
# 0. Один раз: общий базовый образ (CUDA+torch+RUAccent) — см. ../base/README.md
docker build -t reciter-tts-base:cu128 ../base

cd xtts-server
cp .env.example .env
# отредактируй .env: XTTS_API_KEY=$(openssl rand -hex 32), DEFAULT_SPEAKER=default
mkdir -p voices
cp /путь/к/образцу.wav voices/default.wav      # 6–20 сек чистой речи
docker compose up -d --build
# первый старт качает модель (~2 ГБ) в ./cache — подождать
docker compose logs -f | grep -E "API auth|Voices dir|RUAccent|Model loaded"
```

Проверка:

```bash
curl -s http://localhost:8002/health | python3 -m json.tool
# синтез голосом по умолчанию
curl -X POST http://localhost:8002/tts -H "Authorization: Bearer $XTTS_API_KEY" \
  -F "text=Проверка." --output test.wav
```

## Эндпоинты

| Метод | Путь | Назначение |
|------|------|-----------|
| POST | `/tts` | синтез (`text`, `language`, `voice`, `accent`) → WAV |
| POST | `/v1/audio/speech` | OpenAI-совместимый синтез (JSON) |
| POST | `/accent` | текст → текст с ударениями (RUAccent) |
| GET  | `/voices` | список референсов |
| POST | `/voices/upload` | загрузить референс (`name` + `file`) |
| DELETE | `/voices/{name}` | удалить референс |
| GET  | `/health` | статус (устройство, auth, ruaccent, голоса) |

Всё, кроме `/health`, требует `Authorization: Bearer <XTTS_API_KEY>`, если ключ
задан. Управлять голосами удобно прямо из приложения: движок «Свой сервер» →
«Голоса для клонирования» (запись с микрофона или файл).

## Голоса для клонирования

Эталоны лежат в `voices/`, клиент шлёт только текст. Хороший образец: один
голос, без музыки/шума/эха, ровная интонация, **6–20 секунд**. Слишком длинный
или шумный образец ухудшает клон. `DEFAULT_SPEAKER` в `.env` — голос по
умолчанию; загруженный через приложение сервер перекодирует в WAV 24 кГц моно.

## Ударения

Базовая XTTS-v2 метки ударения **не читает** — ставит их сама из контекста
(в основном верно, ошибается на редких словах и омографах). Поэтому метки из
словаря по умолчанию срезаются: `XTTS_SYNTH_STRESS=strip`.

Чтобы XTTS **соблюдала** ударения, нужен русский файнтюн на `+`-разметке,
например [`xttsv2_banana`](https://huggingface.co/Ftfyhh/xttsv2_banana):
положите `model.pth`, `config.json`, `vocab.json` в `./model`, в `.env` —
`XTTS_MODEL_DIR=/app/model` и `XTTS_SYNTH_STRESS=plus`.

`/accent` (нейро-RUAccent) остаётся для других движков; формат ответа —
`XTTS_STRESS_MODE` (`mark`/`plus`/`strip`). Подробнее про форматы — в
[`../server-api.md`](../server-api.md) §3.

## «Странные звуки» в конце фраз

XTTS иногда доклеивает призвук в конце короткой фразы — известная особенность
модели. Сервер лечит это по умолчанию (финальная точка, посентенсный синтез,
срез хвостовой тишины + fade), настраивать ничего не нужно. Если хвост всё же
слышен — поднимите `XTTS_REPETITION_PENALTY` (по умолчанию `6.0`) до `8–12`.
Прочие ручки — в `.env` (`XTTS_TEMPERATURE`, `XTTS_SENT_PAUSE_MS`, `XTTS_TRIM_*`).
