# XTTS-сервер — структура и запуск

Готовый комплект для самостоятельного TTS-сервера с клонированием голоса и
ударениями (RUAccent). Подробная инструкция (безопасность, интернет-доступ,
подключение приложения) — в [`../self-hosted-server.md`](../self-hosted-server.md).
Про вшитый в читалку словарь ударений — в
[`../accent-dictionary.md`](../accent-dictionary.md).

## Структура

```
xtts-server/
├── Dockerfile              # полный образ (PyTorch+CUDA, XTTS, RUAccent, ffmpeg)
├── Dockerfile.snippet      # только фрагмент RUAccent (если правишь свой Dockerfile)
├── docker-compose.yml      # сервис, порты, тома, переменные
├── .env.example            # → скопировать в .env (ключ, голос по умолчанию, RUAccent)
├── server.py               # FastAPI: /tts /accent /voices...
├── Caddyfile               # реверс-прокси HTTPS (опционально)
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
задан. Управлять голосами удобнее прямо из приложения (dev): движок «Свой
сервер» → «Голоса для клонирования».
