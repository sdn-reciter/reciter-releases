# Supertonic-3 сервер (порт 8003)

Лёгкий TTS на [Supertonic-3](https://huggingface.co/Supertone/supertonic-3)
(~99M параметров, ONNX, 31 язык включая русский). Пресетные голоса, без
клонирования по аудио. Обзор и подключение — [`../README.md`](../README.md),
протокол — [`../server-api.md`](../server-api.md).

## Запуск

```bash
docker build -t reciter-tts-base:cu128 ../base   # база, один раз (см. ../base/README.md)
cp .env.example .env
docker compose build && docker compose up -d
docker compose logs -f   # ждать "SERVER_VERSION st-1"; ассеты (~404 МБ) в образ при сборке
```

Проверка:

```bash
curl http://localhost:8003/health
curl -X POST http://localhost:8003/tts \
  -F 'text=Привет! Это тест Супертоника.' -F 'language=ru' -F 'voice=F1' \
  -o test.wav && aplay test.wav
```

## Голоса

Клонирования по аудио нет — только пресеты **F1–F5 / M1–M5** (поле `voice`).
Свой голос делается через [Voice Builder](https://supertone.ai); полученный
`.json` положите в `./styles/` (подхватывается на лету) или загрузите через
`POST /voices/upload`. Загрузка wav вернёт понятную ошибку.

## Ударения — не поддерживаются

Supertonic работает по сырым символам, без G2P — метки ударения (`U+0301`/`+`)
он не читает в принципе, синтез звучит одинаково при любых настройках словаря.
Поэтому `SUPERTONIC_SYNTH_STRESS=strip`. `/accent` (RUAccent) остаётся для
совместимости, но на синтез не влияет.

## Лицензия

Код — MIT, модель — OpenRAIL-M (свободная, с ограничениями на злоупотребления).
