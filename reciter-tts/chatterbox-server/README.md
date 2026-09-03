# Chatterbox Multilingual сервер (порт 8006)

TTS на [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox)
(Resemble AI, ~0.5B, 23 языка включая русский, zero-shot клонирование, контроль
экспрессии). Обзор и подключение — [`../README.md`](../README.md), протокол —
[`../server-api.md`](../server-api.md).

## Запуск

```bash
docker build -t reciter-tts-base:cu128 ../base   # база, один раз (см. ../base/README.md)
cp .env.example .env
docker compose build && docker compose up -d
docker compose logs -f   # ждать "SERVER_VERSION cb-1"; веса (~2 ГБ) в образ при сборке
```

Проверка:

```bash
curl http://localhost:8006/health
curl -X POST http://localhost:8006/tts \
  -F 'text=Привет! Это тест Чаттербокса на русском.' -F 'language=ru' \
  -o test.wav && aplay test.wav
```

## Голоса

Референс **не обязателен** — есть встроенный голос. Для клона загрузите 6–15 с
чистой речи в `./voices/` или через `POST /voices/upload`; выбор голоса — полем
`voice`.

## Ударения — работают (через RUAccent)

Chatterbox обучен на русском с ударениями и читает метки `U+0301`. Ударения
проставляет наш RUAccent в том же формате (`CB_SYNTH_STRESS=mark`), поэтому
словарь реально влияет на произношение.

## Настройки (`.env`)

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `CHATTERBOX_LANG` | `ru` | Язык синтеза (23 языка). |
| `CB_EXAGGERATION` | `0.5` | Экспрессия (0.25–1.0). |
| `CB_CFG_WEIGHT` | `0.5` | Сила следования тексту/темпу (0.3–0.7). |
| `CB_SYNTH_STRESS` | `mark` | Формат меток для модели (`mark`/`plus`/`strip`). |
| `CHATTERBOX_DEFAULT_SPEAKER` | — | Голос по умолчанию (пусто → встроенный). |

## Замечания по сборке

- **RTX 50xx (sm_120):** пакет `chatterbox-tts` пинит свою сборку torch;
  Dockerfile переустанавливает cu128 поверх. Если апстрим сменит требования —
  поправьте версии в Dockerfile.
- Веса (~2 ГБ) запекаются в образ при сборке; на старте сети нет.
- Модель ставит на аудио неслышимый вотермарк (Perth) — особенность апстрима.

## Лицензия

Код и веса — MIT (коммерческое использование разрешено).
