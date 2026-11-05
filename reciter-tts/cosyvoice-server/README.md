# Fun-CosyVoice3 сервер (порт 8007)

TTS на [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
(Alibaba FunAudioLLM, 9 языков **с русским**, zero-shot клонирование, стриминг).
Обзор и подключение — [`../README.md`](../README.md), протокол —
[`../server-api.md`](../server-api.md).

## Запуск

```bash
docker build -t reciter-tts-base:cu128 ../base   # база, один раз (см. ../base/README.md)
cp .env.example .env
mkdir -p voices && cp /путь/к/образцу.wav voices/default.wav   # 6–15 с чистой речи
docker compose build && docker compose up -d
docker compose logs -f   # ждать "SERVER_VERSION cv3-1"
```

> Сборка долгая: тянет репозиторий CosyVoice, подмодуль Matcha-TTS, pynini и
> веса (~2 ГБ). Всё запекается в образ, старт быстрый.

Проверка:

```bash
curl http://localhost:8007/health
curl -X POST http://localhost:8007/tts \
  -F 'text=Привет! Это тест Fun-CosyVoice3 на русском.' -F 'language=ru' -F 'voice=default' \
  -o test.wav && aplay test.wav
```

## Голоса

Референс **обязателен** — встроенного голоса нет. Положите wav (6–15 с чистой
речи) в `./voices/` или загрузите через `POST /voices/upload`; транскрипт не
нужен (cross-lingual инференс).

> Модель извлекает признаки только из эталона **≤30 с**. Более длинные образцы
> сервер автоматически обрезает до первых `COSY_PROMPT_MAX_SEC` секунд (по
> умолчанию 18) и сводит в моно — короткий чистый образец и так даёт лучший клон.

## Ударения — не поддерживаются

CosyVoice3 — нейромодель без управления ударением: метки (`U+0301`/`+`) в синтез
не идут (`COSY_SYNTH_STRESS=strip`), словарь на неё не влияет. `/accent` остаётся
для совместимости.

## Замечания по сборке (самый хрупкий движок)

- **pynini/WeTextProcessing** ставятся колёсами pip. Если колесо под вашу
  платформу не найдётся — поставьте pynini через conda в своём образе.
- **torch:** Dockerfile переустанавливает cu128 (sm_120) поверх — при конфликте
  версий правьте пины в Dockerfile.
- **deepspeed удаляется** после установки (на импорте зовёт `nvcc`, которого нет
  в базе; инференсу не нужен).
- **openai-whisper** ставится заранее без build-изоляции (setuptools<81).
- Веса — в `/opt/models/Fun-CosyVoice3-0.5B` (запечены в образ).

## Лицензия

Apache-2.0 (коммерческое использование разрешено).
