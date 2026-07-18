# Fun-CosyVoice3 сервер — качественный мультиязычный клон (Apache-2.0)

TTS-сервер на [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)
(Alibaba FunAudioLLM, **Apache-2.0**, 9 языков **с русским**, zero-shot
клонирование, стриминг). Альтернатива тяжёлому VoxCPM. API совпадает с
остальными серверами Reciter — см. [`../server-api.md`](../server-api.md).

> Раньше здесь стоял CosyVoice**2**-0.5B — он русский официально НЕ поддерживал
> и озвучивал кириллицу «по-китайски». Версия **3** добавила русский; класс
> `CosyVoice3` в том же репозитории, миграция — смена модели и класса.

| Сервер | Порт |
|---|---|
| XTTS-v2 | 8002 |
| Supertonic | 8003 |
| VoxCPM | 8004 |
| ESpeech (F5) | 8005 |
| Chatterbox | 8006 |
| **Fun-CosyVoice3** | **8007** |

## Запуск

```bash
# 0. Один раз: общий базовый образ — см. ../base/README.md
docker build -t reciter-tts-base:cu128 ../base

mkdir -p /mnt/data/reciter-tts/cosyvoice-server && cd /mnt/data/reciter-tts/cosyvoice-server
# скопируйте сюда файлы этой папки
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

В приложении: «Параметры → Чтение вслух → Свой сервер», адрес `http://<IP-ПК>:8007`.

## Голоса

Референс **обязателен** — Fun-CosyVoice3-0.5B клонирует по образцу (встроенного
голоса нет). Загрузите 6–15 с чистой речи (менеджер голосов в приложении или
`POST /voices/upload`). Сервер использует **cross-lingual** инференс: нужен
только аудио-образец, транскрипт не требуется.

> CosyVoice извлекает признаки только из эталона **≤30 с** (жёсткий предел
> модели). Более длинные образцы сервер автоматически обрезает до первых
> `COSY_PROMPT_MAX_SEC` секунд (по умолчанию 18) и сводит в моно — короткий
> чистый образец и так даёт лучший клон.

## Ударения — ограничение

CosyVoice3 — нейромодель без управления ударением: метки (`U+0301`/`+`) в
синтез не идут (`COSY_SYNTH_STRESS=strip`), словарь на неё не влияет. `/accent`
остаётся для совместимости. Управляемое ударение — [ESpeech (F5)](../espeech-server/README.md).

## Замечания по сборке (это самый хрупкий движок)

- **pynini/WeTextProcessing** ставятся колёсами pip (в базе нет conda). Если
  колесо под вашу платформу не найдётся — поставьте pynini через conda в своём
  образе или уберите зависимость нормализатора.
- **torch:** requirements CosyVoice может тянуть другую сборку torch;
  Dockerfile переустанавливает cu128 (sm_120) поверх. При конфликте версий
  правьте пины в Dockerfile.
- **deepspeed удаляется** после установки: он нужен только для обучения, а на
  импорте зовёт `nvcc` (его нет в CUDA-runtime базе) — иначе сервер падал на
  старте. Инференсу CosyVoice3 deepspeed не нужен.
- **openai-whisper** ставится заранее без build-изоляции (setuptools<81), иначе
  его сборка падает на `pkg_resources`.
- Веса кладутся в `/opt/models/Fun-CosyVoice3-0.5B` (запечены в образ).

## Лицензия

Код и веса Fun-CosyVoice3 — **Apache-2.0** (коммерческое использование
разрешено). Записано в `docs/licenses.md` и на экране «Компоненты и лицензии».
