# Chatterbox Multilingual сервер — современная замена XTTS (MIT)

TTS-сервер на [Chatterbox Multilingual](https://huggingface.co/ResembleAI/chatterbox)
(Resemble AI, ~0.5B, **MIT**, 23 языка включая русский, zero-shot клонирование,
контроль экспрессии). Активный репозиторий — в отличие от заброшенного
XTTS-v2 (CPML). API совпадает с остальными серверами Reciter
(`/tts`, `/accent`, `/voices`, `/health`) — см. [`../server-api.md`](../server-api.md).

| Сервер | Порт |
|---|---|
| XTTS-v2 | 8002 |
| Supertonic | 8003 |
| VoxCPM | 8004 |
| ESpeech (F5) | 8005 |
| **Chatterbox** | **8006** |

## Запуск

```bash
# 0. Один раз: общий базовый образ (CUDA+torch+RUAccent) — см. ../base/README.md
docker build -t reciter-tts-base:cu128 ../base

mkdir -p /mnt/data/reciter-tts/chatterbox-server && cd /mnt/data/reciter-tts/chatterbox-server
# скопируйте сюда файлы этой папки
cp .env.example .env
docker compose build && docker compose up -d
docker compose logs -f   # ждать "SERVER_VERSION cb-1"
```

Проверка:

```bash
curl http://localhost:8006/health
curl -X POST http://localhost:8006/tts \
  -F 'text=Привет! Это тест Чаттербокса на русском.' -F 'language=ru' \
  -o test.wav && aplay test.wav
```

В приложении: «Параметры → Чтение вслух → Свой сервер», адрес `http://<IP-ПК>:8006`.

## Голоса

Референс **не обязателен** — у Chatterbox есть встроенный голос. Для клона
загрузите 6–15 с чистой речи (менеджер голосов в приложении или
`POST /voices/upload`); выбор голоса — там же.

## Ударения — работают (через RUAccent)

Chatterbox **обучен на русском с ударениями** — его встроенный
`russian_text_stresser` ставит их в формате `U+0301` (комбинируемый акут).
Сам пакет `russian-text-stresser` в образ не ставится (конфликт зависимостей:
spacy 3.6 vs gradio), поэтому ударения проставляет **наш RUAccent в том же
формате `U+0301`** (`CB_SYNTH_STRESS=mark`) — модель их читает (без стрессера
её токенизатор пропускает текст в модель как есть). Словарь ударений
приложения (режим «Словарь»/«Сервер», формат U+0301) тоже доходит и
используется.

> Если хочется именно апстрим-стрессер — можно попробовать
> `pip install russian-text-stresser` в своём образе, но он тянет spacy 3.6 и
> конфликтует с gradio из chatterbox-tts. Наш RUAccent надёжнее и качественнее.

## Настройки (`.env`)

| Переменная | По умолчанию | Что делает |
|---|---|---|
| `CHATTERBOX_LANG` | `ru` | Язык синтеза (23 языка). |
| `CB_EXAGGERATION` | `0.5` | Экспрессия (0.25–1.0). |
| `CB_CFG_WEIGHT` | `0.5` | Сила следования тексту/темпу (0.3–0.7). |
| `CB_SYNTH_STRESS` | `strip` | Метки в синтез (strip — единственно верно). |
| `CHATTERBOX_DEFAULT_SPEAKER` | — | Голос по умолчанию (пусто → встроенный). |

## Замечания по сборке

- **RTX 50xx (sm_120):** пакет `chatterbox-tts` пинит свою сборку torch;
  Dockerfile переустанавливает cu128-сборку поверх — иначе CUDA не заведётся.
  Если апстрим сменит требования к torch, поправьте версии в Dockerfile.
- Веса (~2 ГБ) запекаются в образ при сборке; на старте контейнера сети нет.
- Модель ставит на аудио неслышимый вотермарк (Perth) — особенность апстрима.

## Лицензия

Код и веса — **MIT** (коммерческое использование разрешено). Записано в
`docs/licenses.md` и на экране «Компоненты и лицензии».
