# Словари ударений

Скрипты сборки офлайн-словарей ударений и их описание. Сами словари вшиты в
APK и ставят ударения для всех TTS-движков без сети — как это работает в
приложении, см. [`../accent-dictionary.md`](../accent-dictionary.md).

## Что здесь

| Файл | Назначение |
|---|---|
| [`make_dict_wiktionary.py`](make_dict_wiktionary.py) | Собирает словарь из русского Викисловаря (kaikki.org) → `ruaccent_dict_wiktionary.dict` |
| [`make_dict_ruaccent.py`](make_dict_ruaccent.py) | Собирает словарь из RUAccent → `ruaccent_dict_ruaccent.dict` |

## Два словаря

| Ассет | Источник | Формы | Лицензия | Куда попадает |
|---|---|---|---|---|
| `ruaccent_dict_wiktionary.dict` | Русский Викисловарь (kaikki.org) | ~887 тыс. | ✅ **CC BY-SA** | все сборки (prod + dev) |
| `ruaccent_dict_ruaccent.dict` | RUAccent (Den4ikAI/ruaccent) | ~3.2 млн | ⚠️ **CC BY-NC-ND 4.0** | **только dev** |

RUAccent под NoDerivatives нельзя распространять в публичной сборке — поэтому
его ассет лежит в `app/src/dev/assets` и физически не попадает в prod-APK.
Викисловарь публиковать можно (нужна атрибуция — готова в
`app/src/main/assets/NOTICE-accent-dict.txt`). Правило по лицензиям —
`docs/CONVENTIONS.md` §8.

## Формат

Оба скрипта пишут файл одного формата: UTF-8-строки `ключ\tпозиция` (позиция
ударной гласной), **отсортированные по байтам** — для бинарного поиска по
memory-map, — затем `gzip`. Приложение (`AccentDictionary.kt`) разжимает файл
во внутреннее хранилище, mmap'ит и ищет слово бинарным поиском; `U+0301`
вставляет само.

> **Почему расширение `.dict`, а не `.gz`:** AAPT2 молча распаковывает ассеты
> с расширением `.gz` при упаковке APK и снимает суффикс — приложение не
> находило файл с ожидаемым именем. Нейтральное `.dict` AAPT не трогает.

Проверка результата:

```python
import gzip
with gzip.open('ruaccent_dict_wiktionary.dict', 'rt') as f:
    print(''.join(next(f) for _ in range(5)))   # напр.  привет\t4
```

## Пересборка (Colab или локально)

Оба скрипта пишут файл со «своим» именем и в Colab сами запускают скачивание
(`files.download`); если оно не стартовало — печатают, как забрать файл
вручную. Готовый файл заменить в соответствующей папке ассетов
(`app/src/main/assets/` для Викисловаря, `app/src/dev/assets/` для RUAccent).

**Викисловарь (публичный):**

```bash
python make_dict_wiktionary.py
# качает дамп kaikki (~770 МБ) → ruaccent_dict_wiktionary.dict
```

Если ссылка на дамп вернёт 404 — возьми актуальную «all senses» JSONL-ссылку
со страницы <https://kaikki.org/dictionary/Russian/> и передай `--url <ссылка>`.

**RUAccent (только личное использование / dev):**

```bash
pip install ruaccent
python make_dict_ruaccent.py
# → ruaccent_dict_ruaccent.dict
```

Скачать оба файла отдельной ячейкой в Colab:

```python
from google.colab import files
files.download('ruaccent_dict_wiktionary.dict')
files.download('ruaccent_dict_ruaccent.dict')
```

## Компромиссы

- Омографы (за́мок/замо́к) без нейромодели ставятся «по умолчанию» — редкие
  ошибки на контекстных словах. Максимум качества — режим «Сервер» (нейро-
  RUAccent на своём сервере, см. [`../self-hosted-server.md`](../self-hosted-server.md) §4b).
- Покрытие Викисловаря меньше, чем у RUAccent (887 тыс. против 3.2 млн форм),
  но лицензионно чистое для публикации.
