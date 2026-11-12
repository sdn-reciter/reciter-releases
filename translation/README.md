# Перевод интерфейса Reciter

Reciter можно перевести на свой язык или поправить готовый русский и
английский — без программирования, в обычном текстовом файле. Перевод
загружается в приложение и сразу появляется в выборе языка.

*English version is below.*

## Что понадобится

Само приложение и любой текстовый редактор, который сохраняет файлы в UTF-8.
Перевод живёт на телефоне: его не нужно никуда отправлять.

## Перевод вручную

1. Откройте **Параметры → Язык → Свой перевод интерфейса → Выгрузить**.
   Сохранится файл `reciter-translation.txt` со всеми надписями приложения.
2. У каждой надписи в файле — строки по языкам:

   ```
   # ru: Добавить книгу
   # en: Add book
   lib_add_book.ru = Добавить книгу
   lib_add_book.en = Add book
   ```

   Строки с `#` — исходный текст приложения, для справки.
3. Новый язык — допишите под каждой надписью строку со своим кодом языка
   (`de`, `fr`, `es`, `pl`…), переводя с русского или английского:

   ```
   lib_add_book.de = Buch hinzufügen
   ```

   Поправить русский или английский — измените текст после `=` в строке `.ru`
   или `.en`.
4. Откройте **Загрузить** и выберите файл. Язык появится в **Параметры → Язык**.

Не обязательно переводить всё сразу: надпись без перевода покажется по-английски.

## ИИ-перевод

Языковая модель на телефоне может перевести надписи сама:
**Параметры → Язык → Свой перевод интерфейса → Перевести с помощью ИИ**,
выберите, чем переводить, и язык:

- **ИИ-модель** — языковая модель на телефоне: точнее в коротких надписях,
  но медленно.
- **Google (быстро)** — переводчик Google на телефоне: весь интерфейс за
  несколько минут, модель языка — десятки МБ, скачивается сама.

- Перевод идёт в фоне: весь интерфейс — около часа, дольше, если в это
  время читает голос (модель тогда уступает ему процессор). Ход видно в
  настройках и в уведомлении. Его можно остановить и запустить снова — он
  продолжит с того места, где остановился.
- Переводятся только новые и изменившиеся надписи. Ваши правки в файле
  остаются.
- Строки, переведённые ИИ, помечены «переведено ИИ — проверьте». Их стоит
  перечитать: модель небольшая и в коротких надписях иногда ошибается.
- В списке есть языки, выбранные в системе телефона, — даже если их нет
  среди остальных.
- Нужна языковая модель на телефоне. Если её нет, под кнопкой
  появится строка, где её можно скачать.

## После обновления приложения

С новой версией появляются новые надписи, а некоторые меняются. Ваш перевод
при этом сохраняется. В настройках будет видно, сколько надписей без перевода
и сколько изменилось.

Чтобы дополнить перевод: снова **Выгрузить** — у новых надписей строка вашего
языка будет пустой, а у изменившихся появится пометка с прежним текстом.
Допишите и перечитайте их, затем **Загрузить**. Или запустите ИИ-перевод: он
возьмёт только новое и изменившееся.

## Свои языки по одному

Под кнопками выгрузки — список своих языков. **Дополнить** переводит только
новые и изменившиеся надписи этого языка, **Убрать** удаляет один язык, не
трогая остальные. **Сбросить** удаляет весь свой перевод.

## Правила файла

- Ключ слева от `=` и сам знак `=` не меняйте — меняется только текст справа.
- Подстановки `%1$s`, `%2$d`, `%d` оставьте как есть: на их место приложение
  вставляет числа и названия. Порядок слов вокруг них можно менять.
- Перенос строки внутри надписи пишется как `\n`.
- Сохраняйте файл в UTF-8.
- Пустая строка после `=` означает «нет перевода».

## Если что-то пошло не так

- **«В файле нет строк вида „ключ.язык = текст“»** — файл сохранён не в UTF-8
  или строки испорчены. Выгрузите таблицу заново и правьте только текст после `=`.
- **Надпись осталась английской** — у неё нет строки вашего языка, или в
  переводе потерялась подстановка `%1$s`.
- **Вернуть всё как было** — **Сбросить** в том же разделе: свой перевод
  удалится, надписи станут встроенными.

---

# Translating the Reciter interface

You can translate Reciter into your language, or correct the built-in Russian
and English, in a plain text file — no programming needed. The translation is
loaded into the app and appears in the language list right away.

## What you need

The app and any text editor that saves UTF-8. The translation stays on your
phone; you don't need to send it anywhere.

## Translating by hand

1. Open **Settings → Language → Your own interface translation → Export**.
   This saves `reciter-translation.txt` with every label of the app.
2. Each label has one line per language:

   ```
   # ru: Добавить книгу
   # en: Add book
   lib_add_book.ru = Добавить книгу
   lib_add_book.en = Add book
   ```

   Lines starting with `#` show the app's own text, for reference.
3. For a new language add a line with its code (`de`, `fr`, `es`, `pl`…) under
   each label, translating from Russian or English:

   ```
   lib_add_book.de = Buch hinzufügen
   ```

   To correct Russian or English, change the text after `=` in the `.ru` or
   `.en` line.
4. Tap **Import** and pick the file. The language appears in **Settings → Language**.

You don't have to translate everything at once: an untranslated label is shown
in English.

## AI translation

The on-device language model can translate the labels for you:
**Settings → Language → Your own interface translation → Translate with AI**,
choose the engine and a language:

- **AI model** — the on-device language model: better with short labels, but slow.
- **Google (fast)** — Google's on-device translator: the whole interface in a few
  minutes; the language model is tens of MB and downloads by itself.

- It runs in the background: about an hour for the whole interface, longer
  while a voice is reading aloud (the model then leaves the processor to it);
  progress is shown in settings and in a notification. You can stop it and
  start again — it continues where it stopped.
- Only new and changed labels are translated. Your own edits stay.
- Lines translated by AI are marked "translated by AI — please check". Read
  them through: the model is small and sometimes gets short labels wrong.
- The list includes the languages set in your phone's system, even if they
  are not among the others.
- An on-device language model is needed. If there is none, a row to download it
  appears under the button.

## After an app update

A new version brings new labels and changes some old ones. Your translation is
kept. Settings show how many labels are untranslated and how many have changed.

To bring the translation up to date: **Export** again — new labels have an
empty line for your language, changed ones get a note with the previous text.
Fill them in, read them through, then **Import**. Or run the AI translation: it
takes only what is new or changed.

## Your languages one by one

Below the export buttons is a list of your languages. **Update** translates only
the new and changed labels of that language, **Remove** deletes one language and
leaves the others. **Reset** deletes your whole translation.

## File rules

- Don't change the key left of `=` or the `=` itself — only the text on the right.
- Keep the placeholders `%1$s`, `%2$d`, `%d` as they are: the app puts numbers
  and names there. You may move them within the sentence.
- A line break inside a label is written as `\n`.
- Save the file as UTF-8.
- An empty text after `=` means "no translation".

## If something goes wrong

- **"The file has no 'key.language = text' lines"** — the file is not UTF-8 or
  the lines are damaged. Export the table again and edit only the text after `=`.
- **A label stays in English** — it has no line for your language, or the
  translation lost a placeholder like `%1$s`.
- **Back to the original** — **Reset** in the same section deletes your
  translation; the built-in labels return.
