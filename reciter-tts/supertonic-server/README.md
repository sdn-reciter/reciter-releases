# Supertonic-3 сервер — тест лёгкой модели рядом с XTTS

Экспериментальный TTS-сервер на [Supertonic-3](https://huggingface.co/Supertone/supertonic-3)
(~99M параметров, ONNX, 31 язык включая русский). API полностью совпадает с
XTTS-сервером (`/tts`, `/accent`, `/voices`, `/health`), поэтому приложение
переключается между движками **простой сменой адреса сервера** — оба
контейнера работают одновременно:

| Сервер | Порт | Голоса |
|---|---|---|
| XTTS-v2 | 8002 | клоны по wav-образцу |
| Supertonic-3 | **8003** | пресеты F1–F5 / M1–M5 + JSON-стили Voice Builder |

## Запуск

```bash
# 0. Один раз: общий базовый образ (CUDA+torch+RUAccent) — см. ../base/README.md
docker build -t reciter-tts-base:cu128 ../base

mkdir -p /mnt/data/reciter-tts/supertonic-server && cd /mnt/data/reciter-tts/supertonic-server
# скопируйте сюда файлы этой папки (Dockerfile, docker-compose.yml, server.py, .env.example)
cp .env.example .env
docker compose build && docker compose up -d
# ассеты (~404 МБ) качаются ОДИН РАЗ при сборке (в образ), старт быстрый
docker compose logs -f   # ждать "SERVER_VERSION st-1" и список голосов
```

Проверка:

```bash
curl http://localhost:8003/health
curl -X POST http://localhost:8003/tts \
  -F 'text=Привет! Это тест лёгкой модели Супертоник.' -F 'language=ru' -F 'voice=F1' \
  -o test.wav && aplay test.wav
```

## Переключение в приложении

Настройки чтения → «Свой TTS-сервер» → адрес `http://<IP-ПК>:8003`.
Голос выбирается там же (пресеты F1–F5, M1–M5). Возврат на XTTS — смена
адреса обратно на `:8002`.

## Отличия от XTTS

- **Нет клонирования по аудио.** Загрузка wav в менеджере голосов вернёт
  понятную ошибку. Свой голос делается через
  [Voice Builder](https://supertone.ai) → полученный `.json` кладётся в
  `./styles/` (подхватывается на лету) или загружается через `/voices/upload`.
- **Скорость.** Модель быстрее реального времени даже на CPU; GPU
  (onnxruntime-gpu, CUDAExecutionProvider) включён в образ.
- **Ударения — важное ограничение.** Supertonic работает по СЫРЫМ символам,
  без G2P и без фонетической разметки (это его архитектурная особенность —
  «no grapheme-to-phoneme, no phonetic annotations»). Значит метки ударений он
  **не читает в принципе**: ни словарь на устройстве, ни серверный `/accent`
  не меняют, куда падает ударение. Именно поэтому синтез звучит **одинаково**
  при любых настройках словаря, а ошибочные ударения не исправить. Поэтому
  `SUPERTONIC_SYNTH_STRESS=strip` — единственно верный режим (`mark`/`plus`
  лишь подсунут модели неизвестный символ). Эндпоинт `/accent` (RUAccent)
  остаётся для совместимости API, но на сам синтез Supertonic не влияет.
  **Нужно управляемое ударение — используйте ESpeech (F5), он stress-aware.**
  Текст, уходящий в модель, теперь виден в логе строкой `synth[...] ...`.
- **Лицензия.** Код MIT, модель OpenRAIL-M (свободная, с ограничениями на
  злоупотребления).

## Откат

Эксперимент живёт в отдельной ветке `claude/supertonic-test` и отдельном
контейнере. Откат: `docker compose down` в этой папке — XTTS-сервер на 8002
не затрагивается.
