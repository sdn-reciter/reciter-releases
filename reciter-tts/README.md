# Reciter TTS — свои голосовые серверы

Self-hosted TTS-серверы для приложения Reciter и общий протокол: приложение
шлёт текст — сервер возвращает аудио.

> ## ⚙️ Это примеры под Linux + NVIDIA RTX 50xx
>
> Комплекты собраны и проверены на **Linux (Ubuntu)** с видеокартой
> **NVIDIA GeForce RTX 50xx** (сборка под **CUDA 12.8 / sm_120**). Это рабочая
> отправная точка, а **не универсальная конфигурация**.
>
> На другом железе или ОС файлы, скорее всего, придётся поправить: версии
> CUDA/torch в `Dockerfile`, модель и лимиты VRAM в `.env`, при отсутствии GPU —
> CPU-режим.
>
> **Проще всего — отдайте `Dockerfile`, `docker-compose.yml` и `.env` вашего
> движка ИИ-ассистенту** (ChatGPT / Claude и т.п.) и попросите адаптировать под
> ваши характеристики: модель и серию GPU, объём VRAM, версию CUDA/драйвера,
> запуск на CPU. Так быстрее всего получить рабочий конфиг под своё «железо».

- Протокол (если пишете свой движок) — [`server-api.md`](server-api.md).

## Серверы

| Порт | Папка | Движок | Лицензия | Ударения |
|---|---|---|---|---|
| 8002 | [`xtts-server`](xtts-server/README.md) | XTTS-v2, клон голоса | ⚠ CPML (некоммерч.) | сам; метки — только с файнтюном |
| 8003 | [`supertonic-server`](supertonic-server/README.md) | Supertonic | MIT / OpenRAIL | сам (метки не читает) |
| 8004 | [`voxcpm-server`](voxcpm-server/README.md) | VoxCPM | Apache-2.0 | сам |
| 8005 | [`espeech-server`](espeech-server/README.md) | ESpeech (F5) | Apache-2.0 | управляемые (по меткам) |
| 8006 | [`chatterbox-server`](chatterbox-server/README.md) | Chatterbox | MIT | сам |
| 8007 | [`cosyvoice-server`](cosyvoice-server/README.md) | Fun-CosyVoice3 | Apache-2.0 | сам |

Все, кроме XTTS, — с коммерчески пригодной лицензией. Детали и особенности — в
README каждой папки.

## Запуск

Серверы наследуют общий базовый образ (CUDA + torch + RUAccent) — он собирается
один раз, дальше пересборка движка занимает минуты.

```bash
# 1. База — однократно (порядок и структура на диске — base/README.md)
cd base && docker build -t reciter-tts-base:cu128 . && cd ..

# 2. Нужный движок
cd xtts-server
cp .env.example .env           # ключ, голос по умолчанию
docker compose up -d --build   # первый старт качает модель
```

## API-ключ

Ключ защищает сервер: если он задан, каждый запрос обязан прислать его, иначе
`401`. Пустой ключ = без авторизации (**только** для изолированной домашней
сети). В `.env` каждого движка своя переменная: `XTTS_API_KEY`,
`SUPERTONIC_API_KEY`, `VOXCPM_API_KEY`, `ESPEECH_API_KEY`,
`CHATTERBOX_API_KEY`, `COSYVOICE_API_KEY`.

Сгенерировать ключ и сразу записать его в `.env` (пример для XTTS — подставьте
имя переменной своего движка):

```bash
cd xtts-server
KEY=$(openssl rand -hex 32)                       # случайный ключ (64 hex-символа)
sed -i "s/^XTTS_API_KEY=.*/XTTS_API_KEY=$KEY/" .env   # вписать в .env
echo "Ваш ключ: $KEY"                             # показать — скопируйте в приложение
docker compose up -d --force-recreate             # применить
```

Тот же ключ впишите в приложении (см. ниже). **Посмотреть ключ позже**, если
потеряли:

```bash
grep API_KEY /mnt/data/reciter-tts/xtts-server/.env
```

Сменить ключ — впишите новое значение в `.env`, `docker compose up -d
--force-recreate`, и обновите его в приложении.

## Docker не видит видеокарту

```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

Драйвер видеокарты тут ни при чём: так Docker сообщает, что у него не
подключён NVIDIA Container Toolkit — прослойка, которая пробрасывает карту
внутрь контейнера. Чаще всего это вылезает после обновления Docker или
драйвера, когда настройка слетела. Проверять по порядку:

```bash
nvidia-smi                      # 1. карту видит сама система?
nvidia-ctk --version            # 2. прослойка установлена?
docker info | grep -i runtime   # 3. Docker про неё знает? (нужна строка nvidia)
```

1. Если `nvidia-smi` ругается на несовпадение версий драйвера и библиотеки —
   систему после обновления не перезагружали; перезагрузка лечит.
2. Если `nvidia-ctk` не найден — прослойки нет (или её снёс `apt autoremove`,
   оставив запись о рантайме в настройках Docker). Ставится из репозитория
   NVIDIA, на Ubuntu/Debian:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

3. Зарегистрировать прослойку и перезапустить демон:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all reciter-tts-base:cu128 nvidia-smi   # карта видна из контейнера
```

Последняя команда должна напечатать таблицу `nvidia-smi`. После неё
`docker compose up -d` поднимается как обычно.

Отдельный случай — ошибка про CDI:

```
failed to discover GPU vendor from CDI: no known GPU vendor found
```

Docker с версии 28 ищет карту через CDI — описание устройства, которое
генерирует та же прослойка и в котором записаны версии библиотек драйвера.
Файла нет или он остался от прежнего драйвера; пересоздать и проверить:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list          # должен появиться nvidia.com/gpu=all
```

Описание привязано к версии драйвера, поэтому после каждого его обновления
команду повторяют.

## Подключение в приложении

«Параметры → Чтение вслух → движок **«Свой сервер»**»:

- **Адрес** — `http://<IP>:<порт>` (в LAN или через VPN допустим `http`),
  либо `https://…`.
- **Ключ** — значение `API_KEY` сервера, если задан.
- «Тест голоса» проверяет связь.

Приложение синтезирует по абзацу и подгружает следующие заранее. Скорость и
высоту накладывает само при воспроизведении — сервер синтезирует на нейтральной
скорости.

## Доступ снаружи — Tailscale

Слушаете только вы, поэтому порт в интернет открывать не нужно. Проще и
безопаснее всего — **Tailscale** (WireGuard-сеть, наружу ничего не светится):

```bash
# на ПК с сервером
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4        # частный адрес, напр. 100.101.102.103
```

Поставьте Tailscale на телефон (тот же аккаунт). Адрес в приложении —
`http://100.101.102.103:<порт>`. Трафик внутри туннеля уже шифрован, но
`API_KEY` всё равно задайте — сменить ключ проще, чем перенастраивать VPN.

## Безопасность

- Порт TTS напрямую в интернет **не пробрасывать** — только через Tailscale.
- `API_KEY` задан всегда, даже поверх VPN.
- Cleartext `http://` — только в LAN или внутри туннеля; наружу нельзя.
