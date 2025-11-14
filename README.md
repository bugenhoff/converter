# Telegram DOC→DOCX Converter Bot

Этот репозиторий содержит Telegram-бота, который принимает файлы в устаревшем формате `.doc`, запускает `LibreOffice` в безголовом режиме и отсылает пользователю документ в формате `.docx` с сохраненным форматированием.

## Быстрый старт

1. Установите зависимости в виртуальном окружении:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Установите `LibreOffice` (должен быть доступен в PATH или укажите путь в `LIBREOFFICE_PATH`).
3. Скопируйте `.env.example` → `.env` и заполните значения переменных:

   ```env
   TELEGRAM_BOT_TOKEN=123456:ABCDEF
   LIBREOFFICE_PATH=libreoffice
   TEMP_DIR=./tmp
   LOG_LEVEL=INFO
   ```

4. Запустите бота:

   ```bash
   python bot.py
   ```

## Что делает бот

- Приветствует пользователя и объясняет, что он ожидает файл `.doc` и вернет `.docx`.
- Скачивает документ в `TEMP_DIR`, запускает `LibreOffice` для конвертации и возвращает оригинальное содержимое в новом формате.
- Чистит временные файлы и сообщает об ошибках (например, если LibreOffice возвращает код ошибки).

## Структура проекта

- `bot.py` — точка входа.
- `src/config/settings.py` — загрузка `.env` и настройка путей.
- `src/conversion/converter.py` — модуль конвертации через `libreoffice`.
- `src/bot/handlers.py` — Telegram-хендлеры (команда `/start` и обработка документов).
- `src/bot/app.py` — инициализация `python-telegram-bot` и запуск приложения.
- `tests/` — базовый тест на защиту конвертера.

## Тестирование

```bash
pytest
```

## Примечания

- Бот работает в режиме polling и рассчитан на небольшие файлы (по умолчанию до 20 МБ).
- Убедитесь, что `TEMP_DIR` доступен для записи, и присваивайте уникальные имена файлам внутри одного запроса.