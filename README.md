# Telegram DOC→DOCX Converter Bot

Этот репозиторий содержит Telegram-бота, который принимает `.doc`, `.docx`, `.pdf` и изображения (`.png/.jpg/...`), конвертирует их в `.docx` и отсылает результат пользователю.

## Быстрый старт

1. Установите зависимости в виртуальном окружении (протестировано на Python 3.13):

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
   TESSDATA_PREFIX=/root/tesseract/tessdata/
   OCR_LANGUAGES=rus+eng+uzb+uzb_cyrl
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.2-11b-vision-preview
   GROQ_MAX_TOKENS=8000
   GROQ_BATCH_SIZE=3
   GROQ_IMAGE_MAX_SIDE=800
   GROQ_PDF_IMAGE_DPI=200
   TEMP_DIR=./tmp
   LOG_LEVEL=INFO
   ```

5. Запустите бота:

   ```bash
   python bot.py
   ```


## Сценарий работы бота

1. Пользователь отправляет один или несколько документов (`.doc`, `.docx`, `.pdf`) или изображений (включая `photo` из Telegram).
2. Каждый файл скачивается, помещается в очередь и автоматически обрабатывается после тайм-окна (10 секунд).
3. Бот отправляет готовый `.docx` с подписью `✅ source -> result`.
4. Под каждым отправленным `.docx` появляется inline-кнопка `Транслитерация` (латиница -> кириллица, uz).
5. По клику на кнопку бот отправляет отдельный файл `<name>_cyrillic.docx`.

## Что делает бот

- Принимает `.doc`, `.docx`, `.pdf`, изображения (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`) и Telegram `photo`.
- Для PDF использует надежный pipeline: `pdf2docx` -> OCR (`ocrmypdf` + `pdf2docx`) -> Groq fallback.
- Для изображений конвертирует image -> PDF -> OCR -> DOCX.
- Поддерживает транслитерацию узбекской латиницы в кириллицу для готового DOCX через inline-кнопку.
- Чистит временные файлы и сообщает об ошибках (например, если LibreOffice возвращает код ошибки).

## Структура проекта

- `bot.py` — точка входа.
- `src/config/settings.py` — загрузка `.env` и настройка путей.
- `src/conversion/converter.py` — модуль конвертации `.doc/.pdf/.image` с надежным fallback.
- `src/conversion/transliteration.py` — транслитерация узбекской латиницы -> кириллица в DOCX.
- `src/bot/handlers.py` — Telegram-хендлеры (`/start`, прием документов/фото, callback транслитерации).
- `src/bot/queue.py` — утилиты для хранения очереди файлов на уровне чата.
- `src/bot/batching.py` — сборка ZIP-архива с уже конвертированными документами.
- `src/bot/app.py` — инициализация `python-telegram-bot` и запуск приложения.
- `tests/` — базовый тест на защиту конвертера.

## Тестирование

```bash
pytest
```

## Примечания

- Бот работает в режиме polling и рассчитан на небольшие файлы (по умолчанию до 20 МБ).
- Убедитесь, что `TEMP_DIR` доступен для записи, и присваивайте уникальные имена файлам внутри одного запроса.
- Если LibreOffice установлен через Flatpak (`org.libreoffice.LibreOffice`), бот автоматически запустит его через `flatpak run --command=soffice ...`. При необходимости задайте собственную команду с помощью `LIBREOFFICE_PATH`.
- **PDF обработка**: Бот использует deterministic-конвертацию в приоритете; Groq применяется только как последний fallback.  
- OCR использует Tesseract. Настройте `TESSDATA_PREFIX` и `OCR_LANGUAGES`, чтобы перечислить доступные языки (по умолчанию `rus+eng+uzb+uzb_cyrl`).
- **Транслитерация**: по кнопке `Транслитерация` создается новый `<name>_cyrillic.docx`. Обработка выполняется для текстовых слоев DOCX (paragraph/table/header/footer).
- **Groq limits**: лимиты ответа и размера vision-запроса настраиваются через `GROQ_MAX_TOKENS`, `GROQ_BATCH_SIZE`, `GROQ_IMAGE_MAX_SIDE`, `GROQ_PDF_IMAGE_DPI`.
