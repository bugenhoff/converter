# Roadmap

1. ✅ **Scaffold the project** — defined folders, configs, dependencies, and packaging entries.
2. ✅ **Build conversion + bot modules** — converter, handlers, and application factory are in place.
3. ✅ **Document and secure configuration** — README, `.env.example`, and `.gitignore` describe setup.
4. ✅ **Validate with minimal tests** — pytest документация проверена и все три теста проходят.
5. ✅ **Python 3.13 compatibility** — disabled legacy Updater instantiation (`ApplicationBuilder.updater(None)`) and documented the fix.
6. ✅ **Batch queue processing** — добавлены очередь, ZIP-архивация и inline-кнопка «Обработка» + сопровождающие тесты.
7. ✅ **PTB 21 upgrade** — обновлен `python-telegram-bot` до 21.5 и устранены временные патчи.
8. ✅ **PDF Support** — добавлена поддержка конвертации PDF (включая OCR для сканов) в DOCX с сохранением форматирования.
9. ✅ **Flatpak + OCR hardening** — автоопределение LibreOffice через Flatpak и переработанный конвейер OCR → DOCX с настраиваемыми языками.
10. ✅ **Groq LLM Integration** — добавлена интеллектуальная обработка PDF через vision модели с fallback на OCR метод.
11. ✅ **Queue Timer Fix** — исправлена критическая ошибка в системе очередей: устранена зависимость от Update объекта в асинхронном таймере, переписаны уведомления на context.bot.send_message.
12. ✅ **Memory-first queue redesign** — реализован RAM-буфер (50 МБ лимит/фолбэки), усилено логирование таймера; ждём фидбек после боевых прогонов.
13. ✅ **Fix Silent Processing** — исправлена ошибка отступов в уведомлениях очереди, добавлено расширенное логирование этапов конвертации и Groq LLM.