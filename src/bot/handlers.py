"""Telegram handlers for document conversion flows."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.converter import ConversionError, convert_doc_to_docx

logger = logging.getLogger(__name__)


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Привет! Я принимаю файлы в формате .doc и возвращаю .docx с сохранённым форматированием."
        " Просто отправь документ, и я вышлю обратно конвертированную копию."
    )
    await update.message.reply_text(text)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    document = update.message.document
    if not document or not document.file_name:
        await update.message.reply_text("Пожалуйста, отправь файл в формате .doc.")
        return

    if not document.file_name.lower().endswith(".doc"):
        await update.message.reply_text("Я умею конвертировать только `.doc` файлы.")
        return

    await update.message.reply_text("Получил файл, запускаю LibreOffice для конвертации...")
    temp_dir = settings.temp_dir
    download_path = temp_dir / document.file_name
    converted_path: Path | None = None

    file = await document.get_file()
    await file.download_to_drive(custom_path=download_path)

    try:
        converted_path = await asyncio.to_thread(
            convert_doc_to_docx,
            download_path,
            temp_dir,
            settings.libreoffice_path,
        )

        with converted_path.open("rb") as infile:
            await update.message.reply_document(
                InputFile(infile, filename=converted_path.name),
                caption="Готово — вот ваш файл в формате .docx",
            )
    except ConversionError:
        logger.exception("Conversion failed", exc_info=True)
        await update.message.reply_text(
            "Не удалось конвертировать документ. Проверьте настройки LibreOffice и повторите попытку."
        )
    finally:
        for path in (download_path, converted_path):
            if path and path.exists():
                try:
                    path.unlink()
                except OSError:
                    logger.debug("Не удалось удалить временный файл %s", path)