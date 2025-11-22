"""Telegram handlers for document conversion flows."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from telegram import InputFile, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.converter import ConversionError, convert_doc_to_docx, convert_pdf_to_docx
from .batching import build_zip_archive
from .queue import QUEUE_KEY, enqueue_file, flush_queue

logger = logging.getLogger(__name__)

PROCESS_CALLBACK_DATA = "process_queue"


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    text = (
        "Привет! Пришли мне один или несколько файлов в формате .doc или .pdf (даже сканы) — я конвертирую их в .docx с сохранением"
        " форматирования. После того как загрузишь всё, просто нажми кнопку «Обработка» под моими ответами —"
        " я соберу один архив со всеми файлами без лишнего спама в чат."
    )
    await update.message.reply_text(text)


async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    document = update.message.document
    if not document or not document.file_name:
        await update.message.reply_text("Пожалуйста, отправь файл в формате .doc.")
        return

    if not document.file_name.lower().endswith((".doc", ".pdf")):
        await update.message.reply_text("Я умею конвертировать только `.doc` и `.pdf` файлы.")
        return

    temp_dir = settings.temp_dir
    download_path = _unique_download_path(document.file_name)
    converted_path: Path | None = None

    file = await document.get_file()
    await file.download_to_drive(custom_path=download_path)
    logger.info("Downloaded file %s to %s", document.file_name, download_path)

    try:
        if document.file_name.lower().endswith(".doc"):
            logger.info("Starting DOC conversion for %s", document.file_name)
            converted_path = await asyncio.to_thread(
                convert_doc_to_docx,
                download_path,
                temp_dir,
                settings.libreoffice_path,
            )
        else:
            logger.info("Starting PDF conversion for %s", document.file_name)
            converted_path = await asyncio.to_thread(
                convert_pdf_to_docx,
                download_path,
                temp_dir,
            )
        
        logger.info("Conversion successful for %s -> %s", document.file_name, converted_path)

        arcname = _build_unique_arcname(document.file_name, context)
        queue_size = enqueue_file(context.chat_data, converted_path, arcname)
        await update.message.reply_text(
            _queue_update_message(document.file_name, queue_size),
            reply_markup=_build_process_keyboard(),
        )
    except ConversionError:
        logger.exception("Conversion failed", exc_info=True)
        await update.message.reply_text(
            "Не удалось конвертировать документ. Проверьте настройки LibreOffice и повторите попытку."
        )
        if converted_path and converted_path.exists():
            converted_path.unlink(missing_ok=True)
    finally:
        if download_path.exists():
            try:
                download_path.unlink()
            except OSError:
                logger.debug("Не удалось удалить временный файл %s", download_path)


async def process_queue_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query:
        await query.answer()

    message = update.effective_message
    if not message:
        return

    documents = flush_queue(context.chat_data)
    existing_docs = [(path, name) for path, name in documents if path.exists()]
    missing = len(documents) - len(existing_docs)

    if missing:
        logger.warning("Skipped %d missing files from queue", missing)

    if not existing_docs:
        await message.reply_text(
            "Очередь пуста. Отправь .doc файлы и нажми кнопку «Обработка», когда все будут готовы.",
            reply_markup=_build_process_keyboard(),
        )
        return

    try:
        archive_path = await asyncio.to_thread(build_zip_archive, existing_docs, settings.temp_dir)
    except ValueError:
        await message.reply_text(
            "Не удалось сформировать архив: список файлов пуст. Попробуй отправить документы заново.",
            reply_markup=_build_process_keyboard(),
        )
        for path, arcname in existing_docs:
            enqueue_file(context.chat_data, path, arcname)
        return

    try:
        with archive_path.open("rb") as archive_file:
            await message.reply_document(
                InputFile(archive_file, filename=archive_path.name),
                caption=f"Готово: {len(existing_docs)} файл(ов) в одном архиве.",
            )
    except Exception:
        logger.exception("Failed to send archive", exc_info=True)
        for path, arcname in existing_docs:
            enqueue_file(context.chat_data, path, arcname)
        raise
    finally:
        for path, _ in existing_docs:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    logger.debug("Не удалось удалить файл %s", path)

        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                logger.debug("Не удалось удалить архив %s", archive_path)


def _unique_download_path(original_file_name: str) -> Path:
    safe_name = Path(original_file_name).name or "document.doc"
    return settings.temp_dir / f"{uuid4().hex}_{safe_name}"


def _build_unique_arcname(original_file_name: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    queue = context.chat_data.get(QUEUE_KEY, [])
    existing = {item.get("arcname") for item in queue if item.get("arcname")}
    base = Path(original_file_name).stem or "document"
    suffix = ".docx"
    candidate = f"{base}{suffix}"
    counter = 1
    while candidate in existing:
        candidate = f"{base}_{counter}{suffix}"
        counter += 1
    return candidate


def _queue_update_message(original_file_name: str, queue_size: int) -> str:
    if queue_size == 1:
        return (
            f"Файл {original_file_name} сконвертирован и добавлен в очередь."
            " Когда закончишь загружать документы, нажми кнопку «Обработка», и я соберу архив с результатами."
        )
    return (
        f"Файл {original_file_name} сконвертирован. В очереди {queue_size} файл(ов)."
        " Когда загрузишь всё, нажми кнопку «Обработка» — я отправлю один архив без спама."
    )


def _build_process_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("Обработка", callback_data=PROCESS_CALLBACK_DATA)]]
    )