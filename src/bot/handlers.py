"""Telegram handlers for document conversion flows."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import ContextTypes

from ..conversion.memory_processor import memory_processor
from .auth import require_auth, log_user_access
from .file_queue import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, queue_manager

logger = logging.getLogger(__name__)


@require_auth
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, "start_command")

    text = (
        "Привет! 👋\n\n"
        "Пришли мне один или несколько файлов в формате .doc или .pdf (даже сканы) — "
        "я автоматически конвертирую их в .docx с сохранением форматирования.\n\n"
        "🕐 Просто отправляй файлы — я жду 10 секунд после каждого файла, "
        "затем автоматически начинаю обработку всей группы и отправляю готовые DOCX файлы!"
    )
    await update.message.reply_text(text)


@require_auth
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    document = update.message.document
    if not document or not document.file_name:
        await update.message.reply_text("Пожалуйста, отправь файл в формате .doc или .pdf.")
        return
    
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, f"upload_file: {document.file_name}")

    if not document.file_name.lower().endswith((".doc", ".pdf")):
        await update.message.reply_text("Я умею конвертировать только `.doc` и `.pdf` файлы.")
        return

    if document.file_size and document.file_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"⚠️ {document.file_name} весит больше {MAX_FILE_SIZE_MB} МБ и не будет обработан."
        )
        return

    telegram_file = await document.get_file()
    file_bytes = await telegram_file.download_as_bytearray()
    actual_size = len(file_bytes)

    if actual_size == 0:
        await update.message.reply_text("⚠️ Получен пустой файл. Проверьте документ и попробуйте снова.")
        return

    if actual_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"⚠️ {document.file_name} весит {actual_size / (1024 * 1024):.1f} МБ — лимит {MAX_FILE_SIZE_MB} МБ."
        )
        return

    try:
        memory_handle = memory_processor.store_bytes(bytes(file_bytes), document.file_name)
    except MemoryError:
        await update.message.reply_text(
            "⚠️ Временная память заполнена. Подождите окончания текущей обработки и попробуйте снова."
        )
        return
    finally:
        del file_bytes

    file_type = "pdf" if document.file_name.lower().endswith(".pdf") else "doc"

    logger.info(
        "Loaded file %s (%d bytes) for user %s entirely in memory",
        document.file_name,
        actual_size,
        update.effective_user.id if update.effective_user else "unknown",
    )

    try:
        await queue_manager.add_file(
            update=update,
            context=context,
            memory_handle=memory_handle,
            original_name=document.file_name,
            file_type=file_type,
            file_size=actual_size,
        )
    except Exception:
        memory_handle.release()
        raise


@require_auth
async def process_queue_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Legacy handler for old 'process' button - now shows info about new system."""
    query = update.callback_query
    if query:
        await query.answer()

    message = update.effective_message
    if not message:
        return
    
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, "legacy_process_queue")

    await message.reply_text(
        "ℹ️ Система обработки обновлена!\n\n"
        "Теперь файлы обрабатываются автоматически:\n"
        "• Отправляй файлы как обычно\n" 
        "• Жду 10 секунд после каждого файла\n"
        "• Автоматически конвертирую и отправляю готовые DOCX\n\n"
        "Кнопка больше не нужна — всё происходит автоматически! 🚀"
    )