"""Telegram handlers for document conversion flows."""

from __future__ import annotations

import logging
from pathlib import Path

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from ..conversion.transliteration import transliterate_docx_bytes
from ..conversion.memory_processor import memory_processor
from .auth import check_user_access, log_user_access, require_auth
from .file_queue import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, queue_manager

logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@require_auth
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, "start_command")

    text = (
        "Привет! 👋\n\n"
        "Пришли мне один или несколько файлов в формате .doc, .docx, .pdf или изображение "
        "(.png/.jpg и т.д.) — "
        "я автоматически конвертирую их в .docx с сохранением форматирования.\n\n"
        "🕐 Просто отправляй файлы — я жду 10 секунд после каждого файла, "
        "затем автоматически начинаю обработку всей группы и отправляю готовые DOCX файлы!"
    )
    await update.message.reply_text(text)


def _detect_file_type(file_name: str) -> str | None:
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        return "pdf"
    if lower_name.endswith(".docx"):
        return "docx"
    if lower_name.endswith(".doc"):
        return "doc"
    if lower_name.endswith(SUPPORTED_IMAGE_EXTENSIONS):
        return "image"
    return None


async def _enqueue_uploaded_bytes(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    file_name: str,
    payload: bytes,
    file_type: str,
) -> None:
    actual_size = len(payload)
    if actual_size == 0:
        await update.message.reply_text("⚠️ Получен пустой файл. Проверьте документ и попробуйте снова.")
        return

    if actual_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"⚠️ {file_name} весит {actual_size / (1024 * 1024):.1f} МБ — лимит {MAX_FILE_SIZE_MB} МБ."
        )
        return

    try:
        memory_handle = memory_processor.store_bytes(payload, file_name)
    except MemoryError:
        await update.message.reply_text(
            "⚠️ Временная память заполнена. Подождите окончания текущей обработки и попробуйте снова."
        )
        return

    logger.info(
        "Loaded file %s (%d bytes) for user %s entirely in memory",
        file_name,
        actual_size,
        update.effective_user.id if update.effective_user else "unknown",
    )

    try:
        await queue_manager.add_file(
            update=update,
            context=context,
            memory_handle=memory_handle,
            original_name=file_name,
            file_type=file_type,
            file_size=actual_size,
        )
    except Exception:
        memory_handle.release()
        raise


@require_auth
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    document = update.message.document
    if not document or not document.file_name:
        await update.message.reply_text(
            "Пожалуйста, отправь файл в формате .doc, .docx, .pdf или изображение (.png/.jpg)."
        )
        return
    
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, f"upload_file: {document.file_name}")

    file_type = _detect_file_type(document.file_name)
    if not file_type:
        await update.message.reply_text(
            "Я умею конвертировать только `.doc`, `.docx`, `.pdf` и изображения "
            "(`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`)."
        )
        return

    if document.file_size and document.file_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"⚠️ {document.file_name} весит больше {MAX_FILE_SIZE_MB} МБ и не будет обработан."
        )
        return

    telegram_file = await document.get_file()
    file_bytes = await telegram_file.download_as_bytearray()
    try:
        await _enqueue_uploaded_bytes(
            update=update,
            context=context,
            file_name=document.file_name,
            payload=bytes(file_bytes),
            file_type=file_type,
        )
    finally:
        del file_bytes


@require_auth
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    photo = update.message.photo[-1]
    file_name = f"photo_{photo.file_unique_id}.jpg"
    user = update.effective_user
    if user:
        log_user_access(user.id, user.username, f"upload_photo: {file_name}")

    if photo.file_size and photo.file_size > MAX_FILE_SIZE_BYTES:
        await update.message.reply_text(
            f"⚠️ Фото весит больше {MAX_FILE_SIZE_MB} МБ и не будет обработано."
        )
        return

    telegram_file = await photo.get_file()
    file_bytes = await telegram_file.download_as_bytearray()
    try:
        await _enqueue_uploaded_bytes(
            update=update,
            context=context,
            file_name=file_name,
            payload=bytes(file_bytes),
            file_type="image",
        )
    finally:
        del file_bytes


async def transliteration_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return

    user = update.effective_user
    if not user:
        await query.answer("Не удалось определить пользователя", show_alert=True)
        return

    if not check_user_access(user.id):
        await query.answer("🚫 Доступ запрещён", show_alert=True)
        return

    token = query.data.split(":", 1)[1] if ":" in query.data else ""
    if not token:
        await query.answer("Некорректный токен транслитерации", show_alert=True)
        return

    job, error = queue_manager.consume_transliteration_job(token, user.id)
    if error == "forbidden":
        await query.answer("Кнопка принадлежит другому пользователю", show_alert=True)
        return
    if error == "not_found" or not job:
        await query.answer("Ссылка устарела. Запустите конвертацию заново.", show_alert=True)
        return

    chat_id = update.effective_chat.id if update.effective_chat else None
    if not chat_id:
        await query.answer("Не удалось определить чат", show_alert=True)
        return

    log_user_access(user.id, user.username, f"transliteration_request: {job.docx_name}")

    try:
        telegram_file = await context.bot.get_file(job.file_id)
        docx_bytes = await telegram_file.download_as_bytearray()
        transliterated = transliterate_docx_bytes(bytes(docx_bytes))
        output_name = f"{Path(job.docx_name).stem}_cyrillic.docx"

        await context.bot.send_document(
            chat_id=chat_id,
            document=InputFile(transliterated, filename=output_name),
            caption=f"✅ Транслитерация готова: {output_name}",
        )
        await query.answer("Транслитерация завершена")
    except Exception as exc:
        logger.exception("Transliteration callback failed", exc_info=exc)
        await query.answer("Ошибка при транслитерации", show_alert=True)


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
