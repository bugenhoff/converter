"""Обработчики Telegram: /start, приём файлов и фото, кнопка транслитерации."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import InputFile, ReplyParameters, Update
from telegram.ext import ContextTypes

from ..conversion.pipeline import IMAGE_EXTENSIONS, detect_kind
from ..conversion.transliteration import transliterate_docx_bytes
from .auth import log_user_access, require_auth
from .processing import IncomingFile, batch_processor

logger = logging.getLogger(__name__)

# Больше Bot API скачать не даёт.
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

START_TEXT = (
    "Привет! 👋\n\n"
    "Пришли один или несколько файлов .doc, .docx, .pdf или фото/скан страницы "
    f"({', '.join(IMAGE_EXTENSIONS)}) — верну .docx с сохранением оформления.\n\n"
    "Несколько файлов, отправленных подряд, обрабатываются вместе и параллельно. "
    "Под каждым готовым документом есть кнопка «Транслитерация» (латиница → кириллица)."
)
UNSUPPORTED_TEXT = (
    "Я умею конвертировать только .doc, .docx, .pdf и изображения "
    f"({', '.join(IMAGE_EXTENSIONS)})."
)


@require_auth
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(START_TEXT)


@require_auth
async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    document = message.document if message else None
    if not document or not document.file_name:
        return

    log_user_access(update.effective_user.id, update.effective_user.username, f"upload {document.file_name}")
    kind = detect_kind(document.file_name)
    if not kind:
        await message.reply_text(UNSUPPORTED_TEXT)
        return
    await _enqueue(update, context, document, document.file_name, kind)


@require_auth
async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.photo:
        return
    photo = message.photo[-1]
    file_name = f"photo_{photo.file_unique_id}.jpg"
    log_user_access(update.effective_user.id, update.effective_user.username, f"upload {file_name}")
    await _enqueue(update, context, photo, file_name, "image")


async def _enqueue(update: Update, context: ContextTypes.DEFAULT_TYPE, media, file_name: str, kind: str) -> None:
    message = update.message
    if media.file_size and media.file_size > MAX_FILE_SIZE_BYTES:
        await message.reply_text(f"⚠️ {file_name} больше {MAX_FILE_SIZE_MB} МБ — Telegram не даёт ботам скачать такой файл.")
        return

    telegram_file = await media.get_file()
    payload = bytes(await telegram_file.download_as_bytearray())
    if not payload:
        await message.reply_text(f"⚠️ {file_name}: пустой файл.")
        return

    await batch_processor.add(
        context.bot,
        update.effective_chat.id,
        IncomingFile(name=file_name, kind=kind, payload=payload, message_id=message.message_id),
    )


@require_auth
async def transliteration_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Транслитерирует документ из сообщения, под которым нажата кнопка."""
    query = update.callback_query
    message = query.message
    # У сообщений старше 48 часов Telegram отдаёт только id, без документа.
    document = getattr(message, "document", None)
    if document is None:
        await query.answer("Сообщение устарело — отправьте файл заново.", show_alert=True)
        return

    await query.answer("Транслитерирую…")
    log_user_access(update.effective_user.id, update.effective_user.username, f"translit {document.file_name}")
    try:
        telegram_file = await document.get_file()
        docx_bytes = bytes(await telegram_file.download_as_bytearray())
        transliterated = await asyncio.to_thread(transliterate_docx_bytes, docx_bytes)
        output_name = f"{Path(document.file_name or 'document.docx').stem}_cyrillic.docx"
        await context.bot.send_document(
            chat_id=message.chat.id,
            document=InputFile(transliterated, filename=output_name),
            caption=f"✅ Транслитерация: {output_name}",
            reply_parameters=ReplyParameters(message.message_id, allow_sending_without_reply=True),
        )
    except Exception:
        logger.exception("Transliteration failed")
        await context.bot.send_message(message.chat.id, "❌ Ошибка при транслитерации")


async def stale_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопки старых версий бота."""
    if update.callback_query:
        await update.callback_query.answer("Кнопка устарела — просто отправьте файл.")
