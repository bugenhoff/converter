"""Обработчики личного чата: /start, приём файлов и фото, кнопка транслитерации."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from telegram import InputFile, ReplyParameters, Update
from telegram.ext import ContextTypes

from ..conversion.pipeline import IMAGE_EXTENSIONS, detect_kind
from ..conversion.transliteration import transliterate_docx_bytes
from .admin import admin_keyboard
from .auth import is_super_admin, log_user_access, require_auth
from .processing import enqueue_media

logger = logging.getLogger(__name__)

START_TEXT = (
    "Привет! 👋\n\n"
    "Пришли один или несколько файлов .doc, .docx, .wps, .pdf или фото/скан страницы "
    f"({', '.join(IMAGE_EXTENSIONS)}) — верну .docx с сохранением оформления.\n\n"
    "Несколько файлов, отправленных подряд, обрабатываются вместе и параллельно. "
    "Под каждым готовым документом есть кнопка «Транслитерация» (латиница → кириллица)."
)
UNSUPPORTED_TEXT = (
    "Я умею конвертировать только .doc, .docx, .wps, .pdf и изображения "
    f"({', '.join(IMAGE_EXTENSIONS)})."
)


@require_auth
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        keyboard = admin_keyboard() if is_super_admin(update.effective_user.id) else None
        await update.message.reply_text(START_TEXT, reply_markup=keyboard)


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
    await enqueue_media(
        context.bot,
        message,
        media,
        file_name,
        kind,
        message_id=message.message_id,
        user_id=update.effective_user.id,
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
