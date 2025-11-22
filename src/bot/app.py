"""Application factory that wires handlers into python-telegram-bot."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..config.settings import settings
from .handlers import document_handler, process_queue_handler, start_handler

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def build_application() -> Application:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        .build()
    )
    application.add_handler(CommandHandler("start", start_handler))
    # Keep legacy callback handler for old buttons
    application.add_handler(CallbackQueryHandler(process_queue_handler))
    doc_filter = filters.Document.FileExtension("doc") | filters.Document.FileExtension("pdf")
    application.add_handler(MessageHandler(doc_filter, document_handler))
    application.add_error_handler(_error_handler)
    return application


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Произошла непредвиденная ошибка. Попробуйте позже.")


def main() -> None:
    _configure_logging()
    application = build_application()
    application.run_polling(allowed_updates=None)