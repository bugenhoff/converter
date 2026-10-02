"""Сборка приложения python-telegram-bot и запуск."""

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
from .handlers import (
    document_handler,
    photo_handler,
    stale_button_handler,
    start_handler,
    transliteration_handler,
)
from .processing import TRANSLIT_CALLBACK

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_application() -> Application:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        # Пока скачивается один файл, бот принимает следующие.
        .concurrent_updates(True)
        .build()
    )
    application.add_handler(CommandHandler(["start", "help"], start_handler))
    # Старые кнопки имели вид translit:<token>, новые — просто translit.
    application.add_handler(CallbackQueryHandler(transliteration_handler, pattern=rf"^{TRANSLIT_CALLBACK}"))
    application.add_handler(CallbackQueryHandler(stale_button_handler))
    application.add_handler(MessageHandler(filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_error_handler(_error_handler)
    return application


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Произошла непредвиденная ошибка. Попробуйте позже.")


def main() -> None:
    _configure_logging()
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан — заполните .env")
    logger.info(
        "Starting bot: provider=%s model=%s mode=%s concurrency=%d",
        settings.llm_provider,
        settings.llm_model,
        settings.pdf_conversion_mode,
        settings.llm_concurrency,
    )
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)
