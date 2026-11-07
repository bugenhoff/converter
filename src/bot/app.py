"""Сборка приложения python-telegram-bot и запуск."""

from __future__ import annotations

import logging

from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeChat,
    Update,
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..config.settings import settings
from .admin import CALLBACK as ADMIN_CALLBACK
from .admin import admin_callback_handler, admin_command, admin_text_handler
from .group import bot_membership_handler, group_handler, migration_handler
from .handlers import (
    document_handler,
    photo_handler,
    stale_button_handler,
    start_handler,
    transliteration_handler,
)
from .processing import TRANSLIT_CALLBACK
from .state import bot_state

logger = logging.getLogger(__name__)


class _DropPdf2docxInfo(logging.Filter):
    """pdf2docx пишет ход работы в корневой логгер — по строке на страницу."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or "pdf2docx" not in record.pathname


def _configure_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(_DropPdf2docxInfo())
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_application() -> Application:
    application = (
        ApplicationBuilder()
        .token(settings.telegram_token)
        # Пока скачивается один файл, бот принимает следующие.
        .concurrent_updates(True)
        .post_init(_set_commands)
        .build()
    )
    private = filters.ChatType.PRIVATE
    groups = filters.ChatType.GROUPS

    application.add_handler(CommandHandler(["start", "help"], start_handler, filters=private))
    application.add_handler(CommandHandler("admin", admin_command, filters=private))
    # Старые кнопки имели вид translit:<token>, новые — просто translit.
    application.add_handler(CallbackQueryHandler(transliteration_handler, pattern=rf"^{TRANSLIT_CALLBACK}"))
    application.add_handler(CallbackQueryHandler(admin_callback_handler, pattern=rf"^{ADMIN_CALLBACK}(:|$)"))
    application.add_handler(CallbackQueryHandler(stale_button_handler))
    application.add_handler(MessageHandler(private & filters.Document.ALL, document_handler))
    application.add_handler(MessageHandler(private & filters.PHOTO, photo_handler))
    # Кнопка «Администрирование» и ввод своего лимита (остальным текст в личке не нужен).
    application.add_handler(MessageHandler(private & filters.TEXT & ~filters.COMMAND, admin_text_handler))

    application.add_handler(MessageHandler(groups & filters.StatusUpdate.MIGRATE, migration_handler))
    # Только новые сообщения: правка подписи не должна запускать конвертацию повторно.
    application.add_handler(
        MessageHandler(groups & filters.UpdateType.MESSAGE & ~filters.StatusUpdate.ALL, group_handler)
    )
    application.add_handler(ChatMemberHandler(bot_membership_handler, ChatMemberHandler.MY_CHAT_MEMBER))
    application.add_error_handler(_error_handler)
    return application


async def _set_commands(application: Application) -> None:
    try:
        await application.bot.set_my_commands(
            [BotCommand("start", "Как пользоваться")], scope=BotCommandScopeAllPrivateChats()
        )
        await application.bot.set_my_commands(
            [
                BotCommand("convert", "Сконвертировать файл (ответом на сообщение с файлом)"),
                BotCommand("help", "Что умеет бот"),
            ],
            scope=BotCommandScopeAllGroupChats(),
        )
    except Exception:
        logger.warning("Failed to set bot commands", exc_info=True)
    # /admin видят только супер-админы (если уже писали боту).
    for admin_id in settings.super_admin_ids:
        try:
            await application.bot.set_my_commands(
                [BotCommand("start", "Как пользоваться"), BotCommand("admin", "Администрирование")],
                scope=BotCommandScopeChat(admin_id),
            )
        except Exception as exc:
            logger.info("Cannot set commands for super admin %d: %s", admin_id, exc)


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Произошла непредвиденная ошибка. Попробуйте позже.")


def main() -> None:
    _configure_logging()
    if not settings.telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан — заполните .env")
    logger.info(
        "Starting bot: provider=%s model=%s mode=%s concurrency=%d groups=%d auto=%s super_admins=%d",
        settings.llm_provider,
        settings.llm_model,
        settings.pdf_conversion_mode,
        settings.llm_concurrency,
        len(bot_state().allowed_groups()),
        settings.group_auto_convert,
        len(settings.super_admin_ids),
    )
    build_application().run_polling(allowed_updates=Update.ALL_TYPES)
