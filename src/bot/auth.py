"""Доступ к боту только для пользователей из ALLOWED_USER_IDS."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from telegram import Update
from telegram.ext import ContextTypes

from ..config.settings import settings

logger = logging.getLogger(__name__)
ACCESS_DENIED = "🚫 Доступ запрещён"


def check_user_access(user_id: int) -> bool:
    if not settings.allowed_users_only:
        return True
    return user_id in settings.allowed_user_ids


def require_auth(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> Any:
        user = update.effective_user
        if not user:
            return None
        if not check_user_access(user.id):
            logger.info("Access denied for user %d (@%s)", user.id, user.username or "-")
            if update.callback_query:
                await update.callback_query.answer(ACCESS_DENIED, show_alert=True)
            elif update.effective_message:
                await update.effective_message.reply_text(ACCESS_DENIED)
            return None
        return await func(update, context, *args, **kwargs)

    return wrapper


def log_user_access(user_id: int, username: str | None, action: str) -> None:
    logger.info("User %d (@%s): %s", user_id, username or "-", action)
