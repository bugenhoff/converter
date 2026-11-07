"""Доступ к боту.

- Личный чат — пользователи из ALLOWED_USER_IDS и супер-админы.
- Группа из ALLOWED_CHAT_IDS или подключённая супер-админом — все её
  участники, но только в этой группе: в личке они по-прежнему не допущены,
  если их нет в ALLOWED_USER_IDS.
- Группу, убранную супер-админом, бот не обслуживает, пока супер-админ
  снова не вызовет его в ней.
- ALLOWED_USERS_ONLY=false открывает бота всем и везде (кроме убранных групп).
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from telegram import Chat, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from .state import bot_state

logger = logging.getLogger(__name__)
ACCESS_DENIED = "🚫 Доступ запрещён"
GROUP_CHAT_TYPES = (Chat.GROUP, Chat.SUPERGROUP)


def is_super_admin(user_id: int) -> bool:
    return user_id in settings.super_admin_ids


def check_user_access(user_id: int) -> bool:
    """Доступ в личном чате."""
    if not settings.allowed_users_only or is_super_admin(user_id):
        return True
    return user_id in settings.allowed_user_ids


def check_group_access(chat_id: int) -> bool:
    return bot_state().is_group_allowed(chat_id)


def check_chat_access(chat_type: str, chat_id: int, user_id: int) -> bool:
    if chat_type == Chat.PRIVATE:
        return check_user_access(user_id)
    if chat_type in GROUP_CHAT_TYPES:
        return check_group_access(chat_id)
    return False


def require_auth(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> Any:
        user, chat = update.effective_user, update.effective_chat
        if not user or not chat:
            return None
        if not check_chat_access(chat.type, chat.id, user.id):
            logger.info("Access denied for user %d (@%s) in chat %d", user.id, user.username or "-", chat.id)
            if update.callback_query:
                await update.callback_query.answer(ACCESS_DENIED, show_alert=True)
            elif update.effective_message and chat.type == Chat.PRIVATE:
                await update.effective_message.reply_text(ACCESS_DENIED)
            return None
        return await func(update, context, *args, **kwargs)

    return wrapper


def log_user_access(user_id: int, username: str | None, action: str) -> None:
    logger.info("User %d (@%s): %s", user_id, username or "-", action)
