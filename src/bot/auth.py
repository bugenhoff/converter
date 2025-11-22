"""Authorization middleware for telegram bot."""

import logging
from functools import wraps
from typing import Callable, Any

from telegram import Update
from telegram.ext import ContextTypes

from ..config.settings import settings

logger = logging.getLogger(__name__)


def check_user_access(user_id: int) -> bool:
    """Check if user has access to the bot."""
    if not settings.allowed_users_only:
        # If public access is enabled, allow everyone
        return True
    
    # Check if user is in allowed list
    return user_id in settings.allowed_user_ids


def require_auth(func: Callable) -> Callable:
    """Decorator to require authorization for handler functions."""
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs) -> Any:
        if not update.effective_user:
            logger.warning("Received update without user information")
            return
        
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        if not check_user_access(user_id):
            logger.info(
                "Access denied for user %d (@%s)", 
                user_id, username
            )
            await update.message.reply_text("🚫 Доступ запрещён")
            return
        
        logger.debug(
            "Access granted for user %d (@%s)", 
            user_id, username
        )
        
        return await func(update, context, *args, **kwargs)
    
    return wrapper


def log_user_access(user_id: int, username: str | None, action: str) -> None:
    """Log user access for monitoring."""
    logger.info(
        "User %d (@%s) performed action: %s", 
        user_id, username or "Unknown", action
    )