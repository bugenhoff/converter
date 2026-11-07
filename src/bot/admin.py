"""Панель супер-админа в личке: группы с доступом, дневной лимит, сброс счётчиков."""

from __future__ import annotations

import html
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from .auth import ACCESS_DENIED, is_super_admin
from .state import bot_state

logger = logging.getLogger(__name__)

ADMIN_BUTTON = "⚙️ Администрирование"
CALLBACK = "adm"
LIMIT_PRESETS = (5, 10, 20, 30, 50, 100)
AWAITING_LIMIT = "awaiting_limit"
MAX_LIMIT = 100000


def admin_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная кнопка внизу чата супер-админа."""
    return ReplyKeyboardMarkup([[ADMIN_BUTTON]], resize_keyboard=True, is_persistent=True)


def _button(text: str, *parts: object) -> InlineKeyboardButton:
    return InlineKeyboardButton(text, callback_data=":".join([CALLBACK, *map(str, parts)]))


def _limit_text(limit: int) -> str:
    return "без лимита" if not limit else f"{limit} файлов"


def home_view() -> tuple[str, InlineKeyboardMarkup]:
    state = bot_state()
    files, users = state.usage_today()
    text = (
        "<b>⚙️ Администрирование</b>\n\n"
        f"Групп с доступом: {len(state.allowed_groups())}\n"
        f"Лимит в группах: {_limit_text(state.daily_limit)} в сутки на участника\n"
        f"Сегодня: {files} файлов от {users} участников\n"
        f"<i>Сутки начинаются в 00:00, {html.escape(state.zone_name)}</i>"
    )
    keyboard = [
        [_button("🗂 Группы", "groups"), _button("📊 Лимит", "limit")],
        [_button("♻️ Сбросить счётчики", "reset")],
    ]
    return text, InlineKeyboardMarkup(keyboard)


def groups_view() -> tuple[str, InlineKeyboardMarkup]:
    groups = bot_state().allowed_groups()
    if groups:
        text = (
            "<b>🗂 Группы с доступом</b>\n\n"
            "Нажмите на группу, чтобы убрать доступ. Вернуть его — вызвать бота в группе."
        )
    else:
        text = "<b>🗂 Группы с доступом</b>\n\nПока нет. Чтобы подключить группу, вызовите в ней бота."
    rows = [[_button(f"❌ {_shorten(title) or chat_id}", "rm", chat_id)] for chat_id, title in groups.items()]
    rows.append([_button("« Назад", "home")])
    return text, InlineKeyboardMarkup(rows)


def limit_view() -> tuple[str, InlineKeyboardMarkup]:
    limit = bot_state().daily_limit
    text = (
        "<b>📊 Дневной лимит</b>\n\n"
        f"Сейчас: {_limit_text(limit)}. Сколько файлов в сутки может сконвертировать "
        "один участник в группах (супер-админов лимит не касается):"
    )
    presets = [_button(f"✓ {n}" if n == limit else str(n), "limit", n) for n in LIMIT_PRESETS]
    rows = [presets[i:i + 3] for i in range(0, len(presets), 3)]
    rows.append([_button("✓ Без лимита" if not limit else "Без лимита", "limit", 0), _button("✏️ Своё число", "limit", "custom")])
    rows.append([_button("« Назад", "home")])
    return text, InlineKeyboardMarkup(rows)


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user, message = update.effective_user, update.effective_message
    if not user or not message or not is_super_admin(user.id):
        return
    context.user_data.pop(AWAITING_LIMIT, None)
    text, markup = home_view()
    await message.reply_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)


async def admin_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Кнопка «Администрирование» и ввод своего лимита."""
    user, message = update.effective_user, update.effective_message
    if not user or not message or not message.text or not is_super_admin(user.id):
        return
    text = message.text.strip()
    if text == ADMIN_BUTTON:
        await admin_command(update, context)
        return
    if not context.user_data.get(AWAITING_LIMIT):
        return
    if not text.isdigit() or int(text) > MAX_LIMIT:
        await message.reply_text(f"Нужно целое число от 0 до {MAX_LIMIT} (0 — без лимита).")
        return
    context.user_data.pop(AWAITING_LIMIT, None)
    bot_state().set_daily_limit(int(text))
    logger.info("Super admin %d set daily limit to %s", user.id, text)
    view, markup = home_view()
    await message.reply_text(f"✅ Лимит: {_limit_text(int(text))} в сутки.\n\n{view}", parse_mode=ParseMode.HTML, reply_markup=markup)


async def admin_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query, user = update.callback_query, update.effective_user
    if not query or not user:
        return
    if not is_super_admin(user.id):
        await query.answer(ACCESS_DENIED, show_alert=True)
        return
    parts = (query.data or CALLBACK).split(":")[1:]
    action, args = (parts[0], parts[1:]) if parts else ("home", [])
    state = bot_state()
    note, alert, view = None, False, home_view

    if action == "groups":
        view = groups_view
    elif action == "rm" and args:
        chat_id = int(args[0])
        title = state.title(chat_id) or str(chat_id)
        state.remove_group(chat_id)
        logger.info("Super admin %d removed group %d (%s)", user.id, chat_id, title)
        note, alert, view = f"Группа «{title}» убрана. Вернуть: вызовите бота в ней.", True, groups_view
    elif action == "limit" and args and args[0] == "custom":
        context.user_data[AWAITING_LIMIT] = True
        await query.answer()
        await query.message.reply_text("Пришлите число — сколько файлов в сутки на участника (0 — без лимита).")
        return
    elif action == "limit" and args:
        state.set_daily_limit(int(args[0]))
        logger.info("Super admin %d set daily limit to %s", user.id, args[0])
        note = f"Лимит: {_limit_text(state.daily_limit)}"
    elif action == "limit":
        view = limit_view
    elif action == "reset":
        files, users = state.usage_today()
        state.reset_usage()
        logger.info("Super admin %d reset usage (%d files, %d users)", user.id, files, users)
        note = f"Счётчики сброшены ({files} файлов, {users} участников)"

    context.user_data.pop(AWAITING_LIMIT, None)
    await query.answer(note, show_alert=alert)
    text, markup = view()
    try:
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=markup)
    except BadRequest as exc:
        if "not modified" not in str(exc).lower():
            raise


def _shorten(title: str, limit: int = 40) -> str:
    return title if len(title) <= limit else title[: limit - 1] + "…"
