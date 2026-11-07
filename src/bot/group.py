"""Бот в группе: PDF → DOCX и DOCX/DOC/WPS → PDF для всех участников.

Группа должна быть в ALLOWED_CHAT_IDS. Бота вызывают:

- файлом с подписью, где упомянут бот (@имя) или есть команда (/convert);
- упоминанием или командой в ответ на сообщение с файлом — конвертируется
  файл из того сообщения;
- файлом в ответ на сообщение бота.

С GROUP_AUTO_CONVERT=true конвертируется каждый подходящий файл без вызова.
Тогда бот должен видеть все сообщения группы: быть администратором или
с выключенным privacy mode (BotFather → /setprivacy → Disable).
"""

from __future__ import annotations

import logging
import time

from telegram import ChatMember, Message, MessageEntity, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.pipeline import GROUP_KINDS, detect_kind
from .auth import GROUP_CHAT_TYPES, check_group_access, log_user_access
from .processing import enqueue_media

logger = logging.getLogger(__name__)

ALBUM_TTL_SECONDS = 120
HELP_COMMANDS = ("start", "help")

UNSUPPORTED = "В группе я конвертирую только PDF → DOCX и DOCX, DOC, WPS → PDF."
NOT_CONNECTED = (
    "🚫 Эта группа не подключена к боту.\n"
    "Чтобы подключить, добавьте её ID <code>{chat_id}</code> в ALLOWED_CHAT_IDS "
    "на сервере и перезапустите бота."
)


def group_help(username: str) -> str:
    lines = [
        "Я конвертирую документы:",
        "• PDF → DOCX",
        "• DOCX, DOC, WPS → PDF",
        "",
    ]
    if settings.group_auto_convert:
        lines.append("Просто пришлите файл в группу.")
    else:
        lines.append(
            f"Пришлите файл с подписью @{username} или ответьте на сообщение с файлом "
            f"упоминанием @{username} (или командой /convert)."
        )
    return "\n".join(lines)


class AlbumTracker:
    """Файлы одного альбома приходят отдельными сообщениями, подпись — только у одного.

    Файлы без вызова запоминаются; если позже (или раньше) в том же альбоме
    вызвали бота, конвертируется весь альбом.
    """

    def __init__(self, ttl: float = ALBUM_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._waiting: dict[str, tuple[float, list]] = {}
        self._called: dict[str, float] = {}

    def remember(self, album_id: str, item) -> None:
        self._expire()
        self._waiting.setdefault(album_id, (time.monotonic(), []))[1].append(item)

    def call(self, album_id: str) -> list:
        """Отмечает альбом вызванным и отдаёт запомненные файлы."""
        self._expire()
        self._called[album_id] = time.monotonic()
        return self._waiting.pop(album_id, (0.0, []))[1]

    def is_called(self, album_id: str | None) -> bool:
        self._expire()
        return album_id is not None and album_id in self._called

    def _expire(self) -> None:
        deadline = time.monotonic() - self._ttl
        for key in [k for k, (at, _) in self._waiting.items() if at < deadline]:
            del self._waiting[key]
        for key in [k for k, at in self._called.items() if at < deadline]:
            del self._called[key]


albums = AlbumTracker()


def explicit_call(message: Message, username: str, bot_id: int) -> tuple[bool, str | None]:
    """(бота упомянули или дали ему команду, имя команды без «/» и «@бот»)."""
    entities = message.parse_entities() if message.text else message.parse_caption_entities()
    name = (username or "").lower()
    mentioned, command = False, None
    for entity, text in entities.items():
        if entity.type == MessageEntity.MENTION and text.lower() == f"@{name}":
            mentioned = True
        elif entity.type == MessageEntity.TEXT_MENTION and entity.user and entity.user.id == bot_id:
            mentioned = True
        elif entity.type == MessageEntity.BOT_COMMAND and command is None:
            cmd, _, target = text.lstrip("/").partition("@")
            # Команда без @ адресована всем ботам группы, с чужим @ — не нам.
            if not target or target.lower() == name:
                command = cmd.lower()
    return mentioned or command is not None, command


def replies_to_bot(message: Message, bot_id: int) -> bool:
    reply = message.reply_to_message
    return bool(reply and reply.from_user and reply.from_user.id == bot_id)


async def group_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message, user, chat = update.message, update.effective_user, update.effective_chat
    if not message or not user or not chat or chat.type not in GROUP_CHAT_TYPES:
        return
    bot = context.bot
    explicit, command = explicit_call(message, bot.username, bot.id)
    own = message.document
    album = message.media_group_id

    # Что конвертировать: (документ, id сообщения, на которое ответить).
    targets: list[tuple] = []
    if own is not None:
        called = explicit or replies_to_bot(message, bot.id) or albums.is_called(album)
        if album and called:
            targets.extend(albums.call(album))
        if called or settings.group_auto_convert:
            targets.append((own, message.message_id))
        elif album:
            albums.remember(album, (own, message.message_id))
            return
        else:
            return
    elif explicit:
        replied = message.reply_to_message
        if replied is not None and replied.document is not None:
            targets.append((replied.document, message.message_id))
    else:
        return

    if not check_group_access(chat.id):
        logger.info("Group %d (%s) is not allowed; user %d", chat.id, chat.title or "-", user.id)
        if explicit:
            await message.reply_text(NOT_CONNECTED.format(chat_id=chat.id), parse_mode=ParseMode.HTML)
        return

    if not targets:
        if message.photo:
            await message.reply_text(UNSUPPORTED)
        else:
            await message.reply_text(group_help(bot.username))
        return

    for document, reply_to in targets:
        file_name = document.file_name or "document"
        kind = detect_kind(file_name)
        if kind not in GROUP_KINDS:
            if explicit or not settings.group_auto_convert:
                await message.reply_text(f"⚠️ {file_name}: {UNSUPPORTED}")
            continue
        log_user_access(user.id, user.username, f"group {chat.id} upload {file_name}")
        await enqueue_media(
            bot,
            message,
            document,
            file_name,
            kind,
            message_id=reply_to,
            user_id=user.id,
            group=True,
        )


async def bot_membership_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Бота добавили в группу: подсказка или ID для подключения."""
    change = update.my_chat_member
    if not change or change.chat.type not in GROUP_CHAT_TYPES:
        return
    absent = (ChatMember.LEFT, ChatMember.BANNED)
    if change.old_chat_member.status not in absent or change.new_chat_member.status in absent:
        return
    chat = change.chat
    allowed = check_group_access(chat.id)
    logger.info(
        "Bot added to %s %r (id %d) by user %d, allowed=%s",
        chat.type, chat.title, chat.id, change.from_user.id, allowed,
    )
    text = group_help(context.bot.username) if allowed else NOT_CONNECTED.format(chat_id=chat.id)
    try:
        await context.bot.send_message(chat.id, text, parse_mode=ParseMode.HTML if not allowed else None)
    except Exception:
        logger.exception("Failed to greet group %d", chat.id)


async def migration_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Группа стала супергруппой — у неё новый ID, старый в ALLOWED_CHAT_IDS больше не подходит."""
    message = update.effective_message
    if message and message.migrate_to_chat_id:
        logger.warning(
            "Group %d migrated to supergroup %d — update ALLOWED_CHAT_IDS",
            message.chat_id, message.migrate_to_chat_id,
        )
