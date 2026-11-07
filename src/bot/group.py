"""Бот в группе: PDF → DOCX и DOCX/DOC/WPS → PDF для всех участников.

Группа должна быть в ALLOWED_CHAT_IDS или подключена супер-админом: если
супер-админ вызвал бота в незнакомой группе, она молча получает доступ.
В остальных незнакомых группах бот молчит. Бота вызывают:

- файлом с подписью, где упомянут бот (@имя) или есть команда (/convert);
- упоминанием или командой в ответ на сообщение с файлом — конвертируется
  файл из того сообщения;
- файлом в ответ на сообщение бота.

У участников дневной лимит файлов (у супер-админов — нет).

С GROUP_AUTO_CONVERT=true конвертируется каждый подходящий файл без вызова.
Тогда бот должен видеть все сообщения группы: быть администратором или
с выключенным privacy mode (BotFather → /setprivacy → Disable).
"""

from __future__ import annotations

import functools
import logging
import time

from telegram import ChatMember, Message, MessageEntity, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.pipeline import GROUP_KINDS, detect_kind
from .auth import GROUP_CHAT_TYPES, check_group_access, is_super_admin, log_user_access
from .processing import enqueue_media
from .state import bot_state

logger = logging.getLogger(__name__)

ALBUM_TTL_SECONDS = 120
HELP_COMMANDS = ("start", "help")

UNSUPPORTED = "В группе я конвертирую только PDF → DOCX и DOCX, DOC, WPS → PDF."
LIMIT_REACHED = "⛔ Дневной лимит — {limit} файлов на участника — исчерпан, пропущено: {skipped}. Счётчик обнулится в полночь."
# Подтверждение супер-админу, что группа подключена, — реакцией, без сообщения.
CONNECTED_REACTION = "👌"


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
    called = explicit
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

    state = bot_state()
    super_admin = is_super_admin(user.id)
    if not check_group_access(chat.id):
        if not (super_admin and called):
            logger.info("Group %d (%s) is not allowed; user %d", chat.id, chat.title or "-", user.id)
            return
        state.activate_group(chat.id, chat.title, user.id)
        logger.info("Group %d (%s) connected by super admin %d", chat.id, chat.title or "-", user.id)
        if not targets:
            await _acknowledge(message)
            return
    else:
        state.remember_title(chat.id, chat.title)

    if not targets:
        if message.photo:
            await message.reply_text(UNSUPPORTED)
        else:
            await message.reply_text(group_help(bot.username))
        return

    skipped = 0
    for document, reply_to in targets:
        file_name = document.file_name or "document"
        kind = detect_kind(file_name)
        if kind not in GROUP_KINDS:
            if explicit or not settings.group_auto_convert:
                await message.reply_text(f"⚠️ {file_name}: {UNSUPPORTED}")
            continue
        refund = None
        if not super_admin:
            if not state.try_consume(user.id):
                skipped += 1
                continue
            refund = functools.partial(state.refund, user.id)
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
            on_failure=refund,
        )
    if skipped:
        logger.info("User %d hit the daily limit in group %d, skipped %d", user.id, chat.id, skipped)
        await message.reply_text(LIMIT_REACHED.format(limit=state.daily_limit, skipped=skipped))


async def _acknowledge(message: Message) -> None:
    try:
        await message.set_reaction(CONNECTED_REACTION)
    except Exception as exc:
        # Реакции в группе могут быть запрещены — тогда просто молча.
        logger.debug("Reaction failed: %s", exc)


async def bot_membership_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Бота добавили в группу или убрали из неё — только запись в лог, без сообщений в чат."""
    change = update.my_chat_member
    if not change or change.chat.type not in GROUP_CHAT_TYPES:
        return
    absent = (ChatMember.LEFT, ChatMember.BANNED)
    joined = change.old_chat_member.status in absent and change.new_chat_member.status not in absent
    left = change.new_chat_member.status in absent and change.old_chat_member.status not in absent
    if joined or left:
        chat = change.chat
        logger.info(
            "Bot %s %s %r (id %d) by user %d, allowed=%s",
            "added to" if joined else "removed from",
            chat.type, chat.title, chat.id, change.from_user.id, check_group_access(chat.id),
        )


async def migration_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Группа стала супергруппой — у неё новый ID; доступ переносится на него."""
    message = update.effective_message
    if not message or not message.migrate_to_chat_id:
        return
    old, new = message.chat_id, message.migrate_to_chat_id
    carried = bot_state().migrate_group(old, new)
    logger.warning(
        "Group %d migrated to supergroup %d, access carried over: %s%s",
        old, new, carried, " — update ALLOWED_CHAT_IDS in .env" if old in settings.allowed_chat_ids else "",
    )
