"""Бот в группе: PDF → DOCX и DOCX/DOC/WPS → PDF для всех участников.

Группа должна быть в ALLOWED_CHAT_IDS или подключена супер-админом: если
супер-админ вызвал бота в незнакомой группе, она молча получает доступ.
В остальных незнакомых группах бот молчит. Бота вызывают:

- файлом с подписью, где упомянут бот (@имя) или есть команда (/convert);
- упоминанием или командой в ответ на сообщение с файлом — конвертируется
  файл из того сообщения, а если он из альбома — весь альбом;
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
from collections import OrderedDict
from dataclasses import dataclass, field

from telegram import ChatMember, Message, MessageEntity, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.pipeline import GROUP_KINDS, detect_kind
from .auth import GROUP_CHAT_TYPES, check_group_access, is_super_admin, log_user_access
from .processing import enqueue_media
from .state import bot_state

logger = logging.getLogger(__name__)

# Файлы альбомов помнятся сутки: на альбом могут ответить и через несколько часов.
ALBUM_TTL_SECONDS = 24 * 3600
MAX_ALBUMS = 1000
VISIBILITY_TTL_SECONDS = 600
HELP_COMMANDS = ("start", "help")

UNSUPPORTED = "В группе я конвертирую только PDF → DOCX и DOCX, DOC, WPS → PDF."
LIMIT_REACHED = "⛔ Дневной лимит — {limit} файлов на участника — исчерпан, пропущено: {skipped}. Счётчик обнулится в полночь."
PARTIAL_ALBUM = (
    "ℹ️ Конвертирую только этот файл: остальные файлы альбома мне не видны. "
    "Чтобы конвертировался весь альбом, сделайте бота администратором группы."
)
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


@dataclass
class _Album:
    updated: float
    files: dict[int, object] = field(default_factory=dict)  # id сообщения → документ
    called: bool = False
    done: set[int] = field(default_factory=set)


class AlbumTracker:
    """Файлы альбомов, которые видел бот: альбом приходит отдельными сообщениями.

    - вызов в подписи одного файла → конвертируется весь альбом, в том числе
      файлы, пришедшие позже;
    - вызов ответом на любой файл альбома → конвертируется весь альбом.

    Файлы без вызова бот получает, только если он администратор группы или
    у него выключен privacy mode.
    """

    def __init__(self, ttl: float = ALBUM_TTL_SECONDS, limit: int = MAX_ALBUMS) -> None:
        self._ttl = ttl
        self._limit = limit
        self._albums: OrderedDict[tuple[int, str], _Album] = OrderedDict()

    def add(self, chat_id: int, album_id: str, message_id: int, document) -> None:
        album = self._get(chat_id, album_id, create=True)
        album.files[message_id] = document

    def files(self, chat_id: int, album_id: str) -> list[tuple[object, int]]:
        album = self._get(chat_id, album_id)
        return sorted(((doc, mid) for mid, doc in album.files.items()), key=lambda item: item[1]) if album else []

    def call(self, chat_id: int, album_id: str) -> list[tuple[object, int]]:
        """Отмечает альбом вызванным и отдаёт ещё не сконвертированные файлы."""
        album = self._get(chat_id, album_id, create=True)
        album.called = True
        fresh = [(doc, mid) for doc, mid in self.files(chat_id, album_id) if mid not in album.done]
        album.done.update(mid for _, mid in fresh)
        return fresh

    def mark_done(self, chat_id: int, album_id: str, message_id: int) -> None:
        self._get(chat_id, album_id, create=True).done.add(message_id)

    def is_called(self, chat_id: int, album_id: str | None) -> bool:
        album = self._get(chat_id, album_id) if album_id else None
        return bool(album and album.called)

    def _get(self, chat_id: int, album_id: str, create: bool = False) -> _Album | None:
        now = time.monotonic()
        while self._albums and next(iter(self._albums.values())).updated < now - self._ttl:
            self._albums.popitem(last=False)
        key = (chat_id, album_id)
        album = self._albums.get(key)
        if album is None and create:
            album = self._albums[key] = _Album(now)
            while len(self._albums) > self._limit:
                self._albums.popitem(last=False)
        if album is not None:
            album.updated = now
            self._albums.move_to_end(key)
        return album


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

    if own is not None and album:
        albums.add(chat.id, album, message.message_id, own)

    # Что конвертировать: (документ, id сообщения, на которое ответить).
    targets: list[tuple] = []
    partial_album = False
    called = explicit
    if own is not None:
        called = explicit or replies_to_bot(message, bot.id) or albums.is_called(chat.id, album)
        if called and album:
            targets = albums.call(chat.id, album)
        elif called or settings.group_auto_convert:
            targets = [(own, message.message_id)]
            if album:
                albums.mark_done(chat.id, album, message.message_id)
        else:
            return
    elif explicit:
        replied = message.reply_to_message
        if replied is not None and replied.document is not None:
            if replied.media_group_id:
                # Ответ на файл из альбома — весь альбом, который видел бот.
                albums.add(chat.id, replied.media_group_id, replied.message_id, replied.document)
                files = albums.files(chat.id, replied.media_group_id)
                targets = [(document, message.message_id) for document, _ in files]
                partial_album = len(files) == 1
            else:
                targets = [(replied.document, message.message_id)]
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
    if partial_album and not await _sees_all_messages(bot, chat.id):
        await message.reply_text(PARTIAL_ALBUM)


_visibility: dict[int, tuple[float, bool]] = {}


async def _sees_all_messages(bot, chat_id: int) -> bool:
    """Видит ли бот все сообщения группы: администратор или privacy mode выключен."""
    if getattr(bot, "can_read_all_group_messages", False):
        return True
    cached = _visibility.get(chat_id)
    if cached and cached[0] > time.monotonic() - VISIBILITY_TTL_SECONDS:
        return cached[1]
    try:
        member = await bot.get_chat_member(chat_id, bot.id)
        sees = member.status in (ChatMember.ADMINISTRATOR, ChatMember.OWNER)
    except Exception as exc:
        logger.debug("Cannot check bot status in %d: %s", chat_id, exc)
        return True
    _visibility[chat_id] = (time.monotonic(), sees)
    return sees


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
