"""Бот в группе: когда он считает себя вызванным и что конвертирует."""

import asyncio
import datetime
from types import SimpleNamespace

import pytest
from telegram import Chat, Document, ExternalReplyInfo, Message, MessageEntity, MessageOriginUser, PhotoSize, Update, User

from src.bot import group

BOT = User(id=999, first_name="Converter", is_bot=True, username="conv_bot")
ALICE = User(id=1, first_name="Alice", is_bot=False)
ALLOWED = Chat(id=-100, type=Chat.SUPERGROUP, title="Офис")
STRANGER = Chat(id=-200, type=Chat.SUPERGROUP, title="Чужая")
NOW = datetime.datetime(2026, 11, 7, tzinfo=datetime.timezone.utc)


class FakeBot:
    username = BOT.username
    id = BOT.id

    can_read_all_group_messages = False

    def __init__(self):
        self.messages = []
        self.reactions = []
        self.status = "member"

    async def get_chat_member(self, chat_id, user_id, **kwargs):
        return SimpleNamespace(status=self.status)

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(text)

    async def set_message_reaction(self, chat_id, message_id, reaction=None, **kwargs):
        self.reactions.append((message_id, reaction))
        return True


def _document(name):
    return Document(file_id=f"id-{name}", file_unique_id=f"u-{name}", file_name=name)


def _entities(text):
    found = []
    for word in text.split():
        offset = text.index(word)
        if word.startswith("@"):
            found.append(MessageEntity(MessageEntity.MENTION, offset, len(word)))
        elif word.startswith("/"):
            found.append(MessageEntity(MessageEntity.BOT_COMMAND, offset, len(word)))
    return found


def _message(message_id, *, chat=ALLOWED, text=None, caption=None, document=None, photo=None,
             reply_to=None, album=None, sender=ALICE, bot=None, external=None):
    message = Message(
        message_id=message_id,
        date=NOW,
        chat=chat,
        from_user=sender,
        text=text,
        entities=_entities(text) if text else None,
        caption=caption,
        caption_entities=_entities(caption) if caption else None,
        document=document,
        photo=photo,
        reply_to_message=reply_to,
        external_reply=external,
        media_group_id=album,
    )
    if bot is not None:
        message.set_bot(bot)
    return message


ADMIN = User(id=42, first_name="Boss", is_bot=False)


@pytest.fixture
def env(monkeypatch, fresh_state):
    monkeypatch.setattr(group.settings, "allowed_users_only", True)
    monkeypatch.setattr(group.settings, "allowed_chat_ids", frozenset({ALLOWED.id}))
    monkeypatch.setattr(group.settings, "group_auto_convert", False)
    monkeypatch.setattr(group.settings, "super_admin_ids", frozenset({ADMIN.id}))
    monkeypatch.setattr(group.settings, "daily_file_limit", 20)
    monkeypatch.setattr(group, "albums", group.AlbumTracker())
    monkeypatch.setattr(group, "_visibility", {})
    queued, refunds = [], []

    async def fake_enqueue(bot, notify, media, file_name, kind, *, message_id, user_id, group=False, on_failure=None):
        queued.append((file_name, kind, message_id, group))
        refunds.append(on_failure)

    monkeypatch.setattr(group, "enqueue_media", fake_enqueue)
    bot = FakeBot()
    return SimpleNamespace(bot=bot, queued=queued, refunds=refunds, state=fresh_state, context=SimpleNamespace(bot=bot))


def run(env, *messages):
    async def scenario():
        for message in messages:
            message.set_bot(env.bot)
            await group.group_handler(Update(update_id=message.message_id, message=message), env.context)

    asyncio.run(scenario())


def test_file_with_mention_in_caption_is_converted(env):
    run(env, _message(10, caption="@conv_bot", document=_document("Отчёт.pdf")))
    assert env.queued == [("Отчёт.pdf", "pdf", 10, True)]


def test_file_without_call_is_ignored(env):
    run(env, _message(10, document=_document("a.docx")))
    assert env.queued == []
    assert env.bot.messages == []


def test_auto_mode_converts_every_supported_file(env, monkeypatch):
    monkeypatch.setattr(group.settings, "group_auto_convert", True)
    run(env, _message(10, document=_document("a.wps")), _message(11, document=_document("t.xlsx")))
    assert env.queued == [("a.wps", "wps", 10, True)]
    assert env.bot.messages == []


def test_command_in_reply_converts_replied_file(env):
    original = _message(5, document=_document("Письмо.doc"), sender=User(id=2, first_name="Bob", is_bot=False))
    run(env, _message(6, text="/convert@conv_bot", reply_to=original))
    assert env.queued == [("Письмо.doc", "doc", 6, True)]


def test_command_for_another_bot_is_ignored(env):
    original = _message(5, document=_document("a.pdf"))
    run(env, _message(6, text="/convert@other_bot", reply_to=original))
    assert env.queued == []
    assert env.bot.messages == []


def test_file_sent_as_reply_to_bot_is_converted(env):
    bot_message = _message(3, text="Я конвертирую документы", sender=BOT)
    run(env, _message(4, document=_document("a.docx"), reply_to=bot_message))
    assert env.queued == [("a.docx", "docx", 4, True)]


def test_plain_reply_to_bot_does_not_trigger_help(env):
    bot_message = _message(3, text="готово", sender=BOT)
    run(env, _message(4, text="спасибо", reply_to=bot_message))
    assert env.bot.messages == []


def test_mention_without_file_shows_help(env):
    run(env, _message(7, text="@conv_bot что умеешь?"))
    assert env.queued == []
    assert "PDF → DOCX" in env.bot.messages[0]


def test_photo_is_not_supported_in_group(env):
    photo = (PhotoSize(file_id="p", file_unique_id="p", width=10, height=10),)
    run(env, _message(7, caption="@conv_bot", photo=photo))
    assert env.queued == []
    assert env.bot.messages == [group.UNSUPPORTED]


def test_unsupported_file_type_is_reported(env):
    run(env, _message(8, caption="@conv_bot", document=_document("table.xlsx")))
    assert env.queued == []
    assert "table.xlsx" in env.bot.messages[0]


def test_unknown_group_stays_silent_for_members(env):
    run(env, _message(9, chat=STRANGER, caption="@conv_bot", document=_document("a.pdf")))
    assert env.queued == []
    assert env.bot.messages == []
    assert not env.state.is_group_allowed(STRANGER.id)


def test_unknown_group_is_silent_without_call(env, monkeypatch):
    monkeypatch.setattr(group.settings, "group_auto_convert", True)
    run(env, _message(9, chat=STRANGER, document=_document("a.pdf")))
    assert env.queued == []
    assert env.bot.messages == []


def test_album_is_converted_when_any_file_calls_the_bot(env):
    run(
        env,
        _message(20, document=_document("1.pdf"), album="A"),
        _message(21, document=_document("2.pdf"), album="A", caption="@conv_bot"),
        _message(22, document=_document("3.docx"), album="A"),
        _message(23, document=_document("other.pdf"), album="B"),
    )
    assert env.queued == [
        ("1.pdf", "pdf", 20, True),
        ("2.pdf", "pdf", 21, True),
        ("3.docx", "docx", 22, True),
    ]


def test_album_tracker_forgets_old_albums():
    tracker = group.AlbumTracker(ttl=-1)
    tracker.add(-100, "A", 1, "file")
    assert tracker.files(-100, "A") == []
    assert not tracker.is_called(-100, "A")


def test_album_tracker_keeps_only_recent_albums():
    tracker = group.AlbumTracker(limit=2)
    for album in "ABC":
        tracker.add(-100, album, 1, album)
    assert tracker.files(-100, "A") == []
    assert tracker.files(-100, "C") == [("C", 1)]


def _album(env_chat=ALLOWED, ids=(90, 91), album="F"):
    return [_message(i, chat=env_chat, document=_document(f"{i}.wps"), album=album) for i in ids]


def test_reply_to_album_converts_every_file_the_bot_has_seen(env):
    first, second = _album()
    run(env, first, second, _message(92, text="/convert@conv_bot", reply_to=first))
    assert env.queued == [("90.wps", "wps", 92, True), ("91.wps", "wps", 92, True)]
    assert env.bot.messages == []


def test_reply_to_album_the_bot_cannot_see_explains_why(env):
    first, _ = _album()
    run(env, _message(92, text="/convert@conv_bot", reply_to=first))
    assert env.queued == [("90.wps", "wps", 92, True)]
    assert env.bot.messages == [group.PARTIAL_ALBUM]


def test_no_album_hint_when_bot_is_admin(env):
    env.bot.status = "administrator"
    first, _ = _album()
    run(env, _message(92, text="/convert@conv_bot", reply_to=first))
    assert env.bot.messages == []


def test_auto_mode_does_not_convert_album_twice(env, monkeypatch):
    monkeypatch.setattr(group.settings, "group_auto_convert", True)
    first, second = _album()
    run(env, first, second, _message(93, document=_document("93.wps"), album="F", caption="@conv_bot"))
    assert [q[0] for q in env.queued] == ["90.wps", "91.wps", "93.wps"]


def test_explicit_call_reads_command_name():
    message = _message(1, text="/Convert@Conv_Bot please")
    assert group.explicit_call(message, "conv_bot", BOT.id) == (True, "convert")


def test_super_admin_call_connects_group_silently_and_converts(env):
    run(
        env,
        _message(30, chat=STRANGER, caption="@conv_bot", document=_document("a.pdf"), sender=ADMIN),
        _message(31, chat=STRANGER, caption="@conv_bot", document=_document("b.docx")),
    )
    assert env.state.is_group_allowed(STRANGER.id)
    assert env.state.title(STRANGER.id) == "Чужая"
    assert [q[0] for q in env.queued] == ["a.pdf", "b.docx"]
    assert env.bot.messages == []


def test_bare_super_admin_call_answers_with_reaction_only(env):
    run(env, _message(32, chat=STRANGER, text="@conv_bot", sender=ADMIN))
    assert env.state.is_group_allowed(STRANGER.id)
    assert env.bot.messages == []
    assert [message_id for message_id, _ in env.bot.reactions] == [32]


def test_super_admin_file_without_call_does_not_connect_group(env, monkeypatch):
    monkeypatch.setattr(group.settings, "group_auto_convert", True)
    run(env, _message(33, chat=STRANGER, document=_document("a.pdf"), sender=ADMIN))
    assert not env.state.is_group_allowed(STRANGER.id)
    assert env.queued == []


def test_removed_group_is_silent_until_super_admin_calls(env):
    env.state.remove_group(ALLOWED.id)
    run(env, _message(34, caption="@conv_bot", document=_document("a.pdf")))
    assert env.queued == [] and env.bot.messages == []
    run(env, _message(35, text="/convert@conv_bot", sender=ADMIN))
    assert env.state.is_group_allowed(ALLOWED.id)
    run(env, _message(36, caption="@conv_bot", document=_document("b.pdf")))
    assert [q[0] for q in env.queued] == ["b.pdf"]


def test_daily_limit_per_member(env):
    env.state.set_daily_limit(2)
    album = [_message(40 + i, document=_document(f"{i}.pdf"), album="L", caption="@conv_bot" if i == 0 else None)
             for i in range(3)]
    run(env, *album)
    assert [q[0] for q in env.queued] == ["0.pdf", "1.pdf"]
    assert env.bot.messages and "лимит" in env.bot.messages[-1].lower()
    # Другой участник считает свои файлы отдельно, супер-админ — без лимита.
    run(env, _message(50, caption="@conv_bot", document=_document("bob.pdf"), sender=User(id=2, first_name="Bob", is_bot=False)))
    for i in range(3):
        run(env, _message(60 + i, caption="@conv_bot", document=_document(f"boss{i}.pdf"), sender=ADMIN))
    assert len(env.queued) == 6
    assert env.state.used(ADMIN.id) == 0


def test_failed_conversion_returns_file_to_limit(env):
    env.state.set_daily_limit(1)
    run(env, _message(70, caption="@conv_bot", document=_document("a.pdf")))
    assert env.state.used(ALICE.id) == 1
    env.refunds[0]()
    assert env.state.used(ALICE.id) == 0
    run(env, _message(71, caption="@conv_bot", document=_document("b.pdf")))
    assert [q[0] for q in env.queued] == ["a.pdf", "b.pdf"]


def test_migration_carries_access_to_supergroup(env):
    env.state.activate_group(-555, "Старая", ADMIN.id)
    migrated = Message(message_id=80, date=NOW, chat=Chat(id=-555, type=Chat.GROUP), migrate_to_chat_id=-100555)
    asyncio.run(group.migration_handler(Update(update_id=80, message=migrated), env.context))
    assert env.state.is_group_allowed(-100555)
    assert not env.state.is_group_allowed(-555)


def _old_group_reply(**media):
    """Сообщение из истории группы до превращения в супергруппу — другой чат."""
    return ExternalReplyInfo(
        origin=MessageOriginUser(date=NOW, sender_user=User(id=3, first_name="Андрей", is_bot=False)),
        chat=Chat(id=-4001, type=Chat.GROUP, title="FullProCleaning"),
        message_id=12,
        **media,
    )


def test_reply_to_file_from_before_supergroup_migration_is_converted(env):
    reply = _message(100, text="/convert@conv_bot", external=_old_group_reply(document=_document("Facades.wps")))
    run(env, reply)
    assert env.queued == [("Facades.wps", "wps", 100, True)]
    assert env.bot.messages == []


def test_reply_to_message_without_file_says_so(env):
    original = _message(101, text="Готовый образец коммерческого предложения")
    run(env, _message(102, text="/convert@conv_bot", reply_to=original))
    assert env.queued == []
    assert env.bot.messages == [group.NO_DOCUMENT_IN_REPLY]


def test_reply_to_photo_is_unsupported(env):
    photo = (PhotoSize(file_id="p", file_unique_id="p", width=10, height=10),)
    run(env, _message(103, text="/convert@conv_bot", external=_old_group_reply(photo=photo)))
    assert env.bot.messages == [group.UNSUPPORTED]
