"""Бот в группе: когда он считает себя вызванным и что конвертирует."""

import asyncio
import datetime
from types import SimpleNamespace

import pytest
from telegram import Chat, Document, Message, MessageEntity, PhotoSize, Update, User

from src.bot import group

BOT = User(id=999, first_name="Converter", is_bot=True, username="conv_bot")
ALICE = User(id=1, first_name="Alice", is_bot=False)
ALLOWED = Chat(id=-100, type=Chat.SUPERGROUP, title="Офис")
STRANGER = Chat(id=-200, type=Chat.SUPERGROUP, title="Чужая")
NOW = datetime.datetime(2026, 11, 7, tzinfo=datetime.timezone.utc)


class FakeBot:
    username = BOT.username
    id = BOT.id

    def __init__(self):
        self.messages = []

    async def send_message(self, chat_id, text, **kwargs):
        self.messages.append(text)


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
             reply_to=None, album=None, sender=ALICE, bot=None):
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
        media_group_id=album,
    )
    if bot is not None:
        message.set_bot(bot)
    return message


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(group.settings, "allowed_users_only", True)
    monkeypatch.setattr(group.settings, "allowed_chat_ids", frozenset({ALLOWED.id}))
    monkeypatch.setattr(group.settings, "group_auto_convert", False)
    monkeypatch.setattr(group, "albums", group.AlbumTracker())
    queued = []

    async def fake_enqueue(bot, notify, media, file_name, kind, *, message_id, user_id, group=False):
        queued.append((file_name, kind, message_id, group))

    monkeypatch.setattr(group, "enqueue_media", fake_enqueue)
    bot = FakeBot()
    return SimpleNamespace(bot=bot, queued=queued, context=SimpleNamespace(bot=bot))


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


def test_unknown_group_gets_its_id_only_when_calling(env):
    run(env, _message(9, chat=STRANGER, caption="@conv_bot", document=_document("a.pdf")))
    assert env.queued == []
    assert str(STRANGER.id) in env.bot.messages[0]


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
    tracker.remember("A", "file")
    assert tracker.call("A") == []
    assert not tracker.is_called("A")


def test_explicit_call_reads_command_name():
    message = _message(1, text="/Convert@Conv_Bot please")
    assert group.explicit_call(message, "conv_bot", BOT.id) == (True, "convert")
