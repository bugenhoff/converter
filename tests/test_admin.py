"""Панель супер-админа: кнопки и ввод своего лимита."""

import asyncio
from types import SimpleNamespace

import pytest

from src.bot import admin
from src.config.settings import settings


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.answers = []
        self.edits = []
        self.replies = []
        self.message = SimpleNamespace(reply_text=self._reply)

    async def _reply(self, text, **kwargs):
        self.replies.append(text)

    async def answer(self, text=None, show_alert=False):
        self.answers.append((text, show_alert))

    async def edit_message_text(self, text, **kwargs):
        self.edits.append((text, kwargs.get("reply_markup")))


class FakeMessage:
    def __init__(self, text):
        self.text = text
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append((text, kwargs.get("reply_markup")))


@pytest.fixture
def env(monkeypatch, fresh_state):
    monkeypatch.setattr(settings, "super_admin_ids", frozenset({42}))
    monkeypatch.setattr(settings, "allowed_users_only", True)
    monkeypatch.setattr(settings, "allowed_chat_ids", frozenset())
    fresh_state.activate_group(-100, "Офис", by=42)
    return SimpleNamespace(state=fresh_state, context=SimpleNamespace(user_data={}))


def press(env, data, user_id=42):
    query = FakeQuery(data)
    update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=user_id))
    asyncio.run(admin.admin_callback_handler(update, env.context))
    return query


def send(env, text, user_id=42):
    message = FakeMessage(text)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=user_id), effective_message=message)
    asyncio.run(admin.admin_text_handler(update, env.context))
    return message


def buttons(markup):
    return [(button.text, button.callback_data) for row in markup.inline_keyboard for button in row]


def test_only_super_admin_can_press_buttons(env):
    query = press(env, "adm:reset", user_id=7)
    assert query.answers == [(admin.ACCESS_DENIED, True)]
    assert query.edits == []


def test_admin_button_opens_panel(env):
    message = send(env, admin.ADMIN_BUTTON)
    text, markup = message.replies[0]
    assert "Групп с доступом: 1" in text
    assert ("📊 Лимит", "adm:limit") in buttons(markup)


def test_other_users_text_is_ignored(env):
    assert send(env, admin.ADMIN_BUTTON, user_id=7).replies == []


def test_remove_group_from_list(env):
    listing = press(env, "adm:groups")
    assert ("❌ Офис", "adm:rm:-100") in buttons(listing.edits[0][1])
    query = press(env, "adm:rm:-100")
    assert not env.state.is_group_allowed(-100)
    assert query.answers[0][1] is True
    assert "Пока нет" in query.edits[0][0]


def test_limit_presets_and_unlimited(env):
    view = press(env, "adm:limit")
    assert ("✓ 20", "adm:limit:20") in buttons(view.edits[0][1])
    press(env, "adm:limit:50")
    assert env.state.daily_limit == 50
    press(env, "adm:limit:0")
    assert env.state.daily_limit == 0


def test_custom_limit_is_typed_as_number(env):
    press(env, "adm:limit:custom")
    assert send(env, "много").replies[0][0].startswith("Нужно целое число")
    reply = send(env, "15").replies[0][0]
    assert env.state.daily_limit == 15
    assert reply.startswith("✅ Лимит: 15")
    # Флаг ожидания снят — следующее число уже не меняет лимит.
    send(env, "3")
    assert env.state.daily_limit == 15


def test_reset_counters_for_everyone(env):
    env.state.try_consume(1)
    env.state.try_consume(2)
    query = press(env, "adm:reset")
    assert env.state.usage_today() == (0, 0)
    assert "2 файлов, 2 участников" in query.answers[0][0]
