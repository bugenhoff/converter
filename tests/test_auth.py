import pytest

from src.bot import auth


@pytest.fixture
def restricted(monkeypatch):
    monkeypatch.setattr(auth.settings, "allowed_users_only", True)
    monkeypatch.setattr(auth.settings, "allowed_user_ids", frozenset({1, 2}))
    monkeypatch.setattr(auth.settings, "allowed_chat_ids", frozenset({-1001}))


def test_restricted_mode_allows_only_listed_users(restricted):
    assert auth.check_user_access(1)
    assert not auth.check_user_access(3)


def test_public_mode_allows_everyone(monkeypatch):
    monkeypatch.setattr(auth.settings, "allowed_users_only", False)
    monkeypatch.setattr(auth.settings, "allowed_user_ids", frozenset())
    monkeypatch.setattr(auth.settings, "allowed_chat_ids", frozenset())
    assert auth.check_user_access(42)
    assert auth.check_chat_access("supergroup", -5, 42)


def test_member_of_allowed_group_can_use_it_only_there(restricted):
    assert auth.check_chat_access("supergroup", -1001, 77)
    assert auth.check_chat_access("group", -1001, 77)
    assert not auth.check_chat_access("private", 77, 77)


def test_listed_user_is_not_enough_for_unknown_group(restricted):
    assert auth.check_chat_access("private", 1, 1)
    assert not auth.check_chat_access("supergroup", -2002, 1)


def test_channels_are_never_served(restricted):
    assert not auth.check_chat_access("channel", -1001, 1)
