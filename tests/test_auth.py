from src.bot import auth


def test_restricted_mode_allows_only_listed_users(monkeypatch):
    monkeypatch.setattr(auth.settings, "allowed_users_only", True)
    monkeypatch.setattr(auth.settings, "allowed_user_ids", frozenset({1, 2}))
    assert auth.check_user_access(1)
    assert not auth.check_user_access(3)


def test_public_mode_allows_everyone(monkeypatch):
    monkeypatch.setattr(auth.settings, "allowed_users_only", False)
    monkeypatch.setattr(auth.settings, "allowed_user_ids", frozenset())
    assert auth.check_user_access(42)
