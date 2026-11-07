import datetime
import json

import pytest

from src.bot import state as state_module
from src.bot.state import BotState


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setattr(state_module.settings, "allowed_users_only", True)
    monkeypatch.setattr(state_module.settings, "allowed_chat_ids", frozenset({-1}))
    monkeypatch.setattr(state_module.settings, "daily_file_limit", 20)
    return state_module.settings


def test_state_survives_restart(tmp_path, settings):
    path = tmp_path / "data" / "state.json"
    first = BotState(path)
    first.activate_group(-100, "Офис", by=42)
    first.remove_group(-1)
    first.set_daily_limit(7)
    assert first.try_consume(5)

    second = BotState(path)
    assert second.allowed_groups() == {-100: "Офис"}
    assert not second.is_group_allowed(-1)
    assert second.daily_limit == 7
    assert second.used(5) == 1


def test_env_group_can_be_removed_and_returned(tmp_path, settings):
    state = BotState(tmp_path / "s.json")
    assert state.is_group_allowed(-1)
    state.remove_group(-1)
    assert not state.is_group_allowed(-1)
    state.activate_group(-1, "Из .env", by=42)
    assert state.is_group_allowed(-1)


def test_removed_group_is_closed_even_in_public_mode(tmp_path, settings, monkeypatch):
    monkeypatch.setattr(settings, "allowed_users_only", False)
    state = BotState(tmp_path / "s.json")
    assert state.is_group_allowed(-777)
    state.remove_group(-777)
    assert not state.is_group_allowed(-777)


def test_limit_counts_per_user_and_zero_means_unlimited(tmp_path, settings):
    state = BotState(tmp_path / "s.json")
    state.set_daily_limit(2)
    assert state.try_consume(1) and state.try_consume(1)
    assert not state.try_consume(1)
    assert state.try_consume(2)
    assert state.usage_today() == (3, 2)
    state.set_daily_limit(0)
    assert all(state.try_consume(1) for _ in range(50))


def test_default_limit_comes_from_settings(tmp_path, settings):
    assert BotState(tmp_path / "s.json").daily_limit == 20


def test_reset_and_refund(tmp_path, settings):
    state = BotState(tmp_path / "s.json")
    state.try_consume(1)
    state.try_consume(1)
    state.refund(1)
    assert state.used(1) == 1
    state.refund(9)
    state.reset_usage()
    assert state.usage_today() == (0, 0)


def test_counters_start_over_next_day(tmp_path, settings, monkeypatch):
    day = [datetime.datetime(2026, 11, 7, 23, 59, tzinfo=datetime.timezone.utc)]
    monkeypatch.setattr(state_module, "_now", lambda zone: day[0])
    state = BotState(tmp_path / "s.json", "UTC")
    state.set_daily_limit(1)
    assert state.try_consume(1)
    assert not state.try_consume(1)
    day[0] += datetime.timedelta(minutes=2)
    assert state.try_consume(1)


def test_day_follows_configured_timezone(tmp_path, settings, monkeypatch):
    # 20:00 UTC — уже следующие сутки в Ташкенте (UTC+5).
    monkeypatch.setattr(
        state_module, "_now", lambda zone: datetime.datetime(2026, 11, 7, 20, 0, tzinfo=datetime.timezone.utc).astimezone(zone)
    )
    state = BotState(tmp_path / "s.json", "Asia/Tashkent")
    state.try_consume(1)
    saved = json.loads((tmp_path / "s.json").read_text())
    assert saved["usage"]["date"] == "2026-11-08"


def test_unknown_timezone_falls_back_to_utc(tmp_path, settings):
    assert BotState(tmp_path / "s.json", "Mars/Olympus").zone_name == "UTC"


def test_broken_state_file_starts_empty(tmp_path, settings):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    state = BotState(path)
    assert state.allowed_groups() == {-1: ""}
    state.activate_group(-5, "Новая", by=1)
    assert json.loads(path.read_text())["groups"]["-5"]["title"] == "Новая"


def test_titles_are_remembered_for_env_groups(tmp_path, settings):
    state = BotState(tmp_path / "s.json")
    state.remember_title(-1, "Бухгалтерия")
    assert BotState(tmp_path / "s.json").allowed_groups() == {-1: "Бухгалтерия"}
