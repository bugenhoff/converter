import pytest

from src.bot import state as state_module


@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    """Каждый тест — с чистым состоянием во временной папке, а не в ./data."""
    monkeypatch.setattr(state_module.settings, "super_admin_ids", frozenset())
    state = state_module.BotState(tmp_path / "state.json", "UTC")
    state_module.use_state(state)
    yield state
    state_module.use_state(None)
