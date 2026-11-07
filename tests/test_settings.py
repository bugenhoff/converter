import pytest

from src.config import settings as settings_module


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "tmp"))
    for key in ("PDF_CONVERSION_MODE", "LLM_PROVIDER", "ALLOWED_USER_IDS", "LLM_RETRIES", "GROQ_RETRY_PER_TASK"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.mark.parametrize(("raw", "expected"), [("groq_only", "llm_only"), ("groq_first", "llm_first"), ("reliability_first", "reliability_first")])
def test_legacy_pdf_modes_are_accepted(env, raw, expected):
    env.setenv("PDF_CONVERSION_MODE", raw)
    assert settings_module.load_settings().pdf_conversion_mode == expected


def test_unknown_mode_is_rejected(env):
    env.setenv("PDF_CONVERSION_MODE", "magic")
    with pytest.raises(RuntimeError, match="PDF_CONVERSION_MODE"):
        settings_module.load_settings()


def test_provider_selects_key_model_and_url(env):
    env.setenv("LLM_PROVIDER", "openrouter")
    env.setenv("OPENROUTER_API_KEY", "or-key")
    env.setenv("OPENROUTER_MODEL", "vendor/model")
    loaded = settings_module.load_settings()
    assert loaded.llm_api_key == "or-key"
    assert loaded.llm_model == "vendor/model"
    assert loaded.llm_base_url == "https://openrouter.ai/api/v1"


def test_user_ids_and_legacy_retry_name(env):
    env.setenv("ALLOWED_USER_IDS", "1, 2,3")
    env.setenv("GROQ_RETRY_PER_TASK", "5")
    loaded = settings_module.load_settings()
    assert loaded.allowed_user_ids == frozenset({1, 2, 3})
    assert loaded.llm_retries == 5


def test_group_settings(env):
    env.setenv("ALLOWED_CHAT_IDS", "-1001234567890, -42")
    env.setenv("GROUP_AUTO_CONVERT", "yes")
    loaded = settings_module.load_settings()
    assert loaded.allowed_chat_ids == frozenset({-1001234567890, -42})
    assert loaded.group_auto_convert is True


def test_group_settings_default_to_closed(env):
    env.delenv("ALLOWED_CHAT_IDS", raising=False)
    env.delenv("GROUP_AUTO_CONVERT", raising=False)
    loaded = settings_module.load_settings()
    assert loaded.allowed_chat_ids == frozenset()
    assert loaded.group_auto_convert is False


def test_auto_pdf_mode_is_accepted(env):
    env.setenv("PDF_CONVERSION_MODE", "auto")
    assert settings_module.load_settings().pdf_conversion_mode == "auto"
