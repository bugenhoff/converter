"""Настройки бота из переменных окружения и `.env`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PDF_MODES = ("llm_only", "llm_first", "reliability_first")
# Названия режимов из времён, когда распознавание умело только Groq.
LEGACY_PDF_MODES = {"groq_only": "llm_only", "groq_first": "llm_first"}

LLM_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _first_set(*keys: str) -> str | None:
    for key in keys:
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            return raw.strip()
    return None


def _int(*keys: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _first_set(*keys)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{keys[0]} должно быть целым числом, получено {raw!r}") from exc
    return _check_range(keys[0], value, minimum, maximum)


def _float(key: str, *, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _first_set(key)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{key} должно быть числом, получено {raw!r}") from exc
    return _check_range(key, value, minimum, maximum)


def _check_range(key, value, minimum, maximum):
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{key} должно быть >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{key} должно быть <= {maximum}")
    return value


def _bool(key: str, default: bool) -> bool:
    raw = _first_set(key)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _user_ids(key: str) -> frozenset[int]:
    raw = _str(key)
    try:
        return frozenset(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise RuntimeError(f"{key}: ожидаются числа через запятую") from exc


def _pdf_mode() -> str:
    mode = _str("PDF_CONVERSION_MODE", "llm_only").lower()
    mode = LEGACY_PDF_MODES.get(mode, mode)
    if mode not in PDF_MODES:
        raise RuntimeError(f"PDF_CONVERSION_MODE должен быть одним из: {', '.join(PDF_MODES)}")
    return mode


def _llm_provider() -> str:
    provider = _str("LLM_PROVIDER", "groq").lower()
    if provider not in LLM_BASE_URLS:
        raise RuntimeError(f"LLM_PROVIDER должен быть одним из: {', '.join(LLM_BASE_URLS)}")
    return provider


@dataclass
class Settings:
    telegram_token: str
    allowed_users_only: bool
    allowed_user_ids: frozenset[int]
    log_level: str
    temp_dir: Path
    batch_window_seconds: float
    max_parallel_files: int

    libreoffice_path: str
    pdf_conversion_mode: str
    tessdata_prefix: str
    ocr_languages: str

    llm_provider: str
    groq_api_key: str
    groq_model: str
    openrouter_api_key: str
    openrouter_model: str
    openrouter_reasoning_effort: str
    openrouter_provider_sort: str
    llm_concurrency: int
    llm_max_tokens_per_page: int
    llm_image_max_side: int
    llm_retries: int
    llm_timeout_seconds: float

    # Groq и OpenRouter отдают OpenAI-совместимый API, отличаются адрес, ключ и модель.
    @property
    def llm_model(self) -> str:
        return self.openrouter_model if self.llm_provider == "openrouter" else self.groq_model

    @property
    def llm_api_key(self) -> str:
        return self.openrouter_api_key if self.llm_provider == "openrouter" else self.groq_api_key

    @property
    def llm_api_key_name(self) -> str:
        return "OPENROUTER_API_KEY" if self.llm_provider == "openrouter" else "GROQ_API_KEY"

    @property
    def llm_base_url(self) -> str:
        return LLM_BASE_URLS[self.llm_provider]


def load_settings() -> Settings:
    temp_dir = Path(_str("TEMP_DIR", "./tmp")).expanduser()
    temp_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        telegram_token=_str("TELEGRAM_BOT_TOKEN"),
        allowed_users_only=_bool("ALLOWED_USERS_ONLY", True),
        allowed_user_ids=_user_ids("ALLOWED_USER_IDS"),
        log_level=_str("LOG_LEVEL", "INFO").upper(),
        temp_dir=temp_dir,
        batch_window_seconds=_float("BATCH_WINDOW_SECONDS", default=3.0, minimum=0.0, maximum=60.0),
        max_parallel_files=_int("MAX_PARALLEL_FILES", default=3, minimum=1, maximum=20),
        libreoffice_path=_str("LIBREOFFICE_PATH", "libreoffice"),
        pdf_conversion_mode=_pdf_mode(),
        tessdata_prefix=_str("TESSDATA_PREFIX"),
        ocr_languages=_str("OCR_LANGUAGES", "rus+eng+uzb+uzb_cyrl"),
        llm_provider=_llm_provider(),
        groq_api_key=_str("GROQ_API_KEY"),
        groq_model=_str("GROQ_MODEL", "qwen/qwen3.8-27b"),
        openrouter_api_key=_str("OPENROUTER_API_KEY"),
        openrouter_model=_str("OPENROUTER_MODEL", "openai/gpt-6-luna"),
        # Пусто — глубина рассуждений по умолчанию у модели; none отключает их.
        openrouter_reasoning_effort=_str("OPENROUTER_REASONING_EFFORT").lower(),
        # throughput — самые быстрые провайдеры модели, latency — с самым быстрым
        # первым токеном, price — самые дешёвые; пусто — балансировка OpenRouter.
        openrouter_provider_sort=_str("OPENROUTER_PROVIDER_SORT").lower(),
        llm_concurrency=_int("LLM_CONCURRENCY", default=6, minimum=1, maximum=32),
        llm_max_tokens_per_page=_int(
            "LLM_MAX_TOKENS_PER_PAGE", default=8000, minimum=1024, maximum=131072
        ),
        llm_image_max_side=_int("LLM_IMAGE_MAX_SIDE", default=1600, minimum=512, maximum=4096),
        llm_retries=_int("LLM_RETRIES", "GROQ_RETRY_PER_TASK", default=2, minimum=0, maximum=10),
        llm_timeout_seconds=_float("LLM_TIMEOUT", default=180.0, minimum=10.0, maximum=1800.0),
    )


settings = load_settings()
