"""Configuration helpers that load secrets from `.env`."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _load_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(f"Mandatory environment variable {key} is missing")
    return value  # type: ignore[no-any-return]


def _load_env_int(
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = os.environ.get(key)
    if raw is None or not raw.strip():
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError(f"Mandatory integer environment variable {key} is invalid") from exc

    if minimum is not None and value < minimum:
        raise RuntimeError(f"{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{key} must be <= {maximum}")
    return value


@dataclass
class Settings:
    telegram_token: str
    libreoffice_path: str
    tessdata_prefix: str
    ocr_languages: str
    groq_api_key: str
    groq_model: str
    groq_max_tokens: int
    groq_batch_size: int
    groq_image_max_side: int
    groq_pdf_image_dpi: int
    temp_dir: Path
    log_level: str
    allowed_users_only: bool
    allowed_user_ids: list[int]

    def __post_init__(self) -> None:
        self.temp_dir = Path(self.temp_dir).expanduser()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse allowed user IDs from comma-separated string
        if isinstance(self.allowed_user_ids, str):
            if self.allowed_user_ids.strip():
                try:
                    self.allowed_user_ids = [
                        int(uid.strip()) 
                        for uid in self.allowed_user_ids.split(',') 
                        if uid.strip()
                    ]
                except ValueError:
                    raise RuntimeError("Invalid ALLOWED_USER_IDS format. Use comma-separated integers.")
            else:
                self.allowed_user_ids = []


settings = Settings(
    telegram_token=_load_env("TELEGRAM_BOT_TOKEN", required=True),
    libreoffice_path=_load_env("LIBREOFFICE_PATH", default="libreoffice"),
    tessdata_prefix=_load_env("TESSDATA_PREFIX", default="/root/tesseract/tessdata/"),
    ocr_languages=_load_env("OCR_LANGUAGES", default="rus+eng+uzb+uzb_cyrl"),
    groq_api_key=_load_env("GROQ_API_KEY", default=""),
    groq_model=_load_env("GROQ_MODEL", default="llama-3.2-11b-vision-preview"),
    groq_max_tokens=_load_env_int("GROQ_MAX_TOKENS", default=8000, minimum=256, maximum=32768),
    groq_batch_size=_load_env_int("GROQ_BATCH_SIZE", default=3, minimum=1, maximum=10),
    groq_image_max_side=_load_env_int("GROQ_IMAGE_MAX_SIDE", default=800, minimum=256, maximum=3000),
    groq_pdf_image_dpi=_load_env_int("GROQ_PDF_IMAGE_DPI", default=200, minimum=72, maximum=600),
    temp_dir=Path(_load_env("TEMP_DIR", default="./tmp")),
    log_level=_load_env("LOG_LEVEL", default="INFO"),
    allowed_users_only=_load_env("ALLOWED_USERS_ONLY", default="true").lower() == "true",
    allowed_user_ids=_load_env("ALLOWED_USER_IDS", default=""),
)
