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


@dataclass
class Settings:
    telegram_token: str
    libreoffice_path: str
    tessdata_prefix: str
    temp_dir: Path
    log_level: str

    def __post_init__(self) -> None:
        self.temp_dir = Path(self.temp_dir).expanduser()
        self.temp_dir.mkdir(parents=True, exist_ok=True)


settings = Settings(
    telegram_token=_load_env("TELEGRAM_BOT_TOKEN", required=True),
    libreoffice_path=_load_env("LIBREOFFICE_PATH", default="libreoffice"),
    tessdata_prefix=_load_env("TESSDATA_PREFIX", default="/root/tesseract/tessdata/"),
    temp_dir=Path(_load_env("TEMP_DIR", default="./tmp")),
    log_level=_load_env("LOG_LEVEL", default="INFO"),
)