"""Utilities to convert legacy DOC/PDF files to modern DOCX documents."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
import shlex
import shutil
from functools import lru_cache

try:
    import ocrmypdf
    from pdf2docx import Converter
except ImportError:
    ocrmypdf = None
    Converter = None

try:
    from .groq_converter import convert_pdf_to_docx_via_groq, GroqConversionError
except ImportError:
    convert_pdf_to_docx_via_groq = None
    GroqConversionError = None

from ..config.settings import settings

logger = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Signals that LibreOffice failed to convert the document."""


def convert_doc_to_docx(
    source_path: Path,
    output_dir: Path,
    libreoffice_bin: str = "libreoffice",
) -> Path:
    """Run LibreOffice headless converter and return the converted path."""

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = _resolve_libreoffice_command(libreoffice_bin) + [
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise ConversionError(
            "LibreOffice failed to convert document:"
            f" exit={process.returncode}, stderr={process.stderr.strip()}"
        )

    converted_path = output_dir / f"{source_path.stem}.docx"
    if not converted_path.exists():
        raise ConversionError("LibreOffice reported success but output file is missing")

    return converted_path


def convert_pdf_to_docx(source_path: Path, output_dir: Path) -> Path:
    """Convert PDF to DOCX using Groq LLM with OCR fallback."""
    
    # Try Groq LLM first (if available and configured)
    if convert_pdf_to_docx_via_groq and settings.groq_api_key:
        try:
            logger.info("Attempting PDF conversion via Groq LLM")
            return convert_pdf_to_docx_via_groq(source_path, output_dir)
        except (GroqConversionError, Exception) as e:
            logger.warning("Groq LLM conversion failed (%s), falling back to OCR method", e)
    else:
        logger.info("Groq not configured, using OCR method")
    
    # Fallback to OCR method
    return _convert_pdf_to_docx_ocr(source_path, output_dir)


def _convert_pdf_to_docx_ocr(source_path: Path, output_dir: Path) -> Path:
    """Convert PDF to DOCX using OCR and layout reconstruction (fallback method)."""
    if ocrmypdf is None or Converter is None:
        raise ImportError("ocrmypdf and pdf2docx are required for PDF conversion")

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if settings.tessdata_prefix:
        os.environ.setdefault("TESSDATA_PREFIX", settings.tessdata_prefix)

    docx_path = output_dir / f"{source_path.stem}.docx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        searchable_pdf = Path(tmp_dir) / "searchable.pdf"
        logger.info("Running OCR on %s", source_path)

        try:
            ocrmypdf.ocr(
                str(source_path),
                str(searchable_pdf),
                language=settings.ocr_languages,
                force_ocr=True,
                optimize=0,
                deskew=True,
                progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - library-specific errors
            raise ConversionError(f"OCR failed for {source_path.name}: {exc}") from exc

        logger.info("Starting PDF->DOCX conversion for %s", source_path)

        try:
            cv = Converter(str(searchable_pdf))
            cv.convert(str(docx_path), start=0, end=None)
            cv.close()
        except Exception as exc:
            raise ConversionError(f"PDF to DOCX conversion failed: {exc}") from exc

    if not docx_path.exists():
        raise ConversionError("Output file is missing after conversion")
    if docx_path.stat().st_size == 0:
        raise ConversionError("Conversion produced an empty DOCX file")

    return docx_path


def _resolve_libreoffice_command(libreoffice_hint: str | None) -> list[str]:
    """Return an executable command list for LibreOffice, handling Flatpak installs."""

    if libreoffice_hint:
        command = _normalize_command(libreoffice_hint)
        resolved = _ensure_executable(command)
        if resolved:
            return resolved
        logger.warning(
            "LibreOffice command '%s' is not available; falling back to auto-detection",
            libreoffice_hint,
        )

    for candidate in ("libreoffice", "soffice"):
        command = _normalize_command(candidate)
        resolved = _ensure_executable(command)
        if resolved:
            return resolved

    flatpak_command = _detect_flatpak_libreoffice()
    if flatpak_command:
        logger.info("Using Flatpak-installed LibreOffice")
        return list(flatpak_command)

    raise ConversionError(
        "LibreOffice executable was not found. Install LibreOffice or set LIBREOFFICE_PATH."
    )


def _normalize_command(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    return parts


def _ensure_executable(command: list[str]) -> list[str] | None:
    if not command:
        return None

    executable = command[0]
    exec_path = Path(executable)
    if exec_path.is_file() and os.access(exec_path, os.X_OK):
        return [str(exec_path), *command[1:]]

    resolved = shutil.which(executable)
    if resolved:
        return [resolved, *command[1:]]

    return None


@lru_cache(maxsize=1)
def _detect_flatpak_libreoffice() -> tuple[str, ...] | None:
    flatpak = shutil.which("flatpak")
    if not flatpak:
        return None

    try:
        probe = subprocess.run(
            [flatpak, "info", "org.libreoffice.LibreOffice"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if probe.returncode != 0:
        return None

    return (
        flatpak,
        "run",
        "--command=soffice",
        "org.libreoffice.LibreOffice",
    )