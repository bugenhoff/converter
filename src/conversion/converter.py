"""Utilities to convert legacy DOC/PDF files to modern DOCX documents."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
import shlex
import shutil
from functools import lru_cache

try:
    import ocrmypdf
except ImportError:
    ocrmypdf = None

try:
    from pdf2docx import Converter
except ImportError:
    Converter = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from .groq_converter import convert_pdf_to_docx_via_groq, GroqConversionError
except ImportError:
    convert_pdf_to_docx_via_groq = None
    GroqConversionError = None

from ..config.settings import settings

logger = logging.getLogger(__name__)

DOCX_EXPORT_TARGETS = (
    "docx",
    "docx:Office Open XML Text",
    "docx:MS Word 2007 XML",
)


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
    started_at = time.time()
    existing_outputs = {path.resolve() for path in _iter_docx_outputs(output_dir)}

    libreoffice_command = _resolve_libreoffice_command(libreoffice_bin)
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    for export_target in DOCX_EXPORT_TARGETS:
        command = libreoffice_command + [
            "--headless",
            "--convert-to",
            export_target,
            "--outdir",
            str(output_dir),
            str(source_path),
        ]

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        attempts.append((export_target, process))

        converted_path = _find_libreoffice_output(
            source_path=source_path,
            output_dir=output_dir,
            stdout=process.stdout,
            existing_outputs=existing_outputs,
            started_at=started_at,
        )
        if converted_path is not None:
            break
    else:
        converted_path = None

    if converted_path is None:
        available = ", ".join(path.name for path in _iter_docx_outputs(output_dir)) or "none"
        raise ConversionError(
            "LibreOffice did not produce a DOCX file:"
            f" attempts={_format_libreoffice_attempts(attempts)},"
            f" output_dir={output_dir}, docx_files={available}"
        )
    if converted_path.stat().st_size == 0:
        raise ConversionError(f"LibreOffice produced an empty DOCX file: {converted_path}")

    return converted_path


def _format_libreoffice_attempts(
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]]
) -> str:
    return "; ".join(
        (
            f"target={target!r}, exit={process.returncode}, "
            f"stdout={process.stdout.strip()!r}, stderr={process.stderr.strip()!r}"
        )
        for target, process in attempts
    )


def _iter_docx_outputs(output_dir: Path) -> list[Path]:
    """Return DOCX-like files, handling case-sensitive filesystems."""
    if not output_dir.exists():
        return []
    return [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".docx"
    ]


def _find_libreoffice_output(
    *,
    source_path: Path,
    output_dir: Path,
    stdout: str,
    existing_outputs: set[Path],
    started_at: float,
) -> Path | None:
    expected_path = output_dir / f"{source_path.stem}.docx"
    if expected_path.exists():
        return expected_path

    for parsed_path in _parse_libreoffice_output_paths(stdout, output_dir):
        if parsed_path.exists() and parsed_path.suffix.lower() == ".docx":
            return parsed_path

    docx_outputs = _iter_docx_outputs(output_dir)
    new_outputs = [
        path
        for path in docx_outputs
        if path.resolve() not in existing_outputs or path.stat().st_mtime >= started_at
    ]
    if not new_outputs:
        return None

    for path in new_outputs:
        if path.stem.casefold() == source_path.stem.casefold():
            return path

    if len(new_outputs) == 1:
        return new_outputs[0]

    return max(new_outputs, key=lambda path: path.stat().st_mtime)


def _parse_libreoffice_output_paths(stdout: str, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for line in stdout.splitlines():
        if "->" not in line:
            continue
        raw_path = line.split("->", 1)[1].strip()
        if " using" in raw_path:
            raw_path = raw_path.split(" using", 1)[0].strip()
        raw_path = raw_path.strip("'\"")
        if not raw_path:
            continue
        parsed_path = Path(raw_path)
        if not parsed_path.is_absolute():
            parsed_path = output_dir / parsed_path
        paths.append(parsed_path)
    return paths


def convert_pdf_to_docx(source_path: Path, output_dir: Path) -> Path:
    """Convert PDF to DOCX according to PDF_CONVERSION_MODE."""

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    mode = settings.pdf_conversion_mode
    logger.info("PDF conversion mode=%s", mode)

    if mode == "groq_only":
        return _convert_pdf_to_docx_groq(source_path, output_dir)
    if mode == "groq_first":
        try:
            return _convert_pdf_to_docx_groq(source_path, output_dir)
        except Exception as exc:
            logger.warning("Groq-first conversion failed (%s), falling back to reliability pipeline", exc)
            return _convert_pdf_to_docx_reliability(source_path, output_dir)

    return _convert_pdf_to_docx_reliability(source_path, output_dir)


def convert_image_to_docx(source_path: Path, output_dir: Path) -> Path:
    """Convert image files (png/jpg/...) to DOCX according to PDF_CONVERSION_MODE."""

    if Image is None:
        raise ImportError("Pillow is required for image conversion")

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_pdf = Path(tmp_dir) / f"{source_path.stem}.pdf"
        try:
            with Image.open(source_path) as image:
                if image.mode in ("RGBA", "LA", "P"):
                    image = image.convert("RGB")
                image.save(temp_pdf, "PDF", resolution=300.0)
        except Exception as exc:
            raise ConversionError(f"Failed to prepare image for OCR: {exc}") from exc

        if settings.pdf_conversion_mode == "groq_only":
            docx_path = convert_pdf_to_docx(temp_pdf, output_dir)
        else:
            docx_path = _convert_pdf_to_docx_ocr(temp_pdf, output_dir)
        final_path = output_dir / f"{source_path.stem}.docx"
        if docx_path != final_path:
            docx_path.replace(final_path)
        return final_path


def _convert_pdf_to_docx_groq(source_path: Path, output_dir: Path) -> Path:
    if not (convert_pdf_to_docx_via_groq and settings.llm_api_key):
        raise ConversionError(f"LLM conversion is not available: configure {settings.llm_api_key_name}")
    try:
        logger.info("Attempting PDF conversion via Groq LLM")
        return convert_pdf_to_docx_via_groq(source_path, output_dir)
    except (GroqConversionError, Exception) as exc:
        raise ConversionError(f"Groq conversion failed: {exc}") from exc


def _convert_pdf_to_docx_reliability(source_path: Path, output_dir: Path) -> Path:
    try:
        logger.info("Attempting direct PDF->DOCX conversion via pdf2docx")
        return _convert_pdf_to_docx_direct(source_path, output_dir)
    except Exception as exc:
        logger.warning("Direct pdf2docx conversion failed (%s), trying OCR pipeline", exc)

    try:
        logger.info("Attempting OCR PDF->DOCX conversion")
        return _convert_pdf_to_docx_ocr(source_path, output_dir)
    except Exception as exc:
        logger.warning("OCR PDF conversion failed (%s)", exc)

    try:
        logger.info("Attempting final fallback PDF conversion via Groq LLM")
        return _convert_pdf_to_docx_groq(source_path, output_dir)
    except Exception as exc:
        logger.warning("Groq LLM conversion failed (%s)", exc)

    raise ConversionError(
        "Failed to convert PDF using direct, OCR and Groq fallback pipelines"
    )


def _convert_pdf_to_docx_direct(source_path: Path, output_dir: Path) -> Path:
    """Convert PDF to DOCX directly via pdf2docx without OCR."""
    if Converter is None:
        raise ImportError("pdf2docx is required for direct PDF conversion")

    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_path = output_dir / f"{source_path.stem}.docx"

    cv = None
    try:
        cv = Converter(str(source_path))
        cv.convert(str(docx_path), start=0, end=None)
    except Exception as exc:
        raise ConversionError(f"Direct PDF to DOCX conversion failed: {exc}") from exc
    finally:
        if cv is not None:
            cv.close()

    if not docx_path.exists():
        raise ConversionError("Direct conversion reported success but output file is missing")
    if docx_path.stat().st_size == 0:
        raise ConversionError("Direct conversion produced an empty DOCX file")

    return docx_path


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

        cv = None
        try:
            cv = Converter(str(searchable_pdf))
            cv.convert(str(docx_path), start=0, end=None)
        except Exception as exc:
            raise ConversionError(f"PDF to DOCX conversion failed: {exc}") from exc
        finally:
            if cv is not None:
                cv.close()

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
