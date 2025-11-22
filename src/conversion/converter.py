"""Utilities to convert legacy DOC files to modern DOCX via LibreOffice."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

try:
    import ocrmypdf
    from pdf2docx import Converter
except ImportError:
    ocrmypdf = None
    Converter = None

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

    command = [
        libreoffice_bin,
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
    """Convert PDF to DOCX using OCR and layout reconstruction."""
    if ocrmypdf is None or Converter is None:
        raise ImportError("ocrmypdf and pdf2docx are required for PDF conversion")

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. OCR the PDF (create a temp file for the OCR'd version)
    ocr_pdf_path = output_dir / f"{source_path.stem}_ocr.pdf"

    try:
        # We use force_ocr=True because the user specified "images"
        # and we want to ensure we have a good text layer for pdf2docx.
        ocrmypdf.ocr(
            input_file=source_path,
            output_file=ocr_pdf_path,
            language="rus+eng",
            force_ocr=True,
            progress_bar=False,
        )
    except Exception as e:
        # Clean up if OCR failed but file was created
        if ocr_pdf_path.exists():
            ocr_pdf_path.unlink()
        raise ConversionError(f"OCR failed: {e}") from e

    # 2. Convert OCR'd PDF to DOCX
    docx_path = output_dir / f"{source_path.stem}.docx"

    try:
        cv = Converter(str(ocr_pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
    except Exception as e:
        raise ConversionError(f"PDF to DOCX conversion failed: {e}") from e
    finally:
        # Cleanup intermediate OCR file
        if ocr_pdf_path.exists():
            ocr_pdf_path.unlink()

    if not docx_path.exists():
        raise ConversionError("Output file is missing after conversion")

    return docx_path