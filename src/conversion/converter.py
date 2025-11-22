"""Utilities to convert legacy DOC files to modern DOCX via LibreOffice."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

try:
    import ocrmypdf
    import pikepdf
    from pdf2docx import Converter
except ImportError:
    ocrmypdf = None
    pikepdf = None
    Converter = None

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
    if ocrmypdf is None or Converter is None or pikepdf is None:
        raise ImportError("ocrmypdf, pikepdf and pdf2docx are required for PDF conversion")

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare environment for Tesseract
    env = os.environ.copy()
    if settings.tessdata_prefix:
        env["TESSDATA_PREFIX"] = settings.tessdata_prefix

    # Create a temp dir for processing pages
    with tempfile.TemporaryDirectory() as temp_pages_dir:
        temp_pages_path = Path(temp_pages_dir)
        
        logger.info("Converting PDF to images: %s", source_path)
        # 1. Convert PDF to images
        # pdftoppm -png -r 300 source_path temp_pages_path/page
        try:
            subprocess.run(
                ["pdftoppm", "-png", "-r", "300", str(source_path), str(temp_pages_path / "page")],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            raise ConversionError(f"Failed to convert PDF to images: {e.stderr}") from e
        
        # 2. OCR each image to text-only PDF
        pdf_pages = []
        # Sort files numerically: page-1.png, page-2.png, ...
        image_files = sorted(
            temp_pages_path.glob("page-*.png"),
            key=lambda p: int(p.stem.split('-')[-1])
        )
        
        logger.info("Starting OCR for %d pages...", len(image_files))
        
        for img_path in image_files:
            # tesseract output filename should not have extension, it adds .pdf
            output_base = img_path.with_suffix("")
            pdf_page_path = img_path.with_suffix(".pdf")
            
            cmd = [
                "tesseract", str(img_path), str(output_base),
                "-l", "rus+eng",
                "-c", "textonly_pdf=1",
                "pdf"
            ]
            
            try:
                subprocess.run(cmd, env=env, check=True, capture_output=True)
                if pdf_page_path.exists():
                    pdf_pages.append(pdf_page_path)
                else:
                    logger.error("Tesseract did not produce output for %s", img_path)
            except subprocess.CalledProcessError as e:
                logger.error("OCR failed for page %s: %s", img_path, e.stderr)
                raise ConversionError(f"OCR failed for page {img_path.name}") from e

        if not pdf_pages:
            raise ConversionError("No pages were successfully processed")

        # 3. Merge PDFs
        merged_pdf_path = output_dir / f"{source_path.stem}_textonly.pdf"
        logger.info("Merging %d text-only PDF pages...", len(pdf_pages))
        
        try:
            with pikepdf.new() as pdf:
                for page_path in pdf_pages:
                    with pikepdf.open(page_path) as p:
                        pdf.pages.extend(p.pages)
                pdf.save(merged_pdf_path)
        except Exception as e:
            raise ConversionError(f"Failed to merge PDF pages: {e}") from e

    # 4. Convert to DOCX
    docx_path = output_dir / f"{source_path.stem}.docx"
    logger.info("Starting PDF->DOCX conversion for %s", merged_pdf_path)

    try:
        cv = Converter(str(merged_pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
        logger.info("PDF->DOCX conversion completed: %s", docx_path)
    except Exception as e:
        raise ConversionError(f"PDF to DOCX conversion failed: {e}") from e
    finally:
        # Cleanup intermediate merged PDF
        if merged_pdf_path.exists():
            merged_pdf_path.unlink()

    if not docx_path.exists():
        raise ConversionError("Output file is missing after conversion")

    return docx_path