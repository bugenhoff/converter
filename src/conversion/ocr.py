"""Запасной путь без LLM: pdf2docx по текстовому слою или OCR через Tesseract."""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path

from PIL import Image

from ..config.settings import settings
from .errors import ConversionError
from .pages import pdf_has_text_layer

logger = logging.getLogger(__name__)


def pdf_to_docx_direct(pdf_bytes: bytes) -> bytes:
    """pdf2docx по текстовому слою; у сканов слоя нет, и результат был бы из картинок."""
    if not pdf_has_text_layer(pdf_bytes):
        raise ConversionError("В PDF нет текстового слоя")
    with tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix="direct_") as tmp:
        source = Path(tmp) / "source.pdf"
        source.write_bytes(pdf_bytes)
        return _pdf2docx(source, Path(tmp) / "result.docx")


def pdf_to_docx_ocr(pdf_bytes: bytes) -> bytes:
    try:
        import ocrmypdf
    except ImportError as exc:
        raise ConversionError("Для OCR нужен пакет ocrmypdf") from exc

    if settings.tessdata_prefix:
        os.environ.setdefault("TESSDATA_PREFIX", settings.tessdata_prefix)

    with tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix="ocr_") as tmp:
        source = Path(tmp) / "source.pdf"
        searchable = Path(tmp) / "searchable.pdf"
        source.write_bytes(pdf_bytes)
        try:
            ocrmypdf.ocr(
                str(source),
                str(searchable),
                language=settings.ocr_languages,
                force_ocr=True,
                optimize=0,
                deskew=True,
                progress_bar=False,
            )
        except Exception as exc:
            raise ConversionError(f"OCR не удался: {exc}") from exc
        return _pdf2docx(searchable, Path(tmp) / "result.docx")


def image_to_docx_ocr(image_bytes: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, "PDF", resolution=300.0)
    except Exception as exc:
        raise ConversionError(f"Не удалось подготовить изображение для OCR: {exc}") from exc
    return pdf_to_docx_ocr(buffer.getvalue())


def _pdf2docx(source: Path, target: Path) -> bytes:
    try:
        from pdf2docx import Converter
    except ImportError as exc:
        raise ConversionError("Для этого режима нужен пакет pdf2docx") from exc

    converter = None
    try:
        converter = Converter(str(source))
        converter.convert(str(target), start=0, end=None)
    except Exception as exc:
        raise ConversionError(f"pdf2docx не справился: {exc}") from exc
    finally:
        if converter is not None:
            converter.close()

    if not target.exists() or target.stat().st_size == 0:
        raise ConversionError("pdf2docx не создал DOCX")
    return target.read_bytes()
