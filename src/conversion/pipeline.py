"""Единая точка входа: файл любого поддерживаемого типа → DOCX."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from ..config.settings import settings
from .docx_builder import PageContent, build_docx
from .errors import ConversionError
from .layout import plain_text
from .libreoffice import doc_to_docx
from .llm import LLMError, transcribe_pages
from .ocr import image_to_docx_ocr, pdf_to_docx_direct, pdf_to_docx_ocr
from .pages import PdfPages, image_to_page

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

StatusCallback = Callable[[str], None]


@dataclass
class ConversionResult:
    docx: bytes
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)


def detect_kind(file_name: str) -> str | None:
    name = file_name.lower()
    if name.endswith(".pdf"):
        return "pdf"
    if name.endswith(".docx"):
        return "docx"
    if name.endswith(".doc"):
        return "doc"
    if name.endswith(IMAGE_EXTENSIONS):
        return "image"
    return None


def convert_document(
    payload: bytes,
    file_name: str,
    kind: str,
    on_status: StatusCallback | None = None,
) -> ConversionResult:
    started = time.monotonic()
    status = on_status or (lambda _: None)
    if kind == "docx":
        result = ConversionResult(payload)
    elif kind == "doc":
        status("LibreOffice")
        result = ConversionResult(doc_to_docx(payload, file_name))
    elif kind == "pdf":
        result = _convert_pdf(payload, status)
    elif kind == "image":
        result = _convert_image(payload, status)
    else:
        raise ConversionError(f"Неподдерживаемый тип файла: {file_name}")

    logger.info(
        "Converted %s (%s, pages=%s) in %.1fs, warnings=%s",
        file_name,
        kind,
        result.pages,
        time.monotonic() - started,
        result.warnings,
    )
    return result


def _convert_pdf(payload: bytes, status: StatusCallback) -> ConversionResult:
    mode = settings.pdf_conversion_mode
    if mode == "llm_only":
        return _pdf_via_llm(payload, status)
    if mode == "llm_first":
        try:
            return _pdf_via_llm(payload, status)
        except ConversionError as exc:
            logger.warning("LLM conversion failed (%s), falling back to OCR", exc)
            status("OCR")
            return ConversionResult(pdf_to_docx_ocr(payload))

    for name, convert in (("pdf2docx", pdf_to_docx_direct), ("OCR", pdf_to_docx_ocr)):
        try:
            status(name)
            return ConversionResult(convert(payload))
        except ConversionError as exc:
            logger.warning("%s conversion failed: %s", name, exc)
    return _pdf_via_llm(payload, status)


def _convert_image(payload: bytes, status: StatusCallback) -> ConversionResult:
    if settings.pdf_conversion_mode == "reliability_first":
        status("OCR")
        return ConversionResult(image_to_docx_ocr(payload))
    try:
        page = image_to_page(payload, settings.llm_image_max_side)
        return _transcribe([page], 1, status)
    except ConversionError:
        if settings.pdf_conversion_mode != "llm_first":
            raise
        status("OCR")
        return ConversionResult(image_to_docx_ocr(payload))


def _pdf_via_llm(payload: bytes, status: StatusCallback) -> ConversionResult:
    with PdfPages(payload) as pdf:
        total = len(pdf)
        status(f"стр. 0/{total}")
        pages = (pdf.render(number, settings.llm_image_max_side) for number in range(1, total + 1))
        return _transcribe(pages, total, status)


def _transcribe(pages, total: int, status: StatusCallback) -> ConversionResult:
    results = transcribe_pages(pages, total, lambda done, _: status(f"стр. {done}/{total}"))
    failed = [r.page.number for r in results if r.error]
    if len(failed) == len(results):
        raise LLMError(f"Не распознана ни одна страница: {results[0].error}")
    if not any(plain_text(r.blocks) for r in results if r.blocks):
        raise LLMError("Модель не нашла текста ни на одной странице")

    contents = [
        PageContent(r.page.number, r.page.geometry, r.blocks or [], failed=bool(r.error))
        for r in results
    ]
    warnings = []
    if failed:
        warnings.append(f"не распознаны стр. {', '.join(map(str, failed))}")
    return ConversionResult(build_docx(contents), pages=total, warnings=warnings)
