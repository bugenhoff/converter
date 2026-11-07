"""Единая точка входа: файл любого поддерживаемого типа → DOCX (или PDF для группы)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config.settings import settings
from .docx_builder import PageContent, build_docx
from .errors import ConversionError
from .layout import plain_text
from .libreoffice import doc_to_docx, to_pdf
from .llm import LLMError, transcribe_pages
from .ocr import image_to_docx_ocr, pdf_to_docx_direct, pdf_to_docx_ocr
from .pages import PdfPages, image_to_page
from .pdf_layout import pdf_page_count, pdf_to_docx_layout, profile_pdf
from .word_binary import repair_imported_docx

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# Текстовые документы, которые LibreOffice открывает сам.
WORD_KINDS = ("docx", "doc", "wps")
# В группе: PDF → DOCX, документы Word и WPS → PDF.
GROUP_KINDS = ("pdf", *WORD_KINDS)

StatusCallback = Callable[[str], None]


@dataclass
class ConversionResult:
    data: bytes
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    extension: str = ".docx"


def detect_kind(file_name: str) -> str | None:
    name = file_name.lower()
    for kind in ("pdf", *WORD_KINDS):
        if name.endswith(f".{kind}"):
            return kind
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
    elif kind in ("doc", "wps"):
        status("LibreOffice")
        docx, _ = _word_binary_to_docx(payload, file_name)
        result = ConversionResult(docx)
    elif kind == "pdf":
        result = _convert_pdf(payload, status)
    elif kind == "image":
        result = _convert_image(payload, status)
    else:
        raise ConversionError(f"Неподдерживаемый тип файла: {file_name}")
    _log_result(file_name, kind, result, started)
    return result


def convert_for_group(
    payload: bytes,
    file_name: str,
    kind: str,
    on_status: StatusCallback | None = None,
) -> ConversionResult:
    """Режим группы: PDF → DOCX, DOCX/DOC/WPS → PDF."""
    started = time.monotonic()
    status = on_status or (lambda _: None)
    if kind == "pdf":
        result = _pdf_auto(payload, status)
    elif kind in WORD_KINDS:
        status("LibreOffice → PDF")
        pdf = _word_to_pdf(payload, file_name, kind)
        result = ConversionResult(pdf, pages=pdf_page_count(pdf), extension=".pdf")
    else:
        raise ConversionError(f"В группе конвертируются только PDF, DOCX, DOC и WPS: {file_name}")
    _log_result(file_name, kind, result, started)
    return result


def _word_binary_to_docx(payload: bytes, file_name: str) -> tuple[bytes, list[str]]:
    """.doc/.wps → DOCX через LibreOffice с исправлением того, что он читает неправильно."""
    docx = doc_to_docx(payload, file_name)
    try:
        docx, fixes = repair_imported_docx(docx, payload)
    except Exception:
        # Исправления — улучшение, а не условие: без них документ всё равно отдаётся.
        logger.exception("Repair of LibreOffice import failed for %s", file_name)
        return docx, []
    if fixes:
        logger.info("Repaired LibreOffice import of %s: %s", file_name, "; ".join(fixes))
    return docx, fixes


def _word_to_pdf(payload: bytes, file_name: str, kind: str) -> bytes:
    if kind == "docx":
        return to_pdf(payload, file_name)
    # Бинарный .doc/.wps: если импорт пришлось чинить, PDF делается из исправленного DOCX.
    docx, fixes = _word_binary_to_docx(payload, file_name)
    if fixes:
        return to_pdf(docx, f"{Path(file_name).stem or 'document'}.docx")
    return to_pdf(payload, file_name)


def _log_result(file_name: str, kind: str, result: ConversionResult, started: float) -> None:
    logger.info(
        "Converted %s (%s, pages=%s) in %.1fs, warnings=%s",
        file_name,
        kind,
        result.pages,
        time.monotonic() - started,
        result.warnings,
    )


def _convert_pdf(payload: bytes, status: StatusCallback) -> ConversionResult:
    mode = settings.pdf_conversion_mode
    if mode == "auto":
        return _pdf_auto(payload, status)
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


def _pdf_auto(payload: bytes, status: StatusCallback) -> ConversionResult:
    """Цифровой PDF — вёрсткой через pdf2docx, скан — распознаванием моделью."""
    profile = profile_pdf(payload)
    if not profile.is_digital:
        logger.info("PDF looks scanned (%d of %d pages), using LLM", profile.scan_pages, profile.pages)
        return _pdf_via_llm(payload, status)
    status("перенос вёрстки")
    try:
        return ConversionResult(pdf_to_docx_layout(payload), pages=profile.pages)
    except ConversionError as exc:
        logger.warning("Layout conversion failed (%s), falling back to LLM", exc)
        return _pdf_via_llm(payload, status)


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
