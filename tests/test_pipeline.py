import io

import pytest
from docx import Document

from src.conversion import pipeline
from src.conversion.errors import ConversionError
from src.conversion.layout import Paragraph
from src.conversion.llm import LLMError, PageResult
from src.conversion.pages import PageGeometry, PageImage


@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.doc", "doc"), ("a.DOCX", "docx"), ("a.PDF", "pdf"), ("a.jpeg", "image"), ("a.tiff", "image"), ("a.txt", None)],
)
def test_detect_kind(name, kind):
    assert pipeline.detect_kind(name) == kind


def test_docx_is_passed_through():
    assert pipeline.convert_document(b"docx", "a.docx", "docx").docx == b"docx"


def test_unsupported_kind_raises():
    with pytest.raises(ConversionError):
        pipeline.convert_document(b"x", "a.txt", "txt")


@pytest.fixture
def routes(monkeypatch):
    log = []

    def install(failing=()):
        def make(name):
            def run(*_args):
                log.append(name)
                if name in failing:
                    raise ConversionError(name)
                return name.encode()

            return run

        monkeypatch.setattr(pipeline, "pdf_to_docx_direct", make("direct"))
        monkeypatch.setattr(pipeline, "pdf_to_docx_ocr", make("ocr"))
        llm = make("llm")
        monkeypatch.setattr(pipeline, "_pdf_via_llm", lambda *args: pipeline.ConversionResult(llm(*args)))
        return log

    return install


def test_llm_only_never_uses_fallbacks(monkeypatch, routes):
    monkeypatch.setattr(pipeline.settings, "pdf_conversion_mode", "llm_only")
    log = routes(failing={"llm"})
    with pytest.raises(ConversionError):
        pipeline.convert_document(b"%PDF", "a.pdf", "pdf")
    assert log == ["llm"]


def test_llm_first_falls_back_to_ocr(monkeypatch, routes):
    monkeypatch.setattr(pipeline.settings, "pdf_conversion_mode", "llm_first")
    log = routes(failing={"llm"})
    assert pipeline.convert_document(b"%PDF", "a.pdf", "pdf").docx == b"ocr"
    assert log == ["llm", "ocr"]


def test_reliability_first_order(monkeypatch, routes):
    monkeypatch.setattr(pipeline.settings, "pdf_conversion_mode", "reliability_first")
    log = routes(failing={"direct", "ocr"})
    assert pipeline.convert_document(b"%PDF", "a.pdf", "pdf").docx == b"llm"
    assert log == ["direct", "ocr", "llm"]


def _page(number):
    return PageImage(number, b"", PageGeometry(595.28, 841.89))


def test_partially_failed_document_is_built_with_warning(monkeypatch):
    def fake_transcribe(pages, total, on_progress):
        on_progress(2, total)
        return [PageResult(_page(1), blocks=[Paragraph("ok")]), PageResult(_page(2), error="boom")]

    monkeypatch.setattr(pipeline, "transcribe_pages", fake_transcribe)
    statuses = []
    result = pipeline._transcribe([], 2, statuses.append)
    assert result.warnings == ["не распознаны стр. 2"]
    assert result.pages == 2
    assert statuses == ["стр. 2/2"]
    texts = [p.text for p in Document(io.BytesIO(result.docx)).paragraphs]
    assert texts == ["ok", "[Страница 2 не распознана]"]


def test_fully_failed_document_raises(monkeypatch):
    monkeypatch.setattr(
        pipeline, "transcribe_pages", lambda *_: [PageResult(_page(1), error="boom")]
    )
    with pytest.raises(LLMError, match="ни одна"):
        pipeline._transcribe([], 1, lambda _: None)
