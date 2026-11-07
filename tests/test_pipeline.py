import io
from types import SimpleNamespace

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
    assert pipeline.convert_document(b"docx", "a.docx", "docx").data == b"docx"


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
    assert pipeline.convert_document(b"%PDF", "a.pdf", "pdf").data == b"ocr"
    assert log == ["llm", "ocr"]


def test_reliability_first_order(monkeypatch, routes):
    monkeypatch.setattr(pipeline.settings, "pdf_conversion_mode", "reliability_first")
    log = routes(failing={"direct", "ocr"})
    assert pipeline.convert_document(b"%PDF", "a.pdf", "pdf").data == b"llm"
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
    texts = [p.text for p in Document(io.BytesIO(result.data)).paragraphs]
    assert texts == ["ok", "[Страница 2 не распознана]"]


def test_fully_failed_document_raises(monkeypatch):
    monkeypatch.setattr(
        pipeline, "transcribe_pages", lambda *_: [PageResult(_page(1), error="boom")]
    )
    with pytest.raises(LLMError, match="ни одна"):
        pipeline._transcribe([], 1, lambda _: None)


def test_wps_is_a_word_document():
    assert pipeline.detect_kind("Письмо.WPS") == "wps"


def test_private_wps_goes_through_libreoffice(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "doc_to_docx", lambda payload, name: seen.append(name) or b"docx")
    assert pipeline.convert_document(b"wps", "a.wps", "wps").data == b"docx"
    assert seen == ["a.wps"]


@pytest.mark.parametrize("kind", ["docx", "doc", "wps"])
def test_group_turns_word_documents_into_pdf(monkeypatch, kind):
    monkeypatch.setattr(pipeline, "to_pdf", lambda payload, name: b"%PDF-" + name.encode())
    monkeypatch.setattr(pipeline, "pdf_page_count", lambda pdf: 3)
    result = pipeline.convert_for_group(b"x", f"a.{kind}", kind)
    assert result.data == f"%PDF-a.{kind}".encode()
    assert result.extension == ".pdf"
    assert result.pages == 3


def test_group_rejects_images():
    with pytest.raises(ConversionError):
        pipeline.convert_for_group(b"x", "a.jpg", "image")


@pytest.fixture
def auto_routes(monkeypatch):
    def install(digital=True, layout_fails=False):
        log = []
        monkeypatch.setattr(
            pipeline, "profile_pdf", lambda _: SimpleNamespace(is_digital=digital, pages=2, scan_pages=0 if digital else 2)
        )

        def layout(_):
            log.append("layout")
            if layout_fails:
                raise ConversionError("pdf2docx")
            return b"layout"

        monkeypatch.setattr(pipeline, "pdf_to_docx_layout", layout)
        monkeypatch.setattr(
            pipeline, "_pdf_via_llm", lambda *_: log.append("llm") or pipeline.ConversionResult(b"llm")
        )
        return log

    return install


def test_group_pdf_with_text_layer_keeps_layout(auto_routes):
    log = auto_routes(digital=True)
    result = pipeline.convert_for_group(b"%PDF", "a.pdf", "pdf")
    assert (result.data, result.extension, result.pages) == (b"layout", ".docx", 2)
    assert log == ["layout"]


def test_group_scanned_pdf_goes_to_llm(auto_routes):
    log = auto_routes(digital=False)
    assert pipeline.convert_for_group(b"%PDF", "a.pdf", "pdf").data == b"llm"
    assert log == ["llm"]


def test_layout_failure_falls_back_to_llm(auto_routes):
    log = auto_routes(digital=True, layout_fails=True)
    assert pipeline.convert_for_group(b"%PDF", "a.pdf", "pdf").data == b"llm"
    assert log == ["layout", "llm"]


def test_private_auto_mode_uses_the_same_route(monkeypatch, auto_routes):
    monkeypatch.setattr(pipeline.settings, "pdf_conversion_mode", "auto")
    log = auto_routes(digital=True)
    assert pipeline.convert_document(b"%PDF", "a.pdf", "pdf").data == b"layout"
    assert log == ["layout"]
