"""Smoke tests for the converter logic."""

from pathlib import Path

import pytest

from src.conversion.converter import ConversionError, convert_doc_to_docx, convert_pdf_to_docx


def test_missing_source_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert_doc_to_docx(tmp_path / "missing.doc", tmp_path, "libreoffice")


def test_conversion_failure_is_reported(tmp_path: Path, monkeypatch):
    src = tmp_path / "test.doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.converter._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 1
        stderr = "boom"

    def fake_run(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr("src.conversion.converter.subprocess.run", fake_run)

    with pytest.raises(ConversionError):
        convert_doc_to_docx(src, tmp_path, "libreoffice")


def test_successful_conversion(tmp_path: Path, monkeypatch):
    src = tmp_path / "test.doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.converter._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 0
        stderr = ""

    def fake_run(*args, **kwargs):
        output_path = Path(args[0][-1]).with_suffix(".docx")
        output_path.write_text("converted")
        return FakeResult()

    monkeypatch.setattr("src.conversion.converter.subprocess.run", fake_run)

    converted = convert_doc_to_docx(src, tmp_path, "libreoffice")
    assert converted.exists()
    assert converted.suffix == ".docx"


def test_convert_pdf_groq():
    """Test Groq PDF conversion with real file."""
    from src.conversion.groq_converter import convert_pdf_to_docx_via_groq
    
    pdf_file = Path("387-test.pdf")
    if not pdf_file.exists():
        pytest.skip("PDF test file not found")
        
    output_dir = Path("tmp")
    output_dir.mkdir(exist_ok=True)
    
    try:
        result = convert_pdf_to_docx_via_groq(pdf_file, output_dir)
        assert result.exists()
        assert result.suffix == ".docx"
        assert result.stat().st_size > 0
        print(f"✅ Groq conversion successful: {result}")
    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"Groq conversion failed: {e}")


def test_convert_pdf_prefers_direct_then_ocr(tmp_path: Path, monkeypatch):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 test")
    output_dir = tmp_path / "out"

    direct_calls = {"count": 0}
    ocr_calls = {"count": 0}

    def fake_direct(*_args, **_kwargs):
        direct_calls["count"] += 1
        raise ConversionError("direct failed")

    def fake_ocr(_source, output):
        ocr_calls["count"] += 1
        result = Path(output) / "sample.docx"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("ocr")
        return result

    monkeypatch.setattr("src.conversion.converter._convert_pdf_to_docx_direct", fake_direct)
    monkeypatch.setattr("src.conversion.converter._convert_pdf_to_docx_ocr", fake_ocr)
    monkeypatch.setattr(
        "src.conversion.converter.convert_pdf_to_docx_via_groq",
        lambda *_args, **_kwargs: pytest.fail("Groq must not be called when OCR succeeds"),
    )

    result = convert_pdf_to_docx(source, output_dir)
    assert result.exists()
    assert result.name == "sample.docx"
    assert direct_calls["count"] == 1
    assert ocr_calls["count"] == 1


def test_convert_pdf_uses_groq_only_after_direct_and_ocr_fail(tmp_path: Path, monkeypatch):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4 test")
    output_dir = tmp_path / "out"

    groq_result = output_dir / "sample.docx"

    monkeypatch.setattr(
        "src.conversion.converter._convert_pdf_to_docx_direct",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConversionError("direct failed")),
    )
    monkeypatch.setattr(
        "src.conversion.converter._convert_pdf_to_docx_ocr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ConversionError("ocr failed")),
    )

    def fake_groq(_source, output):
        result = Path(output) / "sample.docx"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("groq")
        return result

    monkeypatch.setattr("src.conversion.converter.convert_pdf_to_docx_via_groq", fake_groq)
    monkeypatch.setattr("src.conversion.converter.settings.groq_api_key", "test-key")

    result = convert_pdf_to_docx(source, output_dir)
    assert result == groq_result
    assert result.exists()
