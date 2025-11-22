"""Smoke tests for the converter logic."""

from pathlib import Path

import pytest

from src.conversion.converter import ConversionError, convert_doc_to_docx


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