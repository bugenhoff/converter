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