from __future__ import annotations

from pathlib import Path

from src.conversion.memory_processor import (
    _convert_doc_in_memory,
    _convert_docx_in_memory,
    _convert_pdf_in_memory,
)


def test_convert_pdf_in_memory_uses_converter_pipeline(monkeypatch, tmp_path: Path):
    def fake_convert_pdf_to_docx(source_path, output_dir):
        out = Path(output_dir) / f"{Path(source_path).stem}.docx"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"docx-result")
        return out

    monkeypatch.setattr(
        "src.conversion.converter.convert_pdf_to_docx",
        fake_convert_pdf_to_docx,
    )

    result = _convert_pdf_in_memory(b"%PDF-1.4 mock", "demo.pdf")
    assert result == b"docx-result"


def test_convert_docx_in_memory_passthrough():
    payload = b"docx-bytes"
    assert _convert_docx_in_memory(payload) == payload


def test_convert_doc_in_memory_preserves_original_temp_name(monkeypatch, tmp_path: Path):
    seen_source_names = []

    def fake_convert_doc_to_docx(source_path, output_dir, libreoffice_bin):
        seen_source_names.append(Path(source_path).name)
        out = Path(output_dir) / f"{Path(source_path).stem}.docx"
        out.write_bytes(b"docx-result")
        return out

    monkeypatch.setattr(
        "src.conversion.converter.convert_doc_to_docx",
        fake_convert_doc_to_docx,
    )

    result = _convert_doc_in_memory(b"doc-bytes", "818 24.12.2025 (1).doc")

    assert result == b"docx-result"
    assert seen_source_names == ["818 24.12.2025 (1).doc"]
