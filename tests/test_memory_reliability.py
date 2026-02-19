from __future__ import annotations

from pathlib import Path

from src.conversion.memory_processor import _convert_docx_in_memory, _convert_pdf_in_memory


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
