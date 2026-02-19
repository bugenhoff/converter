from __future__ import annotations

import io

import pytest

from src.conversion.transliteration import (
    transliterate_docx_bytes,
    transliterate_uz_latin_to_cyr,
)


def test_transliterate_uz_latin_to_cyr_core_rules():
    source = "O‘g‘il Shahar chiroyli tong yaqin YO'Q"
    expected = "Ўғил Шаҳар чиройли тонг яқин ЙЎҚ"
    assert transliterate_uz_latin_to_cyr(source) == expected


def test_transliterate_uz_latin_to_cyr_keeps_mixed_content():
    source = "2026-yil, buyruq #388: o'zgarish yo'q."
    expected = "2026-йил, буйруқ #388: ўзгариш йўқ."
    assert transliterate_uz_latin_to_cyr(source) == expected


def test_transliterate_docx_bytes_preserves_structure():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Shahar bo'limi")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "o‘quvchi"
    table.rows[0].cells[1].text = "ng test"

    source = io.BytesIO()
    document.save(source)

    converted = transliterate_docx_bytes(source.getvalue())
    converted_doc = docx.Document(io.BytesIO(converted))

    assert len(converted_doc.paragraphs) == 1
    assert len(converted_doc.tables) == 1
    assert converted_doc.paragraphs[0].text == "Шаҳар бўлими"
    assert converted_doc.tables[0].rows[0].cells[0].text == "ўқувчи"
    assert converted_doc.tables[0].rows[0].cells[1].text == "нг тест"
