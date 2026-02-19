from __future__ import annotations

from types import SimpleNamespace

import pytest
from docx import Document

from src.conversion import groq_converter
from src.conversion.groq_converter import GroqConversionError


def test_validate_merged_pages_rejects_missing_pages():
    content = {"pages": [{"page_number": 1}, {"page_number": 3}]}
    with pytest.raises(GroqConversionError, match="missing pages"):
        groq_converter._validate_merged_pages(content, expected_pages=3)


def test_normalize_batch_result_extracts_text_from_sections():
    normalized = groq_converter._normalize_batch_result(
        {
            "pages": [
                {
                    "page_number": "1",
                    "sections": [
                        {"content": "Header line"},
                        {
                            "table_data": {
                                "headers": ["Col1", "Col2"],
                                "rows": [["A", "B"]],
                            }
                        },
                    ],
                }
            ]
        }
    )

    page = normalized["pages"][0]
    assert page["page_number"] == 1
    assert "Header line" in page["text"]
    assert "Col1 | Col2" in page["text"]
    assert "A | B" in page["text"]


def test_validate_batch_pages_rejects_suspiciously_short_text():
    with pytest.raises(GroqConversionError, match="suspiciously short"):
        groq_converter._validate_batch_pages(
            {"pages": [{"page_number": 1, "text": "too short"}]},
            expected_pages=[1],
        )


def test_normalize_batch_result_keeps_blocks_and_sanitizes_spaces():
    normalized = groq_converter._normalize_batch_result(
        {
            "pages": [
                {
                    "page_number": 1,
                    "blocks": [
                        {"text": "  HELLO    WORLD  ", "bold": True, "color": "green"},
                        {"text": " ", "bold": False},
                    ],
                }
            ]
        }
    )
    page = normalized["pages"][0]
    assert page["text"] == "HELLO WORLD"
    assert len(page["blocks"]) == 1
    assert page["blocks"][0]["bold"] is True
    assert page["blocks"][0]["color"] == "green"


def test_write_transcription_pages_skips_empty_pages():
    doc = Document()
    groq_converter._write_transcription_pages_to_doc(
        doc,
        {
            "pages": [
                {"page_number": 1, "blocks": []},
                {"page_number": 2, "blocks": [{"text": "Line 1"}, {"text": "Line 2"}]},
            ]
        },
    )
    texts = [p.text for p in doc.paragraphs if p.text.strip()]
    assert texts == ["Line 1", "Line 2"]


def test_process_pil_images_with_groq_fails_on_partial_batches(monkeypatch):
    class DummyGroqClient:
        pass

    monkeypatch.setattr(
        groq_converter,
        "groq",
        SimpleNamespace(Groq=lambda api_key: DummyGroqClient()),
    )

    calls = {"count": 0}

    monkeypatch.setattr(groq_converter.settings, "groq_batch_size", 4)
    monkeypatch.setattr(groq_converter.settings, "groq_min_batch_size", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_image_max_side", 800)
    monkeypatch.setattr(groq_converter.settings, "groq_min_image_max_side", 480)
    monkeypatch.setattr(groq_converter.settings, "groq_image_side_reduction_factor", 0.8)
    monkeypatch.setattr(groq_converter.settings, "groq_retry_per_task", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_max_requests_per_document", 0)

    def fake_process_batch(client, pil_images, start_page, batch_idx, image_max_side):
        calls["count"] += 1
        if len(pil_images) > 2:
            raise RuntimeError("too many pages for one request")
        return {
            "title": "ok",
            "pages": [
                {
                    "page_number": page_number,
                    "text": f"Page {page_number} full transcription " * 4,
                }
                for page_number in range(start_page, start_page + len(pil_images))
            ],
        }

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    merged = groq_converter._process_pil_images_with_groq([object(), object(), object(), object()])
    assert calls["count"] == 3
    assert [page["page_number"] for page in merged["pages"]] == [1, 2, 3, 4]


def test_process_pil_images_with_groq_uses_configurable_batch_size(monkeypatch):
    class DummyGroqClient:
        pass

    monkeypatch.setattr(
        groq_converter,
        "groq",
        SimpleNamespace(Groq=lambda api_key: DummyGroqClient()),
    )
    monkeypatch.setattr(groq_converter.settings, "groq_batch_size", 2)

    start_pages: list[int] = []

    monkeypatch.setattr(groq_converter.settings, "groq_min_batch_size", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_image_max_side", 800)
    monkeypatch.setattr(groq_converter.settings, "groq_min_image_max_side", 480)
    monkeypatch.setattr(groq_converter.settings, "groq_image_side_reduction_factor", 0.8)
    monkeypatch.setattr(groq_converter.settings, "groq_retry_per_task", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_max_requests_per_document", 0)

    def fake_process_batch(client, pil_images, start_page, batch_idx, image_max_side):
        start_pages.append(start_page)
        return {
            "title": "ok",
            "pages": [
                {
                    "page_number": page_number,
                    "text": f"Page {page_number} full transcription " * 4,
                }
                for page_number in range(start_page, start_page + len(pil_images))
            ],
        }

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    merged = groq_converter._process_pil_images_with_groq(
        [object(), object(), object(), object(), object()]
    )

    assert start_pages == [1, 3, 5]
    assert [p["page_number"] for p in merged["pages"]] == [1, 2, 3, 4, 5]


def test_process_pil_images_with_groq_downscales_until_min_side(monkeypatch):
    class DummyGroqClient:
        pass

    monkeypatch.setattr(
        groq_converter,
        "groq",
        SimpleNamespace(Groq=lambda api_key: DummyGroqClient()),
    )

    monkeypatch.setattr(groq_converter.settings, "groq_batch_size", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_min_batch_size", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_image_max_side", 800)
    monkeypatch.setattr(groq_converter.settings, "groq_min_image_max_side", 480)
    monkeypatch.setattr(groq_converter.settings, "groq_image_side_reduction_factor", 0.5)
    monkeypatch.setattr(groq_converter.settings, "groq_retry_per_task", 0)
    monkeypatch.setattr(groq_converter.settings, "groq_max_requests_per_document", 0)

    seen_sides: list[int] = []

    def fake_process_batch(client, pil_images, start_page, batch_idx, image_max_side):
        seen_sides.append(image_max_side)
        if image_max_side > 500:
            raise RuntimeError("context overflow")
        return {
            "title": "ok",
            "pages": [{"page_number": start_page, "text": "Single page transcription " * 4}],
        }

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    merged = groq_converter._process_pil_images_with_groq([object()])
    assert [p["page_number"] for p in merged["pages"]] == [1]
    assert seen_sides == [800, 480]


def test_process_pil_images_with_groq_stops_on_request_limit(monkeypatch):
    class DummyGroqClient:
        pass

    monkeypatch.setattr(
        groq_converter,
        "groq",
        SimpleNamespace(Groq=lambda api_key: DummyGroqClient()),
    )

    monkeypatch.setattr(groq_converter.settings, "groq_batch_size", 4)
    monkeypatch.setattr(groq_converter.settings, "groq_min_batch_size", 1)
    monkeypatch.setattr(groq_converter.settings, "groq_image_max_side", 800)
    monkeypatch.setattr(groq_converter.settings, "groq_min_image_max_side", 480)
    monkeypatch.setattr(groq_converter.settings, "groq_image_side_reduction_factor", 0.8)
    monkeypatch.setattr(groq_converter.settings, "groq_retry_per_task", 0)
    monkeypatch.setattr(groq_converter.settings, "groq_max_requests_per_document", 2)

    def fake_process_batch(client, pil_images, start_page, batch_idx, image_max_side):
        raise RuntimeError("force split")

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    with pytest.raises(GroqConversionError, match="request limit reached"):
        groq_converter._process_pil_images_with_groq([object(), object(), object(), object()])
