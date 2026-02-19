from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.conversion import groq_converter
from src.conversion.groq_converter import GroqConversionError


def test_validate_merged_pages_rejects_missing_pages():
    content = {"pages": [{"page_number": 1}, {"page_number": 3}]}
    with pytest.raises(GroqConversionError, match="missing pages"):
        groq_converter._validate_merged_pages(content, expected_pages=3)


def test_process_pil_images_with_groq_fails_on_partial_batches(monkeypatch):
    class DummyGroqClient:
        pass

    monkeypatch.setattr(
        groq_converter,
        "groq",
        SimpleNamespace(Groq=lambda api_key: DummyGroqClient()),
    )

    calls = {"count": 0}

    monkeypatch.setattr(groq_converter.settings, "groq_batch_size", 3)

    def fake_process_batch(client, pil_images, start_page, batch_idx):
        calls["count"] += 1
        if batch_idx == 1:
            raise RuntimeError("boom")
        return {
            "title": "ok",
            "pages": [{"page_number": start_page, "sections": []}],
        }

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    with pytest.raises(GroqConversionError, match="refusing partial output"):
        groq_converter._process_pil_images_with_groq([object(), object(), object(), object()])

    assert calls["count"] == 2


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

    def fake_process_batch(client, pil_images, start_page, batch_idx):
        start_pages.append(start_page)
        return {
            "title": "ok",
            "pages": [
                {
                    "page_number": page_number,
                    "sections": [],
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
