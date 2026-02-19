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

    def fake_process_batch(client, pil_images, batch_idx):
        calls["count"] += 1
        if batch_idx == 1:
            raise RuntimeError("boom")
        return {
            "title": "ok",
            "pages": [{"page_number": batch_idx * 3 + 1, "sections": []}],
        }

    monkeypatch.setattr(groq_converter, "_process_pil_batch_with_groq", fake_process_batch)

    with pytest.raises(GroqConversionError, match="refusing partial output"):
        groq_converter._process_pil_images_with_groq([object(), object(), object(), object()])

    assert calls["count"] == 2
