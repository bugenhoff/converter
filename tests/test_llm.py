import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.conversion import llm
from src.conversion.layout import Paragraph
from src.conversion.pages import PageGeometry, PageImage

REAL_SLEEP = time.sleep


def _page(number: int) -> PageImage:
    return PageImage(number=number, jpeg=b"jpeg", geometry=PageGeometry(595.28, 841.89))


def _response(content: str, finish_reason: str = "stop"):
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, completion_tokens_details=None)
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake")


class FakeClient:
    def __init__(self, replies=None, delay=0.0):
        self.replies = list(replies or [])
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            reply = self.replies.pop(0) if self.replies else _response('{"blocks": [{"text": "page"}]}')
        REAL_SLEEP(self.delay)
        with self._lock:
            self.active -= 1
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def client(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: fake)
    monkeypatch.setattr(llm.settings, "groq_api_key", "key")
    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    monkeypatch.setattr(llm.settings, "llm_retries", 2)
    # Без пауз между повторами, только в модуле llm.
    monkeypatch.setattr(llm, "time", SimpleNamespace(sleep=lambda _: None, monotonic=time.monotonic))
    return fake


def test_pages_are_requested_in_parallel(monkeypatch, client):
    client.delay = 0.3
    pool = ThreadPoolExecutor(max_workers=6)
    monkeypatch.setattr(llm, "_executor", lambda: pool)
    progress = []

    started = time.monotonic()
    results = llm.transcribe_pages((_page(n) for n in range(1, 7)), 6, lambda done, total: progress.append((done, total)))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0  # последовательно было бы 1,8 с
    assert client.max_active == 6
    assert [r.page.number for r in results] == [1, 2, 3, 4, 5, 6]
    assert all(r.blocks == [Paragraph("page")] for r in results)
    assert sorted(progress) == [(n, 6) for n in range(1, 7)]


def test_invalid_json_is_retried(client):
    client.replies = [_response("not json"), _response('{"blocks": [{"text": "ok"}]}')]
    assert llm.transcribe_page(_page(1)) == [Paragraph("ok")]
    assert len(client.calls) == 2


def test_truncated_answer_retried_with_bigger_budget(monkeypatch, client):
    monkeypatch.setattr(llm.settings, "llm_max_tokens_per_page", 4000)
    client.replies = [_response('{"blocks": [', "length"), _response('{"blocks": [{"text": "ok"}]}')]
    assert llm.transcribe_page(_page(1)) == [Paragraph("ok")]
    assert [call["max_tokens"] for call in client.calls] == [4000, 8000]


def test_blank_page_is_accepted_after_retries(client):
    client.replies = [_response('{"blocks": []}')] * 3
    assert llm.transcribe_page(_page(1)) == []
    assert len(client.calls) == 3


def test_failed_page_does_not_fail_document(monkeypatch, client):
    monkeypatch.setattr(llm, "_executor", lambda: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(llm.settings, "llm_retries", 0)
    client.replies = [_response("garbage"), _response(json.dumps({"blocks": [{"text": "two"}]}))]
    results = llm.transcribe_pages([_page(1), _page(2)], 2)
    assert results[0].error and results[0].blocks is None
    assert results[1].blocks == [Paragraph("two")]


def test_missing_key_raises(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    monkeypatch.setattr(llm.settings, "groq_api_key", "")
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.transcribe_pages([_page(1)], 1)


def test_openrouter_options(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "openrouter")
    monkeypatch.setattr(llm.settings, "openrouter_reasoning_effort", "none")
    monkeypatch.setattr(llm.settings, "openrouter_provider_sort", "throughput")
    assert llm._request_options() == {
        "extra_body": {"reasoning": {"effort": "none"}, "provider": {"sort": "throughput"}}
    }
    monkeypatch.setattr(llm.settings, "llm_provider", "groq")
    assert llm._request_options() == {}
