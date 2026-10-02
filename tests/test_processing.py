import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from src.bot import processing
from src.bot.processing import BatchProcessor, IncomingFile
from src.conversion.errors import ConversionError
from src.conversion.pipeline import ConversionResult


class FakeBot:
    def __init__(self):
        self.sent_messages = []
        self.edits = []
        self.documents = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent_messages.append(text)
        return SimpleNamespace(message_id=100 + len(self.sent_messages))

    async def edit_message_text(self, text, **kwargs):
        self.edits.append(text)

    async def send_document(self, **kwargs):
        self.documents.append(kwargs)


@pytest.fixture(autouse=True)
def fast_settings(monkeypatch):
    monkeypatch.setattr(processing.settings, "batch_window_seconds", 0.05)
    monkeypatch.setattr(processing.settings, "max_parallel_files", 3)


def _file(name, message_id=1, kind="pdf"):
    return IncomingFile(name=name, kind=kind, payload=b"data", message_id=message_id)


def test_files_sent_together_share_one_progress_message_and_run_in_parallel():
    active = 0
    peak = 0
    lock = threading.Lock()

    def convert(payload, name, kind, on_status):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        on_status("стр. 1/1")
        time.sleep(0.2)
        with lock:
            active -= 1
        return ConversionResult(f"docx:{name}".encode(), pages=1)

    async def scenario():
        bot = FakeBot()
        processor = BatchProcessor(convert)
        await processor.add(bot, 1, _file("a.pdf", 10))
        await processor.add(bot, 1, _file("b.pdf", 11))
        await processor.add(bot, 1, _file("c.pdf", 12))
        await processor.wait_idle()
        return bot

    bot = asyncio.run(scenario())
    assert len(bot.sent_messages) == 1
    assert peak == 3
    sent = {doc["document"].filename: doc for doc in bot.documents}
    assert set(sent) == {"a.docx", "b.docx", "c.docx"}
    assert sent["b.docx"]["reply_parameters"].message_id == 11
    button = sent["a.docx"]["reply_markup"].inline_keyboard[0][0]
    assert button.callback_data == processing.TRANSLIT_CALLBACK
    assert "✅ <b>Готово 3/3</b>" in bot.edits[-1]


def test_failed_file_is_reported_and_others_still_sent():
    def convert(payload, name, kind, on_status):
        if name == "bad.pdf":
            raise ConversionError("GROQ_API_KEY не задан")
        return ConversionResult(b"docx", pages=2, warnings=["не распознаны стр. 2"])

    async def scenario():
        bot = FakeBot()
        processor = BatchProcessor(convert)
        await processor.add(bot, 1, _file("good.pdf"))
        await processor.add(bot, 1, _file("bad.pdf"))
        await processor.wait_idle()
        return bot

    bot = asyncio.run(scenario())
    assert [d["document"].filename for d in bot.documents] == ["good.docx"]
    assert "не распознаны стр. 2" in bot.documents[0]["caption"]
    final = bot.edits[-1]
    assert "⚠️ <b>Готово 1/2</b>" in final
    assert "❌ bad.pdf — GROQ_API_KEY не задан" in final


def test_full_batch_starts_without_waiting(monkeypatch):
    monkeypatch.setattr(processing.settings, "batch_window_seconds", 30)

    async def scenario():
        bot = FakeBot()
        processor = BatchProcessor(lambda *a: ConversionResult(b"docx"))
        for index in range(processing.MAX_FILES_PER_BATCH):
            await processor.add(bot, 1, _file(f"{index}.docx", kind="docx"))
        await asyncio.wait_for(processor.wait_idle(), timeout=5)
        return bot

    bot = asyncio.run(scenario())
    assert len(bot.documents) == processing.MAX_FILES_PER_BATCH


def test_progress_text_escapes_html():
    progress = processing._ProgressMessage(FakeBot(), 1, ["<a&b>.pdf"])
    assert "&lt;a&amp;b&gt;.pdf" in progress._render(finished=False)
