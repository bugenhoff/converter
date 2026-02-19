from __future__ import annotations

import asyncio

from src.bot.file_queue import FileQueueManager, UserQueue


class _DummyDocument:
    file_id = "telegram-doc-file-id"


class _DummySentMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id
        self.document = _DummyDocument()


class _DummyBot:
    def __init__(self):
        self.send_calls = []
        self.markup_calls = []

    async def send_document(self, chat_id, document, caption=None, reply_markup=None):
        self.send_calls.append(
            {
                "chat_id": chat_id,
                "caption": caption,
            }
        )
        return _DummySentMessage(message_id=101)

    async def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
        self.markup_calls.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup,
            }
        )


class _DummyContext:
    def __init__(self):
        self.bot = _DummyBot()


def test_send_converted_docx_attaches_transliteration_button():
    manager = FileQueueManager()
    manager.user_queues[42] = UserQueue(user_id=42, chat_id=777)
    context = _DummyContext()

    asyncio.run(
        manager._send_converted_files(
            context=context,
            user_id=42,
            results=[(b"fake-docx-content", "incoming.docx")],
        )
    )

    assert len(context.bot.send_calls) == 1
    assert context.bot.send_calls[0]["caption"] == "✅ incoming.docx → incoming.docx"
    assert len(context.bot.markup_calls) == 1
    callback_data = context.bot.markup_calls[0]["reply_markup"].inline_keyboard[0][0].callback_data
    assert callback_data.startswith("translit:")
