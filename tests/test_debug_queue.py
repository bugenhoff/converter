#!/usr/bin/env python3
"""Debug script to test queue timer functionality."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.bot.file_queue import FileQueueManager
from src.conversion.memory_processor import memory_processor

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class MockUpdate:
    def __init__(self, user_id=12345, chat_id=12345):
        self.effective_user = MockUser(user_id)
        self.effective_chat = MockChat(chat_id)
        self.message = MockMessage()
    
class MockUser:
    def __init__(self, user_id):
        self.id = user_id

class MockChat:
    def __init__(self, chat_id):
        self.id = chat_id

class MockMessage:
    def __init__(self):
        self.message_id = 123
    
    async def reply_text(self, text):
        logger.info("BOT REPLY: %s", text)

class MockBot:
    async def send_message(self, chat_id, text):
        logger.info("BOT MESSAGE to %s: %s", chat_id, text)
    
    async def send_document(self, chat_id, document, caption=None):
        logger.info("BOT DOCUMENT to %s: %s", chat_id, caption)

class MockContext:
    def __init__(self):
        self.bot = MockBot()

async def test_queue_timer():
    """Test the queue timer mechanism."""
    logger.info("🔍 Testing queue timer functionality...")
    
    queue = FileQueueManager()
    update = MockUpdate()
    context = MockContext()

    async def _enqueue_mock_file(label: str) -> None:
        payload = f"mock payload for {label}".encode()
        handle = memory_processor.store_bytes(payload, f"{label}.pdf")
        try:
            await queue.add_file(
                update=update,
                context=context,
                memory_handle=handle,
                original_name=f"{label}.pdf",
                file_type="pdf",
                file_size=len(payload),
            )
        except Exception:
            handle.release()
            raise

    logger.info("📁 Adding first file to queue...")
    await _enqueue_mock_file("test_file1")

    logger.info("⏳ Waiting 5 seconds...")
    await asyncio.sleep(5)

    logger.info("📁 Adding second file to queue...")
    await _enqueue_mock_file("test_file2")

    logger.info("⏳ Waiting for timer to complete (15 seconds)...")
    await asyncio.sleep(15)

    user_queue = queue.user_queues.get(12345)
    if user_queue:
        logger.info(
            "📊 Queue state: %d files, processing: %s",
            len(user_queue.files),
            user_queue.is_processing,
        )
    else:
        logger.info("📊 No queue found for user")

    logger.info("✅ Timer test completed!")

if __name__ == "__main__":
    asyncio.run(test_queue_timer())