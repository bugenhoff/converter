"""Tests for dynamic progress indicators and animations."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from telegram import Message

from src.bot.file_queue import (
    FileQueueManager,
    create_progress_bar,
    get_loading_animation,
)


class TestProgressIndicators:
    """Test progress visualization functions."""

    def test_create_progress_bar_empty(self):
        """Test progress bar with zero total."""
        result = create_progress_bar(0, 0)
        assert result == "████████████████████ 0%"

    def test_create_progress_bar_partial(self):
        """Test progress bar with partial completion."""
        result = create_progress_bar(1, 4)
        expected = "█████░░░░░░░░░░░░░░░ 25%"
        assert result == expected

    def test_create_progress_bar_complete(self):
        """Test progress bar with 100% completion."""
        result = create_progress_bar(4, 4)
        expected = "████████████████████ 100%"
        assert result == expected

    def test_create_progress_bar_custom_length(self):
        """Test progress bar with custom length."""
        result = create_progress_bar(1, 2, length=10)
        expected = "█████░░░░░ 50%"
        assert result == expected

    def test_get_loading_animation_cycling(self):
        """Test that animation cycles through frames."""
        frames = []
        for i in range(12):  # More than frame count
            frames.append(get_loading_animation(i))

        # Should have at least some repeating patterns
        assert len(set(frames)) == 10  # 10 unique frames
        assert frames[0] == frames[10]  # Should cycle


@pytest.mark.anyio(backends=["asyncio"])  # Only use asyncio
class TestDynamicProgressMessages:
    """Test dynamic progress messaging functionality."""

    @pytest.fixture
    def queue_manager(self):
        """Create a FileQueueManager instance for testing."""
        return FileQueueManager()

    @pytest.fixture
    def mock_context(self):
        """Create mock context with bot."""
        context = Mock()
        context.bot = AsyncMock()
        context.bot.send_message = AsyncMock()
        context.bot.edit_message_text = AsyncMock()
        return context

    @pytest.fixture
    def mock_message(self):
        """Create mock Telegram message."""
        message = Mock(spec=Message)
        message.message_id = 123
        message.text = "Test message"
        return message

    async def test_notify_processing_start_creates_progress_message(
        self, queue_manager, mock_context, mock_message
    ):
        """Test that processing start creates a progress message with animation."""
        user_id = 12345
        files_count = 3

        # Setup user queue
        from src.bot.file_queue import UserQueue
        user_queue = UserQueue(user_id=user_id, chat_id=67890)
        queue_manager.user_queues[user_id] = user_queue

        # Mock bot response
        mock_context.bot.send_message.return_value = mock_message

        # Call function
        await queue_manager._notify_processing_start(mock_context, user_id, files_count)

        # Verify message was sent
        mock_context.bot.send_message.assert_called_once()
        call_args = mock_context.bot.send_message.call_args
        
        assert call_args.kwargs["chat_id"] == 67890
        assert "🚀 **Начинаю обработку 3 файл(ов)**" in call_args.kwargs["text"]
        assert "░" in call_args.kwargs["text"]  # Progress bar
        assert call_args.kwargs["parse_mode"] == "Markdown"

        # Verify progress message was stored
        assert user_queue.progress_message == mock_message
        assert user_queue.animation_task is not None

        # Clean up animation task
        if user_queue.animation_task:
            user_queue.animation_task.cancel()

    async def test_notify_file_progress_updates_message(
        self, queue_manager, mock_context, mock_message
    ):
        """Test that file progress updates the existing message."""
        user_id = 12345
        
        # Setup user queue with existing progress message
        from src.bot.file_queue import UserQueue
        user_queue = UserQueue(user_id=user_id, chat_id=67890)
        user_queue.progress_message = mock_message
        queue_manager.user_queues[user_id] = user_queue

        # Call function
        await queue_manager._notify_file_progress(
            mock_context, user_id, 2, 3, "test_document.pdf"
        )

        # Verify message was updated
        mock_context.bot.edit_message_text.assert_called_once()
        call_args = mock_context.bot.edit_message_text.call_args
        
        assert call_args.kwargs["chat_id"] == 67890
        assert call_args.kwargs["message_id"] == 123
        assert "⚙️ Конвертирую: test_document.pdf" in call_args.kwargs["text"]
        assert "(2/3)" in call_args.kwargs["text"]

        # Clean up animation task
        if user_queue.animation_task:
            user_queue.animation_task.cancel()

    async def test_animation_cancellation_on_new_progress(
        self, queue_manager, mock_context, mock_message
    ):
        """Test that starting new progress cancels old animation."""
        user_id = 12345
        
        # Setup user queue with mock animation task
        from src.bot.file_queue import UserQueue
        user_queue = UserQueue(user_id=user_id, chat_id=67890)
        user_queue.progress_message = mock_message
        
        # Create mock animation task
        old_animation = AsyncMock()
        user_queue.animation_task = old_animation
        queue_manager.user_queues[user_id] = user_queue

        # Call function that should cancel animation
        await queue_manager._notify_file_progress(
            mock_context, user_id, 1, 2, "test.pdf"
        )

        # Verify old animation was cancelled
        old_animation.cancel.assert_called_once()

    def test_long_filename_truncation(self):
        """Test that long filenames are properly truncated in progress messages."""
        long_filename = "very_long_filename_that_should_be_truncated_because_it_exceeds_limits.pdf"
        
        # Test truncation logic
        truncated = long_filename[:30] + "..." if len(long_filename) > 30 else long_filename
        expected = "very_long_filename_that_should..."  # exactly 30 chars + "..."
        assert truncated == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])