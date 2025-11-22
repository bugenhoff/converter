"""Advanced file queue management with timing windows and batch processing."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from telegram import InputFile, Update
from telegram.ext import ContextTypes

from ..config.settings import settings
from ..conversion.converter import ConversionError, convert_doc_to_docx, convert_pdf_to_docx
from ..conversion.memory_processor import (
    MemoryHandle,
    convert_file_in_memory,
    memory_processor,
)

logger = logging.getLogger(__name__)

# Queue configuration
PROCESSING_WINDOW_SECONDS = 10
MAX_FILES_PER_BATCH = 10
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@dataclass
class QueuedFile:
    """Represents a file in the processing queue."""

    memory_handle: MemoryHandle
    original_name: str
    file_type: str  # 'doc' or 'pdf'
    user_id: int
    message_id: int
    file_size: int
    timestamp: float = field(default_factory=time.time)
    processed: bool = False


@dataclass
class UserQueue:
    """User-specific file queue with processing state."""
    user_id: int
    files: List[QueuedFile] = field(default_factory=list)
    is_processing: bool = False
    waiting_timer_task: Optional[asyncio.Task] = None
    last_file_time: float = field(default_factory=time.time)
    chat_id: Optional[int] = None  # Store chat_id for sending messages


class FileQueueManager:
    """Manages file queues for all users with automatic processing."""
    
    def __init__(self):
        self.user_queues: Dict[int, UserQueue] = {}
    
    async def add_file(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        memory_handle: MemoryHandle,
        original_name: str,
        file_type: str,
        file_size: int,
    ) -> None:
        """Add a file to user's queue and manage processing timer."""

        user_id = update.effective_user.id
        message_id = update.message.message_id
        chat_id = update.effective_chat.id

        if user_id not in self.user_queues:
            self.user_queues[user_id] = UserQueue(user_id=user_id, chat_id=chat_id)
            logger.debug("Created queue for user %d", user_id)
        else:
            self.user_queues[user_id].chat_id = chat_id

        user_queue = self.user_queues[user_id]

        queued_file = QueuedFile(
            memory_handle=memory_handle,
            original_name=original_name,
            file_type=file_type,
            user_id=user_id,
            message_id=message_id,
            file_size=file_size,
        )

        if len(user_queue.files) >= MAX_FILES_PER_BATCH and not user_queue.is_processing:
            logger.info(
                "User %d exceeded batch size (%d). Triggering immediate processing",
                user_id,
                MAX_FILES_PER_BATCH,
            )
            await self._process_user_queue(context, user_id)
            user_queue.files.append(queued_file)
            await self._start_processing_timer(context, user_id)
        else:
            user_queue.files.append(queued_file)
            user_queue.last_file_time = time.time()

            if user_queue.waiting_timer_task:
                logger.debug("Restarting timer for user %d", user_id)
                user_queue.waiting_timer_task.cancel()

            if not user_queue.is_processing:
                await self._start_processing_timer(context, user_id)

        await self._send_queue_status(update, user_id)
    
    async def _start_processing_timer(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
    ) -> None:
        """Start 10-second waiting timer for batch collection."""

        user_queue = self.user_queues[user_id]

        async def timer_callback() -> None:
            started_at = time.time()
            try:
                await asyncio.sleep(PROCESSING_WINDOW_SECONDS)
                if not user_queue.is_processing and user_queue.files:
                    logger.info(
                        "Batch window elapsed for user %d (waited %.2fs)",
                        user_id,
                        time.time() - started_at,
                    )
                    try:
                        await self._process_user_queue(context, user_id)
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception("Timer callback failed for user %d", user_id)
            except asyncio.CancelledError:
                logger.debug(
                    "Timer cancelled for user %d after %.2fs",
                    user_id,
                    time.time() - started_at,
                )

        user_queue.waiting_timer_task = asyncio.create_task(timer_callback())
        logger.info(
            "Started %ds timer for user %d (queue=%d)",
            PROCESSING_WINDOW_SECONDS,
            user_id,
            len(user_queue.files),
        )
    
    async def _process_user_queue(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
    ) -> None:
        """Process all files in user's queue."""

        user_queue = self.user_queues.get(user_id)
        if not user_queue or not user_queue.files or not user_queue.chat_id:
            logger.debug("Skip processing for user %d: queue empty or chat missing", user_id)
            return

        user_queue.is_processing = True
        files_to_process = user_queue.files.copy()
        user_queue.files.clear()

        if user_queue.waiting_timer_task:
            # Avoid self-cancellation if running inside the timer task
            if user_queue.waiting_timer_task != asyncio.current_task():
                user_queue.waiting_timer_task.cancel()
            user_queue.waiting_timer_task = None

        logger.info("Processing %d file(s) for user %d", len(files_to_process), user_id)

        try:
            await self._notify_processing_start(context, user_id, len(files_to_process))

            results: List[Tuple[bytes, str]] = []
            logger.info("Starting processing loop for %d files", len(files_to_process))
            for idx, queued_file in enumerate(files_to_process, start=1):
                logger.info("Processing file %d/%d: %s", idx, len(files_to_process), queued_file.original_name)
                try:
                    await self._notify_file_progress(
                        context,
                        user_id,
                        idx,
                        len(files_to_process),
                        queued_file.original_name,
                    )
                    logger.info("Progress notification sent for %s", queued_file.original_name)

                    converted_bytes = await self._convert_file(queued_file)
                    logger.info("Conversion finished for %s, result: %s bytes", 
                                queued_file.original_name, 
                                len(converted_bytes) if converted_bytes else "None")
                    
                    if converted_bytes:
                        results.append((converted_bytes, queued_file.original_name))
                        queued_file.processed = True
                        logger.info(
                            "Converted %s (%d bytes) for user %d",
                            queued_file.original_name,
                            len(converted_bytes),
                            user_id,
                        )
                    else:
                        logger.error(
                            "Conversion returned no data for %s", queued_file.original_name
                        )
                        await self._notify_conversion_error(
                            context, user_id, queued_file.original_name
                        )

                except Exception as exc:
                    logger.exception(
                        "Unexpected error processing %s for user %d",
                        queued_file.original_name,
                        user_id,
                        exc_info=exc,
                    )
                    await self._notify_conversion_error(
                        context, user_id, queued_file.original_name
                    )
                finally:
                    queued_file.memory_handle.release()

            if results:
                await self._send_converted_files(context, user_id, results)
            else:
                logger.warning("No results to send for user %d", user_id)
                await context.bot.send_message(
                    chat_id=user_queue.chat_id,
                    text=(
                        "❌ Не удалось конвертировать ни одного файла. "
                        "Проверьте формат и попробуйте снова."
                    ),
                )

        finally:
            user_queue.is_processing = False
            for queued_file in files_to_process:
                queued_file.memory_handle.release()

            memory_processor.log_stats()

            if user_queue.files:
                await self._start_processing_timer(context, user_id)
    
    async def _convert_file(self, queued_file: QueuedFile) -> Optional[bytes]:
        """Convert a single file to DOCX preferring in-memory pipelines."""

        try:
            logger.info(
                "Starting conversion for %s (%s, %d bytes)",
                queued_file.original_name,
                queued_file.file_type,
                queued_file.file_size,
            )
            docx_bytes = convert_file_in_memory(
                queued_file.memory_handle.get_bytes_io(),
                queued_file.file_type,
                queued_file.original_name,
            )
            if docx_bytes:
                return docx_bytes

            logger.info(
                "Memory conversion unavailable for %s, using disk fallback",
                queued_file.original_name,
            )
            memory_processor.stats.fallbacks_to_disk += 1

            if queued_file.file_type == "doc":
                return await asyncio.to_thread(
                    self._convert_doc_from_bytes,
                    queued_file.memory_handle.get_bytes(),
                )

            return await asyncio.to_thread(
                self._convert_pdf_from_bytes,
                queued_file.memory_handle.get_bytes(),
            )

        except ConversionError as exc:
            logger.error(
                "Conversion error for %s: %s",
                queued_file.original_name,
                exc,
            )
            return None
        except Exception as exc:
            logger.exception(
                "Unexpected error converting %s", queued_file.original_name, exc_info=exc
            )
            return None

    @staticmethod
    def _convert_doc_from_bytes(payload: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(
            dir=settings.temp_dir, suffix=".doc", delete=False
        ) as tmp_input:
            tmp_input.write(payload)
            tmp_input_path = Path(tmp_input.name)

        try:
            with tempfile.TemporaryDirectory(dir=settings.temp_dir) as tmp_output:
                output_path = convert_doc_to_docx(
                    tmp_input_path, Path(tmp_output), settings.libreoffice_path
                )
                return output_path.read_bytes()
        finally:
            tmp_input_path.unlink(missing_ok=True)

    @staticmethod
    def _convert_pdf_from_bytes(payload: bytes) -> bytes:
        with tempfile.NamedTemporaryFile(
            dir=settings.temp_dir, suffix=".pdf", delete=False
        ) as tmp_input:
            tmp_input.write(payload)
            tmp_input_path = Path(tmp_input.name)

        try:
            with tempfile.TemporaryDirectory(dir=settings.temp_dir) as tmp_output:
                output_path = convert_pdf_to_docx(tmp_input_path, Path(tmp_output))
                return output_path.read_bytes()
        finally:
            tmp_input_path.unlink(missing_ok=True)
    
    async def _send_queue_status(
        self, 
        update: Update, 
        user_id: int
    ) -> None:
        """Send current queue status to user."""
        user_queue = self.user_queues.get(user_id)
        if not user_queue:
            return
        
        files_count = len(user_queue.files)
        if files_count == 0:
            return
        
        if user_queue.is_processing:
            message = (
                f"⚙️ Обрабатываю файлы… Новый файл добавлен в очередь "
                f"({files_count} шт.)"
            )
        else:
            message = (
                f"📁 Файл добавлен в очередь ({files_count}/{MAX_FILES_PER_BATCH}). "
                f"Жду ещё {PROCESSING_WINDOW_SECONDS} с для одного батча."
            )
        
        try:
            await update.message.reply_text(message)
        except Exception:
            logger.exception("Failed to send queue status to user %d", user_id)
    
    async def _notify_processing_start(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        user_id: int, 
        files_count: int
    ) -> None:
        """Notify user that processing started."""
        user_queue = self.user_queues.get(user_id)
        if not user_queue or not user_queue.chat_id:
            return
            
        try:
            message = f"🚀 Начинаю обработку {files_count} файл(ов)..."
            await context.bot.send_message(chat_id=user_queue.chat_id, text=message)
        except Exception:
            logger.exception("Failed to notify processing start for user %d", user_id)
    
    async def _notify_file_progress(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        user_id: int, 
        current: int, 
        total: int, 
        filename: str
    ) -> None:
        """Notify user about current file processing."""
        user_queue = self.user_queues.get(user_id)
        if not user_queue or not user_queue.chat_id:
            return
            
        message = f"⚙️ Обрабатываю {current}/{total}: {filename}"
        try:
            await context.bot.send_message(chat_id=user_queue.chat_id, text=message)
        except Exception:
            # Don't fail processing if notification fails
            logger.debug("Failed to send progress notification")
    
    async def _notify_conversion_error(
        self, 
        context: ContextTypes.DEFAULT_TYPE, 
        user_id: int, 
        filename: str
    ) -> None:
        """Notify user about conversion error."""
        user_queue = self.user_queues.get(user_id)
        if not user_queue or not user_queue.chat_id:
            return
            
        message = f"❌ Ошибка конвертации: {filename}"
        try:
            await context.bot.send_message(chat_id=user_queue.chat_id, text=message)
        except Exception:
            logger.debug("Failed to send error notification")
    
    async def _send_converted_files(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        user_id: int,
        results: List[Tuple[bytes, str]],
    ) -> None:
        """Send converted DOCX files to user one by one."""

        user_queue = self.user_queues.get(user_id)
        if not user_queue or not user_queue.chat_id:
            return

        success_count = 0
        for docx_bytes, original_name in results:
            docx_name = Path(original_name).with_suffix('.docx').name
            try:
                await context.bot.send_document(
                    chat_id=user_queue.chat_id,
                    document=InputFile(docx_bytes, filename=docx_name),
                    caption=f"✅ {original_name} → {docx_name}",
                )
                success_count += 1
            except Exception as exc:
                logger.error("Failed to send file %s: %s", original_name, exc)
                await self._notify_conversion_error(context, user_id, original_name)

        if success_count > 0:
            try:
                await context.bot.send_message(
                    chat_id=user_queue.chat_id,
                    text=(
                        f"✅ Обработка завершена! Успешно конвертировано: "
                        f"{success_count}/{len(results)} файл(ов)"
                    ),
                )
            except Exception:
                logger.exception("Failed to send completion summary to user %d", user_id)


# Global queue manager instance
queue_manager = FileQueueManager()