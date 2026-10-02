"""Приём файлов пачками, параллельная конвертация и сообщение о прогрессе."""

from __future__ import annotations

import asyncio
import html
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, ReplyParameters
from telegram.error import BadRequest

from ..config.settings import settings
from ..conversion.errors import ConversionError
from ..conversion.pipeline import ConversionResult, convert_document

logger = logging.getLogger(__name__)

MAX_FILES_PER_BATCH = 10
PROGRESS_REFRESH_SECONDS = 2.5
TRANSLIT_CALLBACK = "translit"

Converter = Callable[..., ConversionResult]


@dataclass
class IncomingFile:
    name: str
    kind: str
    payload: bytes
    message_id: int


@dataclass
class _FileState:
    name: str
    icon: str = "🕓"
    note: str = "в очереди"


@dataclass
class _PendingBatch:
    files: list[IncomingFile] = field(default_factory=list)
    timer: asyncio.Task | None = None


class BatchProcessor:
    """Собирает файлы чата в пачку и конвертирует их параллельно.

    Файлы, присланные подряд (альбомом или выделением нескольких), приходят
    отдельными сообщениями; короткое окно ожидания объединяет их в одно
    сообщение о прогрессе.
    """

    def __init__(self, convert: Converter = convert_document) -> None:
        self._convert = convert
        self._pending: dict[int, _PendingBatch] = {}
        self._running: set[asyncio.Task] = set()
        self._slots: asyncio.Semaphore | None = None

    async def add(self, bot: Bot, chat_id: int, file: IncomingFile) -> None:
        batch = self._pending.setdefault(chat_id, _PendingBatch())
        batch.files.append(file)
        if batch.timer:
            batch.timer.cancel()
        if len(batch.files) >= MAX_FILES_PER_BATCH:
            self._start(bot, chat_id)
        else:
            batch.timer = asyncio.create_task(self._start_later(bot, chat_id))

    async def wait_idle(self) -> None:
        """Дождаться всех пачек, в том числе ещё собираемых (для тестов)."""
        while self._pending or self._running:
            tasks = [*self._running, *(b.timer for b in self._pending.values() if b.timer)]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _start_later(self, bot: Bot, chat_id: int) -> None:
        await asyncio.sleep(settings.batch_window_seconds)
        self._start(bot, chat_id)

    def _start(self, bot: Bot, chat_id: int) -> None:
        batch = self._pending.pop(chat_id, None)
        if not batch or not batch.files:
            return
        task = asyncio.create_task(self._process(bot, chat_id, batch.files))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _process(self, bot: Bot, chat_id: int, files: list[IncomingFile]) -> None:
        if self._slots is None:
            self._slots = asyncio.Semaphore(settings.max_parallel_files)
        progress = _ProgressMessage(bot, chat_id, [f.name for f in files])
        await progress.start()
        try:
            await asyncio.gather(*(self._process_file(bot, chat_id, i, f, progress) for i, f in enumerate(files)))
        finally:
            await progress.finish()

    async def _process_file(
        self,
        bot: Bot,
        chat_id: int,
        index: int,
        file: IncomingFile,
        progress: _ProgressMessage,
    ) -> None:
        loop = asyncio.get_running_loop()

        def on_status(note: str) -> None:
            # Вызывается из потока конвертации.
            loop.call_soon_threadsafe(progress.update, index, "⚙️", note)

        async with self._slots:
            progress.update(index, "⚙️", "конвертирую")
            started = time.monotonic()
            try:
                result = await asyncio.to_thread(self._convert, file.payload, file.name, file.kind, on_status)
            except ConversionError as exc:
                logger.error("Conversion of %s failed: %s", file.name, exc)
                progress.update(index, "❌", _short_error(exc))
                return
            except Exception:
                logger.exception("Unexpected error converting %s", file.name)
                progress.update(index, "❌", "внутренняя ошибка")
                return
            elapsed = time.monotonic() - started

        docx_name = f"{Path(file.name).stem}.docx"
        details = [f"{result.pages} стр." if result.pages else "", _format_duration(elapsed)]
        caption = f"✅ {docx_name} · " + " · ".join(d for d in details if d)
        if result.warnings:
            caption += "\n⚠️ " + "; ".join(result.warnings)
        try:
            await bot.send_document(
                chat_id=chat_id,
                document=InputFile(result.docx, filename=docx_name),
                caption=caption,
                reply_parameters=ReplyParameters(file.message_id, allow_sending_without_reply=True),
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Транслитерация", callback_data=TRANSLIT_CALLBACK)]]
                ),
            )
        except Exception:
            logger.exception("Failed to send %s", docx_name)
            progress.update(index, "❌", "не удалось отправить файл")
            return
        progress.update(index, "⚠️" if result.warnings else "✅", "; ".join(result.warnings) or "готово")


class _ProgressMessage:
    """Одно сообщение на пачку; правится не чаще раза в PROGRESS_REFRESH_SECONDS."""

    def __init__(self, bot: Bot, chat_id: int, names: list[str]) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._files = [_FileState(name) for name in names]
        self._message_id: int | None = None
        self._shown = ""
        self._started = time.monotonic()
        self._refresher: asyncio.Task | None = None

    def update(self, index: int, icon: str, note: str) -> None:
        self._files[index].icon = icon
        self._files[index].note = note

    async def start(self) -> None:
        text = self._render(finished=False)
        try:
            message = await self._bot.send_message(self._chat_id, text, parse_mode="HTML")
        except Exception:
            logger.exception("Failed to send progress message")
            return
        self._message_id = message.message_id
        self._shown = text
        self._refresher = asyncio.create_task(self._refresh_loop())

    async def finish(self) -> None:
        if self._refresher:
            self._refresher.cancel()
        await self._edit(self._render(finished=True))

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(PROGRESS_REFRESH_SECONDS)
            await self._edit(self._render(finished=False))

    async def _edit(self, text: str) -> None:
        if self._message_id is None or text == self._shown:
            return
        try:
            await self._bot.edit_message_text(
                text, chat_id=self._chat_id, message_id=self._message_id, parse_mode="HTML"
            )
            self._shown = text
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                logger.debug("Progress edit failed: %s", exc)
        except Exception as exc:
            logger.debug("Progress edit failed: %s", exc)

    def _render(self, finished: bool) -> str:
        elapsed = _format_duration(time.monotonic() - self._started)
        total = len(self._files)
        done = sum(1 for f in self._files if f.icon in {"✅", "⚠️"})
        if not finished:
            header = f"⏳ <b>Обработка {done}/{total}</b> · {elapsed}"
        elif done == 0:
            header = f"❌ <b>Не удалось сконвертировать</b> · {elapsed}"
        else:
            icon = "✅" if done == total else "⚠️"
            header = f"{icon} <b>Готово {done}/{total}</b> · {elapsed}"
        lines = [header, ""]
        for state in self._files:
            name = html.escape(_shorten(state.name))
            lines.append(f"{state.icon} {name} — {html.escape(state.note)}")
        return "\n".join(lines)


def _shorten(name: str, limit: int = 40) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _short_error(exc: Exception, limit: int = 120) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds} с"


batch_processor = BatchProcessor()
