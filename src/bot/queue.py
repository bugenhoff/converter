"""Utilities for chat-specific conversion queues."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

QUEUE_KEY = "doc_queue"


def enqueue_file(
    chat_data: MutableMapping[str, Any], converted_path: Path, arcname: str
) -> int:
    """Push converted file metadata into chat queue and return queue size."""

    queue = chat_data.setdefault(QUEUE_KEY, [])
    queue.append({"path": str(Path(converted_path)), "arcname": arcname})
    return len(queue)


def flush_queue(chat_data: MutableMapping[str, Any]) -> list[tuple[Path, str]]:
    """Return queued files metadata and reset queue."""

    stored = chat_data.pop(QUEUE_KEY, [])
    documents: list[tuple[Path, str]] = []
    for item in stored:
        if not item:
            continue
        raw_path = item.get("path")
        if not raw_path:
            continue
        path = Path(raw_path)
        arcname = item.get("arcname") or path.name
        documents.append((path, arcname))
    return documents