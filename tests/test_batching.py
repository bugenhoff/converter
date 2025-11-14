from pathlib import Path
import zipfile

import pytest

from src.bot.batching import build_zip_archive
from src.bot.queue import QUEUE_KEY, enqueue_file, flush_queue


def test_enqueue_and_flush_queue(tmp_path: Path):
    chat_data: dict[str, list[str]] = {}
    file_path = tmp_path / "file.docx"
    file_path.write_text("data")

    size = enqueue_file(chat_data, file_path, "file.docx")
    assert size == 1
    assert enqueue_file(chat_data, file_path, "second.docx") == 2

    queued = flush_queue(chat_data)
    assert len(queued) == 2
    assert queued[0][1] == "file.docx"
    assert QUEUE_KEY not in chat_data


def test_build_zip_archive(tmp_path: Path):
    doc1 = tmp_path / "a.docx"
    doc2 = tmp_path / "b.docx"
    doc1.write_text("one")
    doc2.write_text("two")

    archive = build_zip_archive([(doc1, "doc-one.docx"), (doc2, "doc-two.docx")], tmp_path)
    assert archive.exists()

    import zipfile

    with zipfile.ZipFile(archive) as zip_file:
        names = set(zip_file.namelist())
        assert names == {"doc-one.docx", "doc-two.docx"}


def test_build_zip_archive_requires_documents(tmp_path: Path):
    with pytest.raises(ValueError):
        build_zip_archive([], tmp_path)