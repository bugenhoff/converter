from __future__ import annotations

from src.bot.file_queue import FileQueueManager, TRANSLITERATION_JOB_TTL_SECONDS


def test_transliteration_job_roundtrip():
    manager = FileQueueManager()
    token = manager.create_transliteration_job(
        user_id=1,
        file_id="file_123",
        docx_name="test.docx",
    )

    job, error = manager.consume_transliteration_job(token, user_id=1)
    assert error is None
    assert job is not None
    assert job.file_id == "file_123"

    second_job, second_error = manager.consume_transliteration_job(token, user_id=1)
    assert second_job is None
    assert second_error == "not_found"


def test_transliteration_job_forbidden_user():
    manager = FileQueueManager()
    token = manager.create_transliteration_job(
        user_id=10,
        file_id="file_999",
        docx_name="secret.docx",
    )

    job, error = manager.consume_transliteration_job(token, user_id=20)
    assert job is None
    assert error == "forbidden"


def test_transliteration_job_cleanup_expired(monkeypatch):
    manager = FileQueueManager()
    token = manager.create_transliteration_job(
        user_id=7,
        file_id="file_777",
        docx_name="expired.docx",
    )
    created_at = manager.transliteration_jobs[token].created_at

    monkeypatch.setattr(
        "src.bot.file_queue.time.time",
        lambda: created_at + TRANSLITERATION_JOB_TTL_SECONDS + 1,
    )
    manager._cleanup_expired_transliteration_jobs()

    job, error = manager.consume_transliteration_job(token, user_id=7)
    assert job is None
    assert error == "not_found"
