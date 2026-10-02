"""LibreOffice: поиск результата, повтор с явным фильтром, диагностика."""

from pathlib import Path

import pytest

from src.conversion.errors import ConversionError
from src.conversion.libreoffice import convert_doc_to_docx, doc_to_docx


def test_missing_source_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert_doc_to_docx(tmp_path / "missing.doc", tmp_path, "libreoffice")


def test_conversion_failure_is_reported(tmp_path: Path, monkeypatch):
    src = tmp_path / "test.doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 1
        stdout = ""
        stderr = "boom"

    def fake_run(*args, **kwargs):
        return FakeResult()

    monkeypatch.setattr("src.conversion.libreoffice.subprocess.run", fake_run)

    with pytest.raises(ConversionError):
        convert_doc_to_docx(src, tmp_path, "libreoffice")


def test_successful_conversion(tmp_path: Path, monkeypatch):
    src = tmp_path / "test.doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        output_path = Path(args[0][-1]).with_suffix(".docx")
        output_path.write_text("converted")
        return FakeResult()

    monkeypatch.setattr("src.conversion.libreoffice.subprocess.run", fake_run)

    converted = convert_doc_to_docx(src, tmp_path, "libreoffice")
    assert converted.exists()
    assert converted.suffix == ".docx"


def test_successful_conversion_finds_libreoffice_renamed_output(tmp_path: Path, monkeypatch):
    src = tmp_path / "input.doc"
    src.write_text("dummy")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*args, **kwargs):
        output_dir.mkdir(exist_ok=True)
        (output_dir / "unexpected-name.docx").write_text("converted")
        return FakeResult()

    monkeypatch.setattr("src.conversion.libreoffice.subprocess.run", fake_run)

    converted = convert_doc_to_docx(src, output_dir, "libreoffice")
    assert converted == output_dir / "unexpected-name.docx"
    assert converted.read_text() == "converted"


def test_conversion_retries_with_explicit_docx_filter(tmp_path: Path, monkeypatch):
    src = tmp_path / "818 24.12.2025 (1).doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        def __init__(self, stdout="", stderr=""):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = stderr

    used_targets = []

    def fake_run(args, **kwargs):
        used_targets.append(args[args.index("--convert-to") + 1])
        if used_targets[-1] == "docx:Office Open XML Text":
            (tmp_path / "818 24.12.2025 (1).docx").write_text("converted")
            return FakeResult()
        return FakeResult(stderr="Error: no export filter")

    monkeypatch.setattr("src.conversion.libreoffice.subprocess.run", fake_run)

    converted = convert_doc_to_docx(src, tmp_path, "libreoffice")
    assert converted.name == "818 24.12.2025 (1).docx"
    assert used_targets == ["docx", "docx:Office Open XML Text"]


def test_missing_libreoffice_output_reports_stdout_and_stderr(tmp_path: Path, monkeypatch):
    src = tmp_path / "test.doc"
    src.write_text("dummy")

    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )

    class FakeResult:
        returncode = 0
        stdout = "convert /tmp/test.doc -> /tmp/missing.docx using filter"
        stderr = "warn"

    monkeypatch.setattr(
        "src.conversion.libreoffice.subprocess.run",
        lambda *_args, **_kwargs: FakeResult(),
    )

    with pytest.raises(ConversionError, match="attempts=.*stdout=.*stderr="):
        convert_doc_to_docx(src, tmp_path / "out", "libreoffice")


def test_doc_to_docx_keeps_original_name(monkeypatch):
    monkeypatch.setattr(
        "src.conversion.libreoffice._resolve_libreoffice_command",
        lambda *_: ["libreoffice"],
    )
    seen = []

    class FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        source = Path(args[-1])
        seen.append(source.name)
        outdir = Path(args[args.index("--outdir") + 1])
        (outdir / f"{source.stem}.docx").write_bytes(b"docx")
        return FakeResult()

    monkeypatch.setattr("src.conversion.libreoffice.subprocess.run", fake_run)

    assert doc_to_docx(b"doc", "818 24.12.2025 (1).doc") == b"docx"
    assert seen == ["818 24.12.2025 (1).doc"]
