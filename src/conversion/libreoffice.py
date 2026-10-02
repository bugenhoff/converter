"""Конвертация .doc в .docx через LibreOffice."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

from ..config.settings import settings
from .errors import ConversionError

logger = logging.getLogger(__name__)

DOCX_EXPORT_TARGETS = (
    "docx",
    "docx:Office Open XML Text",
    "docx:MS Word 2007 XML",
)


_TIMEOUT_SECONDS = 180
# Два soffice с одним профилем пользователя мешают друг другу: второй
# молча отдаёт файл первому и выходит без результата.
_lock = threading.Lock()


def doc_to_docx(payload: bytes, original_name: str) -> bytes:
    """.doc в байтах → .docx в байтах."""
    with tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix="doc_") as tmp:
        source = Path(tmp) / "in" / _safe_doc_name(original_name)
        source.parent.mkdir()
        source.write_bytes(payload)
        converted = convert_doc_to_docx(source, Path(tmp) / "out", settings.libreoffice_path)
        return converted.read_bytes()


def _safe_doc_name(original_name: str) -> str:
    # LibreOffice называет результат по имени исходника, поэтому имя сохраняется.
    name = Path(original_name.replace("/", "_").replace("\\", "_").strip()).name or "document.doc"
    if Path(name).suffix.lower() != ".doc":
        name = f"{Path(name).stem or 'document'}.doc"
    return name


def convert_doc_to_docx(
    source_path: Path,
    output_dir: Path,
    libreoffice_bin: str = "libreoffice",
) -> Path:
    """Run LibreOffice headless converter and return the converted path."""

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file {source_path} was not found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    existing_outputs = {path.resolve() for path in _iter_docx_outputs(output_dir)}

    libreoffice_command = _resolve_libreoffice_command(libreoffice_bin)
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    for export_target in DOCX_EXPORT_TARGETS:
        command = libreoffice_command + [
            "--headless",
            "--convert-to",
            export_target,
            "--outdir",
            str(output_dir),
            str(source_path),
        ]

        try:
            with _lock:
                process = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired as exc:
            raise ConversionError(f"LibreOffice не уложился в {_TIMEOUT_SECONDS} с") from exc
        attempts.append((export_target, process))

        converted_path = _find_libreoffice_output(
            source_path=source_path,
            output_dir=output_dir,
            stdout=process.stdout,
            existing_outputs=existing_outputs,
            started_at=started_at,
        )
        if converted_path is not None:
            break
    else:
        converted_path = None

    if converted_path is None:
        available = ", ".join(path.name for path in _iter_docx_outputs(output_dir)) or "none"
        raise ConversionError(
            "LibreOffice did not produce a DOCX file:"
            f" attempts={_format_libreoffice_attempts(attempts)},"
            f" output_dir={output_dir}, docx_files={available}"
        )
    if converted_path.stat().st_size == 0:
        raise ConversionError(f"LibreOffice produced an empty DOCX file: {converted_path}")

    return converted_path


def _format_libreoffice_attempts(
    attempts: list[tuple[str, subprocess.CompletedProcess[str]]]
) -> str:
    return "; ".join(
        (
            f"target={target!r}, exit={process.returncode}, "
            f"stdout={process.stdout.strip()!r}, stderr={process.stderr.strip()!r}"
        )
        for target, process in attempts
    )


def _iter_docx_outputs(output_dir: Path) -> list[Path]:
    """Return DOCX-like files, handling case-sensitive filesystems."""
    if not output_dir.exists():
        return []
    return [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".docx"
    ]


def _find_libreoffice_output(
    *,
    source_path: Path,
    output_dir: Path,
    stdout: str,
    existing_outputs: set[Path],
    started_at: float,
) -> Path | None:
    expected_path = output_dir / f"{source_path.stem}.docx"
    if expected_path.exists():
        return expected_path

    for parsed_path in _parse_libreoffice_output_paths(stdout, output_dir):
        if parsed_path.exists() and parsed_path.suffix.lower() == ".docx":
            return parsed_path

    docx_outputs = _iter_docx_outputs(output_dir)
    new_outputs = [
        path
        for path in docx_outputs
        if path.resolve() not in existing_outputs or path.stat().st_mtime >= started_at
    ]
    if not new_outputs:
        return None

    for path in new_outputs:
        if path.stem.casefold() == source_path.stem.casefold():
            return path

    if len(new_outputs) == 1:
        return new_outputs[0]

    return max(new_outputs, key=lambda path: path.stat().st_mtime)


def _parse_libreoffice_output_paths(stdout: str, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for line in stdout.splitlines():
        if "->" not in line:
            continue
        raw_path = line.split("->", 1)[1].strip()
        if " using" in raw_path:
            raw_path = raw_path.split(" using", 1)[0].strip()
        raw_path = raw_path.strip("'\"")
        if not raw_path:
            continue
        parsed_path = Path(raw_path)
        if not parsed_path.is_absolute():
            parsed_path = output_dir / parsed_path
        paths.append(parsed_path)
    return paths


def _resolve_libreoffice_command(libreoffice_hint: str | None) -> list[str]:
    """Return an executable command list for LibreOffice, handling Flatpak installs."""

    if libreoffice_hint:
        command = _normalize_command(libreoffice_hint)
        resolved = _ensure_executable(command)
        if resolved:
            return resolved
        logger.warning(
            "LibreOffice command '%s' is not available; falling back to auto-detection",
            libreoffice_hint,
        )

    for candidate in ("libreoffice", "soffice"):
        command = _normalize_command(candidate)
        resolved = _ensure_executable(command)
        if resolved:
            return resolved

    flatpak_command = _detect_flatpak_libreoffice()
    if flatpak_command:
        logger.info("Using Flatpak-installed LibreOffice")
        return list(flatpak_command)

    raise ConversionError(
        "LibreOffice executable was not found. Install LibreOffice or set LIBREOFFICE_PATH."
    )


def _normalize_command(command: str) -> list[str]:
    try:
        parts = shlex.split(command)
    except ValueError:
        return []
    return parts


def _ensure_executable(command: list[str]) -> list[str] | None:
    if not command:
        return None

    executable = command[0]
    exec_path = Path(executable)
    if exec_path.is_file() and os.access(exec_path, os.X_OK):
        return [str(exec_path), *command[1:]]

    resolved = shutil.which(executable)
    if resolved:
        return [resolved, *command[1:]]

    return None


@lru_cache(maxsize=1)
def _detect_flatpak_libreoffice() -> tuple[str, ...] | None:
    flatpak = shutil.which("flatpak")
    if not flatpak:
        return None

    try:
        probe = subprocess.run(
            [flatpak, "info", "org.libreoffice.LibreOffice"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None

    if probe.returncode != 0:
        return None

    return (
        flatpak,
        "run",
        "--command=soffice",
        "org.libreoffice.LibreOffice",
    )
