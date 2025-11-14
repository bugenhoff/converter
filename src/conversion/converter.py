"""Utilities to convert legacy DOC files to modern DOCX via LibreOffice."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ConversionError(RuntimeError):
    """Signals that LibreOffice failed to convert the document."""


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

    command = [
        libreoffice_bin,
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(output_dir),
        str(source_path),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:
        raise ConversionError(
            "LibreOffice failed to convert document:"
            f" exit={process.returncode}, stderr={process.stderr.strip()}"
        )

    converted_path = output_dir / f"{source_path.stem}.docx"
    if not converted_path.exists():
        raise ConversionError("LibreOffice reported success but output file is missing")

    return converted_path