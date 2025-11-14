"""Helpers for packaging converted documents into a single archive."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Iterable


def build_zip_archive(documents: Iterable[tuple[Path, str]], output_dir: Path) -> Path:
    """Create a ZIP archive with given documents and return its path."""

    docs = [(Path(path), arcname) for path, arcname in documents if Path(path).exists()]
    if not docs:
        raise ValueError("Cannot build archive without documents")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"converted_{int(time.time())}.zip"

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path, arcname in docs:
            zip_file.write(path, arcname=arcname)

    return archive_path