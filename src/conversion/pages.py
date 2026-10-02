"""Страницы документа в виде картинок для распознавания и их геометрия."""

from __future__ import annotations

import io
import logging
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

from ..config.settings import settings
from .errors import ConversionError

logger = logging.getLogger(__name__)

A4_PT = (595.28, 841.89)
_PDFINFO_PAGE_SIZE = re.compile(r"^Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", re.MULTILINE)
_JPEG_QUALITY = 85

# Пиксель считается «чернилами», если все каналы темнее порога: так в счёт
# не попадают цветные водяные знаки и синие печати.
_INK_MAX_CHANNEL = 110
# Колонка или строка пикселей — часть текста, если чернил в ней больше 0,4 %;
# больше 60 % — это рамка или тень от сканера.
_MIN_INK_DENSITY = 0.004
_MAX_INK_DENSITY = 0.6
_TEXT_LINE_DENSITY = 0.02
# У Times New Roman высота строчных букв около 0,45 кегля.
_X_HEIGHT_RATIO = 0.45


class PageRenderError(ConversionError):
    """Не удалось превратить документ в картинки страниц."""


@dataclass
class PageGeometry:
    """Размер страницы и то, что удалось измерить по скану, в пунктах."""

    width_pt: float
    height_pt: float
    margins_pt: tuple[float, float, float, float] | None = None  # слева, справа, сверху, снизу
    x_height_pt: float | None = None
    line_pitch_pt: float | None = None


@dataclass
class PageImage:
    number: int
    jpeg: bytes
    geometry: PageGeometry


class PdfPages:
    """PDF во временной папке: число страниц и отрисовка по одной."""

    def __init__(self, pdf_bytes: bytes) -> None:
        self._tmp = tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix="pdf_")
        self._path = Path(self._tmp.name) / "source.pdf"
        self._path.write_bytes(pdf_bytes)
        try:
            self.sizes = _pdf_page_sizes(self._path)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> PdfPages:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.sizes)

    def close(self) -> None:
        self._tmp.cleanup()

    def render(self, number: int, max_side: int) -> PageImage:
        prefix = Path(self._tmp.name) / f"page{number}"
        result = subprocess.run(
            [
                "pdftoppm", "-f", str(number), "-l", str(number),
                "-scale-to", str(max_side),
                "-jpeg", "-jpegopt", f"quality={_JPEG_QUALITY}",
                "-singlefile", str(self._path), str(prefix),
            ],
            capture_output=True,
            text=True,
        )
        jpeg_path = prefix.with_suffix(".jpg")
        if result.returncode != 0 or not jpeg_path.exists():
            raise PageRenderError(f"pdftoppm failed on page {number}: {result.stderr.strip()}")
        jpeg = jpeg_path.read_bytes()
        jpeg_path.unlink()

        width_pt, height_pt = _paper_size(*self.sizes[number - 1])
        with Image.open(io.BytesIO(jpeg)) as image:
            # pdftoppm учитывает поворот страницы, а pdfinfo отдаёт размер без него.
            if (image.width > image.height) != (width_pt > height_pt):
                width_pt, height_pt = height_pt, width_pt
            geometry = measure_page(image, width_pt, height_pt)
        return PageImage(number=number, jpeg=jpeg, geometry=geometry)


def image_to_page(image_bytes: bytes, max_side: int) -> PageImage:
    """Фото или скан страницы: уменьшает до max_side и перекодирует в JPEG."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise PageRenderError(f"Не удалось открыть изображение: {exc}") from exc

    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)

    landscape = image.width > image.height
    width_pt, height_pt = (A4_PT[1], A4_PT[0]) if landscape else A4_PT
    # Поля и кегль меряем, только если картинка похожа на скан целого листа A4,
    # а не на фото с краем стола.
    aspect = max(image.size) / min(image.size)
    if abs(aspect - A4_PT[1] / A4_PT[0]) < 0.05:
        geometry = measure_page(image, width_pt, height_pt)
    else:
        geometry = PageGeometry(width_pt, height_pt)
    return PageImage(number=1, jpeg=buffer.getvalue(), geometry=geometry)


def pdf_has_text_layer(pdf_bytes: bytes) -> bool:
    with tempfile.NamedTemporaryFile(dir=settings.temp_dir, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        result = subprocess.run(["pdftotext", "-q", tmp.name, "-"], capture_output=True, text=True)
    return len("".join(result.stdout.split())) >= 20


def measure_page(image: Image.Image, width_pt: float, height_pt: float) -> PageGeometry:
    """Поля, высота строчных букв и шаг строк по проекциям «чернил» на оси."""
    geometry = PageGeometry(width_pt, height_pt)
    rgb = image.convert("RGB")
    red, green, blue = rgb.split()
    brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    ink = brightest.point(lambda value: 255 if value < _INK_MAX_CHANNEL else 0)
    width, height = ink.size

    columns = _profile(ink, axis="x")
    rows = _profile(ink, axis="y")
    xs = [i for i, d in enumerate(columns) if _MIN_INK_DENSITY < d < _MAX_INK_DENSITY]
    ys = [i for i, d in enumerate(rows) if _MIN_INK_DENSITY < d < _MAX_INK_DENSITY]
    if not xs or not ys:
        return geometry

    pt_per_px_x = width_pt / width
    pt_per_px_y = height_pt / height
    geometry.margins_pt = (
        xs[0] * pt_per_px_x,
        (width - 1 - xs[-1]) * pt_per_px_x,
        ys[0] * pt_per_px_y,
        (height - 1 - ys[-1]) * pt_per_px_y,
    )

    body = _profile(ink.crop((xs[0], 0, xs[-1] + 1, height)), axis="y")
    lines = _text_lines(body)
    if len(lines) >= 3:
        x_heights = []
        for start, end in lines:
            peak = max(body[start:end])
            x_heights.append(sum(1 for d in body[start:end] if d >= peak / 2))
        pitches = [lines[i + 1][0] - lines[i][0] for i in range(len(lines) - 1)]
        geometry.x_height_pt = statistics.median(x_heights) * pt_per_px_y
        geometry.line_pitch_pt = statistics.median(pitches) * pt_per_px_y
    return geometry


def estimate_font_size(x_height_pt: float) -> float:
    return x_height_pt / _X_HEIGHT_RATIO


def _profile(mask: Image.Image, axis: str) -> list[float]:
    """Доля «чернил» в каждой колонке (x) или строке (y) пикселей."""
    width, height = mask.size
    size = (width, 1) if axis == "x" else (1, height)
    squeezed = mask.resize(size, Image.Resampling.BOX)
    return [value / 255 for value in squeezed.tobytes()]


def _text_lines(profile: list[float]) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    start = None
    for y, density in enumerate(profile + [0.0]):
        if density > _TEXT_LINE_DENSITY and start is None:
            start = y
        elif density <= _TEXT_LINE_DENSITY and start is not None:
            if y - start >= 3:
                lines.append((start, y))
            start = None
    return lines


def _pdf_page_sizes(path: Path) -> list[tuple[float, float]]:
    result = subprocess.run(
        ["pdfinfo", "-f", "1", "-l", "100000", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PageRenderError(f"pdfinfo failed: {result.stderr.strip() or result.stdout.strip()}")
    sizes = {int(n): (float(w), float(h)) for n, w, h in _PDFINFO_PAGE_SIZE.findall(result.stdout)}
    if not sizes:
        raise PageRenderError("В PDF нет страниц")
    return [sizes[n] for n in sorted(sizes)]


def _paper_size(width_pt: float, height_pt: float) -> tuple[float, float]:
    """Размер листа для DOCX; невозможные для бумаги размеры заменяются на A4."""
    short, long = sorted((width_pt, height_pt))
    if 250 <= short <= 1300 and long <= 1800:
        return width_pt, height_pt
    return A4_PT if height_pt >= width_pt else (A4_PT[1], A4_PT[0])
