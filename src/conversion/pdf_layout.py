"""PDF с текстовым слоем → DOCX с сохранением вёрстки (pdf2docx + исправления).

Подходит для «цифровых» PDF — выгруженных из Word, браузера, дизайнерских
программ: шрифты, цвета, таблицы, колонки и картинки переносятся как есть.
Сканы сюда не годятся — для них распознавание моделью (``llm.py``).
"""

from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..config.settings import settings
from .docx_repair import DARK_BACKGROUND, LIGHT_TEXT, repair_docx
from .errors import ConversionError

logger = logging.getLogger(__name__)

# Страница — скан, если картинки закрывают больше половины листа, а видимого
# текста почти нет (подпись или штамп ЭЦП поверх скана не в счёт).
SCAN_IMAGE_COVERAGE = 0.5
SCAN_MAX_TEXT_CHARS = 200
MIN_TEXT_CHARS = 20

# Градиентные заливки (логотипы из SVG) pdf2docx не видит. Области с ними
# отрисовываются картинкой, если они не крупнее доли листа и без текста.
_SHADING_MAX_PAGE_SHARE = 0.25
_SHADING_MERGE_GAP_PT = 6.0
_SHADING_DPI = 288
_SAMPLE_ZOOM = 2


@dataclass
class PdfProfile:
    pages: int
    scan_pages: int
    text_chars: int

    @property
    def is_digital(self) -> bool:
        """Документ сделан на компьютере, а не отсканирован."""
        return self.text_chars >= MIN_TEXT_CHARS and self.scan_pages * 2 < self.pages


def profile_pdf(pdf_bytes: bytes) -> PdfProfile:
    pymupdf = _pymupdf()
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ConversionError(f"Не удалось открыть PDF: {exc}") from exc
    with document:
        scan_pages = text_chars = 0
        for page in document:
            chars = _visible_chars(page)
            text_chars += chars
            if chars < SCAN_MAX_TEXT_CHARS and _image_coverage(page) >= SCAN_IMAGE_COVERAGE:
                scan_pages += 1
        return PdfProfile(pages=len(document), scan_pages=scan_pages, text_chars=text_chars)


def pdf_page_count(pdf_bytes: bytes) -> int | None:
    try:
        with _pymupdf().open(stream=pdf_bytes, filetype="pdf") as document:
            return len(document)
    except Exception:
        return None


def pdf_to_docx_layout(pdf_bytes: bytes) -> bytes:
    """Цифровой PDF → DOCX с вёрсткой оригинала."""
    pymupdf = _pymupdf()
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ConversionError(f"Не удалось открыть PDF: {exc}") from exc
    with document:
        backgrounds = light_text_backgrounds(document)
        if rasterize_shadings(document):
            pdf_bytes = document.tobytes(garbage=1, deflate=True)
    return repair_docx(run_pdf2docx(pdf_bytes), backgrounds)


def run_pdf2docx(pdf_bytes: bytes) -> bytes:
    """pdf2docx как есть, без исправлений."""
    try:
        # Импорт здесь: pdf2docx при импорте настраивает корневой логгер.
        from pdf2docx import Converter
    except ImportError as exc:
        raise ConversionError("Нужен пакет pdf2docx") from exc
    _patch_line_joins()

    with tempfile.TemporaryDirectory(dir=settings.temp_dir, prefix="pdf2docx_") as tmp:
        source = Path(tmp) / "source.pdf"
        target = Path(tmp) / "result.docx"
        source.write_bytes(pdf_bytes)
        converter = None
        try:
            converter = Converter(str(source))
            converter.convert(str(target), start=0, end=None)
        except Exception as exc:
            raise ConversionError(f"pdf2docx не справился: {exc}") from exc
        finally:
            if converter is not None:
                converter.close()
        if not target.exists() or target.stat().st_size == 0:
            raise ConversionError("pdf2docx не создал DOCX")
        return target.read_bytes()


def _patch_line_joins() -> None:
    """Пробелы, которых в PDF нет символом, а есть только отступом.

    pdf2docx пишет строки абзаца подряд и рассчитывает, что Word перенесёт
    их там же, где PDF; фрагменты одной строки тоже идут вплотную. Если в
    PDF слова разделены сдвигом, а не пробелом (так делают браузеры и многие
    генераторы), они слипаются при правке и копировании. Пробел в конце
    строки на перенос не влияет.
    """
    from pdf2docx.text.Line import Line

    if getattr(Line, "_joins_patched", False):
        return
    original = Line.make_docx

    def make_docx(self, paragraph):
        spans = getattr(self, "spans", None)
        if spans is None or not hasattr(self, "tab_stop") or not hasattr(self, "line_break"):
            return original(self, paragraph)
        for _ in range(self.tab_stop):
            paragraph.add_run().add_tab()
        previous = None
        for span in spans:
            if previous is not None and _gap_without_space(previous, span):
                _add_space(paragraph)
            span.make_docx(paragraph)
            previous = span
        text = "".join(_span_text(span) for span in spans)
        if self.line_break:
            paragraph.add_run("\n")
        elif text and not text[-1].isspace() and text[-1] not in "-­":
            _add_space(paragraph)

    Line.make_docx = make_docx
    Line._joins_patched = True


def _span_text(span) -> str:
    # У картинок pdf2docx текст — заглушка «<image>».
    return "" if hasattr(span, "image") else getattr(span, "text", "")


def _gap_without_space(left, right) -> bool:
    left_text, right_text = _span_text(left), _span_text(right)
    if not left_text or not right_text or left_text[-1].isspace() or right_text[0].isspace():
        return False
    try:
        gap = right.bbox.x0 - left.bbox.x1
        size = max(left.size, right.size)
    except AttributeError:
        return False
    return gap > 0.15 * size


def _add_space(paragraph) -> None:
    # Оформление как у предыдущего фрагмента, чтобы пробел не менял высоту строки.
    runs = paragraph.runs
    props = runs[-1]._r.rPr if runs else None
    space = paragraph.add_run(" ")
    if props is not None:
        space._r.insert(0, copy.deepcopy(props))


def light_text_backgrounds(document) -> dict[str, str]:
    """Светлый текст и цвет фона под ним по отрисовке страницы: {текст: hex}."""
    found: dict[str, str] = {}
    for page in document:
        spans = [
            span
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip() and _int_luminance(span["color"]) >= LIGHT_TEXT
        ]
        if not spans:
            continue
        pymupdf = _pymupdf()
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(_SAMPLE_ZOOM, _SAMPLE_ZOOM), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        for span in spans:
            fill = _background_under(image, span["bbox"])
            if fill:
                found.setdefault(span["text"].strip(), fill)
    return found


def rasterize_shadings(document) -> int:
    """Градиентные области без текста → картинка поверх; возвращает их число.

    MuPDF отдаёт градиент в ``get_image_info`` как картинку без xref (так же
    выглядят и встроенные inline-картинки — их отрисовка картинкой ничего не
    портит), а pdf2docx берёт только картинки с xref.
    """
    pymupdf = _pymupdf()
    added = 0
    for page in document:
        infos = page.get_image_info(xrefs=True)
        graphics = [pymupdf.Rect(info["bbox"]) & page.rect for info in infos if not info.get("xref")]
        if not graphics:
            continue
        page_area = abs(page.rect)
        busy = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks") if b[4].strip()]
        busy += [pymupdf.Rect(info["bbox"]) for info in infos if info.get("xref")]

        def free(rect) -> bool:
            return not rect.is_empty and not any(rect.intersects(box) for box in busy)

        for region in _merge_rects([rect for rect in graphics if free(rect)], _SHADING_MERGE_GAP_PT):
            if abs(region) > page_area * _SHADING_MAX_PAGE_SHARE or not free(region):
                continue
            pixmap = page.get_pixmap(clip=region, dpi=_SHADING_DPI, alpha=False)
            page.insert_image(region, pixmap=pixmap, overlay=True)
            added += 1
    if added:
        logger.info("Rasterized %d gradient regions", added)
    return added


def _merge_rects(rects: list, gap: float) -> list:
    """Склеивает прямоугольники, между которыми меньше ``gap``: буквы логотипа — в один."""
    regions = [+rect for rect in rects]
    while True:
        merged: list = []
        for rect in regions:
            grown = rect + (-gap, -gap, gap, gap)
            for index, existing in enumerate(merged):
                if grown.intersects(existing):
                    merged[index] = existing | rect
                    break
            else:
                merged.append(rect)
        if len(merged) == len(regions):
            return merged
        regions = merged


def _visible_chars(page) -> int:
    # Тип 3 — невидимый текст (слой OCR поверх скана).
    return sum(
        len(span["chars"])
        for span in page.get_texttrace()
        if span.get("type") != 3 and span.get("opacity", 1) > 0
    )


def _image_coverage(page) -> float:
    # Только настоящие картинки (с xref): градиентная подложка — не скан.
    page_area = abs(page.rect) or 1.0
    rect = _pymupdf().Rect
    covered = sum(
        abs(rect(info["bbox"]) & page.rect)
        for info in page.get_image_info(xrefs=True)
        if info.get("xref")
    )
    return min(covered / page_area, 1.0)


def _background_under(image: Image.Image, bbox) -> str | None:
    x0, y0, x1, y1 = (round(v * _SAMPLE_ZOOM) for v in bbox)
    pad = 2 * _SAMPLE_ZOOM
    box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, image.width), min(y1 + pad, image.height))
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    area = image.crop(box)
    colors = [
        (count, color)
        for count, color in area.getcolors(area.width * area.height) or []
        if _rgb_luminance(color) < LIGHT_TEXT
    ]
    if not colors:
        return None
    _, color = max(colors)
    if _rgb_luminance(color) >= DARK_BACKGROUND:
        return None
    return "{:02X}{:02X}{:02X}".format(*color)


def _int_luminance(value: int) -> float:
    return _rgb_luminance(((value >> 16) & 255, (value >> 8) & 255, value & 255))


def _rgb_luminance(color) -> float:
    red, green, blue = color[:3]
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255


def _pymupdf():
    try:
        import pymupdf
    except ImportError as exc:
        raise ConversionError("Нужен пакет pymupdf") from exc
    return pymupdf
