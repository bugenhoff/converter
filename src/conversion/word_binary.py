"""Бинарные .doc и .wps после импорта LibreOffice: то, что он читает неправильно.

WPS Office сохраняет .wps в формате Word 97 (OLE: WordDocument, 0Table, Data).
На таких файлах LibreOffice ошибается в двух местах:

- цвет границ ячеек берёт из старой 16-цветной палитры Word 97, хотя рядом
  в файле записан полный цвет (sprmTCellTopColor и т. п.): фирменный синий
  становится бирюзовым, тёмно-синий — «navy»;
- размер некоторых картинок: логотип получает ширину в несколько метров
  и уходит за край листа.

Полные цвета читаются прямо из потока WordDocument, а документ правится
после того, как LibreOffice сохранил его в DOCX.
"""

from __future__ import annotations

import copy
import io
import logging
import re
import zipfile

from lxml import etree
from PIL import Image

logger = logging.getLogger(__name__)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PARTS = re.compile(r"^word/(document|header\d*|footer\d*)\.xml$")

# Палитра Word 97 (ico 1–16): в неё LibreOffice сводит цвет границ.
PALETTE = (
    "000000", "0000FF", "00FFFF", "00FF00", "FF00FF", "FF0000", "FFFF00", "FFFFFF",
    "000080", "008080", "008000", "800080", "800000", "808000", "808080", "C0C0C0",
)
# sprm с полным цветом границ: (код, где в операнде лежат COLORREF).
_CELL_COLORS = (0xD61A, 0xD61B, 0xD61C, 0xD61D)      # sprmTCell{Top,Left,Bottom,Right}Color
_BRC = (0xC64E, 0xC64F, 0xC650, 0xC651, 0xCA72)      # sprmPBrc{Top,Left,Bottom,Right}, sprmCBrc
_TABLE_BORDERS = 0xD613                              # sprmTTableBorders: 6 BRC по 8 байт
_BORDER_TAGS = {f"{{{W_NS}}}{tag}" for tag in ("top", "left", "bottom", "right", "start", "end", "insideH", "insideV", "bdr")}
_BORDER_PARENTS = {f"{{{W_NS}}}{tag}" for tag in ("pBdr", "tcBorders", "tblBorders", "rPr")}

# Картинка «не помещается на лист», если больше листа хотя бы в 1,5 раза.
_OVERSIZE = 1.5
EMU_PER_TWIP = 635
A4_TWIPS = (11906, 16838)


def is_word_binary(payload: bytes) -> bool:
    return payload[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def full_border_colors(payload: bytes) -> set[str]:
    """Полные цвета границ, записанные в WordDocument (hex без #)."""
    try:
        import olefile
    except ImportError:
        return set()
    try:
        with olefile.OleFileIO(io.BytesIO(payload)) as ole:
            if not ole.exists("WordDocument"):
                return set()
            stream = ole.openstream("WordDocument").read()
    except Exception as exc:
        logger.debug("Not an OLE Word file: %s", exc)
        return set()
    return colors_in_stream(stream)


def colors_in_stream(stream: bytes) -> set[str]:
    """Полные цвета из sprm границ в потоке WordDocument."""
    colors: set[str] = set()
    for sprm in _CELL_COLORS:
        for operand in _operands(stream, sprm):
            if len(operand) % 4 == 0:
                colors.update(_colorrefs(operand, range(0, len(operand), 4)))
    for sprm in _BRC:
        for operand in _operands(stream, sprm):
            if len(operand) == 8:
                colors.update(_colorrefs(operand, (0,)))
    for operand in _operands(stream, _TABLE_BORDERS):
        if len(operand) == 48:
            colors.update(_colorrefs(operand, range(0, 48, 8)))
    return colors


def palette_replacements(colors: set[str]) -> dict[str, str]:
    """Цвет палитры → полный цвет, если к нему сводится ровно один цвет из файла.

    Если цвет палитры в файле встречается и сам по себе (настоящий бирюзовый),
    его не трогаем.
    """
    candidates: dict[str, set[str]] = {}
    for color in colors:
        candidates.setdefault(nearest_palette(color), set()).add(color)
    return {
        palette: next(iter(found))
        for palette, found in candidates.items()
        if len(found) == 1 and palette not in found
    }


def nearest_palette(color: str) -> str:
    red, green, blue = _rgb(color)
    return min(PALETTE, key=lambda p: sum((a - b) ** 2 for a, b in zip(_rgb(p), (red, green, blue))))


def repair_imported_docx(docx: bytes, original: bytes) -> tuple[bytes, list[str]]:
    """DOCX, который LibreOffice сделал из .doc/.wps → исправленный DOCX и список правок."""
    replacements = palette_replacements(full_border_colors(original)) if is_word_binary(original) else {}
    fixes: list[str] = []
    source = zipfile.ZipFile(io.BytesIO(docx))
    page = _page_size(source)
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            content = source.read(item)
            if _PARTS.match(item.filename):
                root = etree.fromstring(content)
                changed = _recolor_borders(root, replacements)
                if changed:
                    fixes.append(f"цвет границ: {changed}")
                resized = _fit_images(root, source, item.filename, page)
                if resized:
                    fixes.append(f"размер картинок: {resized}")
                if changed or resized:
                    content = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
            # Копия: writestr меняет ZipInfo, и чтение из исходного архива сломалось бы.
            target.writestr(copy.copy(item), content)
    return output.getvalue(), fixes


def _operands(stream: bytes, sprm: int):
    """Операнды переменной длины (байт длины + данные) для каждого вхождения sprm."""
    key = sprm.to_bytes(2, "little")
    start = 0
    while (index := stream.find(key, start)) != -1:
        start = index + 1
        size_at = index + 2
        if size_at >= len(stream):
            break
        size = stream[size_at]
        operand = stream[size_at + 1:size_at + 1 + size]
        if size and len(operand) == size:
            yield operand


def _colorrefs(operand: bytes, offsets) -> list[str]:
    found = []
    for offset in offsets:
        if offset + 4 > len(operand):
            break
        red, green, blue, auto = operand[offset:offset + 4]
        if auto == 0:  # 0xFF — «авто», 0 — заданный цвет; остальное — не COLORREF
            found.append(f"{red:02X}{green:02X}{blue:02X}")
    return found


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _recolor_borders(root: etree._Element, replacements: dict[str, str]) -> int:
    if not replacements:
        return 0
    changed = 0
    key = f"{{{W_NS}}}color"
    for element in root.iter():
        if element.tag not in _BORDER_TAGS:
            continue
        parent = element.getparent()
        if parent is None or parent.tag not in _BORDER_PARENTS:
            continue
        color = (element.get(key) or "").upper()
        if color in replacements:
            element.set(key, replacements[color])
            changed += 1
    return changed


def _page_size(package: zipfile.ZipFile) -> tuple[int, int]:
    """Размер листа в EMU по первому разделу документа."""
    try:
        root = etree.fromstring(package.read("word/document.xml"))
        size = root.find(f".//{{{W_NS}}}pgSz")
        width, height = int(size.get(f"{{{W_NS}}}w")), int(size.get(f"{{{W_NS}}}h"))
    except Exception:
        width, height = A4_TWIPS
    return width * EMU_PER_TWIP, height * EMU_PER_TWIP


def _fit_images(root: etree._Element, package: zipfile.ZipFile, part: str, page: tuple[int, int]) -> int:
    """Картинка больше листа и с искажёнными пропорциями → размер по пропорциям файла.

    Верным считается то измерение, при котором картинка помещается на лист.
    """
    sizes = _image_sizes(package, part)
    fixed = 0
    for extent in root.iter(f"{{{WP_NS}}}extent"):
        frame = extent.getparent()
        blip = frame.find(f".//{{{A_NS}}}blip") if frame is not None else None
        pixels = sizes.get(blip.get(f"{{{R_NS}}}embed")) if blip is not None else None
        if not pixels:
            continue
        try:
            cx, cy = int(extent.get("cx")), int(extent.get("cy"))
        except (TypeError, ValueError):
            continue
        if cx <= page[0] * _OVERSIZE and cy <= page[1] * _OVERSIZE:
            continue
        ratio = _visible_ratio(frame, pixels)
        options = [(round(cy * ratio), cy), (cx, round(cx / ratio))]
        fitting = [(w, h) for w, h in options if w <= page[0] and h <= page[1]]
        if not fitting:
            continue
        width, height = max(fitting, key=lambda size: size[0] * size[1])
        for element in (extent, *frame.iter(f"{{{A_NS}}}ext")):
            element.set("cx", str(width))
            element.set("cy", str(height))
        logger.info("Resized broken image %dx%d EMU -> %dx%d", cx, cy, width, height)
        fixed += 1
    return fixed


def _visible_ratio(frame: etree._Element, pixels: tuple[int, int]) -> float:
    width, height = pixels
    crop = frame.find(f".//{{{A_NS}}}srcRect")
    if crop is not None:
        # Обрезка в тысячных долях процента; отрицательная — поля вокруг картинки.
        def part(name: str) -> float:
            try:
                return int(crop.get(name, "0")) / 100000
            except ValueError:
                return 0.0
        width *= max(1 - part("l") - part("r"), 0.01)
        height *= max(1 - part("t") - part("b"), 0.01)
    return width / height


def _image_sizes(package: zipfile.ZipFile, part: str) -> dict[str, tuple[int, int]]:
    folder, name = part.rsplit("/", 1)
    try:
        rels = etree.fromstring(package.read(f"{folder}/_rels/{name}.rels"))
    except (KeyError, etree.XMLSyntaxError):
        return {}
    sizes = {}
    for rel in rels:
        target = rel.get("Target") or ""
        if "://" in target or not rel.get("Type", "").endswith("/image"):
            continue
        path = target.lstrip("/") if target.startswith("/") else f"{folder}/{target}"
        try:
            with Image.open(io.BytesIO(package.read(path))) as image:
                sizes[rel.get("Id")] = image.size
        except Exception:
            continue
    return sizes
