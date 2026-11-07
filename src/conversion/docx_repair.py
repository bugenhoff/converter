"""Исправление DOCX, который собрал pdf2docx.

pdf2docx пишет разметку, которую Word ещё терпит, а LibreOffice, WPS и
Google Docs показывают криво или теряют часть текста:

- ссылка вложена внутрь run (``w:r/w:hyperlink``) — такой текст пропадает;
- имена шрифтов PostScript (``Inter-SemiBold``, ``TimesNewRomanPSMT``) —
  в системе таких семейств нет, и подставляется шрифт по умолчанию;
- сетка таблицы из равных колонок, а реальные ширины только у ячеек;
- блоки «в две колонки» разделены разрывом раздела «со следующей колонки»,
  который LibreOffice и WPS понимают как разрыв страницы;
- вторая и следующие строки пункта списка начинаются с табуляции вместо
  висячего отступа и разъезжаются, если строка перенеслась иначе;
- заливка под светлым текстом иногда теряется — текст становится белым на белом;
- дробные числа, цвета с ``#`` и порядок элементов не по схеме.
"""

from __future__ import annotations

import copy
import io
import re
import statistics
import zipfile
from typing import Callable

from lxml import etree
from PIL import Image

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_PARTS = re.compile(r"^word/(document|header\d*|footer\d*)\.xml$")
_DECIMAL = re.compile(r"^-?\d+\.\d+$")
_FONT_ATTRS = ("ascii", "hAnsi", "eastAsia", "cs")
_BULLET = re.compile(r"^\s*(\d{1,3}[.)]|[a-zа-я][.)]|[•▪◦●■□–—\-·*➢►✓✔])\s*$", re.IGNORECASE)
# Картинка-маркер списка у pdf2docx — квадратик в несколько пунктов (EMU).
_MAX_BULLET_EMU = 12 * 12700

# Светлый текст и тёмный фон — по относительной яркости (0…1).
LIGHT_TEXT = 0.85
DARK_BACKGROUND = 0.6

# Начертания легче обычного и Medium есть в системе отдельными семействами
# («Inter Light»): если шрифта нет, Word подставит обычный — это не страшно.
# SemiBold и тяжелее пишем как семейство + жирный: без установленного шрифта
# выделение всё равно останется.
_WEIGHTS = (
    ("extralight", "ExtraLight"),
    ("ultralight", "ExtraLight"),
    ("semibold", None),
    ("demibold", None),
    ("extrabold", None),
    ("ultrabold", None),
    ("thin", "Thin"),
    ("light", "Light"),
    ("medium", "Medium"),
    ("black", None),
    ("heavy", None),
    ("bold", None),
)
_REGULAR = ("regular", "roman", "book", "normal", "plain")
# Шрифты, у которых PostScript-имя не сводится к имени семейства правилом.
_KNOWN_FAMILIES = {
    # Стандартные шрифты PDF и их метрически совместимые замены.
    "times": "Times New Roman",
    "helvetica": "Arial",
    "courier": "Courier New",
    "timesnewroman": "Times New Roman",
    "arial": "Arial",
    "arialnarrow": "Arial Narrow",
    "couriernew": "Courier New",
    "symbol": "Symbol",
    "gillsans": "Gill Sans MT",
    "twcen": "Tw Cen MT",
    "dejavusans": "DejaVu Sans",
    "dejavuserif": "DejaVu Serif",
    "dejavusansmono": "DejaVu Sans Mono",
}

# Порядок дочерних элементов по схеме ECMA-376: Word строг к нему.
_ORDER = {
    "rPr": (
        "rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike",
        "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden",
        "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
        "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
        "specVanish", "oMath", "rPrChange",
    ),
    "pPr": (
        "pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
        "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
        "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
        "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
        "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle",
        "rPr", "sectPr", "pPrChange",
    ),
    "tcPr": (
        "cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
        "textDirection", "tcFitText", "vAlign", "hideMark", "headers", "cellIns", "cellDel",
        "cellMerge", "tcPrChange",
    ),
}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


ImageColor = Callable[[str], "str | None"]


def repair_docx(data: bytes, backgrounds: dict[str, str] | None = None) -> bytes:
    """DOCX от pdf2docx → тот же документ с исправленной разметкой.

    ``backgrounds`` — текст светлых фрагментов PDF и цвет фона под ними
    (hex без ``#``), чтобы вернуть потерянную заливку.
    """
    source = zipfile.ZipFile(io.BytesIO(data))
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            content = source.read(item)
            if _PARTS.match(item.filename):
                content = repair_part(content, backgrounds or {}, _image_colors(source, item.filename))
            target.writestr(item, content)
    return output.getvalue()


def repair_part(
    xml: bytes,
    backgrounds: dict[str, str] | None = None,
    image_color: ImageColor | None = None,
) -> bytes:
    root = etree.fromstring(xml)
    _fix_numbers(root)
    _unwrap_hyperlinks(root)
    _fix_fonts(root)
    _fix_table_grids(root)
    _merge_column_sections(root)
    _fix_list_tabs(root, image_color or (lambda _: None))
    _restore_backgrounds(root, backgrounds or {})
    _normalize_order(root)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def _image_colors(package: zipfile.ZipFile, part: str) -> ImageColor:
    """Основной цвет картинки по id связи части документа (для маркеров списка)."""
    folder, name = part.rsplit("/", 1)
    try:
        rels = etree.fromstring(package.read(f"{folder}/_rels/{name}.rels"))
    except (KeyError, etree.XMLSyntaxError):
        return lambda _: None
    targets = {rel.get("Id"): rel.get("Target") for rel in rels}

    def color(rel_id: str) -> str | None:
        target = targets.get(rel_id)
        if not target or "://" in target:
            return None
        path = target.lstrip("/") if target.startswith("/") else f"{folder}/{target}"
        try:
            with Image.open(io.BytesIO(package.read(path))) as image:
                pixels = image.convert("RGBA").getcolors(1 << 16) or []
        except Exception:
            return None
        # Самый частый непрозрачный и не белый цвет.
        inked = [(count, rgba) for count, rgba in pixels if rgba[3] > 128 and sum(rgba[:3]) < 720]
        if not inked:
            return None
        _, (red, green, blue, _) = max(inked)
        return f"{red:02X}{green:02X}{blue:02X}"

    return color


def font_family(name: str) -> tuple[str, bool, bool, bool]:
    """PostScript-имя шрифта → (семейство, жирный, курсив, начертание задано явно).

    «Начертание задано явно» — в имени есть вес (Inter-SemiBold), и признак
    жирности у run нужно выставить по нему, а не по флагам из PDF.
    """
    name = re.sub(r"^[A-Z]{6}\+", "", name.strip())
    base, _, style = re.sub(r"[-,]", "\0", name, count=1).partition("\0")
    style = re.sub(r"(PSMT|MT|PS)$", "", style).lower()
    base_key = re.sub(r"(PSMT|MT|PS)$", "", base.replace(" ", ""))

    family = _KNOWN_FAMILIES.get(base_key.lower())
    if family is None:
        family = base.strip() if " " in base.strip() else _split_camel(base_key)

    italic = any(mark in style for mark in ("italic", "oblique")) or style.endswith("it")
    weight = style.replace("italic", "").replace("oblique", "")
    if weight.endswith("it"):
        weight = weight[:-2]
    if not weight or weight in _REGULAR:
        return family, False, italic, bool(style)
    for key, suffix in _WEIGHTS:
        if key in weight:
            if suffix is None:
                return family, True, italic, True
            return f"{family} {suffix}", False, italic, True
    # Неизвестный суффикс (Inter-Display, Condensed) — оставляем частью имени.
    return f"{family} {_split_camel(style.title())}".strip(), False, italic, False


def luminance(hex_color: str | None) -> float | None:
    if not hex_color or not re.fullmatch(r"#?[0-9A-Fa-f]{6}", hex_color):
        return None
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _split_camel(name: str) -> str:
    # TimesNewRoman → Times New Roman, SegoeUI → Segoe UI, PTSans → PT Sans.
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name).strip()


def _fix_numbers(root: etree._Element) -> None:
    color_attrs = {w("color"), w("fill")}
    for element in root.iter():
        for key, value in element.attrib.items():
            if not key.startswith(f"{{{W_NS}}}"):
                continue
            if _DECIMAL.match(value):
                element.set(key, str(round(float(value))))
            elif key in color_attrs and value.startswith("#"):
                element.set(key, value[1:].upper())


def _unwrap_hyperlinks(root: etree._Element) -> None:
    """``w:r > w:hyperlink > w:r`` → ``w:hyperlink > w:r`` с оформлением внешнего run."""
    for hyperlink in list(root.iter(w("hyperlink"))):
        run = hyperlink.getparent()
        if run is None or run.tag != w("r"):
            continue
        outer_props = run.find(w("rPr"))
        for inner in hyperlink.findall(w("r")):
            inner_props = inner.find(w("rPr"))
            if inner_props is not None:
                inner.remove(inner_props)
            if outer_props is not None:
                inner.insert(0, copy.deepcopy(outer_props))
        run.addprevious(hyperlink)
        if all(child.tag == w("rPr") for child in run):
            run.getparent().remove(run)


def _fix_fonts(root: etree._Element) -> None:
    for fonts in root.iter(w("rFonts")):
        original = fonts.get(w("ascii")) or fonts.get(w("hAnsi"))
        if not original:
            continue
        family, bold, italic, explicit = font_family(original)
        for attr in _FONT_ATTRS:
            fonts.set(w(attr), family)
        props = fonts.getparent()
        if props is None or props.tag != w("rPr"):
            continue
        if explicit:
            _set_toggle(props, "b", bold)
        if italic:
            _set_toggle(props, "i", True)


def _set_toggle(props: etree._Element, tag: str, value: bool) -> None:
    element = props.find(w(tag))
    if element is None:
        if not value:
            return
        element = etree.SubElement(props, w(tag))
    if value:
        element.attrib.pop(w("val"), None)
    else:
        element.set(w("val"), "0")


def _fix_table_grids(root: etree._Element) -> None:
    """Ширины колонок сетки по ширинам ячеек: LibreOffice верит только сетке."""
    for table in root.iter(w("tbl")):
        grid = table.find(w("tblGrid"))
        columns = grid.findall(w("gridCol")) if grid is not None else []
        if not columns:
            continue
        edges = _column_edges(table, len(columns))
        if edges is None:
            continue
        for column, (left, right) in zip(columns, zip(edges, edges[1:])):
            column.set(w("w"), str(round(right - left)))


def _column_edges(table: etree._Element, count: int) -> list[float] | None:
    seen: list[list[int]] = [[] for _ in range(count + 1)]
    for row in table.findall(w("tr")):
        x = col = 0
        for cell in row.findall(w("tc")):
            props = cell.find(w("tcPr"))
            width = props.find(w("tcW")) if props is not None else None
            if width is None or width.get(w("type"), "dxa") != "dxa":
                break
            span_element = props.find(w("gridSpan"))
            try:
                span = int(span_element.get(w("val"))) if span_element is not None else 1
                x += int(width.get(w("w")))
            except (TypeError, ValueError):
                break
            col += span
            if col > count:
                break
            seen[col].append(x)
    if any(not positions for positions in seen[1:]):
        return None
    edges = [0.0] + [statistics.median(positions) for positions in seen[1:]]
    if any(right <= left for left, right in zip(edges, edges[1:])):
        return None
    return edges


def _merge_column_sections(root: etree._Element) -> None:
    """Раздел «со следующей колонки» → разрыв колонки внутри одного раздела.

    pdf2docx пишет каждую колонку отдельным разделом. Свойства раздела лежат
    в его последнем абзаце, поэтому раздел предыдущей колонки снимается, а в
    тот же абзац ставится разрыв колонки.
    """
    body = root.find(w("body"))
    if body is None:
        return
    breaks = [
        (paragraph, paragraph.find(f"{w('pPr')}/{w('sectPr')}"))
        for paragraph in body.iterchildren(w("p"))
        if paragraph.find(f"{w('pPr')}/{w('sectPr')}") is not None
    ]
    final = body.find(w("sectPr"))
    if final is not None:
        breaks.append((None, final))

    for (previous_paragraph, previous), (_, current) in zip(breaks, breaks[1:]):
        start = current.find(w("type"))
        if start is None or start.get(w("val")) != "nextColumn":
            continue
        # Объединённый раздел начинается так же, как начиналась первая колонка.
        previous_start = previous.find(w("type"))
        if previous_start is not None:
            start.set(w("val"), previous_start.get(w("val")))
        else:
            current.remove(start)
        previous.getparent().remove(previous)
        _make_column_break(previous_paragraph)


def _make_column_break(paragraph: etree._Element) -> None:
    # Остаток абзаца после разрыва уходит в начало следующей колонки;
    # высота строки 1 pt, чтобы колонка не съезжала вниз.
    props = _child(paragraph, "pPr", first=True)
    if not any(child.tag == w("r") for child in paragraph):
        for old in props.findall(w("spacing")):
            props.remove(old)
        spacing = etree.SubElement(props, w("spacing"))
        spacing.attrib.update({w("line"): "20", w("lineRule"): "exact", w("before"): "0", w("after"): "0"})
    run = etree.SubElement(paragraph, w("r"))
    etree.SubElement(run, w("br")).set(w("type"), "column")


def _fix_list_tabs(root: etree._Element, image_color: ImageColor) -> None:
    """Пункт списка «маркер, таб, строка 1, таб, строка 2…» → висячий отступ.

    Маркер-картинка заменяется символом «•» её цвета: LibreOffice и WPS
    ставят картинку в первой строке с висячим отступом не туда, и табуляция
    уезжает на следующую позицию.
    """
    for paragraph in root.iter(w("p")):
        props = paragraph.find(w("pPr"))
        stops = props.findall(f"{w('tabs')}/{w('tab')}") if props is not None else []
        runs = paragraph.findall(w("r"))
        # Пробел между маркером и табуляцией ничего не значит — убираем.
        while len(runs) > 2 and _is_bullet_run(runs[0]) and _is_blank_run(runs[1]):
            paragraph.remove(runs.pop(1))
        if len(stops) != 1 or len(runs) < 3 or not _is_bullet_run(runs[0]) or not _is_tab_run(runs[1]):
            continue
        _picture_to_bullet(runs[0], runs[2], image_color)
        continuation = [run for run in runs[2:] if _is_tab_run(run)]
        if not continuation:
            continue
        try:
            stop = int(stops[0].get(w("pos")))
        except (TypeError, ValueError):
            continue
        indent = _child(props, "ind")
        try:
            left = int(indent.get(w("left")) or indent.get(w("start")) or 0)
        except ValueError:
            continue
        if stop <= left:
            continue
        for attr in ("start", "firstLine", "hanging"):
            indent.attrib.pop(w(attr), None)
        indent.set(w("left"), str(stop))
        indent.set(w("hanging"), str(stop - left))
        for run in continuation:
            previous = run.getprevious()
            before = "".join(t.text or "" for t in previous.iter(w("t"))) if previous is not None else ""
            if before[-1:].isspace():
                paragraph.remove(run)
                continue
            tab = run.find(w("tab"))
            text = etree.Element(w("t"))
            text.text = " "
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            tab.addprevious(text)
            run.remove(tab)


def _picture_to_bullet(run: etree._Element, text_run: etree._Element, image_color: ImageColor) -> None:
    drawing = run.find(w("drawing"))
    if drawing is None:
        return
    blip = drawing.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}blip")
    rel_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed") if blip is not None else None
    color = (image_color(rel_id) if rel_id else None) or "000000"

    run.remove(drawing)
    old_props = run.find(w("rPr"))
    if old_props is not None:
        run.remove(old_props)
    # Шрифт и кегль — как у текста пункта, цвет — как у картинки.
    template = text_run.find(w("rPr"))
    props = copy.deepcopy(template) if template is not None else etree.Element(w("rPr"))
    for tag in ("b", "i", "u", "shd", "highlight"):
        for element in props.findall(w(tag)):
            props.remove(element)
    _child(props, "color").set(w("val"), color)
    run.insert(0, props)
    etree.SubElement(run, w("t")).text = "•"


def _is_blank_run(run: etree._Element) -> bool:
    content = [child for child in run if child.tag != w("rPr")]
    return bool(content) and all(child.tag == w("t") and not (child.text or "").strip() for child in content)


def _is_tab_run(run: etree._Element) -> bool:
    content = [child for child in run if child.tag != w("rPr")]
    return len(content) == 1 and content[0].tag == w("tab")


def _is_bullet_run(run: etree._Element) -> bool:
    extent = run.find(".//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent")
    if extent is not None:
        try:
            return max(int(extent.get("cx")), int(extent.get("cy"))) <= _MAX_BULLET_EMU
        except (TypeError, ValueError):
            return False
    text = "".join(t.text or "" for t in run.iter(w("t")))
    return bool(_BULLET.match(text))


def _restore_backgrounds(root: etree._Element, backgrounds: dict[str, str]) -> None:
    """Светлый текст без тёмного фона → заливка ячейки или фона текста из оригинала."""
    if not backgrounds:
        return
    for run in root.iter(w("r")):
        color = run.find(f"{w('rPr')}/{w('color')}")
        light = luminance(color.get(w("val"))) if color is not None else None
        if light is None or light < LIGHT_TEXT:
            continue
        text = "".join(t.text or "" for t in run.iter(w("t"))).strip()
        fill = _background_for(text, backgrounds) if text else None
        if fill is None or _has_dark_background(run):
            continue
        # Заливка всей ячейки — только если в ней нет тёмного текста: pdf2docx
        # иногда кладёт шапку и первую строку таблицы в одну ячейку.
        cell = next(run.iterancestors(w("tc")), None)
        if cell is not None and _only_light_text(cell):
            target = _child(cell, "tcPr", first=True)
        else:
            target = run.find(w("rPr"))
        shading = _child(target, "shd")
        shading.attrib.update({w("val"): "clear", w("color"): "auto", w("fill"): fill})


def _background_for(text: str, backgrounds: dict[str, str]) -> str | None:
    if text in backgrounds:
        return backgrounds[text]
    # pdf2docx режет фрагмент PDF на несколько run по оформлению.
    return next((fill for source, fill in backgrounds.items() if len(text) >= 2 and text in source), None)


def _only_light_text(cell: etree._Element) -> bool:
    for run in cell.iter(w("r")):
        if not "".join(t.text or "" for t in run.iter(w("t"))).strip():
            continue
        color = run.find(f"{w('rPr')}/{w('color')}")
        level = luminance(color.get(w("val"))) if color is not None else 0.0
        if level is None or level < LIGHT_TEXT:
            return False
    return True


def _has_dark_background(run: etree._Element) -> bool:
    candidates = [run.find(f"{w('rPr')}/{w('shd')}")]
    for ancestor in run.iterancestors():
        if ancestor.tag == w("p"):
            candidates.append(ancestor.find(f"{w('pPr')}/{w('shd')}"))
        elif ancestor.tag == w("tc"):
            candidates.append(ancestor.find(f"{w('tcPr')}/{w('shd')}"))
            break
    for shading in candidates:
        if shading is None:
            continue
        level = luminance(shading.get(w("fill")))
        if level is not None and level < DARK_BACKGROUND:
            return True
    return False


def _normalize_order(root: etree._Element) -> None:
    for name, order in _ORDER.items():
        rank = {w(tag): index for index, tag in enumerate(order)}
        for element in root.iter(w(name)):
            children = list(element)
            ordered = sorted(children, key=lambda child: rank.get(child.tag, len(order)))
            if ordered != children:
                for child in children:
                    element.remove(child)
                element.extend(ordered)


def _child(parent: etree._Element, tag: str, first: bool = False) -> etree._Element:
    element = parent.find(w(tag))
    if element is None:
        element = etree.Element(w(tag))
        if first:
            parent.insert(0, element)
        else:
            parent.append(element)
    return element
