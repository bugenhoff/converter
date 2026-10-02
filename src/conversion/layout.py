"""Разметка страницы, которую возвращает модель: абзацы, строки-колонки и таблицы."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Union

ALIGNMENTS = {"left", "center", "right", "justify"}
_ALIGNMENT_ALIASES = {"justified": "justify", "both": "justify", "centre": "center", "centered": "center"}
SIZES = {"small", "normal", "large", "xlarge"}

_INLINE_MARKUP = re.compile(r"\*\*((?s:.+?))\*\*|__([^_]+?)__|\*([^*\s](?:[^*]*?[^*\s])?)\*")


class LayoutError(ValueError):
    """Ответ модели не удалось разобрать."""


@dataclass
class Paragraph:
    text: str
    align: str = "left"
    first_line: bool = False
    indent: float = 0.0  # отступ слева, доля ширины текста
    size: str = "normal"
    bold: bool = False
    gap: bool = False  # заметный отступ перед блоком


@dataclass
class Row:
    """Строка из 2–3 колонок: дата слева и номер справа, должность и подпись."""

    cells: list[str]
    size: str = "normal"
    bold: bool = False
    gap: bool = False


@dataclass
class Table:
    rows: list[list[str]]
    borders: bool = True
    widths: list[float] | None = None
    size: str = "normal"
    gap: bool = False


Block = Union[Paragraph, Row, Table]


@dataclass
class Span:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False


@dataclass
class _Style:
    size: str = "normal"
    bold: bool = False
    gap: bool = False


def parse_page_layout(raw: str) -> list[Block]:
    data = _load_json(raw)
    if isinstance(data, dict) and isinstance(data.get("pages"), list):
        raw_blocks = [b for page in data["pages"] if isinstance(page, dict) for b in page.get("blocks") or []]
    elif isinstance(data, dict):
        raw_blocks = data.get("blocks")
    else:
        raw_blocks = data
    if not isinstance(raw_blocks, list):
        raise LayoutError("В ответе нет списка blocks")

    blocks = [block for item in raw_blocks if isinstance(item, dict) and (block := _parse_block(item))]
    return blocks


def plain_text(blocks: list[Block]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, Paragraph):
            parts.append(block.text)
        elif isinstance(block, Row):
            parts.extend(block.cells)
        else:
            parts.extend(cell for row in block.rows for cell in row)
    return "\n".join("".join(span.text for span in parse_inline(part)) for part in parts)


def parse_inline(text: str) -> list[Span]:
    """Разбивает текст по **жирному**, *курсиву* и __подчёркнутому__."""
    spans: list[Span] = []
    position = 0
    for match in _INLINE_MARKUP.finditer(text):
        if match.start() > position:
            spans.append(Span(text[position:match.start()]))
        bold, underline, italic = match.groups()
        if bold is not None:
            spans.extend(Span(s.text, True, s.italic, s.underline) for s in parse_inline(bold))
        elif underline is not None:
            spans.append(Span(underline, underline=True))
        else:
            spans.append(Span(italic, italic=True))
        position = match.end()
    if position < len(text):
        spans.append(Span(text[position:]))
    return [span for span in spans if span.text]


def _load_json(raw: str) -> Any:
    content = raw.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # Модель иногда добавляет пояснение до или после JSON.
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LayoutError(f"Невалидный JSON: {exc}") from exc
    raise LayoutError("В ответе нет JSON")


def _parse_block(item: dict[str, Any]) -> Block | None:
    style = _Style(size=_size(item.get("size")), bold=item.get("bold") is True, gap=item.get("gap") is True)
    if isinstance(item.get("table"), list):
        return _parse_table(item, style)
    if isinstance(item.get("row"), list):
        cells = [_clean_text(cell) for cell in item["row"]]
        if not any(cells):
            return None
        if len(cells) == 1:
            return Paragraph(cells[0], size=style.size, bold=style.bold, gap=style.gap)
        return Row(cells[:3], size=style.size, bold=style.bold, gap=style.gap)

    text = _clean_text(item.get("text"))
    if not text:
        return None
    return Paragraph(
        text=text,
        align=_alignment(item.get("align")),
        first_line=item.get("first_line") is True,
        indent=_fraction(item.get("indent")),
        size=style.size,
        bold=style.bold,
        gap=style.gap,
    )


def _parse_table(item: dict[str, Any], style: _Style) -> Table | None:
    rows = [[_clean_text(cell) for cell in row] for row in item["table"] if isinstance(row, list)]
    rows = [row for row in rows if any(row)]
    if not rows:
        return None
    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]

    widths = item.get("widths")
    if isinstance(widths, list) and len(widths) == columns:
        try:
            values = [float(width) for width in widths]
        except (TypeError, ValueError):
            values = []
        total = sum(values)
        widths = [value / total for value in values] if values and total > 0 and min(values) > 0 else None
    else:
        widths = None

    return Table(rows, borders=item.get("borders") is not False, widths=widths, size=style.size, gap=style.gap)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(str(part) for part in value if part is not None)
    lines = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("­", "").split("\n")
    return "\n".join(" ".join(line.replace("\t", " ").split()) for line in lines).strip()


def _alignment(value: Any) -> str:
    align = str(value or "left").strip().lower()
    align = _ALIGNMENT_ALIASES.get(align, align)
    return align if align in ALIGNMENTS else "left"


def _size(value: Any) -> str:
    size = str(value or "normal").strip().lower()
    return size if size in SIZES else "normal"


def _fraction(value: Any) -> float:
    try:
        fraction = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    if fraction > 1:  # модель прислала проценты
        fraction /= 100
    return min(max(fraction, 0.0), 0.9)
