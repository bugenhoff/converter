import pytest

from src.conversion.layout import (
    LayoutError,
    Paragraph,
    Row,
    Span,
    Table,
    parse_inline,
    parse_page_layout,
    plain_text,
)


def test_paragraph_defaults_and_normalization():
    blocks = parse_page_layout(
        '{"blocks": [{"text": "  Hello\\t  world  ", "align": "justified", "indent": 50, "size": "huge"}]}'
    )
    assert blocks == [Paragraph("Hello world", align="justify", indent=0.5)]


def test_code_fences_and_surrounding_text_are_ignored():
    raw = 'Here you go:\n```json\n{"blocks": [{"text": "A"}]}\n```'
    assert parse_page_layout(raw) == [Paragraph("A")]


def test_legacy_pages_shape_and_bare_list():
    assert parse_page_layout('{"pages": [{"blocks": [{"text": "A"}]}]}') == [Paragraph("A")]
    assert parse_page_layout('[{"text": "B"}]') == [Paragraph("B")]


def test_rows_and_tables():
    blocks = parse_page_layout(
        '{"blocks": ['
        '{"row": ["Direktor", "M.Olloyorov"], "bold": true, "gap": true},'
        '{"row": ["single"]},'
        '{"table": [["1.", "Name"], ["2."]], "borders": false, "widths": [1, 3]},'
        '{"table": [["", ""]]}'
        "]}"
    )
    assert blocks == [
        Row(["Direktor", "M.Olloyorov"], bold=True, gap=True),
        Paragraph("single"),
        Table([["1.", "Name"], ["2.", ""]], borders=False, widths=[0.25, 0.75]),
    ]


def test_empty_blocks_are_dropped():
    assert parse_page_layout('{"blocks": [{"text": "  "}, {"foo": 1}, "junk"]}') == []


def test_invalid_json_raises():
    with pytest.raises(LayoutError):
        parse_page_layout('{"blocks": [')
    with pytest.raises(LayoutError):
        parse_page_layout("no json here")


def test_inline_markup():
    assert parse_inline("4. Raisi (**S. Rasulev**): *izoh* va __5__") == [
        Span("4. Raisi ("),
        Span("S. Rasulev", bold=True),
        Span("): "),
        Span("izoh", italic=True),
        Span(" va "),
        Span("5", underline=True),
    ]


def test_inline_markup_keeps_lone_asterisks_and_blank_fields():
    assert parse_inline("2 * 3 = 6, ________") == [Span("2 * 3 = 6, ________")]
    assert parse_inline("**bold *it* bold**") == [
        Span("bold ", bold=True),
        Span("it", bold=True, italic=True),
        Span(" bold", bold=True),
    ]


def test_plain_text_strips_markup():
    blocks = [Paragraph("**A** b"), Row(["c", "d"]), Table([["e"]])]
    assert plain_text(blocks) == "A b\nc\nd\ne"
