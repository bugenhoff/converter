"""Исправления разметки pdf2docx: каждая поломка на маленьком фрагменте XML."""

import io
import zipfile

import pytest
from lxml import etree
from PIL import Image

from src.conversion.docx_repair import font_family, repair_docx, repair_part

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
HEAD = (
    f'<w:document xmlns:w="{W}" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><w:body>'
)
TAIL = "</w:body></w:document>"


def repair(body: str, **kwargs) -> etree._Element:
    return etree.fromstring(repair_part((HEAD + body + TAIL).encode(), **kwargs))


def texts(root) -> str:
    return "".join(root.xpath("//w:t/text()", namespaces=NS))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Inter-Regular", ("Inter", False, False, True)),
        ("Inter-SemiBold", ("Inter", True, False, True)),
        ("Inter-Medium", ("Inter Medium", False, False, True)),
        ("ABCDEF+Inter-Light", ("Inter Light", False, False, True)),
        ("TimesNewRomanPSMT", ("Times New Roman", False, False, False)),
        ("TimesNewRomanPS-BoldItalicMT", ("Times New Roman", True, True, True)),
        ("Arial-BoldMT", ("Arial", True, False, True)),
        ("Calibri-Light", ("Calibri Light", False, False, True)),
        ("Calibri,Bold", ("Calibri", True, False, True)),
        ("Helvetica-Oblique", ("Arial", False, True, True)),
        ("SegoeUI", ("Segoe UI", False, False, False)),
        ("PTSans-Regular", ("PT Sans", False, False, True)),
        ("DejaVuSans-Bold", ("DejaVu Sans", True, False, True)),
        ("Times New Roman", ("Times New Roman", False, False, False)),
    ],
)
def test_font_family(name, expected):
    assert font_family(name) == expected


def test_fonts_are_renamed_and_weight_comes_from_name():
    root = repair(
        '<w:p><w:r><w:rPr><w:rFonts w:ascii="Inter-SemiBold" w:hAnsi="Inter-SemiBold"/><w:b w:val="0"/></w:rPr>'
        "<w:t>a</w:t></w:r>"
        '<w:r><w:rPr><w:rFonts w:ascii="Inter-Regular" w:hAnsi="Inter-Regular"/><w:b/></w:rPr><w:t>b</w:t></w:r></w:p>'
    )
    first, second = root.xpath("//w:rPr", namespaces=NS)
    assert {first.find(f"{{{W}}}rFonts").get(f"{{{W}}}{a}") for a in ("ascii", "hAnsi", "eastAsia", "cs")} == {"Inter"}
    assert first.find(f"{{{W}}}b").get(f"{{{W}}}val") is None
    assert second.find(f"{{{W}}}b").get(f"{{{W}}}val") == "0"


def test_hyperlink_nested_in_run_is_unwrapped_with_formatting():
    root = repair(
        '<w:p><w:r><w:rPr><w:b/><w:color w:val="0A8EF5"/></w:rPr>'
        '<w:hyperlink r:id="rId9"><w:r><w:rPr><w:rStyle w:val="Hyperlink"/></w:rPr><w:t>site.uz</w:t></w:r></w:hyperlink>'
        "</w:r></w:p>"
    )
    assert not root.xpath("//w:r/w:hyperlink", namespaces=NS)
    run = root.xpath("//w:p/w:hyperlink/w:r", namespaces=NS)[0]
    assert run.xpath("w:rPr/w:color/@w:val", namespaces=NS) == ["0A8EF5"]
    assert not run.xpath("w:rPr/w:rStyle", namespaces=NS)
    assert texts(root) == "site.uz"


def test_decimal_numbers_and_hash_colors_are_fixed():
    root = repair(
        '<w:tbl><w:tblPr><w:tblInd w:w="0.0" w:type="dxa"/></w:tblPr><w:tblGrid><w:gridCol w:w="100"/></w:tblGrid>'
        '<w:tr><w:tc><w:tcPr><w:tcW w:type="dxa" w:w="100"/>'
        '<w:tcBorders><w:bottom w:sz="6.3999" w:val="single" w:color="#dbe3ed"/></w:tcBorders></w:tcPr>'
        "<w:p/></w:tc></w:tr></w:tbl>"
    )
    border = root.xpath("//w:bottom", namespaces=NS)[0]
    assert border.get(f"{{{W}}}sz") == "6"
    assert border.get(f"{{{W}}}color") == "DBE3ED"
    assert root.xpath("//w:tblInd/@w:w", namespaces=NS) == ["0"]


def _cell(width, span=None):
    span_xml = f'<w:gridSpan w:val="{span}"/>' if span else ""
    return f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{width}"/>{span_xml}</w:tcPr><w:p/></w:tc>'


def test_table_grid_follows_cell_widths():
    root = repair(
        '<w:tbl><w:tblGrid><w:gridCol w:w="5000"/><w:gridCol w:w="5000"/><w:gridCol w:w="5000"/></w:tblGrid>'
        f"<w:tr>{_cell(400)}{_cell(9600, span=2)}</w:tr>"
        f"<w:tr>{_cell(400)}{_cell(3000)}{_cell(6600)}</w:tr>"
        "</w:tbl>"
    )
    assert root.xpath("//w:gridCol/@w:w", namespaces=NS) == ["400", "3000", "6600"]


def test_table_grid_is_left_alone_when_widths_are_unknown():
    root = repair(
        '<w:tbl><w:tblGrid><w:gridCol w:w="5000"/><w:gridCol w:w="5000"/></w:tblGrid>'
        f"<w:tr>{_cell(400)}<w:tc><w:p/></w:tc></w:tr></w:tbl>"
    )
    assert root.xpath("//w:gridCol/@w:w", namespaces=NS) == ["5000", "5000"]


def test_next_column_sections_become_one_section_with_column_break():
    cols = '<w:cols w:num="2" w:equalWidth="0"><w:col w:w="4000"/><w:col w:w="4000"/></w:cols>'
    root = repair(
        '<w:p><w:r><w:t>left</w:t></w:r></w:p>'
        f'<w:p><w:pPr><w:sectPr><w:type w:val="continuous"/>{cols}</w:sectPr></w:pPr></w:p>'
        '<w:p><w:r><w:t>right</w:t></w:r></w:p>'
        f'<w:p><w:pPr><w:sectPr><w:type w:val="nextColumn"/>{cols}</w:sectPr></w:pPr></w:p>'
        '<w:p><w:r><w:t>after</w:t></w:r></w:p>'
        '<w:sectPr><w:type w:val="continuous"/></w:sectPr>'
    )
    sections = root.xpath("//w:sectPr", namespaces=NS)
    assert len(sections) == 2
    assert root.xpath("//w:p/w:pPr/w:sectPr/w:type/@w:val", namespaces=NS) == ["continuous"]
    assert len(root.xpath("//w:br[@w:type='column']", namespaces=NS)) == 1
    assert texts(root) == "leftrightafter"


def test_list_item_tabs_become_hanging_indent():
    root = repair(
        '<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="260"/></w:tabs><w:ind w:left="40" w:firstLine="0"/></w:pPr>'
        "<w:r><w:t>•</w:t></w:r><w:r><w:tab/></w:r>"
        "<w:r><w:t>first line</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>second line</w:t></w:r></w:p>"
    )
    indent = root.xpath("//w:ind", namespaces=NS)[0]
    assert indent.get(f"{{{W}}}left") == "260"
    assert indent.get(f"{{{W}}}hanging") == "220"
    assert indent.get(f"{{{W}}}firstLine") is None
    assert len(root.xpath("//w:tab[not(@w:pos)]", namespaces=NS)) == 1
    assert texts(root) == "•first line second line"


def _docx(document_xml: str, image: bytes) -> bytes:
    buffer = io.BytesIO()
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="media/image1.png"/></Relationships>'
    )
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("word/document.xml", document_xml)
        package.writestr("word/_rels/document.xml.rels", rels)
        package.writestr("word/media/image1.png", image)
    return buffer.getvalue()


def test_picture_bullet_becomes_colored_character():
    png = io.BytesIO()
    Image.new("RGB", (8, 8), (9, 142, 245)).save(png, "PNG")
    body = (
        '<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="260"/></w:tabs><w:ind w:left="40"/></w:pPr>'
        '<w:r><w:drawing><wp:inline><wp:extent cx="63500" cy="50800"/>'
        '<a:graphic><a:graphicData><a:blip r:embed="rId5"/></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'
        '<w:r><w:t xml:space="preserve"> </w:t></w:r><w:r><w:tab/></w:r>'
        '<w:r><w:rPr><w:b/><w:sz w:val="19"/></w:rPr><w:t>item </w:t></w:r><w:r><w:tab/></w:r>'
        "<w:r><w:t>continued</w:t></w:r></w:p>"
    )
    result = repair_docx(_docx(HEAD + body + TAIL, png.getvalue()))
    root = etree.fromstring(zipfile.ZipFile(io.BytesIO(result)).read("word/document.xml"))
    assert not root.xpath("//w:drawing", namespaces=NS)
    bullet = root.xpath("//w:r", namespaces=NS)[0]
    assert bullet.xpath("w:t/text()", namespaces=NS) == ["•"]
    assert bullet.xpath("w:rPr/w:color/@w:val", namespaces=NS) == ["098EF5"]
    assert bullet.xpath("w:rPr/w:sz/@w:val", namespaces=NS) == ["19"]
    assert not bullet.xpath("w:rPr/w:b", namespaces=NS)
    # Пробел перед табом-продолжением уже есть — таб просто убран.
    assert texts(root) == "•item continued"


def test_lost_background_restored_on_cell_with_only_light_text():
    root = repair(
        "<w:tbl><w:tr><w:tc><w:tcPr/>"
        '<w:p><w:r><w:rPr><w:color w:val="FFFFFF"/></w:rPr><w:t>Категория</w:t></w:r></w:p>'
        "</w:tc></w:tr></w:tbl>",
        backgrounds={"Категория": "0A295B"},
    )
    assert root.xpath("//w:tcPr/w:shd/@w:fill", namespaces=NS) == ["0A295B"]


def test_lost_background_goes_to_text_when_cell_has_dark_text_too():
    root = repair(
        "<w:tbl><w:tr><w:tc>"
        '<w:p><w:r><w:rPr><w:color w:val="FFFFFF"/></w:rPr><w:t>Элемент</w:t></w:r></w:p>'
        '<w:p><w:r><w:rPr><w:color w:val="1C2736"/></w:rPr><w:t>Фотоотчёт</w:t></w:r></w:p>'
        "</w:tc></w:tr></w:tbl>",
        backgrounds={"Элемент": "0A295B"},
    )
    assert not root.xpath("//w:tcPr/w:shd", namespaces=NS)
    assert root.xpath("//w:r/w:rPr/w:shd/@w:fill", namespaces=NS) == ["0A295B"]


def test_existing_dark_background_is_kept():
    root = repair(
        '<w:tbl><w:tr><w:tc><w:tcPr><w:shd w:fill="098EF5"/></w:tcPr>'
        '<w:p><w:r><w:rPr><w:color w:val="FFFFFF"/></w:rPr><w:t>01</w:t></w:r></w:p>'
        "</w:tc></w:tr></w:tbl>",
        backgrounds={"01": "000000"},
    )
    assert root.xpath("//w:shd/@w:fill", namespaces=NS) == ["098EF5"]


def test_run_properties_follow_schema_order():
    root = repair(
        '<w:p><w:r><w:rPr><w:w w:val="97.5"/><w:rFonts w:ascii="Arial"/><w:color w:val="000000"/><w:b/></w:rPr>'
        "<w:t>x</w:t></w:r></w:p>"
    )
    order = [etree.QName(child).localname for child in root.xpath("//w:rPr", namespaces=NS)[0]]
    assert order == ["rFonts", "b", "color", "w"]
