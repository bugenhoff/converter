"""Исправление импорта .doc/.wps: полные цвета границ и размер картинок."""

import io
import zipfile

from lxml import etree
from PIL import Image

from src.conversion import word_binary as wb

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W, "wp": wb.WP_NS, "a": wb.A_NS}


def sprm(code: int, operand: bytes) -> bytes:
    return code.to_bytes(2, "little") + bytes([len(operand)]) + operand


def colorref(hex_color: str, auto: int = 0) -> bytes:
    return bytes.fromhex(hex_color) + bytes([auto])


def test_full_colors_are_read_from_border_sprms():
    stream = (
        b"\x00garbage"
        + sprm(0xD61A, colorref("0A8EF5") + colorref("000000", auto=0xFF) + colorref("0B2A5B"))
        + sprm(0xC64F, colorref("E7F1FC") + b"\x06\x01\x00\x00")
        + sprm(0xD613, b"".join(colorref("DCE4EE") + b"\x04\x01\x00\x00" for _ in range(6)))
        + sprm(0xD61B, b"\x01\x02\x03")  # длина не кратна 4 — не цвета
    )
    assert wb.colors_in_stream(stream) == {"0A8EF5", "0B2A5B", "E7F1FC", "DCE4EE"}


def test_palette_color_maps_back_only_when_unambiguous():
    assert wb.nearest_palette("0A8EF5") == "00FFFF"
    assert wb.nearest_palette("0B2A5B") == "000080"
    assert wb.palette_replacements({"0A8EF5", "0B2A5B", "DCE4EE"}) == {
        "00FFFF": "0A8EF5",
        "000080": "0B2A5B",
        "FFFFFF": "DCE4EE",
    }
    # Настоящий бирюзовый в файле — бирюзовый не трогаем.
    assert "00FFFF" not in wb.palette_replacements({"0A8EF5", "00FFFF"})
    # Два цвета сводятся к одному цвету палитры — непонятно, какой вернуть.
    assert "00FFFF" not in wb.palette_replacements({"0A8EF5", "20A0F0"})


def test_non_ole_payload_has_no_colors():
    assert wb.full_border_colors(b"not an ole file") == set()
    assert not wb.is_word_binary(b"PK\x03\x04")


def _docx(body: str, image_size=(635, 206)) -> bytes:
    png = io.BytesIO()
    Image.new("RGB", image_size, (9, 142, 245)).save(png, "PNG")
    document = (
        f'<w:document xmlns:w="{W}" xmlns:wp="{wb.WP_NS}" xmlns:a="{wb.A_NS}" xmlns:r="{wb.R_NS}"><w:body>'
        f'{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr></w:body></w:document>'
    )
    rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="media/image1.png"/></Relationships>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as package:
        # Как у LibreOffice: связи и картинка раньше документа, другой уровень сжатия.
        package.writestr("word/_rels/document.xml.rels", rels)
        package.writestr("word/media/image1.png", png.getvalue())
        package.writestr("word/document.xml", document)
    return buffer.getvalue()


def _root(docx: bytes):
    return etree.fromstring(zipfile.ZipFile(io.BytesIO(docx)).read("word/document.xml"))


WPS = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64


def test_palette_border_colors_are_restored(monkeypatch):
    monkeypatch.setattr(wb, "full_border_colors", lambda payload: {"0A8EF5", "0B2A5B"})
    body = (
        '<w:tbl><w:tr><w:tc><w:tcPr><w:tcBorders><w:top w:val="single" w:color="00FFFF"/>'
        '<w:left w:val="single" w:color="000080"/></w:tcBorders></w:tcPr>'
        '<w:p><w:r><w:rPr><w:color w:val="00FFFF"/></w:rPr><w:t>текст</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
    )
    fixed, fixes = wb.repair_imported_docx(_docx(body), WPS)
    root = _root(fixed)
    assert root.xpath("//w:tcBorders/*/@w:color", namespaces=NS) == ["0A8EF5", "0B2A5B"]
    # Цвет текста — не граница, его не трогаем.
    assert root.xpath("//w:rPr/w:color/@w:val", namespaces=NS) == ["00FFFF"]
    assert fixes == ["цвет границ: 2"]


def _picture(cx, cy):
    return (
        f'<w:p><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/>'
        '<a:graphic><a:graphicData><a:blip r:embed="rId1"/>'
        f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )


def test_absurdly_wide_logo_gets_its_proportions_back(monkeypatch):
    monkeypatch.setattr(wb, "full_border_colors", lambda payload: set())
    fixed, fixes = wb.repair_imported_docx(_docx(_picture(345935935, 628650)), WPS)
    sizes = _root(fixed).xpath("//wp:extent/@cx | //a:ext/@cx", namespaces=NS)
    assert sizes == [str(round(628650 * 635 / 206))] * 2
    assert fixes == ["размер картинок: 1"]


def test_normal_and_intentionally_stretched_images_are_left_alone(monkeypatch):
    monkeypatch.setattr(wb, "full_border_colors", lambda payload: set())
    docx = _docx(_picture(1937000, 628650) + _picture(3000000, 300000))
    fixed, fixes = wb.repair_imported_docx(docx, WPS)
    assert fixes == []
    assert _root(fixed).xpath("//wp:extent/@cx", namespaces=NS) == ["1937000", "3000000"]
