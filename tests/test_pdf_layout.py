"""Цифровой PDF → DOCX: определение сканов, градиенты, фон под светлым текстом, сквозной прогон."""

import io
import re
import zipfile

import pymupdf
import pytest

from src.conversion import pdf_layout
from src.conversion.docx_repair import luminance

# Встроенный шрифт PyMuPDF (Helvetica) без кириллицы — текст латиницей.
WORDS = (
    "Thank you for your interest in our services. We offer professional washing and cleaning "
    "of building facades: glass, metal, composite, stone and combined surfaces."
)
DARK = (0.04, 0.16, 0.36)


@pytest.fixture(autouse=True)
def temp_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_layout.settings, "temp_dir", tmp_path)


def _with_shading(page: pymupdf.Page) -> None:
    """Градиент в прямоугольнике 50,60–150,100 (как логотип из SVG)."""
    doc = page.parent
    shading = doc.get_new_xref()
    doc.update_object(
        shading,
        "<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [50 760 150 760] "
        "/Function << /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> /Extend [true true] >>",
    )
    contents = doc.get_new_xref()
    doc.update_object(contents, "<<>>")
    doc.update_stream(contents, b"q 50 742 100 40 re W n /Sh1 sh Q")
    page.set_contents(contents)
    doc.xref_set_key(page.xref, "Resources", f"<< /Shading << /Sh1 {shading} 0 R >> >>")


def _digital_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    _with_shading(page)
    page.insert_textbox(pymupdf.Rect(50, 150, 400, 260), WORDS, fontsize=11)
    page.draw_rect(pymupdf.Rect(50, 300, 545, 324), color=None, fill=DARK)
    page.insert_text((56, 316), "Category", fontsize=10, color=(1, 1, 1))
    page.insert_text((50, 360), "Site: fullpro.uz", fontsize=11)
    page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(80, 348, 140, 364), "uri": "https://fullpro.uz"})
    return doc.tobytes()


def _scan_pdf(stamp: str = "") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 85), False)
    pixmap.clear_with(230)
    page.insert_image(page.rect, pixmap=pixmap)
    if stamp:
        page.insert_text((40, 820), stamp, fontsize=8)
    return doc.tobytes()


def test_digital_pdf_is_detected():
    profile = pdf_layout.profile_pdf(_digital_pdf())
    assert profile.is_digital
    assert profile.pages == 1


@pytest.mark.parametrize("stamp", ["", "Signed electronically 12.03.2026"])
def test_scan_is_detected_even_with_a_text_stamp(stamp):
    assert not pdf_layout.profile_pdf(_scan_pdf(stamp)).is_digital


def test_gradient_region_is_rasterized_into_real_image():
    doc = pymupdf.open(stream=_digital_pdf())
    page = doc[0]
    assert any(not info.get("xref") for info in page.get_image_info(xrefs=True))
    assert pdf_layout.rasterize_shadings(doc) == 1
    images = [info for info in doc[0].get_image_info(xrefs=True) if info.get("xref")]
    assert len(images) == 1
    assert pymupdf.Rect(images[0]["bbox"]).contains(pymupdf.Rect(51, 61, 149, 99))


def test_gradient_behind_text_is_not_rasterized():
    doc = pymupdf.open()
    page = doc.new_page()
    _with_shading(page)
    page.insert_text((60, 85), "Text over gradient", fontsize=10)
    assert pdf_layout.rasterize_shadings(doc) == 0


def test_light_text_background_is_sampled_from_render():
    backgrounds = pdf_layout.light_text_backgrounds(pymupdf.open(stream=_digital_pdf()))
    assert set(backgrounds) == {"Category"}
    assert luminance(backgrounds["Category"]) < 0.2


def test_end_to_end_keeps_text_links_and_background():
    docx = pdf_layout.pdf_to_docx_layout(_digital_pdf())
    package = zipfile.ZipFile(io.BytesIO(docx))
    xml = package.read("word/document.xml").decode()

    flat = re.sub(r"<w:(tab|br)\b[^>]*/>|</w:p>", "<w:t> </w:t>", xml)
    text = "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", flat))
    for word in WORDS.split() + ["Category", "fullpro.uz"]:
        assert word in text.split(), word
    assert "<w:hyperlink" in xml
    assert not re.search(r"<w:r>(?:(?!</w:r>).)*<w:hyperlink", xml, flags=re.S)
    fills = re.findall(r'w:fill="([0-9A-Fa-f]{6})"', xml)
    assert any(luminance(fill) < 0.6 for fill in fills)
    assert any(name.startswith("word/media/") for name in package.namelist())
