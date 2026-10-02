import io
import shutil

import pytest
from PIL import Image, ImageDraw

from src.conversion.pages import PdfPages, image_to_page, measure_page, pdf_has_text_layer

needs_poppler = pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="poppler-utils не установлен")

# Страница A4 при 100 dpi: поля 3 см слева и 1,5 см справа, строки по 25 px.
WIDTH, HEIGHT = 827, 1169
LEFT, RIGHT, TOP = 118, 827 - 59, 79


def _scan(color=(0, 0, 0)) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    for line in range(30):
        y = TOP + line * 25
        draw.rectangle((LEFT, y, RIGHT, y + 9), fill=color)  # «строчные буквы»
        draw.rectangle((LEFT, y - 4, LEFT + 40, y + 13), fill=color)  # «выносные»
    return image


def _pdf(*images: Image.Image) -> bytes:
    buffer = io.BytesIO()
    images[0].save(buffer, "PDF", resolution=100.0, save_all=True, append_images=list(images[1:]))
    return buffer.getvalue()


def test_measure_page_finds_margins_and_line_metrics():
    geometry = measure_page(_scan(), 595.28, 841.89)
    left, right, top, _ = (value / 28.35 for value in geometry.margins_pt)
    assert left == pytest.approx(3.0, abs=0.1)
    assert right == pytest.approx(1.5, abs=0.1)
    assert top == pytest.approx(1.9, abs=0.15)
    assert geometry.line_pitch_pt == pytest.approx(25 * 0.72, abs=0.5)
    assert geometry.x_height_pt == pytest.approx(10 * 0.72, abs=1.0)


def test_coloured_watermarks_are_not_ink():
    image = _scan()
    ImageDraw.Draw(image).text((5, 5), "VRQ6456", fill=(255, 120, 120))
    ImageDraw.Draw(image).rectangle((780, 1000, 826, 1168), fill=(30, 60, 220))
    geometry = measure_page(image, 595.28, 841.89)
    assert geometry.margins_pt[0] / 28.35 == pytest.approx(3.0, abs=0.1)


def test_blank_page_has_no_measurements():
    geometry = measure_page(Image.new("RGB", (100, 141), "white"), 595.28, 841.89)
    assert geometry.margins_pt is None and geometry.x_height_pt is None


def test_photo_is_downscaled_and_not_measured():
    photo = Image.new("RGB", (4000, 2000), "white")
    buffer = io.BytesIO()
    photo.save(buffer, "PNG")
    page = image_to_page(buffer.getvalue(), max_side=1000)
    with Image.open(io.BytesIO(page.jpeg)) as result:
        assert result.size == (1000, 500)
    assert page.geometry.width_pt > page.geometry.height_pt
    assert page.geometry.margins_pt is None


@needs_poppler
def test_pdf_pages_render_with_sizes():
    landscape = _scan().rotate(90, expand=True)
    with PdfPages(_pdf(_scan(), landscape)) as pdf:
        assert len(pdf) == 2
        first = pdf.render(1, max_side=800)
        second = pdf.render(2, max_side=800)
    with Image.open(io.BytesIO(first.jpeg)) as image:
        assert max(image.size) == 800
    assert round(first.geometry.width_pt) == 595
    assert second.geometry.width_pt > second.geometry.height_pt
    assert first.geometry.margins_pt is not None


@needs_poppler
def test_scanned_pdf_has_no_text_layer():
    assert not pdf_has_text_layer(_pdf(_scan()))
