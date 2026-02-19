from src.bot.handlers import _detect_file_type


def test_detect_file_type_for_documents():
    assert _detect_file_type("file.doc") == "doc"
    assert _detect_file_type("file.docx") == "docx"
    assert _detect_file_type("file.PDF") == "pdf"


def test_detect_file_type_for_images():
    assert _detect_file_type("img.png") == "image"
    assert _detect_file_type("img.JPG") == "image"
    assert _detect_file_type("img.tiff") == "image"


def test_detect_file_type_for_unsupported():
    assert _detect_file_type("file.txt") is None
