"""Real decoders and limit preflight; no OCR engine or network is required."""

import hashlib
from contextlib import closing
from dataclasses import replace

import pytest
from PIL import Image

from metricguard import OcrInputLimits, load_ocr_pages
from metricguard import ocr_inputs as inputs


def test_rgb_hash_alpha_and_source_snapshot(tmp_path):
    path = tmp_path / "alpha.png"
    with Image.new("RGBA", (3, 2), (0, 0, 0, 0)) as image:
        image.putpixel((0, 0), (255, 0, 0, 255))
        image.save(path)
    original = path.read_bytes()
    pages = list(load_ocr_pages(path))
    assert len(pages) == 1
    page = pages[0]
    assert page.source_sha256 == hashlib.sha256(original).hexdigest()
    assert page.pixels[:6] == b"\xff\x00\x00\xff\xff\xff"
    assert (
        page.info.pixel_sha256
        == hashlib.sha256(b"\x00\x00\x00\x03\x00\x00\x00\x02" + page.pixels).hexdigest()
    )
    assert page.renderer.startswith("Pillow/")
    assert not hasattr(page, "reference")


def test_exif_orientation_changes_coordinate_frame(tmp_path):
    path = tmp_path / "oriented.png"
    with Image.new("RGB", (4, 2), "white") as image:
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
    page = next(load_ocr_pages(path))
    assert (page.width, page.height) == (2, 4)


def test_multiframe_preflight_and_stable_order(tmp_path):
    path = tmp_path / "pages.tiff"
    with Image.new("RGB", (4, 3), "red") as first, Image.new("RGB", (5, 2), "blue") as last:
        first.save(path, save_all=True, append_images=[last])
    pages = list(load_ocr_pages(path))
    assert [page.page_index for page in pages] == [0, 1]
    assert [page.pixels[:3] for page in pages] == [b"\xff\x00\x00", b"\x00\x00\xff"]
    assert [(page.width, page.height) for page in pages] == [(4, 3), (5, 2)]
    for limits, message in [
        (OcrInputLimits(max_pages=1), "frame count"),
        (OcrInputLimits(max_total_pixels=21), "total pixel"),
        (OcrInputLimits(max_dimension=4), "raster width"),
        (OcrInputLimits(max_page_pixels=11), "pixel limit"),
    ]:
        # Later frames are checked before yielding even the first page.
        with pytest.raises(ValueError, match=message):
            next(load_ocr_pages(path, limits=limits))


def test_raster_only_pdf_preflight_render_and_close(tmp_path):
    path = tmp_path / "pages.pdf"
    with Image.new("RGB", (24, 12), "red") as first, Image.new("RGB", (24, 12), "white") as last:
        first.save(path, save_all=True, append_images=[last], resolution=72)
    pages = list(load_ocr_pages(path, limits=OcrInputLimits(dpi=144)))
    assert [page.page_index for page in pages] == [0, 1]
    assert all((page.width, page.height) == (48, 24) for page in pages)
    assert all(page.source_type == "pdf" for page in pages)
    assert pages[0].pixels != pages[1].pixels
    assert "dpi=144" in pages[0].renderer
    for limits, message in [
        (OcrInputLimits(max_pages=1), "PDF page count"),
        (OcrInputLimits(max_total_pixels=300, dpi=72), "total pixel"),
        (OcrInputLimits(max_dimension=20, dpi=72), "raster width"),
    ]:
        with pytest.raises(ValueError, match=message):
            next(load_ocr_pages(path, limits=limits))
    with closing(load_ocr_pages(path)) as iterator:
        assert next(iterator).page_index == 0
    # Abandoned decoding releases its own PDF resources and lock.
    assert len(list(load_ocr_pages(path))) == 2


@pytest.mark.parametrize("field,value", [("dpi", True), ("dpi", 601), ("max_pages", 0)])
def test_limit_configuration_is_strict(field, value):
    with pytest.raises(ValueError, match=field):
        replace(OcrInputLimits(), **{field: value})


def test_bad_input_missing_extra_and_unsupported_format(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    with pytest.raises(ValueError, match="cannot decode"):
        list(load_ocr_pages(path))
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        list(load_ocr_pages(path))
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="byte limit"):
        list(load_ocr_pages(path, limits=OcrInputLimits(max_file_bytes=2)))
    with pytest.raises(ValueError, match="cannot decode"):
        list(load_ocr_pages(path))
    path.write_bytes(b"%PDF-invalid")
    with pytest.raises(ValueError, match="cannot decode"):
        list(load_ocr_pages(path))
    with Image.new("RGB", (1, 1)) as image:
        image.save(path, format="GIF")
    with pytest.raises(ValueError, match="unsupported"):
        list(load_ocr_pages(path))
    with pytest.raises(ValueError, match="limits must"):
        list(load_ocr_pages(path, limits="wrong"))
    with pytest.raises(ValueError, match="limits must"):
        list(load_ocr_pages(path, limits=False))

    def missing(_name):
        raise ImportError("not installed")

    monkeypatch.setattr(inputs.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="optional"):
        list(load_ocr_pages(path))


def test_decompression_warning_is_an_error_before_decode(tmp_path, monkeypatch):
    path = tmp_path / "image.png"
    with Image.new("RGB", (4, 4)) as image:
        image.save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="DecompressionBombWarning"):
        list(load_ocr_pages(path))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="DecompressionBombError"):
        list(load_ocr_pages(path))
