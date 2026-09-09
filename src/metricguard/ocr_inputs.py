"""Resource-limited image/PDF raster ingestion, with optional imaging dependencies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import math
import threading
import warnings
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ocr_types import OcrPageImage, _integer

_PDF_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class OcrInputLimits:
    max_file_bytes: int = 64 * 1024 * 1024
    max_pages: int = 100
    max_page_pixels: int = 20_000_000
    max_total_pixels: int = 100_000_000
    max_dimension: int = 10_000
    dpi: int = 200

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("max_file_bytes", 64 * 1024 * 1024),
            ("max_pages", 1000),
            ("max_page_pixels", 20_000_000),
            ("max_total_pixels", 200_000_000),
            ("max_dimension", 10_000),
            ("dpi", 600),
        ):
            _integer(getattr(self, name), name, 1, ceiling)

    def check_size(self, width: int, height: int) -> int:
        _integer(width, "raster width", 1, self.max_dimension)
        _integer(height, "raster height", 1, self.max_dimension)
        pixels = width * height
        if pixels > self.max_page_pixels:
            raise ValueError("OCR page exceeds configured pixel limit")
        return pixels


def _dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ValueError(
            "image/PDF OCR requires MetricGuard's optional [ocr] dependencies"
        ) from error


def _rgb(image: Any, image_module: Any, image_ops: Any) -> Any:
    # Explicit white composition avoids interpreting transparent black as ink.
    oriented = image_ops.exif_transpose(image)
    try:
        rgba = oriented.convert("RGBA")
        try:
            background = image_module.new("RGBA", rgba.size, "white")
            try:
                background.alpha_composite(rgba)
                return background.convert("RGB")
            finally:
                background.close()
        finally:
            rgba.close()
    finally:
        oriented.close()


def _decode_image_pages(
    raw: bytes, source_hash: str, limits: OcrInputLimits
) -> Iterator[OcrPageImage]:
    image_module, image_ops = _dependency("PIL.Image"), _dependency("PIL.ImageOps")
    renderer = f"Pillow/{importlib.metadata.version('Pillow')};exif-white-alpha-rgb/v1"
    with warnings.catch_warnings():
        warnings.simplefilter("error", image_module.DecompressionBombWarning)
        with image_module.open(io.BytesIO(raw)) as image:
            if image.format not in {"PNG", "JPEG", "TIFF", "BMP", "WEBP"}:
                raise ValueError("unsupported OCR image format")
            frames = getattr(image, "n_frames", 1)
            _integer(frames, "image frame count", 1, limits.max_pages)
            total = 0
            for index in range(frames):
                image.seek(index)
                total += limits.check_size(*image.size)
                if total > limits.max_total_pixels:
                    raise ValueError("OCR document exceeds total pixel limit")
            for index in range(frames):
                image.seek(index)
                rgb = _rgb(image, image_module, image_ops)
                try:
                    limits.check_size(*rgb.size)
                    yield OcrPageImage(
                        index, source_hash, "image", rgb.width, rgb.height, rgb.tobytes(), renderer
                    )
                finally:
                    rgb.close()


def _image_pages(raw: bytes, source_hash: str, limits: OcrInputLimits) -> Iterator[OcrPageImage]:
    image_module = _dependency("PIL.Image")
    try:
        yield from _decode_image_pages(raw, source_hash, limits)
    except image_module.DecompressionBombError as error:
        raise ValueError("cannot decode OCR input: DecompressionBombError") from error


def _pdf_pages(raw: bytes, source_hash: str, limits: OcrInputLimits) -> Iterator[OcrPageImage]:
    pdfium = _dependency("pypdfium2")
    _dependency("PIL.Image")
    renderer = f"pypdfium2/{importlib.metadata.version('pypdfium2')};dpi={limits.dpi};rgb/v1"
    # PDFium forbids concurrent API calls even across different documents.
    # This lock covers our instances, not third-party calls to PDFium elsewhere.
    with _PDF_LOCK:
        document = pdfium.PdfDocument(raw)
        try:
            count = len(document)
            _integer(count, "PDF page count", 1, limits.max_pages)
            total = 0
            for index in range(count):
                page = document[index]
                try:
                    width, height = page.get_size()
                    if not math.isfinite(width) or not math.isfinite(height):
                        raise ValueError("PDF page has invalid dimensions")
                    total += limits.check_size(
                        math.ceil(width * limits.dpi / 72), math.ceil(height * limits.dpi / 72)
                    )
                    if total > limits.max_total_pixels:
                        raise ValueError("OCR document exceeds total pixel limit")
                finally:
                    page.close()
            for index in range(count):
                page = document[index]
                try:
                    bitmap = page.render(scale=limits.dpi / 72, fill_color=(255, 255, 255, 255))
                    try:
                        image = bitmap.to_pil()
                        try:
                            rgb = image.convert("RGB")
                            try:
                                limits.check_size(*rgb.size)
                                yield OcrPageImage(
                                    index,
                                    source_hash,
                                    "pdf",
                                    rgb.width,
                                    rgb.height,
                                    rgb.tobytes(),
                                    renderer,
                                )
                            finally:
                                rgb.close()
                        finally:
                            image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            document.close()


def load_ocr_pages(
    path: str | Path, *, limits: OcrInputLimits | None = None
) -> Generator[OcrPageImage, None, None]:
    """Snapshot one file, preflight all page sizes, then decode one page at a time.

    PNG/JPEG/TIFF/BMP/WEBP and standard PDF inputs are supported. Call ``close``
    if abandoning the iterator; ``transcribe_document`` does this automatically.
    No text layer is extracted from PDFs and no network content is fetched.
    """
    active = OcrInputLimits() if limits is None else limits
    if not isinstance(active, OcrInputLimits):
        raise ValueError("limits must be OcrInputLimits")
    try:
        with Path(path).open("rb") as stream, io.BytesIO() as snapshot:
            # A single read(limit) can preallocate the entire 64 MiB allowance
            # even for a tiny file. Grow only as actual bounded chunks arrive.
            while chunk := stream.read(min(65536, active.max_file_bytes + 1 - snapshot.tell())):
                snapshot.write(chunk)
                if snapshot.tell() > active.max_file_bytes:
                    break
            raw = snapshot.getvalue()
        if not raw or len(raw) > active.max_file_bytes:
            raise ValueError("OCR input is empty or exceeds the file byte limit")
        source_hash = hashlib.sha256(raw).hexdigest()
        if raw.startswith(b"%PDF-"):
            yield from _pdf_pages(raw, source_hash, active)
        else:
            yield from _image_pages(raw, source_hash, active)
    except (OSError, RuntimeError, Warning) as error:
        raise ValueError(f"cannot decode OCR input: {type(error).__name__}") from error
