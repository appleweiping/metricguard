"""Actual page recognition and ordered document assembly, separate from scoring."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Protocol

from .ocr_inputs import OcrInputLimits, load_ocr_pages
from .ocr_types import OcrDocumentResult, OcrPageImage, OcrPageResult, OcrProviderIdentity


class PageOcrBackend(Protocol):
    @property
    def identity(self) -> OcrProviderIdentity: ...

    def recognize(self, page: OcrPageImage) -> OcrPageResult: ...


def transcribe_document(
    path: str | Path,
    backend: PageOcrBackend,
    *,
    limits: OcrInputLimits | None = None,
    max_document_text_bytes: int = 16 * 1024 * 1024,
) -> OcrDocumentResult:
    """Recognize one image/multiframe image/PDF without access to gold labels.

    Rasterization and recognition are sequential and bounded. A failure returns
    no partial document; successful earlier local OCR calls are not rolled back.
    Results retain text and geometry, not the decoded raster buffers.
    """
    if (
        type(max_document_text_bytes) is not int
        or not 1 <= max_document_text_bytes <= 64 * 1024 * 1024
    ):
        raise ValueError("document text limit must be an integer in [1, 64 MiB]")
    identity = backend.identity
    if not isinstance(identity, OcrProviderIdentity):
        raise ValueError("OCR backend must expose an immutable provider identity")
    pages: list[OcrPageResult] = []
    text_bytes = 0
    words = 0
    with closing(load_ocr_pages(path, limits=limits)) as source:
        for page in source:
            if backend.identity != identity:
                raise ValueError("OCR backend identity changed before recognition")
            result = backend.recognize(page)
            if not isinstance(result, OcrPageResult) or result.page != page.info:
                raise ValueError("OCR backend returned a result for a different page")
            if result.provider != identity or backend.identity != identity:
                raise ValueError("OCR backend identity changed during recognition")
            text_bytes += result.structured_text_bytes + (3 if pages else 0)
            if text_bytes > max_document_text_bytes:
                raise ValueError("OCR document exceeds its text byte limit")
            words += sum(len(line.words) for line in result.lines)
            if words > 100_000:
                raise ValueError("OCR document exceeds 100,000 words")
            pages.append(result)
    return OcrDocumentResult(tuple(pages))
