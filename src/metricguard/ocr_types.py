"""Gold-free, immutable raster identities and structured OCR output."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} contains invalid Unicode") from error
    return value


def _hash(value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected a lowercase SHA-256 digest")


def _integer(value: int, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class OcrPageInfo:
    page_index: int
    source_sha256: str
    source_type: str
    width: int
    height: int
    pixel_sha256: str
    renderer: str

    def __post_init__(self) -> None:
        _integer(self.page_index, "page_index", 0, 999)
        _integer(self.width, "width", 1, 10_000)
        _integer(self.height, "height", 1, 10_000)
        if self.width * self.height > 20_000_000:
            raise ValueError("OCR page exceeds 20 million pixels")
        for digest in (self.source_sha256, self.pixel_sha256):
            _hash(digest)
        if (
            not isinstance(self.source_type, str)
            or self.source_type not in {"image", "pdf"}
            or not _text(self.renderer, "renderer")
        ):
            raise ValueError("invalid raster source type or renderer")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OcrPageImage:
    """Canonical RGB bytes: no reference transcription or other gold data."""

    page_index: int
    source_sha256: str
    source_type: str
    width: int
    height: int
    pixels: bytes = field(repr=False)
    renderer: str

    def __post_init__(self) -> None:
        _integer(self.width, "width", 1, 10_000)
        _integer(self.height, "height", 1, 10_000)
        if not isinstance(self.pixels, bytes) or len(self.pixels) != self.width * self.height * 3:
            raise ValueError("pixels must contain exactly width * height * 3 RGB bytes")
        _ = self.info  # Validate all immutable metadata as well as pixel dimensions.

    @property
    def info(self) -> OcrPageInfo:
        digest = hashlib.sha256(struct.pack(">II", self.width, self.height))
        digest.update(self.pixels)
        return OcrPageInfo(
            self.page_index,
            self.source_sha256,
            self.source_type,
            self.width,
            self.height,
            digest.hexdigest(),
            self.renderer,
        )


@dataclass(frozen=True, slots=True)
class OcrProviderIdentity:
    provider: str
    version: str
    language: str
    implementation_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        for name in ("provider", "version", "language"):
            if not _text(getattr(self, name), name).strip():
                raise ValueError(f"{name} must not be empty")
        _hash(self.implementation_sha256)
        _hash(self.configuration_sha256)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class OcrWord:
    text: str
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not _text(self.text, "word text").strip():
            raise ValueError("word text must not be empty")
        if not isinstance(self.box, (tuple, list)) or len(self.box) != 4:
            raise ValueError("word box must contain x, y, width, height")
        if any(
            type(v) not in (int, float) or not -10_000 <= v <= 10_000 or not math.isfinite(v)
            for v in self.box
        ):
            raise ValueError("word box coordinates must be finite numbers")
        x, y, width, height = self.box
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("word box must have non-negative origin and positive extent")
        object.__setattr__(self, "box", tuple(float(value) for value in self.box))

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "box": list(self.box)}


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    words: tuple[OcrWord, ...]

    def __post_init__(self) -> None:
        _text(self.text, "line text")
        if not isinstance(self.words, (tuple, list)) or not self.words:
            raise ValueError("an OCR line must contain words")
        if any(not isinstance(word, OcrWord) for word in self.words):
            raise ValueError("line words must be OcrWord values")
        object.__setattr__(self, "words", tuple(self.words))

    @property
    def box(self) -> tuple[float, float, float, float]:
        left = min(word.box[0] for word in self.words)
        top = min(word.box[1] for word in self.words)
        right = max(word.box[0] + word.box[2] for word in self.words)
        bottom = max(word.box[1] + word.box[3] for word in self.words)
        return left, top, right - left, bottom - top

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": list(self.box),
            "words": [word.to_dict() for word in self.words],
        }


@dataclass(frozen=True, slots=True)
class OcrPageResult:
    page: OcrPageInfo
    provider: OcrProviderIdentity
    lines: tuple[OcrLine, ...]
    text_angle: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.page, OcrPageInfo) or not isinstance(
            self.provider, OcrProviderIdentity
        ):
            raise ValueError("OCR results require validated page/provider identities")
        if not isinstance(self.lines, (tuple, list)) or any(
            not isinstance(line, OcrLine) for line in self.lines
        ):
            raise ValueError("OCR result lines must be OcrLine values")
        object.__setattr__(self, "lines", tuple(self.lines))
        if len(self.lines) > 20_000 or sum(len(line.words) for line in self.lines) > 100_000:
            raise ValueError("OCR result exceeds line/word count limits")
        if self.text_angle is not None and (
            type(self.text_angle) not in (int, float)
            or not -180 <= self.text_angle <= 180
            or not math.isfinite(self.text_angle)
        ):
            raise ValueError("text_angle must be a finite angle or null")
        for line in self.lines:
            for word in line.words:
                x, y, width, height = word.box
                if x + width > self.page.width + 0.001 or y + height > self.page.height + 0.001:
                    raise ValueError("word box extends outside its OCR raster")
        if self.structured_text_bytes > 4 * 1024 * 1024:
            raise ValueError("OCR page line/word text exceeds 4 MiB")

    @property
    def structured_text_bytes(self) -> int:
        """Count both native line and word strings, including assembly newlines."""
        return max(0, len(self.lines) - 1) + sum(
            len(line.text.encode("utf-8"))
            + sum(len(word.text.encode("utf-8")) for word in line.words)
            for line in self.lines
        )

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page.to_dict(),
            "provider": self.provider.to_dict(),
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "text_angle": self.text_angle,
            "status": "succeeded",
        }


@dataclass(frozen=True, slots=True)
class OcrDocumentResult:
    """Complete contiguous page sequence; separator offsets use Unicode code points."""

    pages: tuple[OcrPageResult, ...]
    separator: str = "\n\f\n"

    def __post_init__(self) -> None:
        if not isinstance(self.pages, (tuple, list)) or not self.pages:
            raise ValueError("OCR document must contain at least one page")
        if any(not isinstance(page, OcrPageResult) for page in self.pages):
            raise ValueError("document pages must be OcrPageResult values")
        object.__setattr__(self, "pages", tuple(self.pages))
        if tuple(page.page.page_index for page in self.pages) != tuple(range(len(self.pages))):
            raise ValueError("document pages must be complete and in source order")
        if (
            len(
                {
                    (page.page.source_sha256, page.page.source_type, page.provider)
                    for page in self.pages
                }
            )
            != 1
        ):
            raise ValueError("document pages must share their source and provider identity")
        if len(_text(self.separator, "separator")) > 100:
            raise ValueError("page separator exceeds 100 code points")

    @property
    def text(self) -> str:
        return self.separator.join(page.text for page in self.pages)

    def to_dict(self) -> dict[str, Any]:
        offset = 0
        spans = []
        for page in self.pages:
            spans.append(
                {
                    "page_index": page.page.page_index,
                    "start": offset,
                    "end": offset + len(page.text),
                }
            )
            offset += len(page.text) + len(self.separator)
        return {
            "format": "metricguard.ocr-document.v1",
            "status": "complete",
            "text": self.text,
            "separator": self.separator,
            "page_spans": spans,
            "pages": [page.to_dict() for page in self.pages],
        }

    @property
    def digest(self) -> str:
        data = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
