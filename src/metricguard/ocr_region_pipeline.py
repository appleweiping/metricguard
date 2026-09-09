"""Complete, source-bound multiple-crop OCR without access to gold text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ocr_inputs import OcrInputLimits, load_ocr_pages
from .ocr_pipeline import PageOcrBackend
from .ocr_regions import (
    CROP_RENDERER_VERSION,
    OcrRegion,
    OcrRegionLimits,
    OcrRegionPlan,
    detect_vertical_regions,
    iter_region_crops,
    map_region_result,
)
from .ocr_types import (
    OcrLine,
    OcrPageImage,
    OcrPageInfo,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
    _hash,
    _integer,
)

REGION_SEPARATOR = "\n\n"
PAGE_SEPARATOR = "\n\f\n"
MAX_RESULT_BYTES = 64 * 1024 * 1024
RegionPlanner = Callable[[OcrPageImage], OcrRegionPlan]


def _canonical(value: Any) -> str:
    chunks = []
    size = 0
    try:
        for chunk in json.JSONEncoder(
            ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).iterencode(value):
            size += len(chunk.encode("utf-8"))
            if size > MAX_RESULT_BYTES:
                raise ValueError("OCR region result exceeds its serialized byte limit")
            chunks.append(chunk)
    except (TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("OCR region result is not bounded JSON") from error
    return "".join(chunks)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _closed(value: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} has invalid fields")
    return value


def _items(value: Any, maximum: int, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)) or len(value) > maximum:
        raise ValueError(f"{name} must be a bounded array")
    return tuple(value)


def _page_info(value: Any) -> OcrPageInfo:
    return OcrPageInfo(**_closed(value, set(OcrPageInfo.__dataclass_fields__), "page identity"))


def _provider(value: Any) -> OcrProviderIdentity:
    return OcrProviderIdentity(
        **_closed(value, set(OcrProviderIdentity.__dataclass_fields__), "provider identity")
    )


def _recognition(value: Any) -> OcrPageResult:
    data = _closed(
        value, {"page", "provider", "text", "lines", "text_angle", "status"}, "recognition"
    )
    lines = []
    words = 0
    for raw in _items(data["lines"], 20_000, "recognition lines"):
        line = _closed(raw, {"text", "box", "words"}, "recognition line")
        parsed = []
        for item in _items(line["words"], 100_000, "recognition words"):
            words += 1
            if words > 100_000:
                raise ValueError("recognition exceeds word limit")
            word = _closed(item, {"text", "box"}, "recognition word")
            parsed.append(OcrWord(word["text"], word["box"]))
        lines.append(OcrLine(line["text"], tuple(parsed)))
    result = OcrPageResult(
        _page_info(data["page"]), _provider(data["provider"]), tuple(lines), data["text_angle"]
    )
    if _canonical(result.to_dict()) != _canonical(data):
        raise ValueError("recognition contains inconsistent derived fields")
    return result


def _check_geometry_inventory(regions: Iterable[OcrRegionResult]) -> None:
    """Reject geometric amplification before constructing nested export dictionaries."""
    words = lines = 0
    for count, region in enumerate(regions, start=1):
        lines += len(region.recognition.lines)
        words += sum(len(line.words) for line in region.recognition.lines)
        if count > 10_000 or lines > 100_000 or words > 100_000:
            raise ValueError("OCR region result exceeds aggregate region, word or line limits")


@dataclass(frozen=True, slots=True)
class OcrRegionPipelineLimits:
    """Aggregate admission limits, separate from per-plan and input limits."""

    max_total_regions: int = 1_000
    max_total_crop_pixels: int = 100_000_000
    max_document_words: int = 100_000
    max_document_lines: int = 100_000

    def __post_init__(self) -> None:
        for name, maximum in (
            ("max_total_regions", 10_000),
            ("max_total_crop_pixels", 200_000_000),
            ("max_document_words", 100_000),
            ("max_document_lines", 100_000),
        ):
            _integer(getattr(self, name), name, 1, maximum)


@dataclass(frozen=True, slots=True)
class OcrRegionResult:
    """One crop result and its source-frame geometry; no raster buffer retained."""

    source: OcrPageInfo
    plan_digest: str
    region: OcrRegion
    recognition: OcrPageResult

    def __post_init__(self) -> None:
        if not isinstance(self.source, OcrPageInfo) or not isinstance(self.region, OcrRegion):
            raise ValueError("region result requires source and region identities")
        if not isinstance(self.recognition, OcrPageResult):
            raise ValueError("region result requires a complete typed recognition")
        if self.recognition.structured_text_bytes > MAX_RESULT_BYTES:
            raise ValueError("OCR region result exceeds its serialized byte limit")
        _hash(self.plan_digest)
        crop = self.recognition.page
        box = self.region.box
        if (
            (crop.source_sha256, crop.source_type, crop.page_index)
            != (self.source.source_sha256, self.source.source_type, self.source.page_index)
            or (crop.width, crop.height) != (box.width, box.height)
            or crop.renderer != f"{CROP_RENDERER_VERSION}:{self.plan_digest}:{self.region.index}"
            or box.right > self.source.width
            or box.bottom > self.source.height
        ):
            raise ValueError("region recognition does not match its source crop")
        # Recheck translated bounds against the original page, not crop geometry.
        _ = self.source_result
        _canonical(self.to_dict())

    @property
    def region_id(self) -> str:
        return _digest(
            {
                "source": self.source.to_dict(),
                "plan_digest": self.plan_digest,
                "region": self.region.to_dict(),
                "crop_page": self.recognition.page.to_dict(),
            }
        )

    @property
    def source_result(self) -> OcrPageResult:
        left, top = self.region.box.left, self.region.box.top
        return OcrPageResult(
            self.source,
            self.recognition.provider,
            tuple(
                OcrLine(
                    line.text,
                    tuple(
                        OcrWord(
                            word.text,
                            (word.box[0] + left, word.box[1] + top, *word.box[2:]),
                        )
                        for word in line.words
                    ),
                )
                for line in self.recognition.lines
            ),
            self.recognition.text_angle,
        )

    @property
    def text(self) -> str:
        return self.recognition.text

    @property
    def source_lines(self) -> tuple[OcrLine, ...]:
        return self.source_result.lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "metricguard.ocr-region-result.v1",
            "region_id": self.region_id,
            "source": self.source.to_dict(),
            "plan_digest": self.plan_digest,
            "region": self.region.to_dict(),
            "recognition": self.recognition.to_dict(),
            "source_lines": [line.to_dict() for line in self.source_result.lines],
        }

    @classmethod
    def from_dict(cls, value: Any) -> OcrRegionResult:
        data = _closed(
            value,
            {
                "format",
                "region_id",
                "source",
                "plan_digest",
                "region",
                "recognition",
                "source_lines",
            },
            "region result",
        )
        _canonical(data)
        result = cls(
            _page_info(data["source"]),
            data["plan_digest"],
            OcrRegion.from_dict(data["region"]),
            _recognition(data["recognition"]),
        )
        if _canonical(result.to_dict()) != _canonical(data):
            raise ValueError("region result contains inconsistent identity or geometry")
        return result


@dataclass(frozen=True, slots=True)
class OcrRegionPageResult:
    """All planned regions, including explicit all-page omission, in plan order."""

    plan: OcrRegionPlan
    provider: OcrProviderIdentity
    regions: tuple[OcrRegionResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, OcrRegionPlan) or not isinstance(
            self.provider, OcrProviderIdentity
        ):
            raise ValueError("region page requires validated plan and provider identities")
        regions = _items(self.regions, 1024, "page regions")
        plan_digest = self.plan.digest
        if len(regions) != len(self.plan.regions) or any(
            not isinstance(result, OcrRegionResult)
            or result.source != self.plan.source
            or result.plan_digest != plan_digest
            or result.region != planned
            or result.recognition.provider != self.provider
            for result, planned in zip(regions, self.plan.regions, strict=True)
        ):
            raise ValueError("page results must exactly complete their ordered region plan")
        object.__setattr__(self, "regions", regions)
        _check_geometry_inventory(regions)
        if self.structured_text_bytes > MAX_RESULT_BYTES:
            raise ValueError("OCR region page exceeds its serialized byte limit")
        _canonical(self.to_dict())

    @property
    def page_id(self) -> str:
        return self.plan.digest

    @property
    def text(self) -> str:
        return REGION_SEPARATOR.join(region.text for region in self.regions)

    @property
    def status(self) -> str:
        if not self.regions:
            return "omitted_all"
        return (
            "recognized"
            if any(region.recognition.lines for region in self.regions)
            else "recognized_empty"
        )

    @property
    def structured_text_bytes(self) -> int:
        return sum(region.recognition.structured_text_bytes for region in self.regions) + max(
            0, len(self.regions) - 1
        ) * len(REGION_SEPARATOR.encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        offset = 0
        spans = []
        for region in self.regions:
            spans.append(
                {
                    "region_index": region.region.index,
                    "region_id": region.region_id,
                    "start": offset,
                    "end": offset + len(region.text),
                }
            )
            offset += len(region.text) + len(REGION_SEPARATOR)
        return {
            "format": "metricguard.ocr-region-page.v1",
            "page_id": self.page_id,
            "status": self.status,
            "plan": self.plan.to_dict(),
            "provider": self.provider.to_dict(),
            "text": self.text,
            "separator": REGION_SEPARATOR,
            "region_spans": spans,
            "regions": [region.to_dict() for region in self.regions],
        }

    @classmethod
    def from_dict(cls, value: Any) -> OcrRegionPageResult:
        data = _closed(
            value,
            {
                "format",
                "page_id",
                "status",
                "plan",
                "provider",
                "text",
                "separator",
                "region_spans",
                "regions",
            },
            "region page",
        )
        _canonical(data)
        result = cls(
            OcrRegionPlan.from_dict(data["plan"]),
            _provider(data["provider"]),
            tuple(
                OcrRegionResult.from_dict(item) for item in _items(data["regions"], 1024, "regions")
            ),
        )
        if _canonical(result.to_dict()) != _canonical(data):
            raise ValueError("region page contains inconsistent derived fields")
        return result


@dataclass(frozen=True, slots=True)
class OcrRegionDocumentResult:
    """Successful recognition of every planned crop, not proof of full coverage."""

    pages: tuple[OcrRegionPageResult, ...]

    def __post_init__(self) -> None:
        pages = _items(self.pages, 1000, "document pages")
        if not pages or any(not isinstance(page, OcrRegionPageResult) for page in pages):
            raise ValueError("region document requires at least one complete page")
        if tuple(page.plan.source.page_index for page in pages) != tuple(range(len(pages))):
            raise ValueError("region document pages must be contiguous and in source order")
        if (
            len(
                {
                    (
                        page.plan.source.source_sha256,
                        page.plan.source.source_type,
                        page.plan.source.renderer,
                        page.provider,
                    )
                    for page in pages
                }
            )
            != 1
        ):
            raise ValueError("region document pages must share source, renderer and provider")
        object.__setattr__(self, "pages", pages)
        _check_geometry_inventory(region for page in pages for region in page.regions)
        counts = self.counters
        if counts["structured_text_bytes"] > MAX_RESULT_BYTES:
            raise ValueError("OCR region document exceeds its serialized byte limit")
        if (
            counts["source_pixels"] > 200_000_000
            or counts["crop_pixels"] > 200_000_000
            or counts["planned_regions"] > 10_000
            or counts["words"] > 100_000
            or counts["lines"] > 100_000
            or counts["structured_text_bytes"] > 64 * 1024 * 1024
        ):
            raise ValueError("OCR region document exceeds aggregate hard limits")
        _canonical(self.to_dict())

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(page.text for page in self.pages)

    @property
    def provider(self) -> OcrProviderIdentity:
        return self.pages[0].provider

    @property
    def counters(self) -> dict[str, int]:
        regions = tuple(region for page in self.pages for region in page.regions)
        return {
            "pages": len(self.pages),
            "planned_regions": len(regions),
            "recognition_calls": len(regions),
            "completed_regions": len(regions),
            "source_pixels": sum(
                page.plan.source.width * page.plan.source.height for page in self.pages
            ),
            "crop_pixels": sum(page.plan.crop_total_pixels for page in self.pages),
            "uncovered_pixels": sum(page.plan.uncovered_pixels for page in self.pages),
            "lines": sum(len(region.recognition.lines) for region in regions),
            "words": sum(
                len(line.words) for region in regions for line in region.recognition.lines
            ),
            "structured_text_bytes": sum(page.structured_text_bytes for page in self.pages)
            + (len(self.pages) - 1) * len(PAGE_SEPARATOR.encode("utf-8")),
            "detector_blank_pages": sum(page.plan.reason == "blank" for page in self.pages),
            "recognized_empty_pages": sum(page.status == "recognized_empty" for page in self.pages),
            "omitted_all_pages": sum(page.status == "omitted_all" for page in self.pages),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        offset = 0
        spans = []
        for page in self.pages:
            spans.append(
                {
                    "page_index": page.plan.source.page_index,
                    "page_id": page.page_id,
                    "start": offset,
                    "end": offset + len(page.text),
                }
            )
            offset += len(page.text) + len(PAGE_SEPARATOR)
        counts = self.counters
        return {
            "format": "metricguard.ocr-region-document.v1",
            "status": "complete",
            "coverage": "complete"
            if not counts["uncovered_pixels"]
            else (
                "omitted_all"
                if counts["uncovered_pixels"] == counts["source_pixels"]
                else "partial"
            ),
            "source_sha256": self.pages[0].plan.source.source_sha256,
            "provider": self.pages[0].provider.to_dict(),
            "text": self.text,
            "separator": PAGE_SEPARATOR,
            "page_spans": spans,
            "counters": counts,
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, value: Any) -> OcrRegionDocumentResult:
        data = _closed(
            value,
            {
                "format",
                "status",
                "coverage",
                "source_sha256",
                "provider",
                "text",
                "separator",
                "page_spans",
                "counters",
                "pages",
            },
            "region document",
        )
        _canonical(data)
        result = cls(
            tuple(
                OcrRegionPageResult.from_dict(item) for item in _items(data["pages"], 1000, "pages")
            )
        )
        if _canonical(result.to_dict()) != _canonical(data):
            raise ValueError("region document contains inconsistent derived fields")
        return result


def transcribe_region_pages(
    pages: Iterable[OcrPageImage],
    backend: PageOcrBackend,
    *,
    planner: RegionPlanner | None = None,
    limits: OcrInputLimits | None = None,
    region_limits: OcrRegionLimits | None = None,
    aggregate_limits: OcrRegionPipelineLimits | None = None,
    max_document_text_bytes: int = 16 * 1024 * 1024,
) -> OcrRegionDocumentResult:
    """Recognize a contiguous RGB page stream, admitting each whole plan first.

    The caller owns and closes a supplied iterator. Earlier recognizer calls
    cannot be rolled back if a later page fails. No partial result is returned.
    """
    active = OcrInputLimits() if limits is None else limits
    local = OcrRegionLimits() if region_limits is None else region_limits
    aggregate = OcrRegionPipelineLimits() if aggregate_limits is None else aggregate_limits
    if (
        not isinstance(active, OcrInputLimits)
        or not isinstance(local, OcrRegionLimits)
        or not isinstance(aggregate, OcrRegionPipelineLimits)
    ):
        raise ValueError("invalid input, region or aggregate limits")
    _integer(max_document_text_bytes, "document text limit", 1, 64 * 1024 * 1024)
    if planner is not None and not callable(planner):
        raise ValueError("planner must be callable")
    identity = getattr(backend, "identity", None)
    if not isinstance(identity, OcrProviderIdentity) or not callable(
        getattr(backend, "recognize", None)
    ):
        raise ValueError("OCR backend must expose an immutable provider identity")
    results: list[OcrRegionPageResult] = []
    source_key = None
    total_pixels = total_regions = crop_pixels = text_bytes = words = lines = 0
    for index, page in enumerate(pages):
        if not isinstance(page, OcrPageImage) or page.page_index != index:
            raise ValueError("OCR pages must be typed, contiguous and in source order")
        if index >= active.max_pages:
            raise ValueError("OCR document exceeds its page limit")
        key = (page.source_sha256, page.source_type, page.renderer)
        if source_key is not None and key != source_key:
            raise ValueError("OCR pages changed source or renderer identity")
        source_key = key
        total_pixels += active.check_size(page.width, page.height)
        if total_pixels > active.max_total_pixels:
            raise ValueError("OCR document exceeds its total pixel limit")
        if backend.identity != identity:
            raise ValueError("OCR backend identity changed before planning")
        source_info = page.info
        plan = detect_vertical_regions(page, limits=local) if planner is None else planner(page)
        if not isinstance(plan, OcrRegionPlan) or plan.source != source_info:
            raise ValueError("OCR planner returned a plan for a different source page")
        plan_digest = plan.digest
        plan.validate_limits(local)
        total_regions += len(plan.regions)
        crop_pixels += plan.crop_total_pixels
        if (
            total_regions > aggregate.max_total_regions
            or crop_pixels > aggregate.max_total_crop_pixels
        ):
            raise ValueError("OCR document exceeds its aggregate region or crop pixel limit")
        # Separator costs are known before recognition, even for empty pages.
        text_bytes += (len(PAGE_SEPARATOR) if results else 0) + max(0, len(plan.regions) - 1) * len(
            REGION_SEPARATOR
        )
        if text_bytes > max_document_text_bytes:
            raise ValueError("OCR document exceeds its text byte limit")
        if backend.identity != identity:
            raise ValueError("OCR backend identity changed during planning")
        recognized = []
        with closing(iter_region_crops(page, plan, limits=local)) as crops:
            for crop in crops:
                if backend.identity != identity:
                    raise ValueError("OCR backend identity changed before recognition")
                result = backend.recognize(crop.image)
                if not isinstance(result, OcrPageResult):
                    raise ValueError("OCR backend must return a complete typed crop result")
                mapped = map_region_result(crop, result)
                if result.provider != identity or backend.identity != identity:
                    raise ValueError("OCR backend identity changed during recognition")
                text_bytes += result.structured_text_bytes
                words += sum(len(line.words) for line in result.lines)
                lines += len(result.lines)
                if (
                    text_bytes > max_document_text_bytes
                    or words > aggregate.max_document_words
                    or lines > aggregate.max_document_lines
                ):
                    raise ValueError("OCR document exceeds its text, word or line limit")
                region_result = OcrRegionResult(source_info, plan_digest, crop.region, result)
                if region_result.source_result.lines != mapped:
                    raise ValueError("OCR source-coordinate mapping is inconsistent")
                recognized.append(region_result)
        if backend.identity != identity:
            raise ValueError("OCR backend identity changed after page recognition")
        results.append(OcrRegionPageResult(plan, identity, tuple(recognized)))
    if backend.identity != identity:
        raise ValueError("OCR backend identity changed after document recognition")
    return OcrRegionDocumentResult(tuple(results))


def transcribe_region_document(
    path: str | Path,
    backend: PageOcrBackend,
    *,
    planner: RegionPlanner | None = None,
    limits: OcrInputLimits | None = None,
    region_limits: OcrRegionLimits | None = None,
    aggregate_limits: OcrRegionPipelineLimits | None = None,
    max_document_text_bytes: int = 16 * 1024 * 1024,
) -> OcrRegionDocumentResult:
    """Decode one image/multiframe image/PDF and recognize its region plans."""
    with closing(load_ocr_pages(path, limits=limits)) as pages:
        return transcribe_region_pages(
            pages,
            backend,
            planner=planner,
            limits=limits,
            region_limits=region_limits,
            aggregate_limits=aggregate_limits,
            max_document_text_bytes=max_document_text_bytes,
        )
