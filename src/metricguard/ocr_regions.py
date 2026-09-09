"""Pixel-bound region plans and a narrow, gold-free vertical-whitespace heuristic.

Coordinates are integer, half-open source-raster coordinates. No resizing,
deskewing, learned detection, semantic layout, or reading-order inference occurs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, cast

from .ocr_types import OcrLine, OcrPageImage, OcrPageInfo, OcrPageResult, OcrWord

_FORMAT = "metricguard.ocr-region-plan.v1"
CROP_RENDERER_VERSION = "metricguard.rgb-crop.v1"
_MAX_PLAN_BYTES = 1024 * 1024


def _integer(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("region plan must contain valid JSON values") from error


def _closed(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"invalid {name} fields")
    return value


@dataclass(frozen=True, slots=True)
class PixelBox:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            _integer(getattr(self, name), name, 0, 10_000)
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("pixel box must have positive width and height")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in ("left", "top", "right", "bottom")}

    @classmethod
    def from_dict(cls, value: Any) -> PixelBox:
        obj = _closed(value, {"left", "top", "right", "bottom"}, "pixel box")
        return cls(obj["left"], obj["top"], obj["right"], obj["bottom"])


@dataclass(frozen=True, slots=True)
class OcrRegion:
    index: int
    box: PixelBox

    def __post_init__(self) -> None:
        _integer(self.index, "region index", 0, 1023)
        if not isinstance(self.box, PixelBox):
            raise ValueError("region box must be a PixelBox")

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "box": self.box.to_dict()}

    @classmethod
    def from_dict(cls, value: Any) -> OcrRegion:
        obj = _closed(value, {"index", "box"}, "region")
        return cls(obj["index"], PixelBox.from_dict(obj["box"]))


@dataclass(frozen=True, slots=True)
class OcrRegionLimits:
    """Per-plan budgets; a consuming document pipeline must also budget its total."""

    max_regions: int = 64
    max_crop_total_pixels: int = 100_000_000

    def __post_init__(self) -> None:
        _integer(self.max_regions, "max_regions", 1, 1024)
        _integer(self.max_crop_total_pixels, "max_crop_total_pixels", 1, 100_000_000)


def _limits(value: OcrRegionLimits | None) -> OcrRegionLimits:
    if value is None:
        return OcrRegionLimits()
    if not isinstance(value, OcrRegionLimits):
        raise ValueError("limits must be OcrRegionLimits")
    return value


def _order(box: PixelBox, direction: str) -> tuple[int, int, int, int]:
    if direction == "ltr":
        return box.left, box.top, box.right, box.bottom
    return -box.right, box.top, -box.left, box.bottom


@dataclass(frozen=True, slots=True)
class OcrRegionPlan:
    """An explicit ordered selection, not an assertion that omitted pixels are blank.

    Order is column-major: left edge then top for LTR; descending right edge then
    top for RTL. Arbitrary semantic/manual reading order is outside v1's contract.
    """

    source: OcrPageInfo
    regions: tuple[OcrRegion, ...]
    direction: str = "ltr"
    method: str = "explicit.v1"
    reason: str = "explicit"
    allow_omissions: bool = False
    white_threshold: int | None = None
    min_gutter_width: int | None = None
    min_region_width: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, OcrPageInfo):
            raise ValueError("plan source must be OcrPageInfo")
        if not isinstance(self.regions, (tuple, list)) or len(self.regions) > 1024:
            raise ValueError("plan regions must contain at most 1024 regions")
        if any(not isinstance(region, OcrRegion) for region in self.regions):
            raise ValueError("plan regions must be OcrRegion values")
        object.__setattr__(self, "regions", tuple(self.regions))
        if self.direction not in ("ltr", "rtl") or type(self.allow_omissions) is not bool:
            raise ValueError("direction must be ltr/rtl and allow_omissions must be boolean")
        if self.method == "explicit.v1":
            if self.reason != "explicit" or any(
                value is not None
                for value in (self.white_threshold, self.min_gutter_width, self.min_region_width)
            ):
                raise ValueError("explicit plans cannot claim detector settings or outcomes")
        elif self.method == "vertical-whitespace.v1":
            _integer(self.white_threshold, "white_threshold", 1, 255)
            _integer(self.min_gutter_width, "min_gutter_width", 1, 10_000)
            _integer(self.min_region_width, "min_region_width", 1, 10_000)
            if self.allow_omissions or self.reason not in ("blank", "no_split", "split"):
                raise ValueError("vertical plans require a declared non-omitting detector outcome")
            if not self.regions or (len(self.regions) > 1) != (self.reason == "split"):
                raise ValueError("detector outcome disagrees with its region count")
        else:
            raise ValueError("unsupported region planning method")
        if tuple(region.index for region in self.regions) != tuple(range(len(self.regions))):
            raise ValueError("region indices must be contiguous in declared order")
        keys = [_order(region.box, self.direction) for region in self.regions]
        if keys != sorted(keys):
            raise ValueError("regions are not in declared column-major order")
        for i, region in enumerate(self.regions):
            box = region.box
            if box.right > self.source.width or box.bottom > self.source.height:
                raise ValueError("region extends outside its source page")
            if self.method == "vertical-whitespace.v1" and (
                box.top != 0 or box.bottom != self.source.height
            ):
                raise ValueError("vertical regions must span the source height")
            for previous in self.regions[:i]:
                other = previous.box
                if max(box.left, other.left) < min(box.right, other.right) and max(
                    box.top, other.top
                ) < min(box.bottom, other.bottom):
                    raise ValueError("regions must not overlap")
        if self.uncovered_pixels and not self.allow_omissions:
            raise ValueError("uncovered source pixels require explicit allow_omissions=True")
        if len(_canonical(self.to_dict())) > _MAX_PLAN_BYTES:
            raise ValueError("region plan exceeds 1 MiB")

    @property
    def crop_total_pixels(self) -> int:
        return sum(region.box.area for region in self.regions)

    @property
    def uncovered_pixels(self) -> int:
        return self.source.width * self.source.height - self.crop_total_pixels

    def validate_limits(self, limits: OcrRegionLimits | None = None) -> None:
        budget = _limits(limits)
        if len(self.regions) > budget.max_regions:
            raise ValueError("region count exceeds max_regions")
        if self.crop_total_pixels > budget.max_crop_total_pixels:
            raise ValueError("crop pixels exceed max_crop_total_pixels")

    def _body(self) -> dict[str, Any]:
        return {
            "format": _FORMAT,
            "source": self.source.to_dict(),
            "regions": [region.to_dict() for region in self.regions],
            "direction": self.direction,
            "method": self.method,
            "reason": self.reason,
            "allow_omissions": self.allow_omissions,
            "white_threshold": self.white_threshold,
            "min_gutter_width": self.min_gutter_width,
            "min_region_width": self.min_region_width,
            "uncovered_pixels": self.uncovered_pixels,
            "crop_total_pixels": self.crop_total_pixels,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(self._body())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value: Any) -> OcrRegionPlan:
        obj = _closed(
            value,
            {
                "format",
                "source",
                "regions",
                "direction",
                "method",
                "reason",
                "allow_omissions",
                "white_threshold",
                "min_gutter_width",
                "min_region_width",
                "uncovered_pixels",
                "crop_total_pixels",
                "digest",
            },
            "region plan",
        )
        if len(_canonical(obj)) > _MAX_PLAN_BYTES:
            raise ValueError("region plan exceeds 1 MiB")
        if obj["format"] != _FORMAT or not isinstance(obj["regions"], (tuple, list)):
            raise ValueError("invalid region plan format or regions")
        if len(obj["regions"]) > 1024:
            raise ValueError("plan regions must contain at most 1024 regions")
        source = _closed(obj["source"], set(OcrPageInfo.__dataclass_fields__), "source page")
        result = cls(
            OcrPageInfo(**source),
            tuple(OcrRegion.from_dict(item) for item in obj["regions"]),
            obj["direction"],
            obj["method"],
            obj["reason"],
            obj["allow_omissions"],
            obj["white_threshold"],
            obj["min_gutter_width"],
            obj["min_region_width"],
        )
        # Canonical equality also rejects bool/int substitution in derived counters.
        if _canonical(result.to_dict()) != _canonical(obj):
            raise ValueError("region plan identity or derived coverage mismatch")
        return result


def explicit_region_plan(
    page: OcrPageImage,
    boxes: Sequence[PixelBox],
    *,
    direction: str = "ltr",
    allow_omissions: bool = False,
    limits: OcrRegionLimits | None = None,
) -> OcrRegionPlan:
    """Validate user-selected boxes without silently reordering or filling holes."""
    if not isinstance(page, OcrPageImage):
        raise ValueError("page must be OcrPageImage")
    if not isinstance(boxes, (tuple, list)) or len(boxes) > 1024:
        raise ValueError("boxes must be a bounded list or tuple of PixelBox values")
    budget = _limits(limits)
    if len(boxes) > budget.max_regions:
        raise ValueError("region count exceeds max_regions")
    plan = OcrRegionPlan(
        page.info,
        tuple(OcrRegion(i, box) for i, box in enumerate(boxes)),
        direction=direction,
        allow_omissions=allow_omissions,
    )
    plan.validate_limits(budget)
    return plan


def _vertical_boxes(
    page: OcrPageImage, threshold: int, gutter: int, minimum: int, direction: str
) -> tuple[tuple[PixelBox, ...], str]:
    # A column is background only when EVERY channel of EVERY pixel meets the
    # threshold. No OCR text, reference transcript, or model output enters here.
    ink = bytearray(page.width)
    pixels = page.pixels
    for y in range(page.height):
        row = y * page.width * 3
        for x in range(page.width):
            if not ink[x]:
                offset = row + x * 3
                if (
                    pixels[offset] < threshold
                    or pixels[offset + 1] < threshold
                    or pixels[offset + 2] < threshold
                ):
                    ink[x] = 1
    occupied = [x for x, value in enumerate(ink) if value]
    if not occupied:
        return (PixelBox(0, 0, page.width, page.height),), "blank"
    cuts = [0]
    x = occupied[0] + 1
    last = occupied[-1]
    while x < last:
        if ink[x]:
            x += 1
            continue
        start = x
        while x < last and not ink[x]:
            x += 1
        midpoint = (start + x) // 2
        if (
            x - start >= gutter
            and midpoint - cuts[-1] >= minimum
            and page.width - midpoint >= minimum
        ):
            cuts.append(midpoint)
    cuts.append(page.width)
    boxes = tuple(PixelBox(left, 0, right, page.height) for left, right in pairwise(cuts))
    if direction == "rtl":
        boxes = tuple(reversed(boxes))
    return boxes, "split" if len(boxes) > 1 else "no_split"


def detect_vertical_regions(
    page: OcrPageImage,
    *,
    direction: str = "ltr",
    white_threshold: int = 255,
    min_gutter_width: int = 8,
    min_region_width: int = 8,
    limits: OcrRegionLimits | None = None,
) -> OcrRegionPlan:
    """Partition all pixels at qualifying full-height background-column midpoints.

    Narrow cuts are ignored left-to-right; pixels are never discarded. The
    whole page is retained when blank or no valid split exists. LTR/RTL affects
    only strip output order, not segmentation or the native order within a crop.
    """
    if not isinstance(page, OcrPageImage):
        raise ValueError("page must be OcrPageImage")
    if direction not in ("ltr", "rtl"):
        raise ValueError("direction must be ltr or rtl")
    _integer(white_threshold, "white_threshold", 1, 255)
    _integer(min_gutter_width, "min_gutter_width", 1, 10_000)
    _integer(min_region_width, "min_region_width", 1, 10_000)
    budget = _limits(limits)
    if page.width * page.height > budget.max_crop_total_pixels:
        raise ValueError("crop pixels exceed max_crop_total_pixels")
    boxes, reason = _vertical_boxes(
        page, white_threshold, min_gutter_width, min_region_width, direction
    )
    if len(boxes) > budget.max_regions:
        raise ValueError("region count exceeds max_regions")
    plan = OcrRegionPlan(
        page.info,
        tuple(OcrRegion(i, box) for i, box in enumerate(boxes)),
        direction,
        "vertical-whitespace.v1",
        reason,
        False,
        white_threshold,
        min_gutter_width,
        min_region_width,
    )
    plan.validate_limits(budget)
    return plan


@dataclass(frozen=True, slots=True)
class OcrRegionCrop:
    """Source and selected region stay separate from the backend's crop raster."""

    source: OcrPageInfo
    plan_digest: str
    region: OcrRegion
    image: OcrPageImage = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, OcrPageInfo) or not isinstance(self.region, OcrRegion):
            raise ValueError("crop requires source and region identities")
        if (
            not isinstance(self.plan_digest, str)
            or len(self.plan_digest) != 64
            or any(char not in "0123456789abcdef" for char in self.plan_digest)
        ):
            raise ValueError("crop plan_digest must be lowercase SHA-256")
        if not isinstance(self.image, OcrPageImage):
            raise ValueError("crop image must be OcrPageImage")
        box = self.region.box
        if box.right > self.source.width or box.bottom > self.source.height:
            raise ValueError("crop region extends outside its source page")
        if (
            (self.image.width, self.image.height) != (box.width, box.height)
            or self.image.page_index != self.source.page_index
            or self.image.source_sha256 != self.source.source_sha256
            or self.image.source_type != self.source.source_type
            or self.image.renderer
            != f"{CROP_RENDERER_VERSION}:{self.plan_digest}:{self.region.index}"
        ):
            raise ValueError("crop raster metadata disagrees with source/region identity")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "source": self.source.to_dict(),
                    "plan_digest": self.plan_digest,
                    "region": self.region.to_dict(),
                    "crop": self.image.info.to_dict(),
                }
            )
        ).hexdigest()


def _validate_page_plan(
    page: OcrPageImage, plan: OcrRegionPlan, limits: OcrRegionLimits | None
) -> None:
    if not isinstance(page, OcrPageImage) or not isinstance(plan, OcrRegionPlan):
        raise ValueError("crop requires a validated page and plan")
    plan.validate_limits(limits)
    if page.info != plan.source:
        raise ValueError("stale region plan: source page identity mismatch")
    if plan.method == "vertical-whitespace.v1":
        # The constructor checked these integers. Recompute declared detector
        # semantics before consuming a deserialized plan, not merely its hash.
        boxes, reason = _vertical_boxes(
            page,
            cast(int, plan.white_threshold),
            cast(int, plan.min_gutter_width),
            cast(int, plan.min_region_width),
            plan.direction,
        )
        if boxes != tuple(region.box for region in plan.regions) or reason != plan.reason:
            raise ValueError("region plan does not match its declared detector output")


def _crop(page: OcrPageImage, plan: OcrRegionPlan, region: OcrRegion) -> OcrRegionCrop:
    box = region.box
    row_bytes = box.width * 3
    pixels = bytearray(box.area * 3)
    for y in range(box.height):
        start = ((box.top + y) * page.width + box.left) * 3
        pixels[y * row_bytes : (y + 1) * row_bytes] = page.pixels[start : start + row_bytes]
    image = OcrPageImage(
        page.page_index,
        page.source_sha256,
        page.source_type,
        box.width,
        box.height,
        bytes(pixels),
        f"{CROP_RENDERER_VERSION}:{plan.digest}:{region.index}",
    )
    return OcrRegionCrop(plan.source, plan.digest, region, image)


def iter_region_crops(
    page: OcrPageImage, plan: OcrRegionPlan, *, limits: OcrRegionLimits | None = None
) -> Generator[OcrRegionCrop, None, None]:
    """Validate once on first iteration, then materialize one exact RGB crop at a time."""
    _validate_page_plan(page, plan, limits)
    for region in plan.regions:
        yield _crop(page, plan, region)


def crop_ocr_region(
    page: OcrPageImage,
    plan: OcrRegionPlan,
    index: int,
    *,
    limits: OcrRegionLimits | None = None,
) -> OcrRegionCrop:
    _validate_page_plan(page, plan, limits)
    _integer(index, "region index", 0, len(plan.regions) - 1)
    return _crop(page, plan, plan.regions[index])


def map_region_result(crop: OcrRegionCrop, result: OcrPageResult) -> tuple[OcrLine, ...]:
    """Translate crop-local boxes without flattening a region into a full-page result."""
    if not isinstance(crop, OcrRegionCrop) or not isinstance(result, OcrPageResult):
        raise ValueError("mapping requires OcrRegionCrop and OcrPageResult")
    if result.page != crop.image.info:
        raise ValueError("OCR result does not match the exact region crop")
    left, top = crop.region.box.left, crop.region.box.top
    return tuple(
        OcrLine(
            line.text,
            tuple(
                OcrWord(word.text, (word.box[0] + left, word.box[1] + top, *word.box[2:]))
                for word in line.words
            ),
        )
        for line in result.lines
    )
