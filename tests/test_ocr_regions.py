"""Independent pixel/geometry oracles, not OCR accuracy tests."""

import hashlib
import json
import struct
from dataclasses import FrozenInstanceError, replace

import pytest

from metricguard.ocr_regions import (
    OcrRegion,
    OcrRegionCrop,
    OcrRegionLimits,
    OcrRegionPlan,
    PixelBox,
    crop_ocr_region,
    detect_vertical_regions,
    explicit_region_plan,
    iter_region_crops,
    map_region_result,
)
from metricguard.ocr_types import OcrLine, OcrPageImage, OcrPageResult, OcrProviderIdentity, OcrWord


def page(width=12, height=3, occupied=(1, 2, 9, 10)):
    pixels = bytes(
        channel
        for _y in range(height)
        for x in range(width)
        for channel in ((0, 0, 0) if x in occupied else (255, 255, 255))
    )
    return OcrPageImage(7, "a" * 64, "pdf", width, height, pixels, "authored-rgb.v1")


def detect(image, **kwargs):
    return detect_vertical_regions(image, min_gutter_width=2, min_region_width=2, **kwargs)


def provider():
    return OcrProviderIdentity("authored", "1", "en-US", "b" * 64, "c" * 64)


def test_midpoint_partition_preserves_every_source_pixel():
    image = page()
    plan = detect(image)
    assert [r.box for r in plan.regions] == [PixelBox(0, 0, 6, 3), PixelBox(6, 0, 12, 3)]
    assert plan.reason == "split"
    assert plan.uncovered_pixels == 0 and plan.crop_total_pixels == 36
    crops = list(iter_region_crops(image, plan))
    for y in range(3):
        assert (
            b"".join(c.image.pixels[y * 18 : (y + 1) * 18] for c in crops)
            == image.pixels[y * 36 : (y + 1) * 36]
        )
    assert image.page_index == crops[0].image.page_index == crops[1].image.page_index == 7
    assert crops[0].digest != crops[1].digest
    assert crops[0].source == image.info
    assert crops[0].plan_digest == plan.digest


def test_rtl_reverses_strips_not_pixels_or_native_order():
    image = page()
    ltr, rtl = detect(image), detect(image, direction="rtl")
    assert [r.box for r in rtl.regions] == [r.box for r in reversed(ltr.regions)]
    assert [r.index for r in rtl.regions] == [0, 1]
    assert ltr.digest != rtl.digest
    assert (
        crop_ocr_region(image, ltr, 1).image.pixels == crop_ocr_region(image, rtl, 0).image.pixels
    )


@pytest.mark.parametrize(
    "occupied,reason", [((), "blank"), (tuple(range(12)), "no_split"), ((5,), "no_split")]
)
def test_blank_dark_and_unsplittable_pages_are_never_dropped(occupied, reason):
    image = page(occupied=occupied)
    plan = detect(image)
    assert plan.reason == reason
    assert len(plan.regions) == 1
    assert plan.regions[0].box == PixelBox(0, 0, 12, 3)
    assert crop_ocr_region(image, plan, 0).image.pixels == image.pixels
    assert plan.uncovered_pixels == 0


def test_faint_and_color_content_remain_ink_by_default():
    pixels = bytes([255, 255, 254] * 12)
    image = OcrPageImage(0, "a" * 64, "image", 12, 1, pixels, "test")
    assert detect(image).reason == "no_split"
    loosened = detect(image, white_threshold=254)
    assert loosened.reason == "blank"
    # Even a permissive blank classification cannot skip faint pixels.
    assert crop_ocr_region(image, loosened, 0).image.pixels == pixels


def test_one_dark_bridge_blocks_a_full_height_gutter():
    image = page()
    pixels = image.pixels[:36] + bytes(36) + image.pixels[72:]
    assert detect(replace(image, pixels=pixels)).reason == "no_split"


def test_midpoint_odd_width_tie_is_floor_and_margins_not_cuts():
    image = page(width=11, occupied=(1, 9))
    plan = detect(image)
    assert [r.box for r in plan.regions] == [PixelBox(0, 0, 5, 3), PixelBox(5, 0, 11, 3)]


def test_narrow_cuts_are_ignored_without_omissions():
    image = page(width=12, occupied=(0, 5, 10))
    plan = detect_vertical_regions(image, min_gutter_width=2, min_region_width=4)
    assert [r.box for r in plan.regions] == [PixelBox(0, 0, 8, 3), PixelBox(8, 0, 12, 3)]
    assert plan.crop_total_pixels == 36
    assert (
        detect_vertical_regions(image, min_gutter_width=2, min_region_width=7).reason == "no_split"
    )


def test_exact_rgb_crop_uses_rows_not_contiguous_source_slice():
    pixels = bytes(range(60))
    image = OcrPageImage(2, "a" * 64, "image", 5, 4, pixels, "test-rgb")
    plan = explicit_region_plan(image, [PixelBox(1, 1, 4, 3)], allow_omissions=True)
    crop = crop_ocr_region(image, plan, 0)
    expected = bytes(list(range(18, 27)) + list(range(33, 42)))
    assert crop.image.pixels == expected
    assert (
        crop.image.info.pixel_sha256
        == hashlib.sha256(struct.pack(">II", 3, 2) + expected).hexdigest()
    )
    assert plan.uncovered_pixels == 14
    assert crop.image.renderer == f"metricguard.rgb-crop.v1:{plan.digest}:0"
    assert image.pixels == pixels


def test_mapping_preserves_text_word_order_and_translates_only():
    image = page(width=12, height=8)
    plan = explicit_region_plan(image, [PixelBox(6, 2, 12, 8)], allow_omissions=True)
    crop = crop_ocr_region(image, plan, 0)
    line = OcrLine(
        "é 🏮\r\n{{literal}}", (OcrWord("🏮", (3, 2, 1, 2)), OcrWord("é", (0.5, 1, 1.5, 1)))
    )
    result = OcrPageResult(crop.image.info, provider(), (line,), 12)
    mapped = map_region_result(crop, result)
    assert mapped[0].text == line.text
    assert [w.text for w in mapped[0].words] == ["🏮", "é"]
    assert [w.box for w in mapped[0].words] == [(9, 4, 1, 2), (6.5, 3, 1.5, 1)]
    assert result.text_angle == 12
    assert result.lines == (line,)
    assert map_region_result(crop, replace(result, lines=())) == ()


def test_wrong_crop_result_even_with_same_geometry_is_rejected():
    image = page()
    plan = detect(image)
    left, right = list(iter_region_crops(image, plan))
    assert left.image.width == right.image.width
    result = OcrPageResult(right.image.info, provider(), ())
    with pytest.raises(ValueError, match="exact region crop"):
        map_region_result(left, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("page_index", 8),
        ("source_sha256", "b" * 64),
        ("source_type", "image"),
        ("renderer", "changed"),
    ],
)
def test_entire_source_identity_not_only_pixels_is_pinned(field, value):
    image = page()
    plan = detect(image)
    with pytest.raises(ValueError, match="stale region plan"):
        list(iter_region_crops(replace(image, **{field: value}), plan))


def test_changed_pixels_and_same_byte_count_different_dimensions_rejected():
    image = page()
    plan = detect(image)
    for changed in (
        replace(image, pixels=b"\x01" + image.pixels[1:]),
        replace(image, width=6, height=6),
    ):
        with pytest.raises(ValueError, match="stale region plan"):
            crop_ocr_region(changed, plan, 0)


def test_detector_claim_is_recomputed_not_just_a_self_asserted_digest():
    image = page()
    genuine = detect(image)
    forged = replace(genuine, regions=(OcrRegion(0, PixelBox(0, 0, 12, 3)),), reason="no_split")
    restored = OcrRegionPlan.from_dict(forged.to_dict())
    with pytest.raises(ValueError, match="declared detector output"):
        list(iter_region_crops(image, restored))


@pytest.mark.parametrize(
    "box",
    [
        (True, 0, 2, 2),
        (0, 0.0, 2, 2),
        (-1, 0, 2, 2),
        (0, 0, 10001, 2),
        (0, 0, 10**500, 2),
        (2, 0, 2, 2),
        (0, 2, 2, 1),
    ],
)
def test_invalid_pixel_geometry_is_rejected(box):
    with pytest.raises(ValueError):
        PixelBox(*box)


@pytest.mark.parametrize(
    "boxes,message",
    [
        ([PixelBox(0, 0, 8, 3), PixelBox(7, 0, 12, 3)], "overlap"),
        ([PixelBox(0, 0, 6, 3), PixelBox(0, 0, 6, 3)], "overlap"),
        ([PixelBox(0, 0, 13, 3)], "outside"),
        ([PixelBox(0, 0, 12, 4)], "outside"),
        ([PixelBox(6, 0, 12, 3), PixelBox(0, 0, 6, 3)], "order"),
    ],
)
def test_explicit_plan_rejects_unsafe_geometry_and_order(boxes, message):
    with pytest.raises(ValueError, match=message):
        explicit_region_plan(page(), boxes, allow_omissions=True)


def test_touching_edges_four_rectangles_are_valid_column_major():
    image = page(height=4)
    boxes = [
        PixelBox(0, 0, 6, 2),
        PixelBox(0, 2, 6, 4),
        PixelBox(6, 0, 12, 2),
        PixelBox(6, 2, 12, 4),
    ]
    plan = explicit_region_plan(image, boxes)
    assert plan.uncovered_pixels == 0 and plan.crop_total_pixels == 48
    rtl = explicit_region_plan(image, boxes[2:] + boxes[:2], direction="rtl")
    assert rtl.regions[0].box == boxes[2]
    with pytest.raises(ValueError, match="order"):
        explicit_region_plan(image, [boxes[0], boxes[2], boxes[1], boxes[3]])


def test_omissions_including_entire_page_must_be_explicit():
    image = page()
    for boxes in ([], [PixelBox(1, 0, 12, 3)]):
        with pytest.raises(ValueError, match="uncovered"):
            explicit_region_plan(image, boxes)
    empty = explicit_region_plan(image, [], allow_omissions=True)
    assert empty.uncovered_pixels == 36 and empty.crop_total_pixels == 0
    assert empty.reason == "explicit" and empty.allow_omissions
    assert list(iter_region_crops(image, empty)) == []
    with pytest.raises(ValueError, match="region index"):
        crop_ocr_region(image, empty, 0)


def test_direct_constructor_indices_and_types_are_not_a_validation_bypass():
    image = page()
    with pytest.raises(ValueError, match="contiguous"):
        OcrRegionPlan(image.info, [OcrRegion(1, PixelBox(0, 0, 12, 3))])
    with pytest.raises(ValueError, match="OcrRegion values"):
        OcrRegionPlan(image.info, ["not-a-region"])
    with pytest.raises(ValueError, match="source"):
        OcrRegionPlan(None, [])
    with pytest.raises(ValueError):
        OcrRegion(True, PixelBox(0, 0, 1, 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "auto"},
        {"allow_omissions": 1},
        {"method": "learned"},
        {"white_threshold": 255},
        {"reason": "blank"},
    ],
)
def test_direct_plan_has_closed_policy_semantics(kwargs):
    plan = explicit_region_plan(page(), [PixelBox(0, 0, 12, 3)])
    with pytest.raises(ValueError):
        replace(plan, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"white_threshold": True},
        {"white_threshold": 0},
        {"white_threshold": 256},
        {"min_gutter_width": 0},
        {"min_gutter_width": 1.5},
        {"min_region_width": 10**500},
        {"direction": "automatic"},
        {"limits": {}},
    ],
)
def test_detector_configuration_rejects_bool_nonfinite_and_unbounded(kwargs):
    with pytest.raises(ValueError):
        detect_vertical_regions(page(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_regions": True},
        {"max_regions": 0},
        {"max_regions": 1025},
        {"max_crop_total_pixels": 0},
        {"max_crop_total_pixels": 100_000_001},
        {"max_crop_total_pixels": float("inf")},
    ],
)
def test_budget_types_and_hard_bounds(kwargs):
    with pytest.raises(ValueError):
        OcrRegionLimits(**kwargs)


def test_count_and_crop_budgets_fail_instead_of_truncating():
    image = page()
    plan = detect(image)
    for budget, error in (
        (OcrRegionLimits(max_regions=1), "max_regions"),
        (OcrRegionLimits(max_crop_total_pixels=35), "max_crop_total_pixels"),
    ):
        with pytest.raises(ValueError, match=error):
            detect(image, limits=budget)
        with pytest.raises(ValueError, match=error):
            explicit_region_plan(image, [r.box for r in plan.regions], limits=budget)
        with pytest.raises(ValueError, match=error):
            list(iter_region_crops(image, plan, limits=budget))
    assert len(list(iter_region_crops(image, plan, limits=OcrRegionLimits(2, 36)))) == 2


def test_region_values_roundtrip_and_are_immutable():
    plan = detect(page())
    payload = json.loads(json.dumps(plan.to_dict()))
    restored = OcrRegionPlan.from_dict(payload)
    assert restored == plan
    assert restored.digest == plan.digest
    payload["regions"][0]["box"]["left"] = 1
    assert plan.regions[0].box.left == 0
    with pytest.raises(FrozenInstanceError):
        restored.direction = "rtl"
    assert OcrRegion.from_dict(plan.regions[0].to_dict()) == plan.regions[0]
    assert PixelBox.from_dict(plan.regions[0].box.to_dict()) == plan.regions[0].box


@pytest.mark.parametrize(
    "field,value",
    [
        ("format", "v2"),
        ("digest", "0" * 64),
        ("uncovered_pixels", False),
        ("crop_total_pixels", 36.0),
        ("regions", {}),
        ("direction", "rtl"),
    ],
)
def test_persisted_plan_rejects_rehashed_or_shape_tampering(field, value):
    payload = detect(page()).to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        OcrRegionPlan.from_dict(payload)


@pytest.mark.parametrize("target", ["plan", "source", "region", "box"])
def test_persisted_plan_rejects_unknown_fields_at_every_level(target):
    payload = detect(page()).to_dict()
    chosen = {
        "plan": payload,
        "source": payload["source"],
        "region": payload["regions"][0],
        "box": payload["regions"][0]["box"],
    }[target]
    chosen["ignored"] = "must-not-be-ignored"
    with pytest.raises(ValueError, match="fields"):
        OcrRegionPlan.from_dict(payload)


def test_complete_plan_serialization_budget_includes_source_metadata(monkeypatch):
    import metricguard.ocr_regions as module

    plan = detect(page())
    count = len(
        json.dumps(
            plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    monkeypatch.setattr(module, "_MAX_PLAN_BYTES", count)
    assert OcrRegionPlan.from_dict(plan.to_dict()) == plan
    monkeypatch.setattr(module, "_MAX_PLAN_BYTES", count - 1)
    with pytest.raises(ValueError, match="1 MiB"):
        OcrRegionPlan.from_dict(plan.to_dict())
    with pytest.raises(ValueError, match="1 MiB"):
        replace(plan)


def test_crop_direct_metadata_contract_rejects_invalid_identity():
    image = page()
    crop = crop_ocr_region(image, detect(image), 0)
    for kwargs in (
        {"plan_digest": "wrong"},
        {"image": image},
        {"source": None},
        {"region": OcrRegion(0, PixelBox(0, 0, 13, 3))},
    ):
        with pytest.raises(ValueError):
            replace(crop, **kwargs)
    with pytest.raises(ValueError):
        OcrRegionCrop(crop.source, crop.plan_digest, crop.region, None)


def test_mapping_and_cropping_reject_wrong_public_argument_types():
    image = page()
    plan = detect(image)
    crop = crop_ocr_region(image, plan, 0)
    for bad in (True, -1, 2, 0.0):
        with pytest.raises(ValueError):
            crop_ocr_region(image, plan, bad)
    with pytest.raises(ValueError):
        crop_ocr_region(None, plan, 0)
    with pytest.raises(ValueError):
        map_region_result(crop, None)
    with pytest.raises(ValueError):
        explicit_region_plan(image, iter([]))
    with pytest.raises(ValueError):
        detect_vertical_regions(None)


def test_empty_plan_still_verifies_source_before_zero_crop_success():
    image = page()
    plan = explicit_region_plan(image, [], allow_omissions=True)
    with pytest.raises(ValueError, match="stale"):
        list(iter_region_crops(replace(image, renderer="changed"), plan))
