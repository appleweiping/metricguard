"""Gold-free scripted recognition: independent geometry, Unicode and admission checks."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from metricguard import ocr_region_pipeline as pipeline
from metricguard.ocr_inputs import OcrInputLimits
from metricguard.ocr_region_pipeline import (
    OcrRegionDocumentResult,
    OcrRegionPageResult,
    OcrRegionPipelineLimits,
    OcrRegionResult,
    transcribe_region_document,
    transcribe_region_pages,
)
from metricguard.ocr_regions import (
    OcrRegionLimits,
    OcrRegionPlan,
    PixelBox,
    explicit_region_plan,
)
from metricguard.ocr_types import (
    OcrLine,
    OcrPageImage,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
)

IDENTITY = OcrProviderIdentity("scripted-test", "1", "und", "a" * 64, "b" * 64)


def page(index=0, *, width=12, height=8, blank=False):
    pixels = (
        b"\xff" * width * height * 3
        if blank
        else bytes(channel for y in range(height) for x in range(width) for channel in (x, y, 0))
    )
    return OcrPageImage(index, "c" * 64, "image", width, height, pixels, "authored-rgb/v1")


def halves(image, *, direction="ltr"):
    boxes = (
        PixelBox(0, 0, image.width // 2, image.height),
        PixelBox(image.width // 2, 0, image.width, image.height),
    )
    return explicit_region_plan(
        image,
        boxes if direction == "ltr" else boxes[::-1],
        direction=direction,
    )


def recognized(image, text="sample", *, identity=IDENTITY):
    lines = () if text is None else (OcrLine(text, (OcrWord("native token", (0.5, 1, 2, 3)),)),)
    return OcrPageResult(image.info, identity, lines, text_angle=0.0)


class Backend:
    def __init__(self, callback=None):
        self.identity = IDENTITY
        self.calls = []
        self.callback = callback

    def recognize(self, image):
        self.calls.append(image)
        if self.callback is not None:
            return self.callback(image)
        return recognized(image, f"native result {len(self.calls)}")


def test_multiple_crops_are_real_rgb_copies_with_source_word_coordinates_and_exact_spans():
    originals = (page(), page(1))
    native_texts = iter(("A\U0001f600e\u0301", "second\r\nline", "literal {x}", "final"))
    backend = Backend(lambda image: recognized(image, next(native_texts)))
    result = transcribe_region_pages(originals, backend, planner=halves)
    assert len(backend.calls) == 4
    for index, crop in enumerate(backend.calls):
        original = originals[index // 2]
        left = 0 if index % 2 == 0 else 6
        expected = b"".join(
            original.pixels[(y * 12 + left) * 3 : (y * 12 + left + 6) * 3] for y in range(8)
        )
        assert crop.pixels == expected
        assert crop.width == 6 and crop.height == 8
        assert not hasattr(crop, "reference")
        region = result.pages[index // 2].regions[index % 2]
        assert region.recognition.page == crop.info
        assert region.source_lines[0].words[0].box == (left + 0.5, 1.0, 2.0, 3.0)
        assert region.recognition.lines[0].words[0].box == (0.5, 1.0, 2.0, 3.0)
        assert region.source_result.page == original.info
        assert not hasattr(region, "pixels")
    expected_pages = ("A\U0001f600e\u0301\n\nsecond\r\nline", "literal {x}\n\nfinal")
    assert result.text == "\n\f\n".join(expected_pages)
    exported = result.to_dict()
    for page_value, span in zip(exported["pages"], exported["page_spans"], strict=True):
        assert result.text[span["start"] : span["end"]] == page_value["text"]
        for region, region_span in zip(
            page_value["regions"], page_value["region_spans"], strict=True
        ):
            assert (
                page_value["text"][region_span["start"] : region_span["end"]]
                == region["recognition"]["text"]
            )
    # This is a code-point boundary, not the UTF-8 byte count of the first crop.
    assert exported["pages"][0]["region_spans"][0]["end"] == 4
    assert exported["counters"]["recognition_calls"] == 4
    assert exported["counters"]["source_pixels"] == exported["counters"]["crop_pixels"] == 192
    assert exported["counters"]["lines"] == exported["counters"]["words"] == 4
    expected_text_bytes = (
        sum(
            len(region.text.encode()) + len(b"native token")
            for item in result.pages
            for region in item.regions
        )
        + 2
        + 2
        + 3
    )
    assert result.counters["structured_text_bytes"] == expected_text_bytes
    assert result.provider == IDENTITY and exported["coverage"] == "complete"


def test_declared_rtl_controls_crop_order_without_reordering_native_lines():
    backend = Backend(
        lambda image: OcrPageResult(
            image.info,
            IDENTITY,
            (
                OcrLine("bottom first", (OcrWord("b", (0, 5, 1, 1)),)),
                OcrLine("top last", (OcrWord("t", (0, 0, 1, 1)),)),
            ),
        )
    )
    result = transcribe_region_pages(
        (page(),), backend, planner=lambda image: halves(image, direction="rtl")
    )
    assert [image.pixels[0] for image in backend.calls] == [6, 0]
    assert result.pages[0].regions[0].source_lines[0].words[0].box == (6.0, 5.0, 1.0, 1.0)
    assert result.text == "bottom first\ntop last\n\nbottom first\ntop last"


def test_all_white_detector_keeps_real_full_page_recognition_and_explicit_reason():
    backend = Backend(lambda image: recognized(image, None))
    result = transcribe_region_pages((page(blank=True), page(1, blank=True)), backend)
    assert len(backend.calls) == 2 and all(image.width == 12 for image in backend.calls)
    assert result.text == "\n\f\n"
    assert result.counters["detector_blank_pages"] == 2
    assert result.counters["recognized_empty_pages"] == 2
    assert result.counters["omitted_all_pages"] == 0
    assert result.to_dict()["coverage"] == "complete"
    assert all(item.plan.reason == "blank" for item in result.pages)


def test_explicit_all_page_omission_is_not_claimed_blank_or_recognized():
    backend = Backend()
    result = transcribe_region_pages(
        (page(), page(1)),
        backend,
        planner=lambda image: explicit_region_plan(image, (), allow_omissions=True),
    )
    assert not backend.calls and result.provider == IDENTITY
    assert result.text == "\n\f\n"
    assert result.to_dict()["coverage"] == "omitted_all"
    assert result.counters["omitted_all_pages"] == 2
    assert result.counters["detector_blank_pages"] == result.counters["recognized_empty_pages"] == 0
    assert result.counters["uncovered_pixels"] == 192
    assert all(item.to_dict()["region_spans"] == [] for item in result.pages)


def test_partial_crop_maps_nonzero_xy_and_reports_omitted_area():
    result = transcribe_region_pages(
        (page(),),
        Backend(),
        planner=lambda image: explicit_region_plan(
            image,
            (PixelBox(3, 2, 9, 7),),
            allow_omissions=True,
        ),
    )
    assert result.pages[0].regions[0].source_lines[0].words[0].box == (3.5, 3.0, 2.0, 3.0)
    assert result.to_dict()["coverage"] == "partial"
    assert result.counters["crop_pixels"] == 30 and result.counters["uncovered_pixels"] == 66


@pytest.mark.parametrize(
    "limits", [OcrRegionLimits(max_regions=1), OcrRegionLimits(max_crop_total_pixels=95)]
)
def test_complete_plan_budget_is_admitted_before_first_crop_call(limits):
    backend = Backend()
    with pytest.raises(ValueError, match="exceed"):
        transcribe_region_pages((page(),), backend, planner=halves, region_limits=limits)
    assert backend.calls == []


@pytest.mark.parametrize(
    "limits",
    [
        OcrRegionPipelineLimits(max_total_regions=3),
        OcrRegionPipelineLimits(max_total_crop_pixels=191),
    ],
)
def test_global_plan_admission_fails_before_any_call_on_later_page(limits):
    backend = Backend()
    with pytest.raises(ValueError, match="aggregate"):
        transcribe_region_pages((page(), page(1)), backend, planner=halves, aggregate_limits=limits)
    assert len(backend.calls) == 2


@pytest.mark.parametrize(
    "limits", [OcrInputLimits(max_pages=1), OcrInputLimits(max_total_pixels=191)]
)
def test_raster_stream_aggregate_limits_cover_empty_region_pages(limits):
    backend = Backend()
    with pytest.raises(ValueError, match="limit"):
        transcribe_region_pages(
            (page(), page(1)),
            backend,
            limits=limits,
            planner=lambda image: explicit_region_plan(image, (), allow_omissions=True),
        )
    assert not backend.calls


@pytest.mark.parametrize(
    "pages",
    [
        (),
        (page(1),),
        (page(), page()),
        ("not raster",),
        (page(), replace(page(1), source_sha256="d" * 64)),
        (page(), replace(page(1), renderer="different")),
        (page(), replace(page(1), source_type="pdf")),
    ],
)
def test_bad_source_sequence_rejected_even_if_every_region_is_explicitly_omitted(pages):
    backend = Backend()
    with pytest.raises(ValueError):
        transcribe_region_pages(
            pages,
            backend,
            planner=lambda image: explicit_region_plan(image, (), allow_omissions=True),
        )
    assert not backend.calls


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("limits", {}),
        ("region_limits", {}),
        ("aggregate_limits", {}),
        ("max_document_text_bytes", True),
        ("max_document_text_bytes", 0),
        ("max_document_text_bytes", 64 * 1024 * 1024 + 1),
        ("planner", "not callable"),
    ],
)
def test_invalid_options_fail_without_consuming_source_or_calling_provider(keyword, value):
    backend = Backend()
    consumed = []

    def source():
        consumed.append(True)
        yield page()

    with pytest.raises(ValueError):
        transcribe_region_pages(source(), backend, **{keyword: value})
    assert not consumed and not backend.calls


@pytest.mark.parametrize("field", list(OcrRegionPipelineLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 10**400])
def test_aggregate_configuration_rejects_noninteger_and_unbounded_values(field, value):
    with pytest.raises(ValueError):
        OcrRegionPipelineLimits(**{field: value})


def test_text_byte_inventory_exact_boundary_and_separator_preflight():
    image = page()
    first = transcribe_region_pages((image,), Backend(), planner=halves)
    size = first.counters["structured_text_bytes"]
    assert (
        transcribe_region_pages(
            (image,), Backend(), planner=halves, max_document_text_bytes=size
        ).digest
        == first.digest
    )
    with pytest.raises(ValueError, match="text"):
        transcribe_region_pages(
            (image,), Backend(), planner=halves, max_document_text_bytes=size - 1
        )
    backend = Backend()
    with pytest.raises(ValueError, match="text byte"):
        transcribe_region_pages((image,), backend, planner=halves, max_document_text_bytes=1)
    assert not backend.calls


@pytest.mark.parametrize(
    "limits",
    [OcrRegionPipelineLimits(max_document_words=1), OcrRegionPipelineLimits(max_document_lines=1)],
)
def test_output_count_limits_return_no_partial_result(limits):
    backend = Backend()
    with pytest.raises(ValueError, match="word or line"):
        transcribe_region_pages((page(),), backend, planner=halves, aggregate_limits=limits)
    assert len(backend.calls) == 2


@pytest.mark.parametrize(
    "mode",
    [
        "untyped",
        "wrong_source",
        "wrong_pixel",
        "wrong_renderer",
        "wrong_provider",
        "identity_drift",
        "raised",
    ],
)
def test_bad_recognition_never_yields_a_complete_document_or_continues(mode):
    backend = Backend()

    def callback(image):
        result = recognized(image)
        if mode == "raised":
            raise RuntimeError("synthetic local recognizer failure")
        if mode == "untyped":
            return {"status": "partial"}
        if mode == "identity_drift":
            backend.identity = replace(IDENTITY, version="changed")
        elif mode == "wrong_provider":
            result = replace(result, provider=replace(IDENTITY, version="changed"))
        else:
            fields = {
                "wrong_source": {"source_sha256": "d" * 64},
                "wrong_pixel": {"pixel_sha256": "d" * 64},
                "wrong_renderer": {"renderer": "different"},
            }
            if mode in fields:
                result = replace(result, page=replace(result.page, **fields[mode]))
        return result

    backend.callback = callback
    with pytest.raises((ValueError, RuntimeError)):
        transcribe_region_pages((page(),), backend, planner=halves)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("mode", ["identity", "wrong_page", "untyped"])
def test_planning_identity_drift_and_wrong_page_plan_fail_before_recognition(mode):
    backend = Backend()

    def planner(image):
        if mode == "identity":
            backend.identity = replace(IDENTITY, version="changed")
        if mode == "untyped":
            return {}
        return halves(replace(image, source_sha256="d" * 64) if mode == "wrong_page" else image)

    with pytest.raises(ValueError):
        transcribe_region_pages((page(),), backend, planner=planner)
    assert not backend.calls


def test_invalid_provider_identity_fails_before_source_iteration():
    backend = Backend()
    backend.identity = {}
    with pytest.raises(ValueError, match="identity"):
        transcribe_region_pages((), backend)


def test_source_decoder_is_closed_after_mid_document_failure(monkeypatch):
    closed = []

    def decoder(*_args, **_kwargs):
        try:
            yield page()
            yield page(1)
        finally:
            closed.append(True)

    monkeypatch.setattr(pipeline, "load_ocr_pages", decoder)

    def fail(_image):
        raise RuntimeError("fixture")

    with pytest.raises(RuntimeError):
        transcribe_region_document("unused", Backend(fail), planner=halves)
    assert closed == [True]


def test_document_roundtrip_immutability_and_independent_digest():
    result = transcribe_region_pages((page(),), Backend(), planner=halves)
    exported = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    restored = OcrRegionDocumentResult.from_dict(exported)
    assert restored == result
    expected = hashlib.sha256(
        json.dumps(
            exported, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert expected == restored.digest
    exported["pages"][0]["recognition"] = "not part of immutable result"
    assert restored == result
    with pytest.raises(FrozenInstanceError):
        result.pages = ()
    counts = result.counters
    counts["pages"] = 999
    assert result.counters["pages"] == 1


@pytest.mark.parametrize(
    "mode",
    [
        "format",
        "unknown",
        "counter_bool",
        "counter",
        "span",
        "geometry",
        "region_id",
        "native_text",
        "page_status",
        "separator",
    ],
)
def test_serialized_derived_fields_are_recomputed_not_trusted(mode):
    data = copy.deepcopy(transcribe_region_pages((page(),), Backend(), planner=halves).to_dict())
    region = data["pages"][0]["regions"][0]
    if mode == "format":
        data["format"] = "future"
    elif mode == "unknown":
        data["extra"] = True
    elif mode == "counter_bool":
        data["counters"]["pages"] = True
    elif mode == "counter":
        data["counters"]["words"] = 0
    elif mode == "span":
        data["page_spans"][0]["end"] -= 1
    elif mode == "geometry":
        region["source_lines"][0]["words"][0]["box"][0] += 1
    elif mode == "region_id":
        region["region_id"] = "e" * 64
    elif mode == "native_text":
        region["recognition"]["text"] = "unearned replacement"
    elif mode == "page_status":
        data["pages"][0]["status"] = "recognized_empty"
    else:
        data["separator"] = "!"
    with pytest.raises(ValueError):
        OcrRegionDocumentResult.from_dict(data)


def test_result_constructors_cannot_accept_partial_reordered_or_cross_plan_regions():
    result = transcribe_region_pages((page(),), Backend(), planner=halves)
    item = result.pages[0]
    for regions in ((), item.regions[::-1], (item.regions[0], item.regions[0])):
        with pytest.raises(ValueError, match="exactly complete"):
            OcrRegionPageResult(item.plan, item.provider, regions)
    with pytest.raises(ValueError):
        replace(item.regions[0], plan_digest="not a digest")
    with pytest.raises(ValueError):
        replace(item.regions[0], source=replace(item.plan.source, width=4))
    with pytest.raises(ValueError):
        OcrRegionDocumentResult(())
    with pytest.raises(ValueError):
        OcrRegionResult(item.plan.source, item.plan.digest, item.plan.regions[0], {})


def test_wrong_crop_renderer_is_rejected_even_when_rebinding_derived_identity():
    result = transcribe_region_pages((page(),), Backend(), planner=halves)
    region = result.pages[0].regions[0]
    wrong = replace(region.recognition, page=replace(region.recognition.page, renderer="forged"))
    with pytest.raises(ValueError, match="source crop"):
        replace(region, recognition=wrong)
    data = region.to_dict()
    data["recognition"] = wrong.to_dict()
    # Recompute the declared ID: rejection must enforce the crop relation,
    # not merely notice that the old digest was left unchanged.
    identity = {
        "source": region.source.to_dict(),
        "plan_digest": region.plan_digest,
        "region": region.region.to_dict(),
        "crop_page": wrong.page.to_dict(),
    }
    data["region_id"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="source crop"):
        OcrRegionResult.from_dict(data)


def test_result_envelope_limit_rejects_whole_export_not_truncated_output(monkeypatch):
    result = transcribe_region_pages((page(),), Backend(), planner=halves)
    size = len(
        json.dumps(
            result.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )
    monkeypatch.setattr(pipeline, "MAX_RESULT_BYTES", size)
    assert OcrRegionDocumentResult.from_dict(result.to_dict()) == result
    monkeypatch.setattr(pipeline, "MAX_RESULT_BYTES", size - 1)
    with pytest.raises(ValueError, match="serialized byte"):
        OcrRegionDocumentResult.from_dict(result.to_dict())


def test_actual_png_decoder_and_default_split_plan_feed_real_crops_to_scripted_backend(tmp_path):
    from PIL import Image

    path = tmp_path / "original-columns.png"
    with Image.new("RGB", (40, 10), "white") as image:
        image.putpixel((2, 2), (0, 0, 0))
        image.putpixel((37, 2), (0, 0, 0))
        image.save(path)
    backend = Backend()
    result = transcribe_region_document(path, backend)
    assert len(backend.calls) == 2
    assert result.pages[0].plan.reason == "split"
    assert result.to_dict()["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert all(crop.source_type == "image" for crop in backend.calls)
    assert sum(crop.width * crop.height for crop in backend.calls) == 400


def test_actual_raster_pdf_decoder_retains_page_order_and_blank_page(tmp_path):
    from PIL import Image

    path = tmp_path / "original-pages.pdf"
    with (
        Image.new("RGB", (24, 12), "black") as first,
        Image.new("RGB", (24, 12), "white") as second,
    ):
        first.save(path, save_all=True, append_images=[second], resolution=72)
    backend = Backend(lambda image: recognized(image, None))
    result = transcribe_region_document(path, backend, limits=OcrInputLimits(dpi=72))
    assert len(backend.calls) == 2
    assert [crop.page_index for crop in backend.calls] == [0, 1]
    assert all(crop.source_type == "pdf" for crop in backend.calls)
    assert result.pages[0].plan.reason == "no_split"
    assert result.pages[1].plan.reason == "blank"
    assert result.counters["recognized_empty_pages"] == 2
    assert result.counters["detector_blank_pages"] == 1


@pytest.mark.parametrize("value", [None, {}, "backend"])
def test_missing_backend_protocol_is_a_controlled_preflight_error(value):
    with pytest.raises(ValueError, match="identity"):
        transcribe_region_pages((), value)


@pytest.mark.parametrize("level", ["document", "page", "region"])
def test_result_readers_reject_unbounded_or_non_json_values(level):
    result = transcribe_region_pages((page(),), Backend(), planner=halves)
    if level == "document":
        cls, value = OcrRegionDocumentResult, result.to_dict()
        value["pages"] = [None] * 1001
    elif level == "page":
        cls, value = OcrRegionPageResult, result.pages[0].to_dict()
        value["regions"] = {}
    else:
        cls, value = OcrRegionResult, result.pages[0].regions[0].to_dict()
        value["source_lines"] = object()
    with pytest.raises(ValueError):
        cls.from_dict(value)


def test_result_readers_count_words_across_lines_before_materializing_unbounded_results():
    region = (
        transcribe_region_pages((page(),), Backend(), planner=halves).pages[0].regions[0].to_dict()
    )
    line = region["recognition"]["lines"][0]
    line["words"] = line["words"] * 50_001
    region["recognition"]["lines"] = [line, line]
    with pytest.raises(ValueError, match="word limit"):
        OcrRegionResult.from_dict(region)


def test_document_constructor_enforces_hard_source_budget_without_allocating_fake_pixels():
    pages = []
    for index in range(11):
        metadata = replace(page().info, page_index=index, width=10_000, height=2_000)
        plan = OcrRegionPlan(metadata, (), allow_omissions=True)
        pages.append(OcrRegionPageResult(plan, IDENTITY, ()))
    with pytest.raises(ValueError, match="aggregate hard"):
        OcrRegionDocumentResult(tuple(pages))


def test_document_constructor_rejects_out_of_order_and_mixed_provider_empty_pages():
    first = OcrRegionPageResult(
        explicit_region_plan(page(), (), allow_omissions=True), IDENTITY, ()
    )
    second = OcrRegionPageResult(
        explicit_region_plan(page(1), (), allow_omissions=True), IDENTITY, ()
    )
    with pytest.raises(ValueError, match="contiguous"):
        OcrRegionDocumentResult((second, first))
    with pytest.raises(ValueError, match="share source"):
        OcrRegionDocumentResult(
            (first, replace(second, provider=replace(IDENTITY, version="other")))
        )
    with pytest.raises(ValueError):
        OcrRegionPageResult({}, IDENTITY, ())
    with pytest.raises(ValueError):
        OcrRegionResult({}, first.plan.digest, "not region", {})


def test_identity_drift_after_last_source_yield_is_rejected_even_without_region_calls():
    backend = Backend()

    def source():
        yield page()
        backend.identity = replace(IDENTITY, version="changed")

    with pytest.raises(ValueError, match="after document"):
        transcribe_region_pages(
            source(),
            backend,
            planner=lambda image: explicit_region_plan(image, (), allow_omissions=True),
        )
    assert not backend.calls


def test_backend_identity_changed_by_next_page_loader_is_rejected_before_planning():
    backend = Backend()

    def source():
        yield page()
        backend.identity = replace(IDENTITY, version="changed")
        yield page(1)

    with pytest.raises(ValueError, match="before planning"):
        transcribe_region_pages(source(), backend)
    assert len(backend.calls) == 1
