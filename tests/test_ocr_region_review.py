"""Independent boundary regressions for the region OCR integration."""

from __future__ import annotations

import argparse
import io
import sys

import pytest

from metricguard import ocr_region_cli as cli
from metricguard import ocr_region_pipeline as pipeline
from metricguard.ocr_regions import PixelBox, explicit_region_plan, iter_region_crops
from metricguard.ocr_types import (
    OcrLine,
    OcrPageImage,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
)


def test_closed_stdout_after_completed_recognition_is_controlled(monkeypatch):
    completed = []

    def execute(_args):
        completed.append(True)
        return b'{"completed":true}\n', None

    output = io.StringIO()
    output.close()
    diagnostic = io.StringIO()
    monkeypatch.setattr(cli, "_execute", execute)
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", diagnostic)
    assert cli.run_region_transcribe_command(argparse.Namespace()) == 0
    assert completed == [True]
    assert "recognition completed" in diagnostic.getvalue()
    assert "stdout could not be delivered" in diagnostic.getvalue()


def test_closed_stderr_during_command_failure_is_controlled_and_private(monkeypatch):
    def failed(_args):
        raise ValueError("PRIVATE_SOURCE_AND_EXCEPTION")

    diagnostic = io.StringIO()
    diagnostic.close()
    output = io.StringIO()
    monkeypatch.setattr(cli, "_execute", failed)
    monkeypatch.setattr(sys, "stderr", diagnostic)
    monkeypatch.setattr(sys, "stdout", output)
    assert cli.run_region_transcribe_command(argparse.Namespace()) == 2
    assert not output.getvalue()


def test_direct_page_result_rejects_over_budget_text_before_joining_it(monkeypatch):
    image = OcrPageImage(0, "a" * 64, "image", 32, 2, bytes(32 * 2 * 3), "authored")
    plan = explicit_region_plan(image, [PixelBox(x, 0, x + 1, 2) for x in range(32)])
    identity = OcrProviderIdentity("fixture", "1", "und", "b" * 64, "c" * 64)
    text = "x" * 4000
    regions = tuple(
        pipeline.OcrRegionResult(
            image.info,
            plan.digest,
            crop.region,
            OcrPageResult(
                crop.image.info,
                identity,
                (OcrLine(text, (OcrWord("x", (0, 0, 1, 1)),)),),
            ),
        )
        for crop in iter_region_crops(image, plan)
    )
    assert len(regions) * len(text) == 128_000

    def forbidden_join(_self):
        raise AssertionError("oversized aggregate text must be rejected before allocation")

    monkeypatch.setattr(pipeline, "MAX_RESULT_BYTES", 65_536)
    monkeypatch.setattr(pipeline.OcrRegionPageResult, "text", property(forbidden_join))
    with pytest.raises(ValueError):
        pipeline.OcrRegionPageResult(plan, identity, regions)


def test_direct_document_rejects_over_budget_text_before_joining_pages(monkeypatch):
    identity = OcrProviderIdentity("fixture", "1", "und", "b" * 64, "c" * 64)
    pages = []
    for index in range(2):
        image = OcrPageImage(index, "a" * 64, "image", 1, 1, bytes(3), "authored")
        plan = explicit_region_plan(image, [PixelBox(0, 0, 1, 1)])
        crop = next(iter_region_crops(image, plan))
        region = pipeline.OcrRegionResult(
            image.info,
            plan.digest,
            crop.region,
            OcrPageResult(
                crop.image.info,
                identity,
                (OcrLine("x" * 4000, (OcrWord("x", (0, 0, 1, 1)),)),),
            ),
        )
        pages.append(pipeline.OcrRegionPageResult(plan, identity, (region,)))
    assert sum(len(page.text) for page in pages) == 8000

    def forbidden_join(_self):
        raise AssertionError("oversized document text must be rejected before allocation")

    monkeypatch.setattr(pipeline, "MAX_RESULT_BYTES", 6000)
    monkeypatch.setattr(pipeline.OcrRegionDocumentResult, "text", property(forbidden_join))
    with pytest.raises(ValueError):
        pipeline.OcrRegionDocumentResult(tuple(pages))


def test_direct_page_bounds_geometry_before_building_nested_word_dicts(monkeypatch):
    image = OcrPageImage(0, "a" * 64, "image", 2, 1, bytes(6), "authored")
    plan = explicit_region_plan(image, [PixelBox(0, 0, 1, 1), PixelBox(1, 0, 2, 1)])
    identity = OcrProviderIdentity("fixture", "1", "und", "b" * 64, "c" * 64)
    # Shared immutable words keep the authored fixture small. Expanding them to
    # dictionaries before checking counts is exactly the allocation to prevent.
    words = (OcrWord("x", (0, 0, 1, 1)),) * 50_001
    line = OcrLine("", words)
    regions = tuple(
        pipeline.OcrRegionResult(
            image.info,
            plan.digest,
            crop.region,
            OcrPageResult(crop.image.info, identity, (line,)),
        )
        for crop in iter_region_crops(image, plan)
    )
    assert sum(region.recognition.structured_text_bytes for region in regions) == 100_002

    def forbidden_serialization(_self):
        raise AssertionError("over-limit word geometry must be rejected before serialization")

    monkeypatch.setattr(pipeline.OcrRegionPageResult, "to_dict", forbidden_serialization)
    with pytest.raises(ValueError):
        pipeline.OcrRegionPageResult(plan, identity, regions)
