"""Compare whole-page and region-first native OCR on original authored layouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
import tracemalloc
from contextlib import closing
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

import metricguard
from metricguard import CharacterErrorRate, TextNormalizer, WindowsOcrBackend, WordErrorRate
from metricguard.ocr_inputs import OcrInputLimits, load_ocr_pages
from metricguard.ocr_pipeline import PageOcrBackend, transcribe_document
from metricguard.ocr_region_pipeline import OcrRegionDocumentResult, transcribe_region_document
from metricguard.ocr_regions import OcrRegionPlan, detect_vertical_regions
from metricguard.ocr_types import OcrPageImage, OcrPageResult, OcrProviderIdentity

LEFT = ("ALPHA ROUTE", "BETA LANTERN")
RIGHT = ("GAMMA CHECK", "DELTA QUIET")
SINGLE = ("SINGLE ROUTE", "LANTERN CHECK")
DETECTOR = {"white_threshold": 255, "min_gutter_width": 100, "min_region_width": 200}


def runtime_inventory() -> dict[str, str]:
    package = Path(metricguard.__file__).parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((*package.glob("*.py"), *package.glob("*.ps1")))
    }


def fixtures(directory: Path) -> dict[str, tuple[Path, tuple[str, ...]]]:
    """Write only a new owned directory; transcripts are returned for later scoring."""
    directory.mkdir()
    font = ImageFont.load_default(size=42)
    left_text, right_text = "\n".join(LEFT), "\n".join(RIGHT)
    expected = left_text + "\n\n" + right_text
    with Image.new("RGB", (1200, 320), "white") as columns:
        drawing = ImageDraw.Draw(columns)
        for x, lines in ((40, LEFT), (700, RIGHT)):
            for y, text in zip((90, 175), lines, strict=True):
                drawing.text((x, y), text, font=font, fill="black")
        columns.save(directory / "columns.png")
        with Image.new("RGB", columns.size, "white") as blank:
            columns.save(
                directory / "columns-and-blank.pdf",
                save_all=True,
                append_images=[blank],
                resolution=144,
                title="Original region OCR fixture",
                creationDate=time.gmtime(0),
                modDate=time.gmtime(0),
            )
        with columns.copy() as bridge:
            # A full-width rule deliberately defeats this full-height-gutter
            # heuristic. Keep this limitation case, even if recognition worsens.
            ImageDraw.Draw(bridge).rectangle((20, 25, 1179, 29), fill="black")
            bridge.save(directory / "bridged-columns.png")
    with Image.new("RGB", (1200, 320), "white") as single:
        drawing = ImageDraw.Draw(single)
        for y, text in zip((90, 175), SINGLE, strict=True):
            drawing.text((40, y), text, font=font, fill="black")
        single.save(directory / "single.png")
    return {
        "columns.png": (directory / "columns.png", (expected,)),
        "columns-and-blank.pdf": (directory / "columns-and-blank.pdf", (expected, "")),
        "bridged-columns.png": (directory / "bridged-columns.png", (expected,)),
        "single.png": (directory / "single.png", ("\n".join(SINGLE),)),
    }


class CountingBackend:
    """Observe actual calls without receiving any reference transcription."""

    def __init__(self, delegate: PageOcrBackend) -> None:
        self.delegate = delegate
        self.calls = 0

    @property
    def identity(self) -> OcrProviderIdentity:
        return self.delegate.identity

    def recognize(self, page: OcrPageImage) -> OcrPageResult:
        self.calls += 1
        return self.delegate.recognize(page)


def layout_plan(page: OcrPageImage) -> OcrRegionPlan:
    return detect_vertical_regions(
        page,
        white_threshold=DETECTOR["white_threshold"],
        min_gutter_width=DETECTOR["min_gutter_width"],
        min_region_width=DETECTOR["min_region_width"],
    )


def scores(reference: str, prediction: str) -> dict[str, Any]:
    """Report every outcome; do not assert that the new workflow improves OCR."""
    normalizer = TextNormalizer()
    cer = CharacterErrorRate(normalizer=normalizer).evaluate(reference, prediction)
    wer = WordErrorRate(normalizer=normalizer).evaluate(reference, prediction)
    return {
        "exact_raw": reference == prediction,
        "exact_normalized": normalizer(reference) == normalizer(prediction),
        "cer": cer.score,
        "wer": wer.score,
        "cer_details": dict(cer.details),
        "wer_details": dict(wer.details),
        "cer_undefined_reason": cer.reason,
        "wer_undefined_reason": wer.reason,
    }


def verify_geometry(path: Path, document: OcrRegionDocumentResult) -> dict[str, int]:
    """Independently slice source RGB and rederive offsets instead of using crop helpers."""
    counts = {"pages": 0, "crops": 0, "mapped_words": 0, "page_spans": 0, "region_spans": 0}
    exported = document.to_dict()
    with closing(load_ocr_pages(path, limits=OcrInputLimits(dpi=200))) as pages:
        for page in pages:
            result = document.pages[page.page_index]
            assert result.plan.source == page.info
            assert result.plan.uncovered_pixels == 0
            counts["pages"] += 1
            page_span = exported["page_spans"][page.page_index]
            assert document.text[page_span["start"] : page_span["end"]] == result.text
            counts["page_spans"] += 1
            serialized = result.to_dict()
            for region in result.regions:
                box = region.region.box
                raw = b"".join(
                    page.pixels[
                        ((y * page.width + box.left) * 3) : ((y * page.width + box.right) * 3)
                    ]
                    for y in range(box.top, box.bottom)
                )
                crop_page = region.recognition.page
                reconstructed = OcrPageImage(
                    page.page_index,
                    page.source_sha256,
                    page.source_type,
                    box.width,
                    box.height,
                    raw,
                    crop_page.renderer,
                )
                assert reconstructed.info == crop_page
                counts["crops"] += 1
                native_words = [word for line in region.recognition.lines for word in line.words]
                mapped_words = [word for line in region.source_result.lines for word in line.words]
                assert len(native_words) == len(mapped_words)
                for local, mapped in zip(native_words, mapped_words, strict=True):
                    assert local.text == mapped.text
                    assert mapped.box == (
                        local.box[0] + box.left,
                        local.box[1] + box.top,
                        *local.box[2:],
                    )
                    counts["mapped_words"] += 1
                span = serialized["region_spans"][region.region.index]
                assert result.text[span["start"] : span["end"]] == region.text
                counts["region_spans"] += 1
    assert counts["pages"] == len(document.pages)
    assert counts["crops"] == document.counters["recognition_calls"]
    return counts


def verify(directory: Path) -> dict[str, Any]:
    runtime_before = runtime_inventory()
    script_before = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paths = fixtures(directory)
    sources_before = {
        name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, (path, _) in paths.items()
    }
    with pdfium.PdfDocument(paths["columns-and-blank.pdf"][0]) as pdf:
        text_layers = []
        for index in range(len(pdf)):
            with closing(pdf[index]) as page, closing(page.get_textpage()) as text_page:
                text_layers.append(len(text_page.get_text_range()))
    assert text_layers == [0, 0]
    native = WindowsOcrBackend("en-US")
    rows = []
    checks: dict[str, dict[str, int]] = {}
    timings = {}
    tracemalloc.start()
    try:
        for name, (path, references) in paths.items():
            whole_backend, region_backend = CountingBackend(native), CountingBackend(native)
            started = time.perf_counter()
            whole = transcribe_document(path, whole_backend, limits=OcrInputLimits(dpi=200))
            timings[name + ":whole_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            regions = transcribe_region_document(
                path,
                region_backend,
                limits=OcrInputLimits(dpi=200),
                planner=layout_plan,
            )
            timings[name + ":regions_seconds"] = time.perf_counter() - started
            assert len(whole.pages) == len(regions.pages) == len(references)
            assert region_backend.calls == regions.counters["recognition_calls"]
            assert whole_backend.calls == len(references)
            checks[name] = verify_geometry(path, regions)
            for index, reference in enumerate(references):
                result = regions.pages[index]
                assert whole.pages[index].page == result.plan.source
                rows.append(
                    {
                        "fixture": name,
                        "page_index": index,
                        "reference": reference,
                        "whole_text": whole.pages[index].text,
                        "region_text": result.text,
                        "whole_scores": scores(reference, whole.pages[index].text),
                        "region_scores": scores(reference, result.text),
                        "regions": len(result.regions),
                        "detector_reason": result.plan.reason,
                        "status": result.status,
                        "plan_sha256": result.plan.digest,
                        "whole_recognition_calls": 1,
                        "region_recognition_calls": len(result.regions),
                        "document_sha256": regions.digest,
                    }
                )
            expected_counts = (
                (2, 1) if name.endswith(".pdf") else ((2,) if name == "columns.png" else (1,))
            )
            assert tuple(len(page.regions) for page in regions.pages) == expected_counts
            if name == "bridged-columns.png":
                assert regions.pages[0].plan.reason == "no_split"
            if name.endswith(".pdf"):
                # Blank pixels are known; a recognizer may still hallucinate
                # text. Preserve that adverse result and its undefined rates.
                assert regions.pages[1].plan.reason == "blank"
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert runtime_inventory() == runtime_before, "runtime files changed during measurement"
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_before
    assert sources_before == {
        name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, (path, _) in paths.items()
    }, "source fixtures changed during measurement"
    return {
        "format": "metricguard.region-ocr-benchmark.v1",
        "kind": "original-authored-layouts-with-real-native-recognition",
        "evidence_scope": (
            "Five authored source pages in four files; one column layout repeated as PNG/PDF. "
            "Includes a deliberate gutter-blocking negative case. Not real-world OCR accuracy "
            "or learned layout detection evidence; all scores, including regressions, retained."
        ),
        "gold_boundary": (
            "References enter scoring only after recognition; backend receives RGB and identity."
        ),
        "detector": DETECTOR,
        "cer_wer_normalization": TextNormalizer().to_dict(),
        "font": "Pillow ImageFont.load_default(size=42)",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("Pillow", "pypdfium2")
        },
        "provider": native.identity.to_dict(),
        "pdf_text_layer_characters": text_layers,
        "sources": sources_before,
        "runtime_sha256": runtime_before,
        "runtime_and_fixture_files_unchanged": True,
        "script_sha256": script_before,
        "page_evaluations": rows,
        "independent_checks": checks,
        "timings": timings,
        "peak_traced_python_bytes": peak,
        "memory_scope": (
            "Python allocations during recognition and verification only; not native/child RSS."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new aggregate JSON artifact")
    parser.add_argument("--fixtures", type=Path, help="new directory with an existing parent")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    if args.fixtures is None:
        with tempfile.TemporaryDirectory(prefix="metricguard-region-ocr-") as temporary:
            result = verify(Path(temporary) / "fixtures")
    else:
        result = verify(args.fixtures)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {
                "completed": True,
                "artifact": str(args.output),
                "pages": len(result["page_evaluations"]),
                "synthetic": True,
            }
        )
    )


if __name__ == "__main__":
    main()
