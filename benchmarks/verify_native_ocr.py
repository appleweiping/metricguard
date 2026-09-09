"""Execute Windows OCR on authored PNG, blank PNG and raster-only two-page PDF."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from metricguard import (
    CharacterErrorRate,
    OcrInputLimits,
    TextNormalizer,
    WindowsOcrBackend,
    WordErrorRate,
    transcribe_document,
)

EXPECTED = ("METRIC GUARD 2026", "LOCAL OCR PAGE TWO")


def fixtures(directory: Path) -> dict[str, Path]:
    """Create original synthetic text rasters; no model or reference text file is used."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / name for name in ("authored.png", "two-pages.pdf", "blank.png")}
    if any(path.exists() for path in paths.values()):
        raise ValueError("fixture destinations must not already exist")
    images = []
    try:
        for text in EXPECTED:
            image = Image.new("RGB", (1100, 220), "white")
            ImageDraw.Draw(image).text(
                (40, 60), text, fill="black", font=ImageFont.load_default(size=52)
            )
            images.append(image)
        images[0].save(paths["authored.png"])
        images[0].save(
            paths["two-pages.pdf"],
            save_all=True,
            append_images=images[1:],
            resolution=144,
            title="MetricGuard authored OCR fixture",
            creationDate=time.gmtime(0),
            modDate=time.gmtime(0),
        )
        with Image.new("RGB", (1100, 220), "white") as blank:
            blank.save(paths["blank.png"])
    finally:
        for image in images:
            image.close()
    return paths


def verify(directory: Path) -> dict:
    paths = fixtures(directory)
    timings = {}
    # Fixture construction is excluded. tracemalloc does not measure PDFium,
    # Pillow's native allocations, the PowerShell child, or the Windows engine.
    tracemalloc.start()
    try:
        started = time.perf_counter()
        with pdfium.PdfDocument(paths["two-pages.pdf"]) as document:
            text_layer_characters = []
            for index in range(len(document)):
                page = document[index]
                try:
                    text_page = page.get_textpage()
                    try:
                        text_layer_characters.append(len(text_page.get_text_range()))
                    finally:
                        text_page.close()
                finally:
                    page.close()
        if text_layer_characters != [0, 0]:
            raise ValueError("authored PDF must have exactly two pages without text layers")
        timings["verify_raster_only_pdf_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        backend = WindowsOcrBackend("en-US")
        timings["native_initialization_seconds"] = time.perf_counter() - started
        results = {}
        for name, path in paths.items():
            started = time.perf_counter()
            results[name] = transcribe_document(path, backend, limits=OcrInputLimits(dpi=200))
            timings[f"{name}_decode_and_recognize_seconds"] = time.perf_counter() - started
        if results["authored.png"].text != EXPECTED[0]:
            raise ValueError("authored PNG was not recognized exactly")
        if [page.text for page in results["two-pages.pdf"].pages] != list(EXPECTED):
            raise ValueError("PDF pages were not recognized exactly in source order")
        if results["blank.png"].text != "" or results["blank.png"].pages[0].lines:
            raise ValueError("blank page should remain present with no recognized words")
        cer, wer = CharacterErrorRate(), WordErrorRate(normalizer=TextNormalizer())
        measured = []
        for name in ("authored.png", "two-pages.pdf"):
            for page in results[name].pages:
                expected = EXPECTED[page.page.page_index]
                if not page.lines or not page.lines[0].words:
                    raise ValueError("nonblank fixture must produce structured word geometry")
                measured.append(
                    {
                        "fixture": name,
                        "page_index": page.page.page_index,
                        "reference": expected,
                        "recognized": page.text,
                        "cer": cer.evaluate(expected, page.text).score,
                        "wer": wer.evaluate(expected, page.text).score,
                        "words": sum(len(line.words) for line in page.lines),
                    }
                )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    package = Path(sys.modules["metricguard"].__file__).parent
    runtime_paths = [
        package / name
        for name in (
            "ocr_inputs.py",
            "ocr_types.py",
            "native_ocr.py",
            "ocr_pipeline.py",
            "_windows_ocr.ps1",
        )
    ]
    return {
        "kind": "authored-synthetic-real-recognizer",
        "claim_scope": (
            "One local run: 2 distinct clean English uppercase texts, repeated as PNG/PDF, "
            "plus blank PNG. Not a real-world OCR accuracy benchmark."
        ),
        "gold_boundary": (
            "Expected strings are compared only after recognition; "
            "provider receives RGB raster and source identity only."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("Pillow", "pypdfium2")
        },
        "font": "Pillow ImageFont.load_default(size=52)",
        "provider": backend.identity.to_dict(),
        "installed_languages": backend.available_languages,
        "runtime_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_paths
        },
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {
            name: {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "pdf_text_layer_characters": text_layer_characters,
        "nonblank_page_evaluations": measured,
        "blank_page_preserved": True,
        "document_digests": {name: result.digest for name, result in results.items()},
        "timings": timings,
        "peak_traced_python_bytes": peak,
        "memory_scope": (
            "Fixture generation excluded; Python tracemalloc only, "
            "not native decoder/recognizer/child RSS."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/native-ocr.json"))
    parser.add_argument(
        "--fixtures", type=Path, help="Keep newly authored images/PDF in this directory"
    )
    args = parser.parse_args()
    if args.fixtures is None:
        with tempfile.TemporaryDirectory(prefix="metricguard-native-ocr-") as directory:
            result = verify(Path(directory))
    else:
        result = verify(args.fixtures)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
