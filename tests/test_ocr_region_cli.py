"""Command-level region OCR contracts, with explicit native subprocess opt-in."""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw, ImageFont

from metricguard import ocr_region_cli as cli
from metricguard.ocr_inputs import load_ocr_pages
from metricguard.ocr_region_pipeline import OcrRegionDocumentResult
from metricguard.ocr_regions import OcrRegionPlan, detect_vertical_regions, explicit_region_plan
from metricguard.ocr_types import OcrLine, OcrPageImage, OcrPageResult, OcrProviderIdentity, OcrWord


def invoke(*argv: Any) -> int:
    parser = argparse.ArgumentParser()
    cli.configure_region_transcribe_parser(parser)
    args = parser.parse_args([str(value) for value in argv])
    return int(args.handler(args))


def raster(tmp_path: Path, *, pages: int = 1) -> Path:
    path = tmp_path / ("input.png" if pages == 1 else "input.tiff")
    images = []
    try:
        for index in range(pages):
            image = Image.new("RGB", (40, 20), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((3, 3, 8, 15), fill=(index, 0, 0))
            draw.rectangle((30, 3, 35, 15), fill="black")
            images.append(image)
        images[0].save(path, save_all=True, append_images=images[1:])
    finally:
        for image in images:
            image.close()
    return path


def source_plans(path: Path) -> tuple[OcrRegionPlan, ...]:
    with closing(load_ocr_pages(path)) as pages:
        return tuple(detect_vertical_regions(page) for page in pages)


def write_plans(path: Path, plans: tuple[OcrRegionPlan, ...]) -> Path:
    path.write_text(
        json.dumps(
            {"format": "metricguard.ocr-region-plans.v1", "plans": [p.to_dict() for p in plans]}
        ),
        encoding="utf-8",
    )
    return path


class Backend:
    identity = OcrProviderIdentity("authored-fixture", "1", "en-US", "a" * 64, "b" * 64)

    def __init__(self) -> None:
        self.calls: list[OcrPageImage] = []
        self.fail_at: int | None = None

    def recognize(self, page: OcrPageImage) -> OcrPageResult:
        self.calls.append(page)
        if self.fail_at == len(self.calls):
            raise ValueError("PRIVATE image contents and provider exception")
        text = f"café😀-{len(self.calls)}"
        return OcrPageResult(
            page.info, self.identity, (OcrLine(text, (OcrWord(text, (1, 2, 2, 3)),)),)
        )


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[Backend, list[Any]]:
    instance = Backend()
    constructors: list[Any] = []

    def create(*args: Any, **kwargs: Any) -> Backend:
        constructors.append((args, kwargs))
        return instance

    monkeypatch.setattr(cli, "WindowsOcrBackend", create)
    return instance, constructors


def test_detector_cli_emits_private_roundtrippable_unicode_and_geometry(
    tmp_path: Path, backend: tuple[Backend, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    source = raster(tmp_path)
    assert invoke(source, "--direction", "rtl", "--min-gutter-width", 10) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "metricguard.ocr-region-transcription.v1"
    assert report["private_data"] is True
    document = OcrRegionDocumentResult.from_dict(report["document"])
    assert document.digest == report["document_digest"]
    assert document.text == "café😀-1\n\ncafé😀-2"
    page = document.pages[0]
    assert page.plan.direction == "rtl"
    assert page.plan.white_threshold == 255
    assert page.plan.min_gutter_width == 10
    assert page.plan.regions[0].box.left > page.plan.regions[1].box.left
    assert page.regions[0].source_lines[0].words[0].box[0] == page.plan.regions[0].box.left + 1
    assert len(backend[0].calls) == 2
    assert backend[1] == [(("en-US",), {"timeout": 30.0, "max_output_bytes": 4 * 1024 * 1024})]


def test_explicit_plans_all_pages_and_exclusive_output(
    tmp_path: Path, backend: tuple[Backend, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    source = raster(tmp_path, pages=2)
    plans = write_plans(tmp_path / "plans.json", source_plans(source))
    output = tmp_path / "report.json"
    assert invoke(source, "--plans", plans, "--output", output) == 0
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_bytes())
    assert report["document"]["counters"]["recognition_calls"] == 4
    assert report["document"]["counters"]["pages"] == 2
    assert not list(tmp_path.glob(".metricguard-regions-*"))
    original = output.read_bytes()
    assert invoke(source, "--plans", plans, "--output", output) == 2
    assert output.read_bytes() == original
    assert len(backend[0].calls) == 4


@pytest.mark.parametrize(
    "flag", ["--direction", "--white-threshold", "--min-gutter-width", "--min-region-width"]
)
def test_detector_config_never_silently_overrides_plan(
    tmp_path: Path, backend: tuple[Backend, list[Any]], flag: str
) -> None:
    source = raster(tmp_path)
    plans = write_plans(tmp_path / "plans.json", source_plans(source))
    assert invoke(source, "--plans", plans, flag, "ltr" if flag == "--direction" else 10) == 2
    assert not backend[1]


@pytest.mark.parametrize(
    "content",
    [
        b"\xff",
        b'{"format":1,"format":2}',
        b'{"x":NaN}',
        b"[]",
        b"{}",
        b'{"format":"metricguard.ocr-region-plans.v1","plans":[]}',
    ],
)
def test_strict_plan_json_fails_before_backend(
    tmp_path: Path, backend: tuple[Backend, list[Any]], content: bytes
) -> None:
    source = raster(tmp_path)
    plans = tmp_path / "bad.json"
    plans.write_bytes(content)
    assert invoke(source, "--plans", plans) == 2
    assert not backend[1]


@pytest.mark.parametrize(
    "mutation", ["duplicate", "order", "other_source", "digest", "derived_bool", "extra_field"]
)
def test_malformed_plan_entries_fail_before_backend(
    tmp_path: Path, backend: tuple[Backend, list[Any]], mutation: str
) -> None:
    source = raster(tmp_path, pages=2)
    first, second = source_plans(source)
    if mutation == "other_source":
        second = replace(second, source=replace(second.source, source_sha256="c" * 64))
    values = [first.to_dict(), second.to_dict()]
    if mutation == "duplicate":
        values[1] = copy.deepcopy(values[0])
    elif mutation == "order":
        values.reverse()
    elif mutation == "digest":
        values[0]["digest"] = "0" * 64
    elif mutation == "derived_bool":
        values[0]["uncovered_pixels"] = False
    elif mutation == "extra_field":
        values[0]["secret"] = "not an allowed field"
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps({"format": "metricguard.ocr-region-plans.v1", "plans": values}))
    assert invoke(source, "--plans", plans) == 2
    assert not backend[1]


@pytest.mark.parametrize("mismatch", ["extra", "missing", "stale_pixels", "later_stale"])
def test_complete_source_plan_matching_publishes_nothing_on_mismatch(
    tmp_path: Path,
    backend: tuple[Backend, list[Any]],
    mismatch: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = raster(tmp_path, pages=2 if mismatch in ("missing", "later_stale") else 1)
    plans = list(source_plans(source))
    if mismatch == "extra":
        plans.append(replace(plans[0], source=replace(plans[0].source, page_index=1)))
    elif mismatch == "missing":
        plans.pop()
    else:
        index = 1 if mismatch == "later_stale" else 0
        plans[index] = replace(
            plans[index], source=replace(plans[index].source, pixel_sha256="c" * 64)
        )
    path = write_plans(tmp_path / "plans.json", tuple(plans))
    output = tmp_path / "output.json"
    assert invoke(source, "--plans", path, "--output", output) == 2
    assert not output.exists()
    assert not capsys.readouterr().out
    assert len(backend[0].calls) == (0 if mismatch == "stale_pixels" else 2)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-pages", 0),
        ("--max-dimension", 0),
        ("--dpi", 601),
        ("--max-file-bytes", 0),
        ("--max-page-pixels", 0),
        ("--max-total-pixels", 0),
        ("--max-regions", 0),
        ("--max-crop-total-pixels", 0),
        ("--max-total-regions", 0),
        ("--max-total-crop-pixels", 0),
        ("--max-document-words", 0),
        ("--max-document-lines", 0),
        ("--max-document-text-bytes", 0),
        ("--white-threshold", 256),
        ("--min-gutter-width", 0),
        ("--min-region-width", 0),
    ],
)
def test_invalid_budgets_precede_backend(
    tmp_path: Path, backend: tuple[Backend, list[Any]], flag: str, value: int
) -> None:
    assert invoke(raster(tmp_path), flag, value) == 2
    assert not backend[1]


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--max-pages", 1),
        ("--max-total-pixels", 1000),
        ("--max-regions", 1),
        ("--max-crop-total-pixels", 500),
        ("--max-total-regions", 3),
        ("--max-total-crop-pixels", 1000),
    ],
)
def test_declared_plan_budgets_preflight_all_pages(
    tmp_path: Path, backend: tuple[Backend, list[Any]], flag: str, value: int
) -> None:
    source = raster(tmp_path, pages=2)
    plans = write_plans(tmp_path / "plans.json", source_plans(source))
    assert invoke(source, "--plans", plans, flag, value) == 2
    assert not backend[1]


def test_late_backend_failure_and_report_text_cap_do_not_publish(
    tmp_path: Path, backend: tuple[Backend, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    source = raster(tmp_path)
    output = tmp_path / "output.json"
    backend[0].fail_at = 2
    assert invoke(source, "--output", output) == 2
    assert not output.exists()
    captured = capsys.readouterr()
    assert not captured.out and "PRIVATE" not in captured.err
    backend[0].fail_at = None
    assert invoke(source, "--max-document-text-bytes", 1, "--output", output) == 2
    assert not output.exists()


def test_empty_explicit_plan_is_not_misreported_as_blank_recognition(
    tmp_path: Path, backend: tuple[Backend, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    source = raster(tmp_path)
    with closing(load_ocr_pages(source)) as pages:
        plan = explicit_region_plan(next(pages), [], allow_omissions=True)
    plans = write_plans(tmp_path / "plans.json", (plan,))
    assert invoke(source, "--plans", plans) == 0
    report = json.loads(capsys.readouterr().out)["document"]
    assert report["coverage"] == "omitted_all"
    assert report["counters"]["recognition_calls"] == 0
    assert report["counters"]["detector_blank_pages"] == 0
    assert not backend[0].calls


def test_input_plan_output_identity_and_missing_parent_protection(
    tmp_path: Path, backend: tuple[Backend, list[Any]]
) -> None:
    source = raster(tmp_path)
    plans = write_plans(tmp_path / "plans.json", source_plans(source))
    alias = tmp_path / "alias"
    os.link(source, alias)
    for output in (source, plans, alias, tmp_path / "missing" / "result.json"):
        assert invoke(source, "--plans", plans, "--output", output) == 2
    assert invoke(source, "--plans", source) == 2
    assert invoke(source, "--plans", alias) == 2
    assert invoke(tmp_path / "missing.png") == 2
    assert invoke(tmp_path) == 2
    assert not backend[1]


def test_racing_output_and_cleanup_failure_are_honest(
    tmp_path: Path,
    backend: tuple[Backend, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = raster(tmp_path)
    output = tmp_path / "output.json"
    real_link = os.link

    def raced(first: Any, second: Any) -> None:
        Path(second).write_text("concurrent user output")
        real_link(first, second)

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", raced)
        assert invoke(source, "--output", output) == 2
    assert output.read_text() == "concurrent user output"
    assert not list(tmp_path.glob(".metricguard-regions-*"))
    output.unlink()
    original_unlink = Path.unlink

    def blocked(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.startswith(".metricguard-regions-"):
            raise OSError("PRIVATE temporary cleanup details")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", blocked)
        assert invoke(source, "--output", output) == 2
    captured = capsys.readouterr()
    assert "artifact or private temporary file may already exist" in captured.err
    assert "PRIVATE" not in captured.err
    assert json.loads(output.read_bytes())["document"]["status"] == "complete"
    temporary = list(tmp_path.glob(".metricguard-regions-*"))
    assert len(temporary) == 1 and temporary[0].samefile(output)
    temporary[0].unlink()


def test_serialized_caps_and_closed_output_streams(
    tmp_path: Path, backend: tuple[Backend, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = raster(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(cli, "_REPORT_BYTES", 10)
        assert invoke(source) == 2
    plans = write_plans(tmp_path / "plans.json", source_plans(source))
    with monkeypatch.context() as patch:
        patch.setattr(cli, "_PLAN_BYTES", 10)
        assert invoke(source, "--plans", plans) == 2

    class Broken(io.StringIO):
        def write(self, content: str) -> int:
            raise OSError("PRIVATE pipe")

    monkeypatch.setattr(sys, "stdout", Broken())
    monkeypatch.setattr(sys, "stderr", Broken())
    assert invoke(source) == 0


def test_real_subprocess_bad_input_is_controlled_utf8(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "metricguard", "transcribe-regions", str(tmp_path / "不存在😀")],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert not result.stdout
    error = result.stderr.decode("utf-8")
    assert "metricguard regions:" in error and "Traceback" not in error


_PORTABLE_CHILD = """
import sys
from metricguard import ocr_region_cli as region_cli
from metricguard.cli import main
from metricguard.ocr_types import OcrLine, OcrWord, OcrPageResult, OcrProviderIdentity
class Backend:
    identity = OcrProviderIdentity('fixture', '1', 'en-US', 'a'*64, 'b'*64)
    def __init__(self, *args, **kwargs): pass
    def recognize(self, page):
        text = 'caf\\xe9\\U0001f600'
        return OcrPageResult(page.info, self.identity,
                             (OcrLine(text, (OcrWord(text, (1, 1, 1, 1)),)),))
region_cli.WindowsOcrBackend = Backend
raise SystemExit(main(['transcribe-regions', sys.argv[1]]))
"""


def test_real_subprocess_unicode_stdout_and_closed_consumer(tmp_path: Path) -> None:
    source = raster(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _PORTABLE_CHILD, str(source)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8")
    assert json.loads(result.stdout.decode("utf-8"))["document"]["text"] == "café😀\n\ncafé😀"
    process = subprocess.Popen(
        [sys.executable, "-c", _PORTABLE_CHILD, str(source)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    process.stdout.close()
    try:
        process.wait(timeout=30)
        error = process.stderr.read().decode("utf-8")
        assert process.returncode == 0
        assert "recognition completed" in error
        assert "Exception ignored" not in error and "Traceback" not in error
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def test_unavailable_native_platform_returns_controlled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from metricguard import native_ocr

    monkeypatch.setattr(native_ocr.sys, "platform", "linux")
    assert invoke(raster(tmp_path)) == 2
    captured = capsys.readouterr()
    assert not captured.out and "metricguard regions:" in captured.err


@pytest.mark.native_ocr
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("METRICGUARD_RUN_NATIVE_OCR") != "1",
    reason="explicit opt-in Windows region OCR CLI; no network or model downloads",
)
def test_real_native_cli_png_and_raster_only_two_page_pdf(tmp_path: Path) -> None:
    import pypdfium2 as pdfium

    expected = (("LEFTONE", "RIGHTTWO"), ("THREE", "FOUR"))
    images = []
    try:
        for words in expected:
            image = Image.new("RGB", (1200, 220), "white")
            draw = ImageDraw.Draw(image)
            for x, text in zip((40, 680), words, strict=True):
                draw.text((x, 70), text, fill="black", font=ImageFont.load_default(size=48))
            images.append(image)
        png, pdf = tmp_path / "authored.png", tmp_path / "two-pages.pdf"
        images[0].save(png)
        images[0].save(pdf, save_all=True, append_images=images[1:], resolution=144)
    finally:
        for image in images:
            image.close()
    with pdfium.PdfDocument(pdf) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                text = page.get_textpage()
                try:
                    assert text.get_text_range() == ""
                finally:
                    text.close()
            finally:
                page.close()
    for source, expected_pages in ((png, expected[:1]), (pdf, expected)):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "metricguard",
                "transcribe-regions",
                str(source),
                "--min-gutter-width",
                "100",
                "--min-region-width",
                "100",
            ],
            capture_output=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8")
        report = json.loads(result.stdout.decode("utf-8"))
        assert report["private_data"] is True
        parsed = OcrRegionDocumentResult.from_dict(report["document"])
        assert parsed.digest == report["document_digest"]
        # The installed recognizer may insert word spaces (RIGHTTWO -> RIGHT TWO).
        # This integration oracle checks glyph content and region/page order,
        # not zero CER/WER; returned strings are preserved without normalization.
        assert [
            tuple("".join(region.text.split()) for region in page.regions) for page in parsed.pages
        ] == list(expected_pages)
        assert parsed.counters["recognition_calls"] == len(expected_pages) * 2
        assert report["document"]["coverage"] == "complete"
