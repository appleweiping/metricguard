"""Private, exclusive-output command boundary for local region OCR."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .io import _strict_json_loads
from .native_ocr import WindowsOcrBackend
from .ocr_inputs import OcrInputLimits
from .ocr_region_pipeline import OcrRegionPipelineLimits, transcribe_region_document
from .ocr_regions import OcrRegionLimits, OcrRegionPlan, detect_vertical_regions
from .ocr_types import OcrPageImage, _integer

_PLAN_BYTES = 16 * 1024 * 1024
_REPORT_BYTES = 64 * 1024 * 1024


def configure_region_transcribe_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--direction", choices=("ltr", "rtl"))
    parser.add_argument("--white-threshold", type=int)
    parser.add_argument("--min-gutter-width", type=int)
    parser.add_argument("--min-region-width", type=int)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-native-output-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-dimension", type=int, default=10_000)
    parser.add_argument("--max-page-pixels", type=int, default=20_000_000)
    parser.add_argument("--max-total-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-regions", type=int, default=64)
    parser.add_argument("--max-crop-total-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-total-regions", type=int, default=1000)
    parser.add_argument("--max-total-crop-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-document-words", type=int, default=100_000)
    parser.add_argument("--max-document-lines", type=int, default=100_000)
    parser.add_argument("--max-document-text-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=run_region_transcribe_command)


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve() == second.resolve() or (
        first.exists() and second.exists() and first.samefile(second)
    )


def _paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    source = args.input
    plan = args.plans
    destination = args.output
    for path in (source, plan):
        if path is not None and not path.is_file():
            raise ValueError("input and region-plan paths must be existing regular files")
    if plan is not None and _same_path(source, plan):
        raise ValueError("source and region-plan files must be independent")
    if destination is not None:
        if any(_same_path(destination, path) for path in (source, plan) if path is not None):
            raise ValueError("OCR report must not alias any input")
        if destination.exists() or destination.is_symlink():
            raise ValueError("OCR report destination must be new")
        if not destination.parent.is_dir():
            raise ValueError("OCR report parent directory must already exist")
    return source.resolve(), plan.resolve() if plan is not None else None, destination


def _read_plan(path: Path) -> Any:
    with path.open("rb") as stream, io.BytesIO() as buffer:
        while chunk := stream.read(min(65536, _PLAN_BYTES + 1 - buffer.tell())):
            buffer.write(chunk)
            if buffer.tell() > _PLAN_BYTES:
                raise ValueError("region-plan file exceeds 16 MiB")
        raw = buffer.getvalue()
    return _strict_json_loads(raw.decode("utf-8"))


def _plans(
    path: Path,
    inputs: OcrInputLimits,
    regions: OcrRegionLimits,
    aggregate: OcrRegionPipelineLimits,
) -> tuple[OcrRegionPlan, ...]:
    value = _read_plan(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "plans"}
        or value["format"] != "metricguard.ocr-region-plans.v1"
        or not isinstance(value["plans"], list)
        or not 1 <= len(value["plans"]) <= inputs.max_pages
    ):
        raise ValueError("invalid source-bound region-plan envelope")
    plans = tuple(OcrRegionPlan.from_dict(item) for item in value["plans"])
    if tuple(plan.source.page_index for plan in plans) != tuple(range(len(plans))):
        raise ValueError("region plans must list every page once in source order")
    if len({(p.source.source_sha256, p.source.source_type, p.source.renderer) for p in plans}) != 1:
        raise ValueError("region plans must share one source and renderer identity")
    pixels = 0
    for plan in plans:
        plan.validate_limits(regions)
        pixels += inputs.check_size(plan.source.width, plan.source.height)
    if pixels > inputs.max_total_pixels:
        raise ValueError("region-plan source pages exceed the total raster budget")
    if sum(len(plan.regions) for plan in plans) > aggregate.max_total_regions:
        raise ValueError("region plans exceed the document region budget")
    if sum(plan.crop_total_pixels for plan in plans) > aggregate.max_total_crop_pixels:
        raise ValueError("region plans exceed the document crop budget")
    return plans


def _render(value: Any) -> bytes:
    output = bytearray()
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    for part in encoder.iterencode(value):
        encoded = part.encode("utf-8")
        if len(output) + len(encoded) + 1 > _REPORT_BYTES:
            raise ValueError("serialized region OCR report exceeds 64 MiB")
        output.extend(encoded)
    output.extend(b"\n")
    return bytes(output)


def _publish(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".metricguard-regions-", dir=path.parent)
    source = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive installation never replaces a racing destination.
        os.link(source, path)
    finally:
        source.unlink(missing_ok=True)


def _silence_failed_stream(stream: Any) -> None:
    try:
        descriptor = stream.fileno()
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except (OSError, ValueError, AttributeError):
        return


def _diagnostic(message: str) -> None:
    try:
        print("metricguard regions: " + message, file=sys.stderr, flush=True)
    except (OSError, UnicodeError, ValueError):
        _silence_failed_stream(sys.stderr)


def _execute(args: argparse.Namespace) -> tuple[bytes, Path | None]:
    source, plan_path, destination = _paths(args)
    inputs = OcrInputLimits(
        max_file_bytes=args.max_file_bytes,
        max_pages=args.max_pages,
        max_page_pixels=args.max_page_pixels,
        max_total_pixels=args.max_total_pixels,
        max_dimension=args.max_dimension,
        dpi=args.dpi,
    )
    regions = OcrRegionLimits(args.max_regions, args.max_crop_total_pixels)
    aggregate = OcrRegionPipelineLimits(
        max_total_regions=args.max_total_regions,
        max_total_crop_pixels=args.max_total_crop_pixels,
        max_document_words=args.max_document_words,
        max_document_lines=args.max_document_lines,
    )
    _integer(args.max_document_text_bytes, "document text bytes", 1, 64 * 1024 * 1024)
    detector_options = (
        args.direction,
        args.white_threshold,
        args.min_gutter_width,
        args.min_region_width,
    )
    if plan_path is not None and any(value is not None for value in detector_options):
        raise ValueError("detector options cannot override a source-bound plan")
    plans = _plans(plan_path, inputs, regions, aggregate) if plan_path is not None else None
    direction = args.direction if args.direction is not None else "ltr"
    threshold = args.white_threshold if args.white_threshold is not None else 255
    gutter = args.min_gutter_width if args.min_gutter_width is not None else 8
    minimum = args.min_region_width if args.min_region_width is not None else 8
    _integer(threshold, "white threshold", 1, 255)
    _integer(gutter, "minimum gutter width", 1, 10_000)
    _integer(minimum, "minimum region width", 1, 10_000)
    consumed = 0

    def planner(page: OcrPageImage) -> OcrRegionPlan:
        nonlocal consumed
        if plans is None:
            return detect_vertical_regions(
                page,
                direction=direction,
                white_threshold=threshold,
                min_gutter_width=gutter,
                min_region_width=minimum,
                limits=regions,
            )
        if consumed >= len(plans) or plans[consumed].source != page.info:
            raise ValueError("region plans do not match the complete rasterized source")
        plan = plans[consumed]
        consumed += 1
        return plan

    backend = WindowsOcrBackend(
        args.language, timeout=args.timeout, max_output_bytes=args.max_native_output_bytes
    )
    document = transcribe_region_document(
        source,
        backend,
        planner=planner,
        limits=inputs,
        region_limits=regions,
        aggregate_limits=aggregate,
        max_document_text_bytes=args.max_document_text_bytes,
    )
    if plans is not None and consumed != len(plans):
        raise ValueError("region-plan file contains unused extra pages")
    return _render(
        {
            "format": "metricguard.ocr-region-transcription.v1",
            "private_data": True,
            "document": document.to_dict(),
            "document_digest": document.digest,
        }
    ), destination


def run_region_transcribe_command(args: argparse.Namespace) -> int:
    """Return only a complete bounded report; never echo raw provider exceptions."""
    try:
        rendered, destination = _execute(args)
        if destination is not None:
            _publish(destination, rendered)
            return 0
    except (ValueError, OSError, TypeError, RuntimeError, RecursionError):
        _diagnostic(
            "command could not complete; no complete report was returned. "
            "If --output was supplied, inspect the destination directory: "
            "an artifact or private temporary file may already exist"
        )
        return 2
    try:
        binary = getattr(sys.stdout, "buffer", None)
        if binary is not None:
            binary.write(rendered)
            binary.flush()
        else:
            print(rendered.decode("utf-8"), end="", flush=True)
    except (OSError, UnicodeError, ValueError):
        _diagnostic("recognition completed, but stdout could not be delivered")
        _silence_failed_stream(sys.stdout)
    return 0
