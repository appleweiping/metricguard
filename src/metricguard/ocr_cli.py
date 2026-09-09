"""Reference-free local transcription CLI and atomic JSON publication."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from .native_ocr import WindowsOcrBackend
from .ocr_inputs import OcrInputLimits
from .ocr_pipeline import transcribe_document


def configure_transcribe_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-page-pixels", type=int, default=20_000_000)
    parser.add_argument("--max-total-pixels", type=int, default=100_000_000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(handler=run_transcribe_command)


def run_transcribe_command(args: argparse.Namespace) -> int:
    source = args.input.resolve()
    destination = args.output.resolve() if args.output is not None else None
    if destination is not None:
        if source == destination or (
            source.exists() and destination.exists() and source.samefile(destination)
        ):
            raise ValueError("OCR output must not overwrite or alias the source input")
        if destination.exists() and not destination.is_file():
            raise ValueError("OCR output must be a file")
    limits = OcrInputLimits(
        max_file_bytes=args.max_file_bytes,
        max_pages=args.max_pages,
        max_page_pixels=args.max_page_pixels,
        max_total_pixels=args.max_total_pixels,
        dpi=args.dpi,
    )
    backend = WindowsOcrBackend(args.language, timeout=args.timeout)
    document = transcribe_document(source, backend, limits=limits)
    rendered = (
        json.dumps(
            document.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    if len(rendered.encode("utf-8")) > 64 * 1024 * 1024:
        raise ValueError("serialized OCR report exceeds 64 MiB")
    if destination is None:
        print(rendered, end="")
        return 0
    temporary: str | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)
    return 0
