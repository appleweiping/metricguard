from __future__ import annotations

import sys

import pytest

from metricguard import (
    CommandOcrBackend,
    OcrImageCase,
    load_ocr_cases,
    run_ocr_backend_benchmark,
    run_ocr_benchmark,
)


def test_ocr_benchmark_reports_cer_wer_and_exact_match(tmp_path) -> None:
    path = tmp_path / "ocr.jsonl"
    path.write_text(
        '{"id":"one","reference":"hello world","prediction":"hello world"}\n'
        '{"id":"empty","reference":"","prediction":"noise"}\n',
        encoding="utf-8",
    )
    report = run_ocr_benchmark(load_ocr_cases(path))
    assert report.cases == 2
    assert report.character_error_rate.skipped_count == 1
    assert report.word_error_rate.skipped_count == 1
    assert report.exact_match_rate == 0.5
    assert report.to_dict()["word_error_rate"]["scored"] == 1


def test_ocr_benchmark_rejects_empty_case_collection() -> None:
    try:
        run_ocr_benchmark(())
    except ValueError as error:
        assert "at least one" in str(error)
    else:  # pragma: no cover
        raise AssertionError("empty OCR benchmark was accepted")


def test_command_backend_is_shell_free_and_feeds_ocr_benchmark(tmp_path) -> None:
    image = tmp_path / "scan.bin"
    image.write_bytes(b"placeholder")
    backend = CommandOcrBackend(
        [sys.executable, "-c", "print('hello world')", "{image}"], max_output_bytes=128
    )
    assert backend(image) == "hello world"
    report = run_ocr_backend_benchmark((OcrImageCase("one", image, "hello world"),), backend)
    assert report.exact_match_rate == 1.0
    with pytest.raises(ValueError, match="placeholder"):
        CommandOcrBackend([sys.executable, "-c", "print('x')"])
    with pytest.raises(ValueError, match="does not exist"):
        backend(tmp_path / "missing.bin")
