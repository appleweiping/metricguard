from __future__ import annotations

from metricguard import load_ocr_cases, run_ocr_benchmark


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
