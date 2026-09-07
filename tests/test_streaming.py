from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from metricguard import EvaluationCase, MetricValue, UndefinedPolicy, evaluate_stream
from metricguard.cli import main
from metricguard.io import CaseFormatError, iter_cases
from metricguard.metrics import build_metric


@dataclass(frozen=True)
class _UndefinedMetric:
    name: str = "undefined"

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        return MetricValue(None, reason="not defined")


@dataclass(frozen=True)
class _FailingMetric:
    name: str = "failing"

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        if reference == "bad":
            raise RuntimeError("bad input")
        return MetricValue(1.0)


def _write_cases(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "reference": "yes", "prediction": "yes", "tags": ["ok"]}),
                json.dumps({"id": "b", "reference": "no", "prediction": "yes", "tags": ["review"]}),
                json.dumps(
                    {"id": "c", "reference": "yes", "prediction": "yes", "tags": ["ok", "review"]}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_iter_cases_is_incremental_and_preserves_duplicate_guard(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    _write_cases(path)
    cases = iter_cases(path)
    assert next(cases).case_id == "a"
    assert [case.case_id for case in cases] == ["b", "c"]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"id":"a","reference":"x","prediction":"x"}\n'
        '{"id":"a","reference":"x","prediction":"x"}\n',
        encoding="utf-8",
    )
    with pytest.raises(CaseFormatError, match="duplicate"):
        tuple(iter_cases(duplicate))


def test_streaming_evaluation_aggregates_without_case_results() -> None:
    cases = (
        EvaluationCase("a", "yes", "yes", tags=("ok",)),
        EvaluationCase("b", "no", "yes", tags=("review",)),
        EvaluationCase("c", "yes", "yes", tags=("ok", "review")),
    )
    report = evaluate_stream(cases, build_metric("exact_match"))
    assert report.case_count == 3
    assert report.scored_count == 3
    assert report.mean_score == pytest.approx(2 / 3)
    assert report.passed
    assert [(item.tag, item.case_count, item.mean_score) for item in report.by_tag] == [
        ("ok", 2, 1.0),
        ("review", 2, 0.5),
    ]
    payload = report.to_dict()
    assert payload["summary"]["errors"] == 0  # type: ignore[index]


def test_streaming_undefined_policies_and_non_fatal_errors() -> None:
    cases = (EvaluationCase("a", "x", "y"),)
    skipped = evaluate_stream(cases, _UndefinedMetric(), undefined_policy=UndefinedPolicy.SKIP)
    assert skipped.skipped_count == 1 and skipped.scored_count == 0
    zero = evaluate_stream(cases, _UndefinedMetric(), undefined_policy=UndefinedPolicy.ZERO)
    assert zero.mean_score == 0.0
    with pytest.raises(ValueError, match="failed"):
        evaluate_stream(cases, _UndefinedMetric())
    failing_cases = (EvaluationCase("bad", "bad", "x"), EvaluationCase("ok", "ok", "x"))
    report = evaluate_stream(failing_cases, _FailingMetric(), fail_fast=False)
    assert report.errors == (("bad", "RuntimeError: bad input"),)
    assert report.scored_count == 1
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_stream((cases[0], cases[0]), build_metric("exact_match"))


def test_stream_run_cli_emits_bounded_aggregate(tmp_path: Path) -> None:
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases)
    output = tmp_path / "report.json"
    assert main(["stream-run", str(cases), "--metric", "exact_match", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 3
    assert payload["by_tag"][0]["tag"] == "ok"
