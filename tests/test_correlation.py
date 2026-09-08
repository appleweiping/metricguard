from __future__ import annotations

import json
from pathlib import Path

import pytest

from metricguard import EvaluationCase, EvaluationSuite, build_metric, correlate_reports
from metricguard.cli import main
from metricguard.correlation import CorrelationReport
from metricguard.models import CaseResult, MetricValue, SuiteReport
from metricguard.service import MetricService


def _reports() -> dict[str, SuiteReport]:
    cases = tuple(
        EvaluationCase(str(index), reference, prediction)
        for index, (reference, prediction) in enumerate(
            [("a", "a"), ("b", "b"), ("c", "x"), ("d", "x")]
        )
    )
    exact = EvaluationSuite(cases).run(build_metric("exact_match"))
    token = EvaluationSuite(cases).run(build_metric("token_f1"))
    return {"token": token, "exact": exact}


def test_correlation_is_order_independent_and_handles_ties() -> None:
    report = correlate_reports(_reports())
    assert isinstance(report, CorrelationReport)
    assert report.metrics == ("exact", "token")
    pair = report.pair("token", "exact")
    assert pair.count == 4
    assert pair.pearson == pytest.approx(1.0)
    assert pair.spearman == pytest.approx(1.0)
    payload = report.to_dict()
    assert payload["pearson"]["exact"]["token"] == pytest.approx(1.0)
    assert payload["pair_counts"]["token"]["exact"] == 4
    with pytest.raises(ValueError, match="two distinct"):
        report.pair("exact", "exact")
    with pytest.raises(KeyError, match="unknown metric pair"):
        report.pair("exact", "missing")


def test_correlation_uses_pairwise_complete_scores_and_reports_constant_vectors() -> None:
    first = SuiteReport(
        "first",
        (
            CaseResult("a", MetricValue(1.0), 1.0),
            CaseResult("b", MetricValue(0.0), 0.0),
            CaseResult("c", MetricValue(None, reason="undefined"), None, skipped=True),
        ),
    )
    second = SuiteReport(
        "second",
        (
            CaseResult("a", MetricValue(2.0), 2.0),
            CaseResult("b", MetricValue(2.0), 2.0),
            CaseResult("c", MetricValue(2.0), 2.0),
        ),
    )
    pair = correlate_reports({"z": second, "a": first}).pair("a", "z")
    assert pair.count == 2
    assert pair.pearson is None
    assert pair.spearman is None
    assert pair.reason == "correlation undefined for a constant score vector"


def test_correlation_rejects_insufficient_pairs_and_bad_inputs() -> None:
    report = SuiteReport("one", (CaseResult("a", MetricValue(1.0), 1.0),))
    other = SuiteReport("two", (CaseResult("a", MetricValue(1.0), 1.0),))
    pair = correlate_reports({"one": report, "two": other}).pairs[0]
    assert pair.count == 1
    assert pair.reason == "fewer than 2 pairwise-complete cases"
    with pytest.raises(ValueError, match="at least 2"):
        correlate_reports({"one": report, "two": other}, minimum_count=1)
    with pytest.raises(ValueError, match="non-empty"):
        correlate_reports({})
    with pytest.raises(ValueError, match="non-empty strings"):
        correlate_reports({"": report, "two": other})
    with pytest.raises(TypeError, match="SuiteReport"):
        correlate_reports({"one": object(), "two": other})  # type: ignore[arg-type]


def test_correlation_cli_and_service(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            json.dumps({"id": str(index), "reference": ref, "prediction": pred})
            for index, (ref, pred) in enumerate([("a", "a"), ("b", "x"), ("c", "c")])
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "correlation.json"
    assert (
        main(
            [
                "correlate",
                str(cases),
                "--metrics",
                "exact_match,token_f1",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"] == ["exact_match", "token_f1"]
    assert capsys.readouterr().out == ""
    assert (
        main(
            [
                "correlate",
                str(cases),
                "--metrics",
                "exact_match,token_f1",
                "--format",
                "markdown",
            ]
        )
        == 0
    )
    assert "Pearson correlation" in capsys.readouterr().out
    result = MetricService().dispatch(
        {
            "operation": "correlate",
            "cases": str(cases),
            "metrics": ["exact_match", "token_f1"],
        }
    )
    assert result["correlation"]["schema_version"] == 1


def test_correlation_rejects_duplicate_case_ids() -> None:
    report = SuiteReport(
        "bad",
        (
            CaseResult("a", MetricValue(1.0), 1.0),
            CaseResult("a", MetricValue(0.0), 0.0),
        ),
    )
    other = SuiteReport("other", (CaseResult("a", MetricValue(1.0), 1.0),))
    with pytest.raises(ValueError, match="duplicate case ID"):
        correlate_reports({"bad": report, "other": other})
