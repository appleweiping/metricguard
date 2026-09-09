"""Cache results must be interchangeable with a fresh evaluation."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from metricguard import EvaluationCase, MetricRunner, MetricValue, UndefinedPolicy, build_metric
from metricguard.metrics import BUILTIN_METRIC_NAMES
from metricguard.ranking import RANKING_METRICS


def test_configuration_changes_invalidate_scores(tmp_path: Path) -> None:
    scenarios = [
        ("exact_match", {"normalizer": {"lowercase": True}}, "HELLO", "hello"),
        ("numeric_equivalence", {"absolute_tolerance": "0.1"}, "1", "1.01"),
        ("recall_at_k", {"k": 1}, {"a": 1}, ["b", "a"]),
    ]
    for name, options, reference, prediction in scenarios:
        path = tmp_path / f"{name}.jsonl"
        data = [EvaluationCase("same", reference, prediction)]
        first = MetricRunner(data, build_metric(name)).run(cache=path)
        metric = build_metric({"kind": name, **options})
        changed = MetricRunner(data, metric).run(cache=path)
        fresh = MetricRunner(data, metric).run()
        assert changed.cached == 0 and changed.results == fresh.results
        assert first.mean_score != changed.mean_score
        resumed = MetricRunner(data, metric).run(cache=path)
        assert resumed.cached == 1 and resumed.results == fresh.results


def test_raw_undefined_result_is_resolved_by_each_current_policy(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("missing", "not numeric", "1")]
    metric = build_metric("numeric_equivalence")
    for policy in (UndefinedPolicy.ZERO, UndefinedPolicy.ONE, UndefinedPolicy.SKIP):
        report = MetricRunner(data, metric, undefined_policy=policy).run(cache=path)
        expected = MetricRunner(data, metric, undefined_policy=policy).run()
        assert report.results == expected.results
        assert report.cached == (0 if policy is UndefinedPolicy.ZERO else 1)
    with pytest.raises(ValueError, match=r"undefined.*missing"):
        MetricRunner(data, metric).run(cache=path)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["score"] is None
    assert "resolved_score" not in row and "skipped" not in row


def test_numeric_context_invalidates_cache_and_is_shared_by_workers(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("a", "1.1234567", "1")]
    metric = build_metric({"kind": "numeric_equivalence", "absolute_tolerance": "0.12345"})
    with localcontext() as context:
        context.prec = 3
        first = MetricRunner(data, metric, workers=2).run(cache=path)
        assert first.mean_score == MetricRunner(data, metric).run().mean_score == 1.0
        context.prec = 4
        changed = MetricRunner(data, metric, workers=2).run(cache=path)
        assert changed.cached == 0
        assert changed.mean_score == MetricRunner(data, metric).run().mean_score == 0.0
        assert MetricRunner(data, metric).run(cache=path).cached == 1


def test_undefined_failure_is_checkpointed_for_later_policy(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("missing", {}, [])]
    metric = build_metric("ndcg_at_k")
    with pytest.raises(ValueError, match="undefined"):
        MetricRunner(data, metric).run(cache=path)
    resumed = MetricRunner(data, metric, undefined_policy=UndefinedPolicy.ZERO).run(cache=path)
    assert resumed.cached == 1 and resumed.mean_score == 0.0


class CountingMetric:
    name = "custom"

    def __init__(self, *, multiplier: float = 1.0, fail: bool = False) -> None:
        self.multiplier = multiplier
        self.fail = fail
        self.calls: list[str] = []

    def cache_identity(self) -> dict[str, Any]:
        # Operational counters and a transient backend error are not configuration.
        return {"revision": 1, "multiplier": self.multiplier}

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        self.calls.append(str(reference))
        if reference == "bad" and self.fail:
            raise RuntimeError("backend unavailable")
        return MetricValue(self.multiplier * float(reference == prediction))


@pytest.mark.parametrize("workers", [1, 3])
def test_exceptions_are_retried_and_not_hidden_by_resume(tmp_path: Path, workers: int) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("a", "good", "good"), EvaluationCase("b", "bad", "bad")]
    broken = CountingMetric(fail=True)
    first = MetricRunner(data, broken, workers=workers, undefined_policy=UndefinedPolicy.ZERO).run(
        cache=path, fail_fast=False
    )
    assert first.errors == (("b", "RuntimeError: backend unavailable"),)
    assert first.mean_score == 0.5
    still_broken = CountingMetric(fail=True)
    with pytest.raises(ValueError, match="backend unavailable"):
        MetricRunner(data, still_broken, workers=workers).run(cache=path)
    assert still_broken.calls == ["bad"]
    fixed = CountingMetric()
    resumed = MetricRunner(data, fixed, workers=workers).run(cache=path)
    assert fixed.calls == ["bad"]
    assert resumed.cached == 1 and resumed.mean_score == 1.0 and not resumed.errors


def test_fail_fast_preserves_completed_work_and_stops_serial_evaluation(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase(name, name, name) for name in ("first", "bad", "last")]
    metric = CountingMetric(fail=True)
    with pytest.raises(ValueError, match="backend unavailable"):
        MetricRunner(data, metric).run(cache=path)
    assert metric.calls == ["first", "bad"]
    fixed = CountingMetric()
    resumed = MetricRunner(data, fixed).run(cache=path)
    assert resumed.cached == 1 and fixed.calls == ["bad", "last"]


def test_plugin_identity_covers_configuration_and_requires_explicit_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("a", "x", "x")]
    MetricRunner(data, CountingMetric()).run(cache=path)
    changed = MetricRunner(data, CountingMetric(multiplier=0.5)).run(cache=path)
    assert changed.cached == 0 and changed.mean_score == 0.5

    class NoIdentity:
        name = "custom"

        def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
            return MetricValue(0.25)

    assert MetricRunner(data, NoIdentity()).run().mean_score == 0.25
    before = path.read_bytes()
    with pytest.raises(ValueError, match="require cache_identity"):
        MetricRunner(data, NoIdentity()).run(cache=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("identity", [None, {}, "revision", {"x": object()}, {1: "x"}])
def test_plugin_cache_identity_must_be_lossless_json(tmp_path: Path, identity: Any) -> None:
    class InvalidIdentity(CountingMetric):
        def cache_identity(self) -> Any:
            return identity

    with pytest.raises(ValueError):
        MetricRunner([], InvalidIdentity()).run(cache=tmp_path / "scores.jsonl")


def test_uncached_execution_preserves_non_json_metric_inputs(tmp_path: Path) -> None:
    data = [EvaluationCase("a", Decimal("1"), Decimal("1.0"))]
    metric = build_metric("numeric_equivalence")
    assert MetricRunner(data, metric).run().mean_score == 1.0
    with pytest.raises(ValueError, match=r"case 'a'.*JSON"):
        MetricRunner(data, metric).run(cache=tmp_path / "scores.jsonl")


def test_all_builtin_identities_can_resume(tmp_path: Path) -> None:
    for name in BUILTIN_METRIC_NAMES:
        metric = build_metric(name)
        data = [EvaluationCase("case", "1", "1")]
        if name in RANKING_METRICS:
            data = [EvaluationCase("case", {"a": 1}, ["a"])]
        path = tmp_path / f"{name}.jsonl"
        first = MetricRunner(data, metric).run(cache=path)
        second = MetricRunner(data, build_metric(name)).run(cache=path)
        assert second.cached == 1 and second.results == first.results


def test_parallel_completion_order_and_periodic_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    second_finished = Event()

    class Reordered(CountingMetric):
        def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
            if reference == "first":
                assert second_finished.wait(5)
            else:
                second_finished.set()
            return super().evaluate(reference, prediction)

    data = [EvaluationCase(name, name, name) for name in ("first", "second")]
    metric = Reordered()
    report = MetricRunner(data, metric, workers=2).run(cache=path, checkpoint_interval=1)
    assert metric.calls == ["second", "first"]
    assert [row.case_id for row in report.results] == ["first", "second"]
    assert MetricRunner(data, metric, workers=2).run(cache=path).cached == 2


def test_cache_snapshots_mutable_plugin_details(tmp_path: Path) -> None:
    class ReusedDetails(CountingMetric):
        def __init__(self) -> None:
            super().__init__()
            self.details: dict[str, Any] = {"observed": []}

        def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
            self.details["observed"].append(reference)
            return MetricValue(1.0, details=self.details)

    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase(name, name, name) for name in ("first", "second")]
    first = MetricRunner(data, ReusedDetails()).run(cache=path)
    assert first.results[0].raw.details == {"observed": ["first"]}
    assert first.results[1].raw.details == {"observed": ["first", "second"]}
    resumed = MetricRunner(data, ReusedDetails()).run(cache=path)
    assert resumed.cached == 2 and resumed.results == first.results


def test_unserializable_output_preserves_previous_successes(tmp_path: Path) -> None:
    class InvalidDetails(CountingMetric):
        def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
            if reference == "bad":
                return MetricValue(1.0, details={"payload": object()})
            return super().evaluate(reference, prediction)

    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase(name, name, name) for name in ("first", "bad")]
    with pytest.raises(ValueError, match="unsupported JSON value"):
        MetricRunner(data, InvalidDetails()).run(cache=path)
    # The invalid case is absent; the already computed score remains resumable.
    resumed = MetricRunner(data[:1], InvalidDetails()).run(cache=path)
    assert resumed.cached == 1 and resumed.mean_score == 1.0


@pytest.mark.parametrize(
    "change, message",
    [
        ({"version": 3}, "version"),
        ({"version": True}, "version"),
        ({"extra": True}, "fields"),
        ({"case_id": " "}, "case_id"),
        ({"fingerprint": "bad"}, "digests"),
        ({"score": True}, "finite number"),
        ({"score": float("nan")}, "finite number"),
        ({"score": None, "reason": None}, "requires a reason"),
        ({"reason": 7}, "reason must"),
        ({"details": []}, "details must"),
        ({"details": {"nested": float("inf")}}, "JSON compliant"),
        ({"score": 0.125}, "checksum mismatch"),
    ],
)
def test_corrupt_cache_fails_with_location_and_preserves_file(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("a", "x", "x")]
    metric = build_metric("exact_match")
    MetricRunner(data, metric).run(cache=path)
    row = json.loads(path.read_text(encoding="utf-8"))
    row.update(change)
    path.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match=f"line 2:.*{message}"):
        MetricRunner(data, metric).run(cache=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "text, message",
    [("{", "Expecting"), ("[]", "JSON object"), ('{"x":1,"x":2}', "duplicate JSON key")],
)
def test_invalid_json_cache_has_context(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=f"line 1:.*{message}"):
        MetricRunner([], build_metric("exact_match")).run(cache=path)


def test_duplicate_rows_are_rejected_and_legacy_rows_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    data = [EvaluationCase("a", "x", "x")]
    metric = build_metric("exact_match")
    MetricRunner(data, metric).run(cache=path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text + text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2:.*duplicate cached"):
        MetricRunner(data, metric).run(cache=path)
    legacy = json.loads(text)
    del legacy["version"]
    legacy["score"] = 0.0
    path.write_text(json.dumps(legacy), encoding="utf-8")
    report = MetricRunner(data, metric).run(cache=path)
    assert report.cached == 0 and report.mean_score == 1.0


def test_atomic_replace_failure_preserves_prior_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scores.jsonl"
    metric = build_metric("exact_match")
    MetricRunner([EvaluationCase("a", "x", "x")], metric).run(cache=path)
    before = path.read_bytes()

    def fail_replace(source: Any, target: Any) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("metricguard._runner_cache.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk unavailable"):
        MetricRunner([EvaluationCase("a", "x", "y")], metric).run(cache=path, checkpoint_interval=1)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_checkpoint_interval_is_validated(interval: Any) -> None:
    with pytest.raises(ValueError, match="checkpoint_interval"):
        MetricRunner([], build_metric("exact_match")).run(checkpoint_interval=interval)


def test_fail_fast_flag_is_strict() -> None:
    with pytest.raises(TypeError, match="fail_fast"):
        MetricRunner([], build_metric("exact_match")).run(fail_fast="yes")  # type: ignore[arg-type]
