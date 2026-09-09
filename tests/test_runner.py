from pathlib import Path

import pytest

from metricguard import EvaluationCase, MetricRunner, UndefinedPolicy, build_metric


def cases() -> list[EvaluationCase]:
    return [EvaluationCase("a", "hello", "hello"), EvaluationCase("b", "hello", "bye")]


def test_runner_order_parallel_and_cache(tmp_path: Path) -> None:
    cache = tmp_path / "results.jsonl"
    metric = build_metric("exact_match")
    first = MetricRunner(cases(), metric, workers=2).run(cache=cache)
    assert [result.case_id for result in first.results] == ["a", "b"]
    assert first.mean_score == 0.5 and first.cached == 0
    second = MetricRunner(cases(), metric, workers=2).run(cache=cache)
    assert second.cached == 2 and second.results == first.results
    changed = [EvaluationCase("a", "hello", "HELLO"), cases()[1]]
    third = MetricRunner(changed, metric).run(cache=cache)
    assert third.cached == 1
    assert third.mean_score == 0.0


def test_undefined_policies_and_errors(tmp_path: Path) -> None:
    metric = build_metric("ndcg_at_k")
    data = [EvaluationCase("undefined", {}, [])]
    with pytest.raises(ValueError, match="undefined"):
        MetricRunner(data, metric).run()
    report = MetricRunner(data, metric, undefined_policy=UndefinedPolicy.SKIP).run(
        cache=tmp_path / "cache"
    )
    assert report.results[0].skipped and report.mean_score is None


def test_fail_fast_false_records_metric_errors() -> None:
    metric = build_metric("ndcg_at_k")
    report = MetricRunner(
        [EvaluationCase("bad", {"x": -1}, [])],
        metric,
        undefined_policy=UndefinedPolicy.SKIP,
    ).run(fail_fast=False)
    assert report.errors[0][0] == "bad" and report.errors[0][1].startswith("ValueError:")


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_invalid_workers(workers: int) -> None:
    with pytest.raises(ValueError, match="workers"):
        MetricRunner(cases(), build_metric("exact_match"), workers=workers)
