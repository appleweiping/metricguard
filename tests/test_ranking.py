import math
from itertools import permutations

import pytest

from metricguard import (
    EvaluationCase,
    EvaluationSuite,
    RankingMetric,
    UndefinedPolicy,
    build_metric,
)
from metricguard.ranking import RANKING_METRICS


@pytest.mark.parametrize(
    "name, expected",
    [
        ("precision_at_k", 2 / 3),
        ("recall_at_k", 2 / 3),
        ("reciprocal_rank_at_k", 1 / 2),
        ("average_precision_at_k", (1 / 2 + 2 / 3) / 3),
    ],
)
def test_hand_computed_binary_metrics(name: str, expected: float) -> None:
    metric = build_metric({"kind": name, "k": 3})
    result = metric.evaluate({"a": 3, "b": 1, "c": 2}, ["missing", "b", "a", "c"])
    assert result.score == pytest.approx(expected)
    assert result.details["unjudged"] == 1


def test_linear_ndcg_and_large_relevance() -> None:
    metric = RankingMetric(k=2)
    expected = (1 + 3 / math.log2(3)) / (3 + 2 / math.log2(3))
    assert metric.evaluate({"a": 3, "b": 2, "c": 1}, ["c", "a"]).score == pytest.approx(expected)
    assert metric.evaluate({"a": 1e308, "b": 1e308}, ["a", "b"]).score == 1
    assert metric.evaluate({"a": 1e-300}, ["a"]).score == 1


def test_ndcg_ideal_ranking_is_maximal() -> None:
    judgments = {"a": 4, "b": 2, "c": 1, "d": 0}
    metric = RankingMetric(k=4)
    for order in permutations(judgments):
        score = metric.evaluate(judgments, list(order)).score
        assert score is not None and 0 <= score <= 1
    assert metric.evaluate(judgments, ["a", "b", "c", "d"]).score == 1


@pytest.mark.parametrize("name", RANKING_METRICS)
def test_empty_rankings_and_undefined_policy(name: str) -> None:
    metric = RankingMetric(name=name)
    assert metric.evaluate({"a": 1}, []).score == 0
    assert metric.evaluate({}, ["a"]).score is None
    cases = [EvaluationCase("q", {}, [])]
    with pytest.raises(ValueError, match="undefined"):
        EvaluationSuite(cases).run(metric)
    result = EvaluationSuite(cases, undefined_policy=UndefinedPolicy.SKIP).run(metric)
    assert result.skipped_count == 1
    assert result.mean_score is None


def test_short_run_precision_and_ap_denominators() -> None:
    assert RankingMetric("precision_at_k", 10).evaluate({"a": 1}, ["a"]).score == 0.1
    assert RankingMetric("average_precision_at_k", 10).evaluate({"a": 1}, ["a"]).score == 1


@pytest.mark.parametrize(
    "reference",
    [
        {"a": True},
        {"a": -1},
        {"a": float("nan")},
        {"a": float("inf")},
        {1: 2},
        {"": 2},
        {"a": 10**400},
    ],
)
def test_invalid_judgments(reference: object) -> None:
    with pytest.raises(ValueError):
        RankingMetric().evaluate(reference, [])


@pytest.mark.parametrize("prediction", [["a", "a"], [""], [1]])
def test_invalid_predictions(prediction: object) -> None:
    with pytest.raises(ValueError):
        RankingMetric(k=1).evaluate({"a": 1}, prediction)


@pytest.mark.parametrize("k", [True, 0, -1, 1.5])
def test_invalid_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RankingMetric(k=k)


def test_configuration_and_input_type_errors() -> None:
    with pytest.raises(ValueError, match="unknown ranking"):
        RankingMetric("unknown")
    with pytest.raises(ValueError, match="unknown ranking options"):
        build_metric({"kind": "ndcg_at_k", "ties": "average"})
    with pytest.raises(TypeError, match="reference"):
        RankingMetric().evaluate([], [])
    with pytest.raises(TypeError, match="prediction"):
        RankingMetric().evaluate({}, "a")
