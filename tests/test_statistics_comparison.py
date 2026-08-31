import pytest

from metricguard.comparison import compare_case_sets
from metricguard.metrics import ExactMatch
from metricguard.models import CaseResult, EvaluationCase, MetricValue, SuiteReport
from metricguard.statistics import (
    BootstrapConfig,
    confidence_interval,
    paired_comparison,
    report_confidence_interval,
    summarize_by_tag,
)
from metricguard.suite import EvaluationSuite


def report(name: str, *scores: float | None) -> SuiteReport:
    return SuiteReport(
        name,
        tuple(
            CaseResult(
                f"c{index}",
                MetricValue(score, None if score is not None else "undefined"),
                score,
                skipped=score is None,
            )
            for index, score in enumerate(scores)
        ),
    )


def test_confidence_interval_is_deterministic_and_degenerate() -> None:
    config = BootstrapConfig(samples=101, confidence=0.9, seed=42)
    first = confidence_interval((0.0, 0.5, 1.0), config)
    assert first == confidence_interval((0.0, 0.5, 1.0), config)
    assert first.point == 0.5
    assert first.lower <= first.point <= first.upper
    constant = confidence_interval((0.25, 0.25), config)
    assert (constant.lower, constant.point, constant.upper) == (0.25, 0.25, 0.25)


def test_report_confidence_interval_ignores_explicitly_skipped_results() -> None:
    interval = report_confidence_interval(report("metric", 1.0, None), BootstrapConfig(samples=5))
    assert interval.point == interval.lower == interval.upper == 1.0


def test_paired_comparison_and_gate() -> None:
    comparison = paired_comparison(
        report("metric", 0.0, 0.5, 1.0),
        report("metric", 0.5, 0.5, 1.0),
        config=BootstrapConfig(samples=101, seed=3),
        minimum_delta=0.1,
    )
    assert comparison.delta.point == pytest.approx(1 / 6)
    assert comparison.baseline_mean == 0.5
    assert comparison.candidate_mean == pytest.approx(2 / 3)
    assert comparison.compared_count == 3
    assert comparison.passed_gate
    assert 0 <= comparison.probability_improvement <= 1
    assert 0 <= comparison.two_sided_p_value <= 1


def test_paired_comparison_optional_lower_bound_gate_and_omissions() -> None:
    comparison = paired_comparison(
        report("metric", 1.0, None),
        report("metric", 1.0, None),
        config=BootstrapConfig(samples=9),
        minimum_lower_bound=0.01,
    )
    assert comparison.omitted_count == 1
    assert not comparison.passed_gate


def test_paired_sign_flip_p_value_is_deterministic_and_nonzero() -> None:
    baseline = report("metric", *(0.0 for _ in range(8)))
    candidate = report("metric", *(1.0 for _ in range(8)))
    config = BootstrapConfig(samples=1_023, seed=11)
    first = paired_comparison(baseline, candidate, config=config)
    second = paired_comparison(baseline, candidate, config=config)
    assert first.two_sided_p_value == second.two_sided_p_value
    assert 0 < first.two_sided_p_value < 0.05
    unchanged = paired_comparison(baseline, baseline, config=BootstrapConfig(samples=17))
    assert unchanged.two_sided_p_value == 1.0


def test_lower_is_better_orients_interval_probability_and_gate() -> None:
    comparison = paired_comparison(
        report("loss", 1.0, 0.8, 0.9),
        report("loss", 0.4, 0.5, 0.6),
        config=BootstrapConfig(samples=101, seed=5),
        direction="lower",
        minimum_delta=0.2,
        minimum_lower_bound=0.1,
    )
    assert comparison.delta.point < 0
    assert comparison.improvement.point > 0
    assert comparison.improvement.lower == -comparison.delta.upper
    assert comparison.probability_improvement > 0.5
    assert comparison.passed_gate
    with pytest.raises(ValueError, match="direction"):
        paired_comparison(
            report("loss", 1.0),
            report("loss", 0.0),
            direction="sideways",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        (report("left", 1.0), report("right", 1.0), "same metric"),
        (report("m", 1.0), SuiteReport("m", ()), "different case IDs"),
        (report("m", None), report("m", 1.0), "only one side"),
        (report("m", None), report("m", None), "at least one resolved"),
    ],
)
def test_paired_comparison_rejects_misalignment(
    baseline: SuiteReport, candidate: SuiteReport, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        paired_comparison(baseline, candidate, config=BootstrapConfig(samples=3))


def test_tag_summaries_are_sorted_and_preserve_skips() -> None:
    cases = (
        EvaluationCase("a", "x", "x", tags=("short", "english")),
        EvaluationCase("b", "x", "y", tags=("english",)),
    )
    suite_report = EvaluationSuite(cases).run(ExactMatch())
    summaries = summarize_by_tag(suite_report, cases)
    assert [summary.tag for summary in summaries] == ["english", "short"]
    assert summaries[0].case_count == summaries[0].scored_count == 2
    assert summaries[0].mean_score == 0.5


def test_compare_case_sets_aligns_by_id_not_file_order() -> None:
    baseline = (
        EvaluationCase("a", "yes", "no"),
        EvaluationCase("b", "yes", "yes"),
    )
    candidate = (
        EvaluationCase("b", "yes", "yes"),
        EvaluationCase("a", "yes", "yes"),
    )
    result = compare_case_sets(
        baseline,
        candidate,
        metric=ExactMatch(),
        bootstrap=BootstrapConfig(samples=21),
    )
    assert result.delta.point == 0.5
    assert result.passed_gate


def test_compare_case_sets_treats_tag_order_as_non_semantic() -> None:
    result = compare_case_sets(
        (EvaluationCase("a", "yes", "no", tags=("quality", "english")),),
        (EvaluationCase("a", "yes", "yes", tags=("english", "quality")),),
        metric=ExactMatch(),
        bootstrap=BootstrapConfig(samples=5),
    )
    assert result.delta.point == 1.0


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ((EvaluationCase("other", "yes", "yes"),), "different IDs"),
        ((EvaluationCase("a", "no", "yes"),), "different references"),
        ((EvaluationCase("a", "yes", "yes", tags=("new",)),), "different tags"),
    ],
)
def test_compare_case_sets_rejects_dataset_drift(
    candidate: tuple[EvaluationCase, ...], message: str
) -> None:
    baseline = (EvaluationCase("a", "yes", "yes"),)
    with pytest.raises(ValueError, match=message):
        compare_case_sets(
            baseline,
            candidate,
            metric=ExactMatch(),
            bootstrap=BootstrapConfig(samples=3),
        )


def test_compare_case_sets_does_not_equate_booleans_and_integers() -> None:
    with pytest.raises(ValueError, match="different references"):
        compare_case_sets(
            (EvaluationCase("a", True, "yes"),),
            (EvaluationCase("a", 1, "yes"),),
            metric=ExactMatch(),
            bootstrap=BootstrapConfig(samples=3),
        )


@pytest.mark.parametrize(
    "config",
    [
        {"samples": 0},
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": float("nan")},
    ],
)
def test_bootstrap_config_rejects_invalid_values(config: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        BootstrapConfig(**config)  # type: ignore[arg-type]


def test_confidence_interval_rejects_empty_nonfinite_and_non_tuple() -> None:
    with pytest.raises(ValueError, match="at least one"):
        confidence_interval(())
    with pytest.raises(ValueError, match="finite"):
        confidence_interval((float("nan"),))
    with pytest.raises(TypeError, match="tuple"):
        confidence_interval([1.0])  # type: ignore[arg-type]


def test_paired_reports_reject_duplicate_ids_and_nonfinite_resolved_scores() -> None:
    duplicate = SuiteReport(
        "m",
        (
            CaseResult("same", MetricValue(1.0), 1.0),
            CaseResult("same", MetricValue(0.0), 0.0),
        ),
    )
    with pytest.raises(ValueError, match="duplicate case ID"):
        paired_comparison(duplicate, duplicate, config=BootstrapConfig(samples=3))

    nonfinite = SuiteReport("m", (CaseResult("a", MetricValue(1.0), float("nan")),))
    finite = SuiteReport("m", (CaseResult("a", MetricValue(1.0), 1.0),))
    with pytest.raises(ValueError, match="finite"):
        paired_comparison(nonfinite, finite, config=BootstrapConfig(samples=3))


def test_tag_summary_rejects_duplicate_report_results() -> None:
    duplicate = SuiteReport(
        "m",
        (
            CaseResult("a", MetricValue(1.0), 1.0),
            CaseResult("a", MetricValue(1.0), 1.0),
        ),
    )
    with pytest.raises(ValueError, match="duplicate case ID"):
        summarize_by_tag(duplicate, (EvaluationCase("a", "x", "x", tags=("tag",)),))
