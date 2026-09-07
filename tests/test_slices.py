import pytest

from metricguard import (
    BootstrapConfig,
    EvaluationCase,
    EvaluationSuite,
    SliceSummary,
    UndefinedPolicy,
    compare_by_metadata,
    summarize_by_metadata,
)
from metricguard.edit_metrics import CharacterErrorRate


def _report():
    cases = (
        EvaluationCase("a", "hello", "hello", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("b", "hello", "hullo", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("c", "hello", "hello", metadata={"cohort": {"lang": "fr"}}),
        EvaluationCase("d", "hello", "hello", metadata={}),
    )
    return cases, EvaluationSuite(cases).run(CharacterErrorRate())


def test_metadata_slices_are_sorted_and_nested() -> None:
    cases, report = _report()
    summaries = summarize_by_metadata(report, cases, "cohort.lang")
    assert summaries == (
        SliceSummary("cohort.lang", "<missing>", 1, 1, 0, 0.0),
        SliceSummary("cohort.lang", "en", 2, 2, 0, 0.1),
        SliceSummary("cohort.lang", "fr", 1, 1, 0, 0.0),
    )
    assert summaries[0].to_dict()["value"] == "<missing>"


def test_slices_support_complex_values_and_minimum_count() -> None:
    cases, _ = _report()
    cases = tuple(
        EvaluationCase(
            case.case_id,
            case.reference,
            case.prediction,
            metadata=(
                {"group": [1, 2]}
                if case.case_id in {"a", "b"}
                else ({"group": "other"} if case.case_id == "c" else {})
            ),
        )
        for case in cases
    )
    report = EvaluationSuite(cases).run(CharacterErrorRate())
    summaries = summarize_by_metadata(report, cases, "group", min_count=2)
    assert len(summaries) == 1
    assert summaries[0].value == "[1,2]"


def test_invalid_population_and_arguments_are_rejected() -> None:
    cases, report = _report()
    for field in ("", " "):
        try:
            summarize_by_metadata(report, cases, field)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid field")
    try:
        summarize_by_metadata(report, cases[:-1], "cohort.lang")
    except ValueError as error:
        assert "same case IDs" in str(error)
    else:
        raise AssertionError("expected population mismatch")


def test_metadata_slice_comparison_is_paired_and_serializable() -> None:
    baseline = (
        EvaluationCase("a", "hello", "hello", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("b", "hello", "hullo", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("c", "hello", "hello", metadata={"cohort": {"lang": "fr"}}),
    )
    candidate = (
        EvaluationCase("a", "hello", "hello", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("b", "hello", "hello", metadata={"cohort": {"lang": "en"}}),
        EvaluationCase("c", "hello", "hello", metadata={"cohort": {"lang": "fr"}}),
    )
    comparisons = compare_by_metadata(
        baseline,
        candidate,
        metric=CharacterErrorRate(),
        field="cohort.lang",
        undefined_policy=UndefinedPolicy.ERROR,
        bootstrap=BootstrapConfig(samples=32, confidence=0.9, seed=7),
        direction="lower",
    )
    assert [item.value for item in comparisons] == ["en", "fr"]
    english = comparisons[0]
    assert english.comparison is not None
    assert english.comparison.compared_count == 2
    assert english.passed
    payload = english.to_dict()
    assert payload["comparison"]["improvement"] > 0
    assert payload["passed"] is True


def test_metadata_slice_comparison_rejects_metadata_drift() -> None:
    baseline = (EvaluationCase("a", "yes", "yes", metadata={"group": "control"}),)
    candidate = (EvaluationCase("a", "yes", "yes", metadata={"group": "treatment"}),)
    with pytest.raises(ValueError, match="changed metadata slice value"):
        compare_by_metadata(
            baseline,
            candidate,
            metric=CharacterErrorRate(),
            field="group",
            undefined_policy=UndefinedPolicy.ERROR,
        )


def test_metadata_slice_comparison_reports_small_slice_errors() -> None:
    baseline = (EvaluationCase("a", "yes", "no", metadata={"group": "tiny"}),)
    candidate = (EvaluationCase("a", "yes", "yes", metadata={"group": "tiny"}),)
    comparisons = compare_by_metadata(
        baseline,
        candidate,
        metric=CharacterErrorRate(),
        field="group",
        undefined_policy=UndefinedPolicy.ERROR,
        min_count=2,
    )
    assert comparisons == ()
