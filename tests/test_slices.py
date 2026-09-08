import pytest

from metricguard import (
    BootstrapConfig,
    EvaluationCase,
    EvaluationSuite,
    MetadataFamilyComparison,
    SliceSummary,
    UndefinedPolicy,
    compare_by_metadata,
    compare_metadata_family,
    correct_slice_p_values,
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


def test_metadata_slice_comparison_validates_slice_arguments() -> None:
    cases = (EvaluationCase("a", "yes", "yes", metadata={"group": "x"}),)
    for field, missing, min_count in (
        ("", "<missing>", 1),
        ("group.", "<missing>", 1),
        ("group", "", 1),
        ("group", "<missing>", 0),
    ):
        with pytest.raises(ValueError):
            compare_by_metadata(
                cases,
                cases,
                metric=CharacterErrorRate(),
                field=field,
                missing=missing,
                min_count=min_count,
                undefined_policy=UndefinedPolicy.ERROR,
            )


def test_metadata_slice_comparison_applies_family_correction_without_changing_gate() -> None:
    baseline = tuple(
        EvaluationCase(f"c{index}", "yes", "no", metadata={"group": group})
        for index, group in enumerate(("a", "a", "b", "b"))
    )
    candidate = tuple(
        EvaluationCase(f"c{index}", "yes", "yes", metadata={"group": group})
        for index, group in enumerate(("a", "a", "b", "b"))
    )
    comparisons = compare_by_metadata(
        baseline,
        candidate,
        metric=CharacterErrorRate(),
        field="group",
        undefined_policy=UndefinedPolicy.ERROR,
        bootstrap=BootstrapConfig(samples=16, seed=3),
    )
    corrected = correct_slice_p_values(comparisons, method="bonferroni", alpha=1.0)
    assert len(corrected) == 2
    assert all(item.adjusted_p_value is not None for item in corrected)
    assert all(item.significance_passed is True for item in corrected)
    assert [item.passed for item in corrected] == [item.passed for item in comparisons]


def test_metadata_slice_correction_validates_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        correct_slice_p_values((), method="holm", alpha=0)


def test_metadata_family_corrects_across_fields() -> None:
    baseline = tuple(
        EvaluationCase(
            f"c{index}",
            "yes",
            "no",
            metadata={"group": group, "language": language},
        )
        for index, (group, language) in enumerate(
            (("a", "en"), ("a", "en"), ("b", "fr"), ("b", "fr"))
        )
    )
    candidate = tuple(
        EvaluationCase(
            case.case_id,
            case.reference,
            "yes",
            metadata=case.metadata,
        )
        for case in baseline
    )
    family = compare_metadata_family(
        baseline,
        candidate,
        metric=CharacterErrorRate(),
        fields=("group", "language"),
        undefined_policy=UndefinedPolicy.ERROR,
        bootstrap=BootstrapConfig(samples=16, seed=2),
        direction="lower",
        correction="holm",
        alpha=1.0,
    )
    assert isinstance(family, MetadataFamilyComparison)
    assert len(family.comparisons) == 4
    assert all(item.adjusted_p_value is not None for item in family.comparisons)
    assert family.passed
    assert family.to_dict()["fields"] == ["group", "language"]


def test_metadata_family_rejects_duplicate_fields() -> None:
    cases = (EvaluationCase("a", "yes", "yes", metadata={"group": "x"}),)
    with pytest.raises(ValueError, match="unique"):
        compare_metadata_family(
            cases,
            cases,
            metric=CharacterErrorRate(),
            fields=("group", "group"),
            undefined_policy=UndefinedPolicy.ERROR,
        )
