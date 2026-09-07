from metricguard import EvaluationCase, EvaluationSuite, SliceSummary, summarize_by_metadata
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
