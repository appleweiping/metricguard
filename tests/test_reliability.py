from __future__ import annotations

import json
from pathlib import Path

import pytest

from metricguard import EvaluationCase, calibration_report
from metricguard.cli import main


def _cases() -> tuple[EvaluationCase, ...]:
    return tuple(
        EvaluationCase(
            f"case-{index}",
            "yes",
            "yes" if correct else "no",
            metadata={"model": {"confidence": confidence}, "correct": correct},
        )
        for index, (confidence, correct) in enumerate(
            ((0.9, True), (0.8, True), (0.2, False), (0.1, False))
        )
    )


def test_calibration_report_is_deterministic_and_keeps_empty_bins() -> None:
    report = calibration_report(_cases(), confidence_field="model.confidence", bins=4)
    assert report.case_count == 4
    assert report.brier_score == pytest.approx(0.025)
    assert report.expected_calibration_error == pytest.approx(0.15)
    assert report.maximum_calibration_error == pytest.approx(0.15)
    assert len(report.bins) == 4
    assert report.bins[0].count == 2
    assert report.bins[1].count == 0
    assert report.bins[3].accuracy == 1.0
    assert report.to_dict()["schema_version"] == 1


def test_calibration_cli_writes_nested_metadata_report(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": case.case_id,
                    "reference": case.reference,
                    "prediction": case.prediction,
                    "metadata": case.metadata,
                }
            )
            for case in _cases()
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "calibration.json"
    assert (
        main(
            [
                "calibrate",
                str(cases_path),
                "--confidence-field",
                "model.confidence",
                "--bins",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["operation"] == "calibration"
    assert payload["case_count"] == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bins": 1}, "bins"),
        ({"confidence_field": "missing"}, "missing"),
        ({"outcome_field": "correct"}, "confidence"),
    ],
)
def test_calibration_rejects_invalid_or_missing_metadata(
    kwargs: dict[str, object], message: str
) -> None:
    cases = _cases()
    if kwargs.get("outcome_field") == "correct":
        cases = tuple(
            EvaluationCase(case.case_id, case.reference, case.prediction, metadata={})
            for case in cases
        )
    with pytest.raises((TypeError, ValueError), match=message):
        calibration_report(cases, **kwargs)  # type: ignore[arg-type]


def test_calibration_rejects_duplicate_ids_and_invalid_values() -> None:
    duplicate = (*_cases(), _cases()[0])
    with pytest.raises(ValueError, match="duplicate"):
        calibration_report(duplicate, confidence_field="model.confidence")
    invalid = EvaluationCase("bad", "x", "y", metadata={"confidence": 1.2, "correct": False})
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_report((invalid,))
    wrong_outcome = EvaluationCase("bad", "x", "y", metadata={"confidence": 0.5, "correct": 1})
    with pytest.raises(ValueError, match="outcome must be boolean"):
        calibration_report((wrong_outcome,))
