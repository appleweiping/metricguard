from __future__ import annotations

import json

import pytest

from metricguard.cli import main
from metricguard.migration import migrate_report


def test_migrate_legacy_report_recomputes_summary() -> None:
    migrated = migrate_report(
        {
            "metric_name": "exact_match",
            "summary": {"case_count": 999},
            "results": [
                {"id": "a", "score": 1.0},
                {"id": "b", "score": None, "reason": "undefined", "skipped": True},
            ],
            "findings": [],
        }
    )
    assert migrated["schema_version"] == 1
    assert migrated["metric"] == "exact_match"
    assert migrated["summary"] == {
        "case_count": 2,
        "scored_count": 1,
        "skipped_count": 1,
        "mean_score": 1.0,
        "passed_contracts": True,
    }
    assert migrated["results"][1]["undefined_reason"] == "undefined"


def test_migrate_report_accepts_v1_and_rejects_bad_shapes(tmp_path) -> None:
    current = {
        "schema_version": 1,
        "metric": "x",
        "results": [
            {
                "case_id": "a",
                "raw_score": 1,
                "resolved_score": 1,
                "undefined_reason": None,
                "skipped": False,
                "details": {},
            }
        ],
        "findings": [],
    }
    assert migrate_report(current)["summary"]["case_count"] == 1
    with pytest.raises(ValueError, match="unsupported"):
        migrate_report({"schema_version": 4})
    with pytest.raises(ValueError, match="results"):
        migrate_report({"metric": "x", "results": {}})

    source = tmp_path / "legacy.json"
    output = tmp_path / "migrated.json"
    source.write_text(json.dumps({"metric": "x", "results": []}), encoding="utf-8")
    assert main(["migrate-report", str(source), str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "object"),
        ({"metric": "x", "results": [{"id": ""}]}, "non-empty id"),
        ({"metric": "x", "results": [1]}, "must be an object"),
        ({"metric": "x", "results": [], "findings": {}}, "findings"),
        ({"schema_version": 1, "metric": "x", "results": [{}]}, "missing fields"),
    ],
)
def test_migrate_report_rejects_invalid_shapes(payload: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        migrate_report(payload)  # type: ignore[arg-type]
