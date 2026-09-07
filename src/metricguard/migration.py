"""Explicit migration of legacy MetricGuard report JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def migrate_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a legacy report shape into schema-version 1."""

    if not isinstance(payload, Mapping):
        raise TypeError("report must be an object")
    version = payload.get("schema_version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version not in {0, 1}:
        raise ValueError("unsupported report schema_version; expected 0 or 1")
    if version == 1:
        metric = payload.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("schema-v1 report metric must be a non-empty string")
        rows = _rows(payload.get("results"))
    else:
        metric = payload.get("metric", payload.get("metric_name"))
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("legacy report requires a non-empty metric or metric_name")
        rows = _legacy_rows(payload.get("results", payload.get("cases", [])))
    findings = _findings(payload.get("findings", []))
    scored = [row for row in rows if row["resolved_score"] is not None]
    mean = sum(float(row["resolved_score"]) for row in scored) / len(scored) if scored else None
    return {
        "schema_version": 1,
        "metric": metric,
        "summary": {
            "case_count": len(rows),
            "scored_count": len(scored),
            "skipped_count": sum(bool(row["skipped"]) for row in rows),
            "mean_score": mean,
            "passed_contracts": not findings,
        },
        "results": rows,
        "findings": findings,
    }


def _legacy_rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("legacy report results must be an array")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"legacy result {index} must be an object")
        case_id = item.get("case_id", item.get("id"))
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"legacy result {index} requires a non-empty id")
        score = item.get("score", item.get("raw_score"))
        resolved = item.get("resolved_score", score)
        skipped = item.get("skipped", resolved is None)
        if not isinstance(skipped, bool):
            raise ValueError(f"legacy result {index} skipped must be boolean")
        rows.append(
            {
                "case_id": case_id,
                "raw_score": score,
                "resolved_score": resolved,
                "undefined_reason": item.get("reason", item.get("undefined_reason")),
                "skipped": skipped,
                "details": item.get("details", {}),
            }
        )
    return rows


def _rows(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("schema-v1 report results must be an array")
    required = {"case_id", "raw_score", "resolved_score", "undefined_reason", "skipped", "details"}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping) or not required.issubset(item):
            raise ValueError(f"schema-v1 result {index} has missing fields")
        if not isinstance(item["case_id"], str) or not item["case_id"].strip():
            raise ValueError(f"schema-v1 result {index} case_id must be non-empty text")
        if not isinstance(item["skipped"], bool):
            raise ValueError(f"schema-v1 result {index} skipped must be boolean")
        rows.append({key: item[key] for key in required})
    return rows


def _findings(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("report findings must be an array")
    return [dict(item) if isinstance(item, Mapping) else {"message": str(item)} for item in raw]
