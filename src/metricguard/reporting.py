"""Stable report serialization."""

from __future__ import annotations

import json
from typing import Any

from .models import SuiteReport


def report_to_dict(report: SuiteReport) -> dict[str, Any]:
    """Convert a report to a versioned JSON-compatible object."""

    return {
        "schema_version": 1,
        "metric": report.metric_name,
        "summary": {
            "case_count": len(report.results),
            "scored_count": report.scored_count,
            "skipped_count": report.skipped_count,
            "mean_score": report.mean_score,
            "passed_contracts": report.passed_contracts,
        },
        "results": [
            {
                "case_id": result.case_id,
                "raw_score": result.raw.score,
                "resolved_score": result.resolved_score,
                "undefined_reason": result.raw.reason,
                "skipped": result.skipped,
                "details": result.raw.details,
            }
            for result in report.results
        ],
        "findings": [
            {
                "rule": finding.rule,
                "severity": finding.severity.value,
                "message": finding.message,
                "case_id": finding.case_id,
                "observed": finding.observed,
            }
            for finding in report.findings
        ],
    }


def render_json(report: SuiteReport) -> str:
    """Render deterministic, human-readable JSON."""

    return (
        json.dumps(
            report_to_dict(report),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_markdown(report: SuiteReport) -> str:
    """Render a compact Markdown audit report."""

    mean = "n/a" if report.mean_score is None else f"{report.mean_score:.6f}"
    status = "PASS" if report.passed_contracts else "FAIL"
    case_summary = (
        f"- Cases: {len(report.results)} "
        f"({report.scored_count} scored, {report.skipped_count} skipped)"
    )
    lines = [
        f"# MetricGuard report: `{_markdown_text(report.metric_name)}`",
        "",
        f"- Contract status: **{status}**",
        case_summary,
        f"- Macro mean: {mean}",
        "",
        "## Cases",
        "",
        "| Case | Raw | Resolved | State |",
        "|---|---:|---:|---|",
    ]
    for result in report.results:
        raw = "undefined" if result.raw.score is None else f"{result.raw.score:.6f}"
        resolved = "—" if result.resolved_score is None else f"{result.resolved_score:.6f}"
        state = "skipped" if result.skipped else (result.raw.reason or "scored")
        lines.append(
            f"| `{_markdown_text(result.case_id)}` | {raw} | {resolved} | {_markdown_text(state)} |"
        )
    lines.extend(["", "## Contract findings", ""])
    if report.findings:
        for finding in report.findings:
            location = f" (`{_markdown_text(finding.case_id)}`)" if finding.case_id else ""
            prefix = (
                f"- **{finding.severity.value.upper()}** "
                f"`{_markdown_text(finding.rule)}`{location}:"
            )
            lines.append(f"{prefix} {_markdown_text(finding.message)}")
    else:
        lines.append("No contract findings.")
    return "\n".join(lines) + "\n"


def _markdown_text(value: str) -> str:
    """Keep untrusted text inside one Markdown cell or inline-code span."""

    return (
        value.replace("&", "&amp;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )
