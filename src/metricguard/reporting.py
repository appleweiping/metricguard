"""Stable report serialization."""

from __future__ import annotations

import json
from typing import Any

from .models import SuiteReport
from .statistics import PairedComparison


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


def comparison_to_dict(comparison: PairedComparison) -> dict[str, Any]:
    """Convert paired statistics to a versioned machine-readable object."""

    return {
        "schema_version": 1,
        "metric": comparison.metric_name,
        "baseline_mean": comparison.baseline_mean,
        "candidate_mean": comparison.candidate_mean,
        "delta": {
            "point": comparison.delta.point,
            "lower": comparison.delta.lower,
            "upper": comparison.delta.upper,
            "confidence": comparison.delta.confidence,
            "bootstrap_samples": comparison.delta.samples,
        },
        "improvement": {
            "direction": comparison.direction,
            "point": comparison.improvement.point,
            "lower": comparison.improvement.lower,
            "upper": comparison.improvement.upper,
        },
        "methods": {
            "confidence_interval": "paired-percentile-bootstrap-v1",
            "p_value": "paired-sign-flip-monte-carlo-v1",
            "p_value_samples": comparison.delta.samples,
        },
        "probability_improvement": comparison.probability_improvement,
        "two_sided_p_value": comparison.two_sided_p_value,
        "compared_count": comparison.compared_count,
        "omitted_count": comparison.omitted_count,
        "gate": {
            "minimum_delta": comparison.minimum_delta,
            "minimum_lower_bound": comparison.minimum_lower_bound,
            "passed": comparison.passed_gate,
        },
    }


def render_comparison_json(comparison: PairedComparison) -> str:
    """Render a paired comparison as deterministic JSON."""

    return (
        json.dumps(
            comparison_to_dict(comparison),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def render_comparison_markdown(comparison: PairedComparison) -> str:
    """Render a paired comparison and its CI gate as Markdown."""

    status = "PASS" if comparison.passed_gate else "FAIL"
    confidence = comparison.delta.confidence * 100
    lines = [
        f"# MetricGuard comparison: `{_markdown_text(comparison.metric_name)}`",
        "",
        f"- Regression gate: **{status}**",
        f"- Compared cases: {comparison.compared_count}",
        f"- Omitted on both sides: {comparison.omitted_count}",
        f"- Optimization direction: {comparison.direction}",
        f"- Baseline macro mean: {comparison.baseline_mean:.6f}",
        f"- Candidate macro mean: {comparison.candidate_mean:.6f}",
        f"- Candidate - baseline (raw): {comparison.delta.point:+.6f}",
        f"- Direction-oriented improvement: {comparison.improvement.point:+.6f}",
        (
            f"- {confidence:g}% paired-bootstrap interval (oriented improvement): "
            f"[{comparison.improvement.lower:+.6f}, {comparison.improvement.upper:+.6f}] "
            f"({comparison.delta.samples} samples)"
        ),
        f"- Bootstrap mass above zero improvement: {comparison.probability_improvement:.6f}",
        (
            f"- Two-sided paired sign-flip p-value: {comparison.two_sided_p_value:.6f} "
            f"({comparison.delta.samples} Monte Carlo samples)"
        ),
        f"- Minimum observed improvement: {comparison.minimum_delta:+.6f}",
    ]
    if comparison.minimum_lower_bound is not None:
        lines.append(f"- Minimum confidence lower bound: {comparison.minimum_lower_bound:+.6f}")
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
