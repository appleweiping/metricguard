"""Reproducible synthetic throughput benchmark for MetricGuard."""

from __future__ import annotations

import argparse
import json
import platform
import time
from statistics import median

from metricguard import (
    BootstrapConfig,
    EvaluationCase,
    EvaluationSuite,
    build_metric,
    paired_comparison,
)


def _measure(operation: object, repeats: int) -> tuple[float, object]:
    durations: list[float] = []
    result: object = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()  # type: ignore[operator]
        durations.append(time.perf_counter() - started)
    return median(durations), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=5_000)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.cases < 1 or args.samples < 1 or args.repeats < 1:
        parser.error("cases, samples, and repeats must be positive")

    baseline_cases = tuple(
        EvaluationCase(
            f"case-{index}",
            "deterministic evaluation protects model quality",
            "deterministic evaluation protects quality" if index % 3 else "wrong answer",
        )
        for index in range(args.cases)
    )
    candidate_cases = tuple(
        EvaluationCase(
            case.case_id,
            case.reference,
            "deterministic evaluation protects model quality" if index % 5 else case.prediction,
        )
        for index, case in enumerate(baseline_cases)
    )
    timings: dict[str, float] = {}
    reports = {}
    for name in ("token_f1", "rouge_l", "sentence_bleu", "levenshtein_similarity"):
        duration, report = _measure(
            lambda metric_name=name: EvaluationSuite(baseline_cases).run(build_metric(metric_name)),
            args.repeats,
        )
        timings[name] = duration
        reports[name] = report

    baseline_report = reports["token_f1"]
    candidate_report = EvaluationSuite(candidate_cases).run(build_metric("token_f1"))
    comparison_seconds, comparison = _measure(
        lambda: paired_comparison(
            baseline_report,
            candidate_report,
            config=BootstrapConfig(samples=args.samples, seed=20260831),
        ),
        args.repeats,
    )
    output = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "workload": {
            "cases": args.cases,
            "bootstrap_samples": args.samples,
            "repeats": args.repeats,
        },
        "median_seconds": {**timings, "paired_comparison": comparison_seconds},
        "checks": {
            "baseline_mean": baseline_report.mean_score,
            "candidate_mean": candidate_report.mean_score,
            "delta": comparison.delta.point,  # type: ignore[attr-defined]
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
