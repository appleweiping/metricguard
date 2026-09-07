"""Command-line interface for repeatable metric checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .document import load_ocr_document_cases, run_document_ocr_benchmark
from .experiment import ExperimentMatrix, ExperimentSpec
from .io import CaseFormatError, iter_cases, load_cases, load_metric_config
from .metrics import build_metric
from .models import UndefinedPolicy
from .ocr import (
    CommandOcrBackend,
    load_ocr_cases,
    load_ocr_image_cases,
    run_ocr_backend_benchmark,
    run_ocr_benchmark,
)
from .registry import MetricRegistry
from .reporting import (
    render_comparison_json,
    render_comparison_markdown,
    render_json,
    render_markdown,
)
from .slices import compare_by_metadata, correct_slice_p_values, summarize_by_metadata
from .statistics import BootstrapConfig, report_confidence_interval, summarize_by_tag
from .streaming import evaluate_stream
from .suite import EvaluationSuite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metricguard",
        description="Run NLP metrics under explicit undefined and behavior contracts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser("list", help="list available metrics")
    list_parser.add_argument(
        "--plugins",
        action="store_true",
        help="explicitly discover metricguard.metrics entry points",
    )
    list_parser.set_defaults(handler=_list_metrics)

    run = subcommands.add_parser("run", help="evaluate a JSON or JSONL case suite")
    run.add_argument("cases", type=Path)
    metric = run.add_mutually_exclusive_group(required=True)
    metric.add_argument("--metric", help="built-in or explicitly loaded plugin metric name")
    metric.add_argument("--metric-config", type=Path)
    run.add_argument(
        "--load-plugins",
        action="store_true",
        help="explicitly discover metricguard.metrics entry points",
    )
    run.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    run.add_argument(
        "--audit", action="store_true", help="audit boundedness, determinism, and identity"
    )
    run.add_argument("--symmetric", action="store_true", help="also require a symmetric metric")
    run.add_argument("--format", choices=["json", "markdown"], default="markdown")
    run.add_argument("--output", type=Path)
    run.set_defaults(handler=_run)
    stream_run = subcommands.add_parser(
        "stream-run", help="evaluate JSONL cases with bounded-memory aggregation"
    )
    stream_run.add_argument("cases", type=Path, help="JSONL case file")
    stream_metric = stream_run.add_mutually_exclusive_group(required=True)
    stream_metric.add_argument("--metric")
    stream_metric.add_argument("--metric-config", type=Path)
    stream_run.add_argument("--load-plugins", action="store_true")
    stream_run.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    stream_run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="record metric and undefined-policy errors instead of stopping",
    )
    stream_run.add_argument("--output", type=Path)
    stream_run.set_defaults(handler=_stream_run)
    confidence = subcommands.add_parser(
        "confidence", help="report deterministic bootstrap uncertainty for one metric suite"
    )
    confidence.add_argument("cases", type=Path)
    confidence_metric = confidence.add_mutually_exclusive_group(required=True)
    confidence_metric.add_argument("--metric")
    confidence_metric.add_argument("--metric-config", type=Path)
    confidence.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    confidence.add_argument("--samples", type=int, default=2_000)
    confidence.add_argument("--confidence", type=float, default=0.95)
    confidence.add_argument("--seed", type=int, default=0)
    confidence.add_argument("--output", type=Path)
    confidence.set_defaults(handler=_confidence)
    slices = subcommands.add_parser(
        "slices", help="summarize metric scores by a nested case metadata field"
    )
    slices.add_argument("cases", type=Path)
    slices_metric = slices.add_mutually_exclusive_group(required=True)
    slices_metric.add_argument("--metric")
    slices_metric.add_argument("--metric-config", type=Path)
    slices.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.SKIP.value,
    )
    slices.add_argument("--field", required=True, help="dotted metadata path, e.g. cohort.language")
    slices.add_argument("--missing", default="<missing>")
    slices.add_argument("--min-count", type=int, default=1)
    slices.add_argument("--output", type=Path)
    slices.set_defaults(handler=_slices)

    compare = subcommands.add_parser(
        "compare", help="compare aligned baseline and candidate prediction files"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare_metric = compare.add_mutually_exclusive_group(required=True)
    compare_metric.add_argument("--metric", help="built-in or explicitly loaded plugin metric name")
    compare_metric.add_argument("--metric-config", type=Path)
    compare.add_argument(
        "--load-plugins",
        action="store_true",
        help="explicitly discover metricguard.metrics entry points",
    )
    compare.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    compare.add_argument(
        "--samples",
        type=int,
        default=2_000,
        help="replicates for both the paired bootstrap and sign-flip test",
    )
    compare.add_argument("--confidence", type=float, default=0.95)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument(
        "--direction",
        choices=("higher", "lower"),
        default="higher",
        help="whether larger or smaller metric values are better",
    )
    compare.add_argument(
        "--minimum-delta",
        type=float,
        default=0.0,
        help="fail if the direction-oriented observed improvement is below this value",
    )
    compare.add_argument(
        "--minimum-lower-bound",
        type=float,
        help="also fail if the confidence interval lower bound is below this value",
    )
    compare.add_argument("--format", choices=["json", "markdown"], default="markdown")
    compare.add_argument("--output", type=Path)
    compare.set_defaults(handler=_compare)
    compare_slices = subcommands.add_parser(
        "compare-slices", help="compare baseline and candidate by a metadata slice"
    )
    compare_slices.add_argument("baseline", type=Path)
    compare_slices.add_argument("candidate", type=Path)
    compare_slices_metric = compare_slices.add_mutually_exclusive_group(required=True)
    compare_slices_metric.add_argument("--metric")
    compare_slices_metric.add_argument("--metric-config", type=Path)
    compare_slices.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    compare_slices.add_argument(
        "--load-plugins",
        action="store_true",
        help="explicitly discover metricguard.metrics entry points",
    )
    compare_slices.add_argument("--field", required=True)
    compare_slices.add_argument("--missing", default="<missing>")
    compare_slices.add_argument("--min-count", type=int, default=1)
    compare_slices.add_argument("--samples", type=int, default=2_000)
    compare_slices.add_argument("--confidence", type=float, default=0.95)
    compare_slices.add_argument("--seed", type=int, default=0)
    compare_slices.add_argument("--direction", choices=("higher", "lower"), default="higher")
    compare_slices.add_argument("--minimum-delta", type=float, default=0.0)
    compare_slices.add_argument("--minimum-lower-bound", type=float)
    compare_slices.add_argument(
        "--p-value-correction",
        choices=("none", "bonferroni", "holm", "benjamini-hochberg"),
        default="none",
        help="adjust the family of slice p-values before reporting significance",
    )
    compare_slices.add_argument("--alpha", type=float, default=0.05)
    compare_slices.add_argument("--output", type=Path)
    compare_slices.set_defaults(handler=_compare_slices)
    matrix = subcommands.add_parser("matrix", help="evaluate several metrics over one case suite")
    matrix.add_argument("cases", type=Path)
    matrix.add_argument("--metrics", required=True, help="comma-separated built-in metric names")
    matrix.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.ERROR.value,
    )
    matrix.add_argument("--workers", type=int, default=1, help="workers per metric")
    matrix.add_argument("--max-workers", type=int, default=1, help="parallel metric cells")
    matrix.add_argument("--cache-dir", type=Path)
    matrix.add_argument("--output", type=Path)
    matrix.set_defaults(handler=_matrix)
    ocr = subcommands.add_parser("ocr", help="evaluate OCR reference/prediction text pairs")
    ocr.add_argument("cases", type=Path)
    ocr.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.SKIP.value,
    )
    ocr.add_argument("--output", type=Path)
    ocr.set_defaults(handler=_ocr)
    ocr_backend = subcommands.add_parser(
        "ocr-backend", help="run an explicit shell-free OCR command over image/reference cases"
    )
    ocr_backend.add_argument("cases", type=Path)
    ocr_backend.add_argument(
        "--command",
        action="append",
        required=True,
        metavar="ARG",
        help="one argv token; repeat for the complete template containing exactly one {image}",
    )
    ocr_backend.add_argument("--timeout", type=float, default=120.0)
    ocr_backend.add_argument("--max-output-bytes", type=int, default=4 * 1024 * 1024)
    ocr_backend.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.SKIP.value,
    )
    ocr_backend.add_argument("--output", type=Path)
    ocr_backend.set_defaults(handler=_ocr_backend)
    ocr_document = subcommands.add_parser(
        "ocr-document", help="evaluate ordered OCR pages with document-level summaries"
    )
    ocr_document.add_argument("cases", type=Path)
    ocr_document.add_argument(
        "--command",
        action="append",
        required=True,
        metavar="ARG",
        help="one argv token; repeat for the complete template containing exactly one {image}",
    )
    ocr_document.add_argument("--timeout", type=float, default=120.0)
    ocr_document.add_argument("--max-output-bytes", type=int, default=4 * 1024 * 1024)
    ocr_document.add_argument(
        "--undefined",
        choices=[policy.value for policy in UndefinedPolicy],
        default=UndefinedPolicy.SKIP.value,
    )
    ocr_document.add_argument("--output", type=Path)
    ocr_document.set_defaults(handler=_ocr_document)
    return parser


def _list_metrics(args: argparse.Namespace) -> int:
    registry = MetricRegistry.with_builtins()
    if args.plugins:
        registry.load_plugins()
    print("\n".join(registry.names))
    return 0


def _run(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config, load_plugins=args.load_plugins)
    suite = EvaluationSuite(
        load_cases(args.cases), undefined_policy=UndefinedPolicy(args.undefined)
    )
    auditor = (
        ContractAuditor(Contract(symmetric=args.symmetric))
        if args.audit or args.symmetric
        else None
    )
    report = suite.run(metric, auditor=auditor)
    rendered = render_json(report) if args.format == "json" else render_markdown(report)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report.passed_contracts else 2


def _stream_run(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config, load_plugins=args.load_plugins)
    report = evaluate_stream(
        iter_cases(args.cases),
        metric,
        undefined_policy=UndefinedPolicy(args.undefined),
        fail_fast=not args.continue_on_error,
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report.passed else 2


def _confidence(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config)
    cases = tuple(load_cases(args.cases))
    report = EvaluationSuite(cases, undefined_policy=UndefinedPolicy(args.undefined)).run(metric)
    interval = report_confidence_interval(
        report,
        BootstrapConfig(samples=args.samples, confidence=args.confidence, seed=args.seed),
    )
    payload = {
        "metric": report.metric_name,
        "cases": len(cases),
        "scored": report.scored_count,
        "mean_score": report.mean_score,
        "confidence_interval": {
            "point": interval.point,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
            "samples": interval.samples,
        },
        "by_tag": [
            {
                "tag": summary.tag,
                "case_count": summary.case_count,
                "scored_count": summary.scored_count,
                "mean_score": summary.mean_score,
            }
            for summary in summarize_by_tag(report, cases)
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _slices(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config)
    cases = tuple(load_cases(args.cases))
    report = EvaluationSuite(cases, undefined_policy=UndefinedPolicy(args.undefined)).run(metric)
    payload = {
        "metric": report.metric_name,
        "field": args.field,
        "slices": [
            summary.to_dict()
            for summary in summarize_by_metadata(
                report,
                cases,
                args.field,
                missing=args.missing,
                min_count=args.min_count,
            )
        ],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _compare(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.baseline, args.candidate, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config, load_plugins=args.load_plugins)
    comparison = compare_case_sets(
        load_cases(args.baseline),
        load_cases(args.candidate),
        metric=metric,
        undefined_policy=UndefinedPolicy(args.undefined),
        bootstrap=BootstrapConfig(
            samples=args.samples,
            confidence=args.confidence,
            seed=args.seed,
        ),
        minimum_delta=args.minimum_delta,
        minimum_lower_bound=args.minimum_lower_bound,
        direction=args.direction,
    )
    rendered = (
        render_comparison_json(comparison)
        if args.format == "json"
        else render_comparison_markdown(comparison)
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if comparison.passed_gate else 2


def _compare_slices(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.baseline, args.candidate, args.metric_config)
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config, load_plugins=args.load_plugins)
    baseline = tuple(load_cases(args.baseline))
    candidate = tuple(load_cases(args.candidate))
    comparisons = compare_by_metadata(
        baseline,
        candidate,
        metric=metric,
        field=args.field,
        undefined_policy=UndefinedPolicy(args.undefined),
        bootstrap=BootstrapConfig(
            samples=args.samples,
            confidence=args.confidence,
            seed=args.seed,
        ),
        missing=args.missing,
        min_count=args.min_count,
        minimum_delta=args.minimum_delta,
        minimum_lower_bound=args.minimum_lower_bound,
        direction=args.direction,
    )
    if args.p_value_correction != "none":
        comparisons = correct_slice_p_values(
            comparisons,
            method=args.p_value_correction,
            alpha=args.alpha,
        )
    payload = {
        "metric": metric.name,
        "field": args.field,
        "slices": [comparison.to_dict() for comparison in comparisons],
        "passed": bool(comparisons) and all(comparison.passed for comparison in comparisons),
        "p_value_correction": args.p_value_correction,
        "alpha": args.alpha,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    significant = args.p_value_correction == "none" or all(
        comparison.significance_passed is True
        for comparison in comparisons
        if comparison.comparison is not None and comparison.error is None
    )
    return 0 if payload["passed"] and significant else 2


def _matrix(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases)
    names = tuple(name.strip() for name in args.metrics.split(",") if name.strip())
    if not names or len(names) != len(set(names)):
        raise ValueError("--metrics must contain unique comma-separated names")
    cases = tuple(load_cases(args.cases))
    policy = UndefinedPolicy(args.undefined)
    specs = tuple(
        ExperimentSpec(
            name, build_metric(name), cases, undefined_policy=policy, workers=args.workers
        )
        for name in names
    )
    results = ExperimentMatrix(specs).run(cache_dir=args.cache_dir, max_workers=args.max_workers)
    rendered = (
        json.dumps(
            {
                "experiments": [
                    {
                        "name": result.name,
                        "metric": result.metric_name,
                        "cases": result.cases,
                        "mean_score": result.mean_score,
                        "cached": result.report.cached,
                        "errors": [list(error) for error in result.report.errors],
                    }
                    for result in results
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _ocr(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases)
    report = run_ocr_benchmark(
        load_ocr_cases(args.cases), undefined_policy=UndefinedPolicy(args.undefined)
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _ocr_backend(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases)
    backend = CommandOcrBackend(
        args.command,
        timeout=args.timeout,
        max_output_bytes=args.max_output_bytes,
    )
    report = run_ocr_backend_benchmark(
        load_ocr_image_cases(args.cases), backend, undefined_policy=UndefinedPolicy(args.undefined)
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _ocr_document(args: argparse.Namespace) -> int:
    _ensure_output_is_distinct(args.output, args.cases)
    backend = CommandOcrBackend(
        args.command,
        timeout=args.timeout,
        max_output_bytes=args.max_output_bytes,
    )
    pages = load_ocr_document_cases(args.cases)

    def transcribe(page) -> str:  # type: ignore[no-untyped-def]
        if page.image is None:
            raise ValueError(f"OCR document page {page.page_id!r} has no image path")
        return backend(page.image)

    report = run_document_ocr_benchmark(
        pages, transcribe, undefined_policy=UndefinedPolicy(args.undefined)
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def _ensure_output_is_distinct(output: Path | None, *inputs: Path | None) -> None:
    """Prevent a report from overwriting an input or metric configuration."""

    if output is None:
        return
    for input_path in inputs:
        if input_path is None:
            continue
        try:
            collision = output.resolve() == input_path.resolve() or (
                output.exists() and input_path.exists() and output.samefile(input_path)
            )
        except (OSError, RuntimeError):
            collision = False
        if collision:
            raise ValueError(f"output path must differ from input path {input_path}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and convert expected input failures to exit code 2."""

    try:
        args = _parser().parse_args(argv)
        return int(args.handler(args))
    except (CaseFormatError, OSError, TypeError, ValueError) as error:
        print(f"metricguard: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
