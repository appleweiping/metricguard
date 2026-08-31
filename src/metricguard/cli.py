"""Command-line interface for repeatable metric checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .io import CaseFormatError, load_cases, load_metric_config
from .metrics import build_metric
from .models import UndefinedPolicy
from .registry import MetricRegistry
from .reporting import (
    render_comparison_json,
    render_comparison_markdown,
    render_json,
    render_markdown,
)
from .statistics import BootstrapConfig
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
