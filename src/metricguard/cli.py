"""Command-line interface for repeatable metric checks."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .contracts import Contract, ContractAuditor
from .io import CaseFormatError, load_cases, load_metric_config
from .metrics import build_metric
from .models import UndefinedPolicy
from .reporting import render_json, render_markdown
from .suite import EvaluationSuite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metricguard",
        description="Run NLP metrics under explicit undefined and behavior contracts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    list_parser = subcommands.add_parser("list", help="list built-in metrics")
    list_parser.set_defaults(handler=_list_metrics)

    run = subcommands.add_parser("run", help="evaluate a JSON or JSONL case suite")
    run.add_argument("cases", type=Path)
    metric = run.add_mutually_exclusive_group(required=True)
    metric.add_argument(
        "--metric", choices=["exact_match", "token_f1", "character_f1", "numeric_equivalence"]
    )
    metric.add_argument("--metric-config", type=Path)
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
    return parser


def _list_metrics(_: argparse.Namespace) -> int:
    print("exact_match\ntoken_f1\ncharacter_f1\nnumeric_equivalence")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = load_metric_config(args.metric_config) if args.metric_config else args.metric
    metric = build_metric(config)
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
