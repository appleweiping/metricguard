"""Loopback JSON service for MetricGuard evaluation and comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast

from .comparison import compare_case_sets
from .correlation import correlate_reports
from .experiment import ExperimentMatrix, ExperimentSpec
from .io import load_cases, load_metric_config
from .metrics import Metric, build_metric
from .models import UndefinedPolicy
from .reliability import calibration_report
from .reporting import comparison_to_dict, report_to_dict
from .slices import compare_by_metadata, compare_metadata_family, summarize_by_metadata
from .statistics import BootstrapConfig
from .suite import EvaluationSuite


class MetricService:
    """Dispatch evaluation requests without changing CLI semantics."""

    def dispatch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be an object")
        operation = request.get("operation")
        undefined = _undefined(request)
        if operation == "run":
            metric = _metric(request)
            cases_path = _path(request, "cases")
            report = EvaluationSuite(load_cases(cases_path), undefined_policy=undefined).run(metric)
            return {"operation": operation, "report": report_to_dict(report)}
        if operation == "compare":
            metric = _metric(request)
            baseline = tuple(load_cases(_path(request, "baseline")))
            candidate = tuple(load_cases(_path(request, "candidate")))
            comparison = compare_case_sets(
                baseline,
                candidate,
                metric=metric,
                undefined_policy=undefined,
                bootstrap=BootstrapConfig(
                    samples=_integer(request, "samples", 2_000, minimum=1),
                    confidence=_number(request, "confidence", 0.95),
                    seed=_integer(request, "seed", 0),
                ),
                minimum_delta=_number(request, "minimum_delta", 0.0),
                minimum_lower_bound=request.get("minimum_lower_bound"),
                direction=request.get("direction", "higher"),
            )
            return {"operation": operation, "comparison": comparison_to_dict(comparison)}
        if operation == "matrix":
            cases = tuple(load_cases(_path(request, "cases")))
            metrics = request.get("metrics")
            if (
                not isinstance(metrics, list)
                or not metrics
                or not all(isinstance(name, str) and name.strip() for name in metrics)
            ):
                raise ValueError("metrics must be a non-empty array of names")
            if len(set(metrics)) != len(metrics):
                raise ValueError("metrics must contain unique names")
            workers = _integer(request, "workers", 1, minimum=1)
            max_workers = _integer(request, "max_workers", 1, minimum=1)
            cache_dir = request.get("cache_dir")
            if cache_dir is not None and not isinstance(cache_dir, str):
                raise ValueError("cache_dir must be a path string or omitted")
            specs = tuple(
                ExperimentSpec(
                    name,
                    build_metric(name),
                    cases,
                    undefined_policy=undefined,
                    workers=workers,
                )
                for name in metrics
            )
            results = ExperimentMatrix(specs).run(cache_dir=cache_dir, max_workers=max_workers)
            leaderboard = ExperimentMatrix.leaderboard(results)
            return {
                "operation": operation,
                "results": [
                    {
                        "name": item.name,
                        "metric": item.metric_name,
                        "cases": item.cases,
                        "mean_score": item.mean_score,
                        "cached": item.report.cached,
                        "errors": list(item.report.errors),
                    }
                    for item in results
                ],
                "leaderboard": [
                    {
                        "rank": item.rank,
                        "name": item.name,
                        "metric": item.metric_name,
                        "mean_score": item.mean_score,
                        "scored_cases": item.scored_cases,
                        "total_cases": item.total_cases,
                        "contract_passed": item.contract_passed,
                    }
                    for item in leaderboard
                ],
            }
        if operation == "correlate":
            cases = tuple(load_cases(_path(request, "cases")))
            metrics = request.get("metrics")
            if (
                not isinstance(metrics, list)
                or len(metrics) < 2
                or not all(isinstance(name, str) and name.strip() for name in metrics)
            ):
                raise ValueError("metrics must be an array of at least two names")
            if len(set(metrics)) != len(metrics):
                raise ValueError("metrics must contain unique names")
            reports = {
                name: EvaluationSuite(cases, undefined_policy=undefined).run(build_metric(name))
                for name in metrics
            }
            result = correlate_reports(
                reports,
                minimum_count=_integer(request, "minimum_count", 2, minimum=2),
            )
            return {"operation": operation, "correlation": result.to_dict()}
        if operation == "calibrate":
            confidence_field = request.get("confidence_field", "confidence")
            outcome_field = request.get("outcome_field", "correct")
            if not isinstance(confidence_field, str):
                raise ValueError("confidence_field must be a field path string")
            if not isinstance(outcome_field, str):
                raise ValueError("outcome_field must be a field path string")
            calibration = calibration_report(
                load_cases(_path(request, "cases")),
                confidence_field=confidence_field,
                outcome_field=outcome_field,
                bins=_integer(request, "bins", 10, minimum=2),
            )
            return {"operation": operation, "report": calibration.to_dict()}
        if operation == "slices":
            cases = tuple(load_cases(_path(request, "cases")))
            field = request.get("field")
            if not isinstance(field, str) or not field.strip():
                raise ValueError("field must be a non-empty metadata path")
            report = EvaluationSuite(cases, undefined_policy=undefined).run(_metric(request))
            summaries = summarize_by_metadata(
                report,
                cases,
                field,
                missing=_string(request, "missing", "<missing>"),
                min_count=_integer(request, "min_count", 1, minimum=1),
            )
            return {
                "operation": operation,
                "summaries": [summary.to_dict() for summary in summaries],
            }
        if operation == "compare_slices":
            field = request.get("field")
            if not isinstance(field, str) or not field.strip():
                raise ValueError("field must be a non-empty metadata path")
            comparisons = compare_by_metadata(
                tuple(load_cases(_path(request, "baseline"))),
                tuple(load_cases(_path(request, "candidate"))),
                metric=_metric(request),
                field=field,
                undefined_policy=undefined,
                bootstrap=_bootstrap(request),
                missing=_string(request, "missing", "<missing>"),
                min_count=_integer(request, "min_count", 1, minimum=1),
                minimum_delta=_number(request, "minimum_delta", 0.0),
                minimum_lower_bound=_optional_number(request, "minimum_lower_bound"),
                direction=_direction(request),
            )
            return {
                "operation": operation,
                "comparisons": [comparison.to_dict() for comparison in comparisons],
            }
        if operation == "compare_family":
            fields = request.get("fields")
            if (
                not isinstance(fields, list)
                or not fields
                or not all(isinstance(field, str) and field.strip() for field in fields)
            ):
                raise ValueError("fields must be a non-empty array of metadata paths")
            family = compare_metadata_family(
                tuple(load_cases(_path(request, "baseline"))),
                tuple(load_cases(_path(request, "candidate"))),
                metric=_metric(request),
                fields=fields,
                undefined_policy=undefined,
                bootstrap=_bootstrap(request),
                missing=_string(request, "missing", "<missing>"),
                min_count=_integer(request, "min_count", 1, minimum=1),
                minimum_delta=_number(request, "minimum_delta", 0.0),
                minimum_lower_bound=_optional_number(request, "minimum_lower_bound"),
                direction=_direction(request),
                correction=_correction(request),
                alpha=_number(request, "alpha", 0.05),
            )
            return {"operation": operation, "family": family.to_dict()}
        raise ValueError(
            "operation must be run, compare, matrix, correlate, calibrate, slices, "
            "compare_slices, or compare_family"
        )


def create_server(
    service: MetricService | None = None, *, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Create a loopback-first JSON server; call ``serve_forever`` to run it."""
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    target = service or MetricService()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/v1/dispatch":
                self._write(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "-1"))
                if size < 0 or size > 4 * 1024 * 1024:
                    raise ValueError("Content-Length must be between 0 and 4194304")
                payload = target.dispatch(json.loads(self.rfile.read(size).decode("utf-8")))
            except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, OSError) as error:
                self._write(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._write(HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _write(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _metric(request: Mapping[str, Any]) -> Metric:
    value = request.get("metric")
    config = request.get("metric_config")
    if value is not None and config is not None:
        raise ValueError("metric and metric_config are mutually exclusive")
    if config is not None:
        if not isinstance(config, str):
            raise ValueError("metric_config must be a path string")
        return build_metric(load_metric_config(Path(config)))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("metric must be a non-empty string or metric_config must be supplied")
    return build_metric(value)


def _undefined(request: Mapping[str, Any]) -> UndefinedPolicy:
    value = request.get("undefined", UndefinedPolicy.ERROR.value)
    if not isinstance(value, str):
        raise ValueError("undefined must be a policy string")
    try:
        return UndefinedPolicy(value)
    except ValueError as error:
        raise ValueError("undefined must be error, skip, zero, or one") from error


def _path(request: Mapping[str, Any], name: str) -> Path:
    value = request.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    return Path(value)


def _integer(
    request: Mapping[str, Any], name: str, default: int, *, minimum: int | None = None
) -> int:
    value = request.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(request: Mapping[str, Any], name: str, default: float) -> float:
    value = request.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_number(request: Mapping[str, Any], name: str) -> float | None:
    value = request.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number or omitted")
    return float(value)


def _string(request: Mapping[str, Any], name: str, default: str) -> str:
    value = request.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _direction(request: Mapping[str, Any]) -> Literal["higher", "lower"]:
    value = _string(request, "direction", "higher")
    if value not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    return cast(Literal["higher", "lower"], value)


def _correction(
    request: Mapping[str, Any],
) -> Literal["none", "bonferroni", "holm", "benjamini-hochberg"]:
    value = _string(request, "correction", "none")
    choices = {"none", "bonferroni", "holm", "benjamini-hochberg"}
    if value not in choices:
        raise ValueError("correction must be none, bonferroni, holm, or benjamini-hochberg")
    return cast(Literal["none", "bonferroni", "holm", "benjamini-hochberg"], value)


def _bootstrap(request: Mapping[str, Any]) -> BootstrapConfig:
    return BootstrapConfig(
        samples=_integer(request, "samples", 2_000, minimum=1),
        confidence=_number(request, "confidence", 0.95),
        seed=_integer(request, "seed", 0),
    )


__all__ = ["MetricService", "create_server"]
