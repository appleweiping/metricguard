"""Loopback JSON service for MetricGuard evaluation and comparison."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .comparison import compare_case_sets
from .experiment import ExperimentMatrix, ExperimentSpec
from .io import load_cases, load_metric_config
from .metrics import Metric, build_metric
from .models import UndefinedPolicy
from .reporting import comparison_to_dict, report_to_dict
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
        raise ValueError("operation must be run, compare, or matrix")


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


__all__ = ["MetricService", "create_server"]
