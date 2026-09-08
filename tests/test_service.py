import json
import threading
import urllib.request

import pytest

from metricguard import MetricService, create_server


def _cases(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "reference": "ok", "prediction": "ok"}),
                json.dumps({"id": "b", "reference": "ok", "prediction": "no"}),
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_metric_service_run_and_compare(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = _cases(tmp_path)
    service = MetricService()
    report = service.dispatch({"operation": "run", "cases": str(cases), "metric": "exact_match"})
    assert report["report"]["summary"]["scored_count"] == 2
    comparison = service.dispatch(
        {
            "operation": "compare",
            "baseline": str(cases),
            "candidate": str(cases),
            "metric": "exact_match",
            "samples": 20,
        }
    )
    assert comparison["comparison"]["compared_count"] == 2


def test_metric_service_accepts_config_and_rejects_bad_requests(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = _cases(tmp_path)
    config = tmp_path / "metric.json"
    config.write_text(json.dumps({"kind": "exact_match"}), encoding="utf-8")
    assert (
        MetricService().dispatch(
            {"operation": "run", "cases": str(cases), "metric_config": str(config)}
        )["report"]["summary"]["case_count"]
        == 2
    )
    service = MetricService()
    with pytest.raises(ValueError):
        service.dispatch({"operation": "run", "cases": str(cases)})
    with pytest.raises(ValueError):
        service.dispatch(
            {
                "operation": "run",
                "cases": str(cases),
                "metric": "exact_match",
                "metric_config": str(config),
            }
        )
    with pytest.raises(ValueError):
        service.dispatch(
            {"operation": "run", "cases": str(cases), "metric": "exact_match", "undefined": "bad"}
        )
    with pytest.raises(ValueError):
        service.dispatch({"operation": "other", "cases": str(cases), "metric": "exact_match"})


def test_metric_service_matrix_is_cached_and_ranked(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = _cases(tmp_path)
    cache = tmp_path / "cache"
    request = {
        "operation": "matrix",
        "cases": str(cases),
        "metrics": ["exact_match", "word_error_rate"],
        "cache_dir": str(cache),
    }
    service = MetricService()
    first = service.dispatch(request)
    second = service.dispatch(request)
    assert len(first["leaderboard"]) == 2
    assert all(row["cached"] == 0 for row in first["results"])
    assert all(row["cached"] == 2 for row in second["results"])


def test_metric_service_rejects_invalid_port() -> None:
    with pytest.raises(ValueError):
        create_server(port=65536)


def test_metric_service_rejects_invalid_matrix_requests(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cases = _cases(tmp_path)
    service = MetricService()
    with pytest.raises(ValueError):
        service.dispatch([])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.dispatch({"operation": "matrix", "cases": str(cases), "metrics": []})
    with pytest.raises(ValueError):
        service.dispatch(
            {"operation": "matrix", "cases": str(cases), "metrics": ["exact_match", "exact_match"]}
        )


def test_metric_service_http_dispatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    server = create_server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {"operation": "run", "cases": str(_cases(tmp_path)), "metric": "exact_match"}
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/v1/dispatch",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read())["operation"] == "run"
    finally:
        server.shutdown()
        server.server_close()
