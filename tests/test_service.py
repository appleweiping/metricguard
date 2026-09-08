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


def test_metric_service_rejects_invalid_port() -> None:
    with pytest.raises(ValueError):
        create_server(port=65536)


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
