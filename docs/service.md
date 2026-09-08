# MetricGuard local service

`MetricService` and `create_server()` provide a framework-free local dispatch
boundary. The service reuses the same metric factory, undefined-value policy,
case alignment, and deterministic bootstrap code as the CLI.

```python
from metricguard import MetricService

result = MetricService().dispatch({
    "operation": "run",
    "cases": "examples/text_cases.jsonl",
    "metric": "exact_match",
})
```

`compare` accepts baseline and candidate paths plus the normal paired bootstrap
options. `create_server()` binds to loopback and serves `POST /v1/dispatch`.
