# Resumable evaluation runner

`MetricRunner` evaluates an ordered case sequence, optionally using a bounded
thread pool and a JSONL checkpoint cache. A cache row is reused only when its
case ID, metric name, reference, and prediction fingerprint match. Results are
returned in input order regardless of worker completion order.

```python
from metricguard import MetricRunner

report = MetricRunner(cases, metric, workers=4).run(cache="results.jsonl")
```

Undefined values follow the selected `UndefinedPolicy`; ordinary metric
exceptions can either fail fast or be returned in `report.errors`. Cache rows
are deterministic JSON and sorted by case ID on write. This runner does not
serialize arbitrary Python objects, does not execute remote providers, and its
thread pool does not make a non-thread-safe custom metric safe. Use process-level
isolation or a provider-specific scheduler for expensive or unsafe backends.
