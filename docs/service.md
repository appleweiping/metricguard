# MetricGuard local service

`MetricService` and `create_server()` provide a framework-free local dispatch
boundary. The service reuses the same metric factory, undefined-value policy,
case alignment, and deterministic bootstrap code as the CLI.

```python
from metricguard import MetricService

result = MetricService().dispatch(
    {
        "operation": "run",
        "cases": "examples/text_cases.jsonl",
        "metric": "exact_match",
    }
)
```

`compare` accepts baseline and candidate paths plus the normal paired bootstrap
options. `create_server()` binds to loopback and serves `POST /v1/dispatch`.

`matrix` accepts a unique `metrics` array and optional `cache_dir`; repeated
requests reuse verified case-level results and return a stable leaderboard.

`calibrate` evaluates confidence/outcome metadata with deterministic equal-width
bins and returns the same Brier, ECE, MCE, and bin schema as the `calibrate`
CLI. Use `confidence_field`, `outcome_field`, and `bins` to select nested
metadata paths and the number of bins.

`ocr` runs the standard text-pair OCR benchmark through the service and returns
character error rate, word error rate, and normalized exact-match summaries.
It uses the same undefined-value policy as the other operations and does not
invoke an image backend.

`slices` runs one metric and returns macro score summaries grouped by a dotted
case-metadata path. `compare_slices` performs paired baseline/candidate
comparisons independently for each value, while `compare_family` applies one
multiple-comparison correction across several metadata fields. These service
operations reuse the CLI's validation, bootstrap, direction, and gate semantics.
