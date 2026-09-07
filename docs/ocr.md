# OCR benchmark workflow

`run_ocr_benchmark()` evaluates recognized text against references with three
complementary views: character error rate, word error rate, and normalized
exact match. Empty references are undefined for error rates and are explicitly
reported according to the selected `UndefinedPolicy` (the default is `SKIP`).

```python
from metricguard import load_ocr_cases, run_ocr_benchmark

report = run_ocr_benchmark(load_ocr_cases("ocr-cases.jsonl"))
print(report.to_dict())
```

The module evaluates text pairs; it intentionally does not pretend to be an
OCR image model or provider. An OCR backend can feed its recognized strings into
the same case format while MetricGuard preserves auditable metric semantics.
