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

The same workflow is available from the CLI:

```bash
metricguard ocr ocr-cases.jsonl --output ocr-report.json
```

The module evaluates text pairs; it intentionally does not pretend to be an
OCR image model or provider. An OCR backend can feed its recognized strings into
the same case format while MetricGuard preserves auditable metric semantics.

When a local OCR executable is available, `CommandOcrBackend` provides a safe
adapter without adding an SDK dependency. Supply an argv template containing
`{image}`; it never invokes a shell, enforces a timeout and UTF-8 output-size
limit, and reports non-zero exits explicitly. `run_ocr_backend_benchmark()`
connects that backend to `OcrImageCase` objects and the same CER/WER/exact-match
report, so backend changes remain comparable under one metric contract.

Image cases can be loaded from JSON or JSONL with `id`, `image`, and `reference`
fields. Relative image paths are resolved against the case file:

```json
{"id":"scan-1","image":"pages/scan-1.png","reference":"hello world"}
```

The complete backend workflow is also available from the CLI. The command is an
argv template, not a shell string, and must contain `{image}`:

```bash
metricguard ocr-backend image-cases.jsonl \
  --command tesseract --command {image} --command stdout \
  --output backend-report.json
```

Repeat `--command` once per argv token so options such as `-c` remain data
rather than being interpreted by MetricGuard's own parser; use the
`--command=-c` spelling for a token that starts with a dash.

Use `--timeout`, `--max-output-bytes`, and `--undefined` to make resource and
empty-reference behavior explicit. A backend failure is reported as a clean
non-zero CLI result rather than being hidden as a metric score.

For manifests containing multiple pages per document, see
[document-level OCR](document-ocr.md). It preserves deterministic page order
and emits both global and per-document reports through `ocr-document`.
