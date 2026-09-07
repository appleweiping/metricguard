# Streaming evaluation

`EvaluationSuite` intentionally returns per-case results. For very large
JSONL corpora, `evaluate_stream` provides a bounded-memory alternative: it
consumes one validated `EvaluationCase` at a time and retains only counters,
tag aggregates, and error IDs. It never stores references, predictions, or
individual metric details.

## CLI

```console
metricguard stream-run cases.jsonl --metric token_f1 --undefined skip \
  --continue-on-error --output stream-report.json
```

`stream-run` accepts JSONL as a true streaming input. JSON arrays remain
available through the regular `run` command. The undefined policies have the
same meaning as `EvaluationSuite`; `--continue-on-error` records metric and
undefined-policy failures and continues rather than stopping at the first
failure. Exit status is zero only when no errors were recorded.

The report contains a schema version, case/scored/skipped counts, macro mean,
error IDs, and deterministic per-tag aggregates. It intentionally omits
per-case scores, so it is safe to use for large-scale smoke gates while the
regular `run` command remains the choice for detailed review artifacts.

## Python API

```python
from metricguard import UndefinedPolicy, evaluate_stream
from metricguard.io import iter_cases
from metricguard.metrics import build_metric

report = evaluate_stream(
    iter_cases("cases.jsonl"),
    build_metric("token_f1"),
    undefined_policy=UndefinedPolicy.SKIP,
)
print(report.case_count, report.mean_score)
```

The JSONL iterator validates each case and rejects duplicate IDs before the
same ID can silently affect an aggregate. Input errors remain fatal even when
metric errors are configured to continue.

