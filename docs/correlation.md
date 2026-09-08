# Cross-metric correlation

`correlate_reports()` compares metric outputs on the same evaluation cases. It
is intended for benchmark diagnostics: a high correlation can reveal redundant
metrics, while a low correlation can identify complementary failure modes. It is
descriptive analysis, not evidence that either metric is valid.

```python
from metricguard import EvaluationSuite, build_metric, correlate_reports

reports = {
    name: EvaluationSuite(cases).run(build_metric(name))
    for name in ("exact_match", "token_f1", "rouge_l")
}
matrix = correlate_reports(reports, minimum_count=2)
print(matrix.to_dict()["spearman"])
```

The implementation uses case IDs for alignment and performs pairwise-complete
analysis. Undefined or skipped values are omitted only from the pair involving
that metric. Pearson correlation is computed on the observed population; the
Spearman implementation assigns average ranks to ties. Fewer than the requested
`minimum_count` complete cases, or a constant score vector, is represented as
`null` with a reason rather than silently reported as zero.

The CLI evaluates a shared suite with several metrics:

```bash
metricguard correlate cases.jsonl \
  --metrics exact_match,token_f1,rouge_l \
  --undefined skip --format markdown --output correlation.md
```

The loopback service accepts the same operation:

```json
{
  "operation": "correlate",
  "cases": "cases.jsonl",
  "metrics": ["exact_match", "token_f1"],
  "minimum_count": 2
}
```

The JSON payload is schema-versioned and contains both symmetric matrices and a
pair list with counts and explanatory reasons, so downstream reports can retain
the audit trail for missing or degenerate comparisons.
