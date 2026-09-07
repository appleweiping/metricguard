# Retrieval and ranking evaluation

MetricGuard can evaluate already-ranked document IDs using five per-query
metrics: `precision_at_k`, `recall_at_k`, `reciprocal_rank_at_k`,
`average_precision_at_k`, and `ndcg_at_k`.

```python
from metricguard import EvaluationCase, EvaluationSuite, build_metric

metric = build_metric({"kind": "ndcg_at_k", "k": 3})
report = EvaluationSuite([
    EvaluationCase("query-1", {"doc-a": 3, "doc-b": 1}, ["doc-b", "doc-a"])
]).run(metric)
assert report.mean_score is not None
```

References map document IDs to finite, nonnegative relevance grades. Predictions
are unique document IDs in best-first order. Unknown IDs are treated as
non-relevant and counted as unjudged in the result details. Duplicate IDs are
errors, including duplicates outside the cutoff. Ranking score ties must be
resolved upstream: these metrics consume an order, not document scores.

## Exact conventions

| Metric | Definition |
|---|---|
| Precision | Positive-relevance hits in the first k positions divided by k |
| Recall | Hits divided by all positive judgments |
| Reciprocal rank | Inverse position of the first positive hit, or zero |
| Average precision | Sum of prefix precisions at positive hits, divided by min(k, all positive judgments) |
| NDCG | Linear relevance divided by log2(rank + 1), summed and normalized by the ideal ordering at k |

Short runs do not reduce precision's denominator. NDCG uses linear, **not
exponential**, gains. Common scaling prevents finite large grades from
overflowing the discounted sums. Binary metrics interpret any positive grade as
relevant. With no positive judgments, all five return an undefined value;
`EvaluationSuite` applies its explicit error/skip/zero/one policy. An empty
prediction with at least one positive judgment receives zero.

The suite reports a macro average of per-query values. For reciprocal rank this
is MRR at the configured cutoff. These documented conventions must be checked
before comparing numbers against another evaluator; MetricGuard does not claim
bit-for-bit equivalence with every TREC or research benchmark implementation.

This adds scoring, not retrieval generation, model serving, benchmark datasets,
or HELMET's full long-context experiment runner.
