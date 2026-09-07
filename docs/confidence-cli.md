# Confidence intervals from the CLI

MetricGuard can expose deterministic uncertainty estimates for a complete case
suite:

```console
metricguard confidence cases.jsonl --metric exact_match \
  --samples 2000 --confidence 0.95 --seed 7
```

The JSON report includes the resolved mean, a percentile bootstrap interval,
and per-tag score summaries. The same case alignment and undefined-value policy
used by `metricguard run` applies here; invalid or entirely unresolved suites
fail rather than silently reporting a misleading interval. The seeded
resampling is reproducible across processes.
