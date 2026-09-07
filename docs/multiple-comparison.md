# Multiple-comparison correction

`compare-slices` evaluates one paired hypothesis per metadata slice. When a
release decision uses those p-values as a family, select an explicit
correction:

```console
metricguard compare-slices baseline.jsonl candidate.jsonl \
  --metric exact_match --field cohort.language \
  --p-value-correction benjamini-hochberg --alpha 0.05 \
  --output slice-report.json
```

Supported methods are:

- `bonferroni`: simple family-wise error control;
- `holm`: step-down family-wise error control with more power;
- `benjamini-hochberg`: false-discovery-rate control.

The report preserves each raw paired p-value and adds `adjusted_p_value` and
`significance_passed`. This significance result is deliberately separate from
MetricGuard's confidence/delta regression gate; a caller can require both for
a release. Input ordering is preserved and ties are resolved by slice order.
