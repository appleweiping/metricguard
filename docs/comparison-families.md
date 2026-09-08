# Comparison families

`compare-slices` corrects p-values within one metadata field. For a report
that explores several dimensions, use `compare-family` so every successful
slice across all requested fields enters one shared correction family:

```bash
metricguard compare-family baseline.json candidate.json \
  --metric token_f1 --fields dataset,language,metadata.domain \
  --p-value-correction holm --alpha 0.05 --output family.json
```

The output preserves field and slice order, reports the adjusted p-value for
each successful comparison, and returns exit code `2` unless every slice
passes both the configured effect gate and the family-level significance gate.
Use `--p-value-correction none` when significance correction is intentionally
not part of the claim; the existing confidence/delta gate remains independent.
