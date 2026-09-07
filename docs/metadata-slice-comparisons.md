# Metadata slice comparisons

`compare-slices` evaluates the same baseline and candidate cases independently
for each metadata value, then applies MetricGuard's paired bootstrap interval and
sign-flip regression gate to every slice. Case IDs, references, and the selected
metadata value must remain aligned; metadata movement is rejected instead of
silently changing a cohort's population.

```bash
metricguard compare-slices baseline.jsonl candidate.jsonl \
  --metric cer --field cohort.language --undefined skip \
  --samples 2000 --minimum-delta 0.01 --minimum-lower-bound 0
```

The JSON report contains one object per slice with its case count, baseline and
candidate means, oriented improvement, confidence bounds, p-value, and gate
status. Use `--direction lower` for error metrics where a smaller value is an
improvement. `--min-count` can suppress underpowered slices; a report with no
eligible slices fails the command so a missing cohort is not mistaken for a
passing check.

The Python API is `compare_by_metadata(...)`, which returns immutable
`SliceComparison` values. The comparison is paired by case ID within each slice,
so the same test population is used on both sides.
