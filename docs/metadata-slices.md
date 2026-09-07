# Metadata slice reports

The `slices` command evaluates one metric and groups resolved scores by a
dotted field in each case's `metadata` object. This makes cohort, language,
source, and annotation-stratum regressions visible without copying raw text.

```bash
metricguard slices cases.jsonl --metric cer --field cohort.language \
  --undefined skip --min-count 5 --output slices.json
```

Values are serialized deterministically; missing or null paths use the explicit
`--missing` bucket. `--min-count` suppresses tiny slices. The Python API is
`summarize_by_metadata(report, cases, field=...)`, returning immutable
`SliceSummary` values with counts and macro means.
