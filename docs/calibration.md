# Confidence calibration

`calibration_report()` measures whether confidence values in case metadata
match binary outcomes. It reports the Brier score plus equal-width expected and
maximum calibration error. Missing or invalid metadata fails rather than
silently dropping cases.

```json
{
  "id": "case-1",
  "reference": "yes",
  "prediction": "yes",
  "metadata": {"model": {"confidence": 0.9}, "correct": true}
}
```

```console
metricguard calibrate cases.jsonl \
  --confidence-field model.confidence \
  --outcome-field correct --bins 10 \
  --output calibration.json
```

The report keeps empty bins so plots and comparisons have stable boundaries.
`gap` is `accuracy - mean_confidence`; positive values indicate
under-confidence in a bin. This is a diagnostic, not proof that a model's
probabilities are valid or that the case set is representative.
