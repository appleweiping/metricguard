# Changelog

All notable changes follow semantic versioning.

## 0.2.0 - 2026-08-31

- Add pairwise ROUGE-L, sentence BLEU, and normalized Levenshtein similarity.
- Add deterministic percentile confidence intervals and paired-bootstrap comparisons.
- Add strict baseline/candidate alignment and configurable CI regression gates.
- Add tag-grouped summaries and versioned comparison JSON/Markdown reports.
- Add an explicit metric registry and opt-in `metricguard.metrics` entry-point discovery.
- Add comparison examples, statistical/plugin documentation, and a reproducible benchmark.
- Use a paired sign-flip randomization p-value, explicit optimization direction,
  atomic plugin registration, and overwrite-safe CLI outputs.

## 0.1.0 - 2026-08-31

- Add strict JSON/JSONL evaluation case loading.
- Add exact, token F1, character F1, and numeric equivalence metrics.
- Add explicit undefined-result policies and behavior contracts.
- Add JSON and Markdown reports, CLI, tests, examples, and CI.
