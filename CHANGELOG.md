# Changelog

All notable changes follow the principles of Keep a Changelog.

## [Unreleased]

## [0.2.0] - 2026-08-31

### Added

- Pairwise ROUGE-L, sentence BLEU, and normalized Levenshtein similarity.
- Deterministic percentile confidence intervals and paired-bootstrap comparisons.
- Strict baseline/candidate alignment and configurable CI regression gates.
- Tag-grouped summaries and versioned comparison JSON/Markdown reports.
- An explicit metric registry and opt-in `metricguard.metrics` entry-point discovery.
- Comparison examples, statistical and plugin documentation, and a reproducible benchmark.

### Changed

- Comparisons use a paired sign-flip randomization p-value and an explicit optimization direction.
- Plugin registration is atomic, and CLI outputs are overwrite-safe.

## [0.1.0] - 2026-08-31

### Added

- Strict JSON/JSONL evaluation case loading.
- Exact, token F1, character F1, and numeric equivalence metrics.
- Explicit undefined-result policies and behavior contracts.
- JSON and Markdown reports, CLI, tests, examples, and CI.
