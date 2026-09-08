# Changelog

## Unreleased

- Add deterministic cross-metric Pearson and Spearman correlation reports with
  pairwise-complete score alignment, explicit missing-data reasons, a CLI
  `correlate` command, and loopback-service support.

All notable changes follow the principles of Keep a Changelog.

## [Unreleased]

- Add deterministic confidence calibration reports with equal-width bins,
  Brier score, expected calibration error, maximum calibration error, and a
  `calibrate` CLI command for nested case metadata.
- Expose confidence calibration through the loopback service with configurable
  nested metadata fields and bin counts.
- Expose metadata-slice summaries and family-corrected comparisons through the
  loopback service.
- Expose the deterministic OCR text-pair benchmark through the loopback
  service, including CER, WER, and exact-match summaries.

- Add a strict loopback JSON/HTTP service for metric runs and paired comparisons.
- Add a service-level matrix operation with resumable per-metric caches and leaderboard output.

- Add `compare-family` and `compare_metadata_family` for one corrected p-value family across metadata dimensions.

- Add `migrate-report` and a public report migration API for legacy JSON report
  shapes with recomputed summaries.

- Add deterministic Bonferroni, Holm, and Benjamini-Hochberg corrections for
  metadata-slice p-value families, exposed through `compare-slices`.

- Add paired, direction-aware metadata slice comparisons and the
  `compare-slices` regression-gate CLI.
- Add deterministic nested-metadata slice summaries and the `slices` CLI.
- Add OCR benchmark orchestration combining CER, WER, and normalized exact-match reports.
- Expose the OCR workflow through the `metricguard ocr` command.
- Add a shell-free, bounded external OCR command backend and image-case benchmark
  adapter for local tools and model-server wrappers.
- Add strict JSON/JSONL image-case loading and an `ocr-backend` CLI that runs
  the external adapter with explicit timeout/output limits.
- Add document-level OCR manifests, deterministic page ordering, per-document
  reports, and the `metricguard ocr-document` CLI.
- Add a `metricguard confidence` CLI for deterministic bootstrap intervals and
  per-tag summaries.
- Add bounded-memory JSONL evaluation through `iter_cases`, `evaluate_stream`,
  and the `metricguard stream-run` CLI with deterministic tag/error aggregates.

### Added

- Five configurable retrieval metrics with explicit cutoff, relevance, short-run,
  duplicate-document, and undefined-result policies, available through the metric registry.
- Ordered parallel metric runner with case fingerprints, resumable JSONL cache, and error policy.
- Ordered metric experiment matrices with isolated resumable cell caches.
- Deterministic experiment leaderboards with tie-aware ranks, score completeness, and cache/error visibility.
- Character and word error-rate metrics with explicit empty-reference semantics for OCR evaluation.

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
