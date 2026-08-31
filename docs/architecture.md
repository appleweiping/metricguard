# Architecture

MetricGuard separates raw metric semantics from suite policy.

## Layers

1. `io` converts JSON/JSONL into typed, frozen `EvaluationCase` records. Nested
   JSON payloads remain ordinary mutable Python values. The loader rejects
   unexpected fields so misspellings cannot silently change a run.
2. `normalizers` provides an immutable normalization policy.
3. `metrics` compares one reference/prediction pair and returns `MetricValue`.
   Metrics may return an undefined value, but do not decide how it affects an aggregate.
4. `suite` applies `UndefinedPolicy`, producing one `CaseResult` per input case.
5. `contracts` performs independent behavioral checks.
6. `registry` owns built-in factories and explicit entry-point discovery.
7. `statistics` performs deterministic paired-bootstrap estimation and a separate
   paired sign-flip randomization test.
8. `comparison` verifies dataset alignment before evaluating two model runs.
9. `reporting` creates versioned dictionaries and JSON/Markdown renderings.

This direction keeps I/O and presentation out of metric implementations. A custom
metric only needs a stable `name` and an `evaluate(reference, prediction)` method.

## Failure boundaries

- File, schema, and case-model validation problems raise `CaseFormatError`.
- Configuration problems raise `ValueError` or `TypeError`.
- Undefined metric outcomes are values until the suite applies its policy.
- CLI input failures and contract failures use exit code 2.

Metric exceptions are not swallowed. A crashing metric is different from a metric
that intentionally declares a comparison undefined.

Plugin discovery is a trust boundary. Importing an entry point executes installed
third-party code, so the default registry contains built-ins only. A caller or CLI
user must opt in before discovery. Registry mutation is atomic, but in-process
plugin imports are executable code and cannot be sandboxed or rolled back.

Paired comparison is a data-integrity boundary. IDs may arrive in a different order,
but references and tag sets must match by ID. One metric configuration evaluates
both sides. Python's equality rule that
`True == 1` is deliberately not used for nested JSON-like references.

## Versioning

Package releases use semantic versioning. JSON reports contain `schema_version`;
package and report versions deliberately evolve independently.
