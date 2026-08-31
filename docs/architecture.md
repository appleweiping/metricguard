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
6. `reporting` creates a versioned dictionary and the two supported renderings.

This direction keeps I/O and presentation out of metric implementations. A custom
metric only needs a stable `name` and an `evaluate(reference, prediction)` method.

## Failure boundaries

- File, schema, and case-model validation problems raise `CaseFormatError`.
- Configuration problems raise `ValueError` or `TypeError`.
- Undefined metric outcomes are values until the suite applies its policy.
- CLI input failures and contract failures use exit code 2.

Metric exceptions are not swallowed. A crashing metric is different from a metric
that intentionally declares a comparison undefined.

## Versioning

Package releases use semantic versioning. JSON reports contain `schema_version`;
package and report versions deliberately evolve independently.
