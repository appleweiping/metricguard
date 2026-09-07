# Report migration

`metricguard migrate-report INPUT OUTPUT` converts the legacy report shape
(missing `schema_version`, or `id`/`score`/`reason` result keys) to the
versioned schema emitted by current reports. The migration recomputes all
summary counts and means from result rows, so stale legacy summaries cannot
survive silently. Existing schema-v1 reports are validated and copied.

The operation is local, deterministic, and never evaluates a metric or changes
case data. It exits 2 for malformed or unsupported report versions.
