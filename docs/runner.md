# Resumable evaluation runner

`MetricRunner` evaluates an ordered case sequence, optionally using a bounded
thread pool and a JSONL checkpoint cache. A cache row is reused only when its
case ID, metric implementation identity, complete configuration, reference, and
prediction fingerprint match. Built-in identities include normalization, ranking
cutoffs, decimal tolerances, the Unicode data version, the numeric decimal context,
and a semantic implementation revision. Worker threads use the caller's decimal
context so precision and traps are consistent with serial evaluation. Results are
returned in input order regardless of worker completion order.

```python
from metricguard import MetricRunner

report = MetricRunner(cases, metric, workers=4).run(cache="results.jsonl")
```

Only raw metric values are cached. Undefined values are resolved using the policy
of the current run, including when a cached undefined result now uses `ERROR`.
Ordinary metric exceptions can either fail fast or be returned in `report.errors`
with `fail_fast=False` and a non-error undefined policy. Exceptions are never
cached, so transient backend failures are retried and remain visible on resume.

Version 2 cache rows carry a SHA-256 integrity checksum and are validated before
evaluation. Invalid rows, duplicate IDs/keys, unsupported versions, and truncated
JSON fail with the file path and line number, leaving the checkpoint unchanged.
Unversioned legacy rows are cache misses because their name-only fingerprints
cannot establish configuration equivalence. A checksum detects accidental edits;
it does not authenticate a cache supplied by somebody else.

Successful results are checkpointed every 100 newly evaluated cases, on normal
completion, and when a run exits with an evaluation or undefined-policy error.
Set `checkpoint_interval=1` in `run()` to persist every successful result, or raise
the interval to reduce the cost of rewriting a large checkpoint. Each checkpoint
is written to a flushed sibling file and atomically replaces the previous file.
An abrupt process termination can lose results since the last checkpoint; rerun
the same workload to compute the missing cases. Parallel workers may finish
in-flight evaluations after a failure, but only consumed results are persisted.
Give each simultaneous runner its own cache file: atomic replacement provides
file integrity, not multi-writer transaction coordination.

Cache rows are deterministic JSON and sorted by case ID. Cached inputs and result
details must use lossless JSON types with string object keys and finite numbers.
Uncached runs still accept other values supported by the metric, such as Decimal
inputs for numeric equivalence. Custom metrics must provide the explicit
[`cache_identity()` contract](plugins.md#resumable-cache-identity) to use a cache.
The runner does not discover plugins or execute remote providers, and its thread
pool does not make a non-thread-safe custom metric safe.

Maintainers must increment the built-in semantic revision in `_runner_cache.py`
when changing metric or normalization semantics; changing the display name or
distribution version alone is not a substitute for versioning cached behavior.
