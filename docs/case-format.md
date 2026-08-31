# Evaluation case format

MetricGuard accepts a JSON array or JSONL stream. Blank JSONL lines are ignored.

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `id` | string | yes | Stable unique identity inside one run |
| `reference` | any JSON value | yes | Expected value; text metrics require a string |
| `prediction` | any JSON value | yes | Candidate value; text metrics require a string |
| `tags` | array of unique strings | no | Caller-defined grouping labels |
| `metadata` | object | no | Caller-owned provenance |

Unknown fields are rejected. Metadata is not included in built-in metric evaluation.
This prevents a metric from accidentally depending on dataset-specific side channels.

## Numeric values

JSON numbers may lose decimal spelling before they reach Python. If exact decimal
spelling matters, store the value as a JSON string and let `numeric_equivalence`
parse it.
