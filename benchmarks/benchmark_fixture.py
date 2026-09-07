"""Benchmark MetricGuard on checked-in evaluation cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

from metricguard import EvaluationSuite, UndefinedPolicy, WordErrorRate, build_metric
from metricguard.io import load_cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "examples" / "baseline_cases.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "benchmarks/results/fixture.json")
    args = parser.parse_args()
    payload = source.read_bytes()
    cases = load_cases(source)
    tracemalloc.start()
    started = time.perf_counter()
    text_report = EvaluationSuite(cases).run(build_metric("token_f1"))
    wer_report = EvaluationSuite(cases, undefined_policy=UndefinedPolicy.SKIP).run(WordErrorRate())
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "kind": "fixture-real",
        "source": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "cases": len(cases),
        "token_f1_mean": text_report.mean_score,
        "word_error_rate_mean": wer_report.mean_score,
        "word_error_rate_skipped": wer_report.skipped_count,
        "elapsed_seconds": elapsed,
        "peak_python_bytes": peak,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
