from pathlib import Path

import pytest

from metricguard import EvaluationCase, ExperimentMatrix, ExperimentSpec, build_metric


def cases() -> tuple[EvaluationCase, ...]:
    return (
        EvaluationCase("a", "hello", "hello"),
        EvaluationCase("b", "hello", "bye"),
    )


def test_matrix_preserves_order_and_reuses_cell_caches(tmp_path: Path) -> None:
    specs = (
        ExperimentSpec("exact", build_metric("exact_match"), cases(), workers=2),
        ExperimentSpec("tokens", build_metric("token_f1"), cases()),
    )
    matrix = ExperimentMatrix(specs)
    first = matrix.run(cache_dir=tmp_path / "cache", max_workers=2)
    second = matrix.run(cache_dir=tmp_path / "cache", max_workers=2)
    assert [item.name for item in first] == ["exact", "tokens"]
    assert first[0].mean_score == 0.5
    assert second[0].report.cached == 2 and second[1].report.cached == 2
    assert dict(ExperimentMatrix.means(first)) == {"exact": 0.5, "tokens": 0.5}


def test_matrix_rejects_duplicate_names_and_invalid_workers() -> None:
    metric = build_metric("exact_match")
    spec = ExperimentSpec("same", metric, cases())
    with pytest.raises(ValueError, match="unique"):
        ExperimentMatrix((spec, spec))
    with pytest.raises(ValueError, match="max_workers"):
        ExperimentMatrix((spec,)).run(max_workers=0)
    with pytest.raises(ValueError, match="name"):
        ExperimentSpec("not valid", metric, cases())
