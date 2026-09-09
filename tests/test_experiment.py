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


def test_same_experiment_names_recompute_changed_configuration(tmp_path: Path) -> None:
    data = (EvaluationCase("a", "HELLO", "hello"),)
    first = ExperimentMatrix(
        [ExperimentSpec("same", build_metric("exact_match"), data, workers=2)]
    ).run(cache_dir=tmp_path, max_workers=2)
    changed = ExperimentMatrix(
        [
            ExperimentSpec(
                "same",
                build_metric({"kind": "exact_match", "normalizer": {"lowercase": True}}),
                data,
                workers=2,
            )
        ]
    ).run(cache_dir=tmp_path, max_workers=2)
    assert first[0].mean_score == 0.0
    assert changed[0].mean_score == 1.0 and changed[0].report.cached == 0


def test_leaderboard_is_deterministic_and_exposes_contract_state(tmp_path: Path) -> None:
    specs = (
        ExperimentSpec("high", build_metric("exact_match"), cases()),
        ExperimentSpec("low", build_metric("token_f1"), cases()),
    )
    results = ExperimentMatrix(specs).run(cache_dir=tmp_path / "cache")
    ranked = ExperimentMatrix.leaderboard(results)
    assert [item.rank for item in ranked] == [1, 1]
    assert [item.name for item in ranked] == ["high", "low"]
    assert all(item.total_cases == 2 and item.scored_cases == 2 for item in ranked)
    assert all(item.contract_passed for item in ranked)
    assert ExperimentMatrix.leaderboard(results, higher_is_better=False)[0].name == "high"


def test_leaderboard_rejects_duplicate_result_names() -> None:
    metric = build_metric("exact_match")
    spec = ExperimentSpec("same", metric, cases())
    result = ExperimentMatrix((spec,)).run()[0]
    with pytest.raises(ValueError, match="unique"):
        ExperimentMatrix.leaderboard((result, result))
