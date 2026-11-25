"""Tests for the per-fit job decomposition: fold fits, selection, final fits."""

import importlib
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from _perturbseq_common import PS_GRIDS

SCRIPTS = str(Path(__file__).resolve().parents[1] / "src/expt/workflow/scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
compare = importlib.import_module("perturbseq_cells_compare")
common = importlib.import_module("_perturbseq_common")


def cell_data(seed=0):
    rng = np.random.default_rng(seed)
    batch = np.repeat(np.arange(1, 9), 30)
    return {
        "obs": {"X": rng.normal(size=(len(batch), 3)), "batch": batch},
        "a": {"X": rng.normal(size=(len(batch), 3)) + [1.5, 0, 0], "batch": batch},
        "b": {"X": rng.normal(size=(len(batch), 3)) + [0, 1.5, 0], "batch": batch},
    }


def test_outer_pairs_and_inner_folds_are_disjoint():
    all_outer = []
    for split in range(4):
        train, test, inner = (
            common.outer_train_batches(split),
            common.BATCH_PAIRS[split],
            common.inner_validation_pairs(split),
        )
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(1, 9))
        assert set(sum(inner, ())) == set(train)
        all_outer.extend(test)
    assert sorted(all_outer) == list(range(1, 9))


def test_fold_fit_scores_one_validation_pair_and_records_failures():
    data = cell_data()
    record = compare.run_fold(data, ["x", "y", "z"], 3, "coarse", 1, 2)
    assert record["status"] == "ok"
    assert np.isfinite(record["score"])
    assert not set(record["train_batches"]) & set(record["validation_batches"])
    assert record["validation_batches"] == [5, 6]

    def failing_fit(method, inputs, params):
        raise ValueError("boom")

    record = compare.run_fold(data, ["x", "y", "z"], 3, "coarse", 1, 2, fit=failing_fit)
    assert record["status"] == "failed"
    assert "boom" in record["error"]["message"]


def test_selection_picks_best_mean_and_reports_all_failed():
    folds = [
        {"candidate": c, "fold": f, "status": "ok", "score": s}
        for c, scores in enumerate([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]])
        for f, s in enumerate(scores)
    ]
    from perturbseq_cells_select import select_candidates

    result = select_candidates(folds, "coarse")
    assert result["status"] == "ok"
    assert result["selected_index"] == 1
    assert result["selected_parameters"] == [1e-3]

    failed = [
        {"candidate": c, "fold": f, "status": "failed", "error": {"message": "x"}}
        for c in range(len(PS_GRIDS["coarse"]))
        for f in range(3)
    ]
    result = select_candidates(failed, "coarse")
    assert result["status"] == "failed"
    assert result["selected_parameters"] is None


def test_final_skips_fitting_when_selection_failed():
    calls = []

    def fit(method, inputs, params):
        calls.append(params)
        return compare.singleton_graph(3)

    data = cell_data()
    selection = {"status": "failed", "selected_parameters": None, "candidates": []}
    result = compare.run_final(data, ["x", "y", "z"], 3, "coarse", selection, fit=fit)
    assert result["status"] == "failed"
    assert not calls
    selection = {"status": "ok", "selected_index": 1, "selected_parameters": [0.001]}
    result = compare.run_final(data, ["x", "y", "z"], 3, "coarse", selection, fit=fit)
    assert result["status"] == "ok"
    assert calls == [(0.001,)]


@pytest.mark.parametrize(
    "method,params",
    [("repare", (0.01, 0.01)), ("utigsp", (0.01, 0.01)), ("coarse_kpc1", (0.01,)), ("coarse_kpc3", (0.01,)), ("coarse_avg", (0.01,)), ("coarse_scored", (1.0, 1.0))],
)
def test_real_method_adapters_on_small_synthetic_data(method, params):
    inputs = {env: rec["X"] for env, rec in cell_data().items()}
    dag = compare.fit_graph(method, inputs, params)
    assert nx.is_directed_acyclic_graph(dag)
    assert sorted(i for node in dag.nodes for i in node) == [0, 1, 2]


@pytest.mark.parametrize(
    "method",
    ["coarse", "coarse_kpc1", "coarse_kpc3", "coarse_avg", "coarse_scored", "repare", "utigsp", "singleton_empty", "joint"],
)
def test_complete_pipeline_on_synthetic_data(method):
    data = cell_data()
    fold_records = [
        compare.run_fold(data, ["x", "y", "z"], 3, method, c, f)
        for c in range(len(common.PS_GRIDS[method]))
        for f in range(3)
    ]
    from perturbseq_cells_select import select_candidates

    selection = select_candidates(fold_records, method)
    final = compare.run_final(data, ["x", "y", "z"], 3, method, selection)
    assert final["status"] == "ok", final.get("error")
    assert np.isfinite(final["heldout_loglik"]["mean_per_gene"])
    assert len(final["blocks"]) > 0
