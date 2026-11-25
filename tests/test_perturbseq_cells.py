"""Tests for the exploratory cell-level Perturb-seq evaluation."""

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

path = (
    Path(__file__).resolve().parents[1]
    / "src/expt/workflow/scripts/perturbseq_cells_fit.py"
)
spec = importlib.util.spec_from_file_location("perturbseq_cells_fit", path)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def test_subset_retains_cells_and_disjoint_batches():
    data = {"obs": {"X": np.arange(16).reshape(8, 2), "batch": np.arange(1, 9)}}
    a = fit.subset(data, (1, 2, 3, 4))
    b = fit.subset(data, (7, 8))
    assert len(a["obs"]) == 4
    assert len(b["obs"]) == 2
    assert not set(a["obs"][:, 0]) & set(b["obs"][:, 0])
    with pytest.raises(ValueError, match="missing batches"):
        fit.subset(data, (9,))


def test_constant_training_feature_is_reported():
    batch = np.repeat(np.arange(1, 9), 2)
    data = {
        "obs": {
            "X": np.column_stack([np.arange(len(batch)), np.zeros(len(batch))]),
            "batch": batch,
        }
    }
    with pytest.raises(ValueError, match="constant training features.*silent_gene"):
        fit.validate_training_variation(data, ["variable_gene", "silent_gene"])


def test_density_is_joint_and_rejects_invalid_partition():
    train = {"obs": np.array([[-1.0, -2.0], [1.0, 2.0], [-1.0, 2.0], [1.0, -2.0]])}
    test = {"obs": np.array([[0.5, 1.0]])}
    graph = fit.singleton_graph(2)
    score = fit.graph_density_score(graph, train, test)["mean_per_gene"]
    variances = np.array([1.0, 4.0]) * (1 + fit.FLOOR)
    expected = np.mean(
        -0.5 * (np.log(2 * np.pi * variances) + test["obs"][0] ** 2 / variances)
    )
    assert score == pytest.approx(expected)
    graph.add_node((0, 1))
    with pytest.raises(ValueError, match="partition"):
        fit.graph_density_score(graph, train, test)
    cyclic = nx.DiGraph([((0,), (1,)), ((1,), (0,))])
    with pytest.raises(ValueError, match="DAG"):
        fit.graph_density_score(cyclic, train, test)


def test_end_to_end_uses_fixed_batch_holdout():
    rng = np.random.default_rng(41)
    batch = np.repeat(np.arange(1, 9), 20)
    data = {
        "obs": {"X": rng.normal(size=(len(batch), 3)), "batch": batch},
        "a": {"X": rng.normal(size=(len(batch), 3)) + [2, 1, 0], "batch": batch},
    }
    result = fit.fit_and_evaluate(data, {"features": ["x", "y", "z"]}, (0.001, 0.1))
    assert result["test_batches"] == [7, 8]
    assert result["train_batches"] == [1, 2, 3, 4, 5, 6]
    assert result["selected_alpha"] in (0.001, 0.1)
    assert set(result["methods"]) == {"coarse", "singleton_empty_dag"}
    assert np.isfinite(result["methods"]["coarse"]["heldout_loglik"]["mean_per_gene"])
