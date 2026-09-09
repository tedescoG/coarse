"""Tests for the causalchamber grid-selection and edge-metric helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "expt" / "workflow" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
from _causalchamber_common import select_oracle_row  # noqa: E402


def _row(alpha, lam, score, ari=0.0, f1=0.0, precision=0.0):
    return {
        "alpha": alpha, "lambda": lam, "score": score,
        "ari": ari, "f1": f1, "precision": precision,
    }


def test_select_oracle_row_uses_ground_truth_over_whole_grid():
    # Best score (first row) loses to best ARI, then F1.
    records = [
        _row(1e-4, 1.0, 100.0, ari=0.2),
        _row(1e-2, 1.0, 10.0, ari=0.9, f1=0.5),
        _row(1e-3, 1.0, 20.0, ari=0.9, f1=0.7),
    ]
    assert select_oracle_row(records) == _row(1e-3, 1.0, 20.0, ari=0.9, f1=0.7)


# Baseline-method helpers (GIES / GnIES / UT-IGSP)

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from _causalchamber_common import (  # noqa: E402
    adjacency_to_singleton_dag,
    all_edge_metrics,
    baseline_env_lists,
    directed_edge_metrics,
    gaussian_bic_score,
    partition_edge_metrics,
    select_utigsp_oracle_row,
    select_utigsp_score_row,
    skeleton_edge_metrics,
    subsample_rows,
)


def test_adjacency_to_singleton_dag_keeps_both_directions_of_undirected_edge():
    # 0 -> 1 directed, 1 - 2 undirected (both entries set)
    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
    dag = adjacency_to_singleton_dag(adj)
    assert set(dag.nodes) == {(0,), (1,), (2,)}
    assert set(dag.edges) == {((0,), (1,)), ((1,), (2,)), ((2,), (1,))}


def _true_chain():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


def test_skeleton_metrics_ignore_orientation_directed_metrics_do_not():
    est = nx.DiGraph()
    est.add_nodes_from([(0,), (1,), (2, 3)])
    est.add_edge((0,), (1,))
    est.add_edge((2, 3), (1,))  # reversed relative to the truth 1 -> 2
    truth = _true_chain()
    directed = partition_edge_metrics(est, truth)
    skeleton = skeleton_edge_metrics(est, truth)
    assert directed["precision"] == 0.5
    assert skeleton == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_directed_metrics_are_index_order_independent_on_singletons():
    # True 3 -> 0 runs backward in index order; the estimate recovers it plus 0 -> 1.
    truth = nx.DiGraph()
    truth.add_nodes_from(range(4))
    truth.add_edges_from([(0, 1), (3, 0), (1, 2)])
    est = adjacency_to_singleton_dag(
        np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    )
    # tp=2, |est|=2, |true|=3
    assert directed_edge_metrics(est, truth) == {
        "precision": 1.0, "recall": pytest.approx(2 / 3), "f1": pytest.approx(0.8),
    }
    # The forward-only partition collapse drops the backward edges.
    forward_only = partition_edge_metrics(est, truth)
    assert forward_only["precision"] == 0.5 and forward_only["recall"] == 0.5


def test_skeleton_metrics_on_singletons():
    est = adjacency_to_singleton_dag(
        np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]])
    )  # 0 - 1 undirected, 3 -> 2 reversed
    truth = _true_chain()
    skel = skeleton_edge_metrics(est, truth)
    # Pairs {0,1} and {2,3} are in the true skeleton; 1-2 is missed.
    assert skel["precision"] == 1.0
    assert skel["recall"] == pytest.approx(2 / 3)


def test_all_edge_metrics_copies_native_triple():
    est = nx.DiGraph()
    est.add_nodes_from([(0,), (1,), (2,), (3,)])
    est.add_edge((1,), (0,))
    truth = _true_chain()
    edges = all_edge_metrics(est, truth, native="edges")
    skel = all_edge_metrics(est, truth, native="skeleton")
    part = all_edge_metrics(est, truth, native="partition")
    assert edges["metric_type"] == "edges"
    assert (edges["precision"], edges["recall"]) == (edges["dir_precision"], edges["dir_recall"])
    assert edges["dir_precision"] == 0.0 and edges["skel_precision"] == 1.0
    assert skel["metric_type"] == "skeleton"
    assert skel["precision"] == skel["skel_precision"] == 1.0
    # Partition-native uses the forward-only collapse: the reversed estimate is a FP
    # and the true 0 -> 1 pair drops out of the recall denominator.
    assert part["metric_type"] == "partition"
    assert part["precision"] == 0.0
    assert part["recall"] == 0.0 and part["dir_recall"] == 0.0
    with pytest.raises(ValueError):
        all_edge_metrics(est, truth, native="cpdag")


def _ut_row(alpha_ci, alpha_inv, bic, f1=0.0, precision=0.0):
    return {
        "alpha_ci": alpha_ci, "alpha_inv": alpha_inv, "bic": bic,
        "f1": f1, "precision": precision,
    }


def test_select_utigsp_score_row_min_bic_then_smaller_alphas():
    records = [
        _ut_row(1e-2, 1e-2, 10.0, f1=1.0),
        _ut_row(1e-3, 1e-1, 5.0),
        _ut_row(1e-3, 1e-2, 5.0),
        _ut_row(1e-4, 1e-1, 5.0),
    ]
    assert select_utigsp_score_row(records) == _ut_row(1e-4, 1e-1, 5.0)


def test_select_utigsp_oracle_row_max_f1_times_precision():
    records = [
        _ut_row(1e-4, 1e-4, 1.0, f1=0.9, precision=0.5),   # 0.45
        _ut_row(1e-2, 1e-2, 2.0, f1=0.8, precision=0.7),   # 0.56, later in grid order
        _ut_row(1e-3, 1e-3, 9.0, f1=0.8, precision=0.7),   # 0.56 tie -> first in grid order
    ]
    # Ties go to the first cell in grid order.
    assert select_utigsp_oracle_row(records) == _ut_row(1e-3, 1e-3, 9.0, f1=0.8, precision=0.7)


def test_baseline_env_lists_grouped_and_ungrouped():
    blocks = {k: np.full((2, 3), i, dtype=float)
              for i, k in enumerate(["obs", "rgb", "pol", "red", "pol_1"])}
    group_targets = {"rgb": {0, 2}, "pol": {1}}
    name_to_idx = {"red": 0, "pol_1": 1, "green": 2}
    data, targets, labels = baseline_env_lists(
        "grouped", blocks, group_targets, ["red", "pol_1"], name_to_idx
    )
    assert labels == ["obs", "rgb", "pol"]
    assert targets == [[], [0, 2], [1]]
    assert [d[0, 0] for d in data] == [0.0, 1.0, 2.0]

    data, targets, labels = baseline_env_lists(
        "ungrouped", blocks, group_targets, ["red", "pol_1"], name_to_idx
    )
    assert labels == ["obs", "red", "pol_1"]
    assert targets == [[], [0], [1]]
    assert [d[0, 0] for d in data] == [0.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        baseline_env_lists("other", blocks, group_targets, [], name_to_idx)


def test_subsample_rows_is_a_no_op_below_cap_and_deterministic_above():
    rng = np.random.default_rng(0)
    small = rng.normal(size=(10, 2))
    assert subsample_rows(small, max_rows=10, seed=0) is small
    big = rng.normal(size=(50, 2))
    a = subsample_rows(big, max_rows=20, seed=3)
    b = subsample_rows(big, max_rows=20, seed=3)
    assert a.shape == (20, 2)
    np.testing.assert_array_equal(a, b)


def test_gaussian_bic_prefers_true_parent_on_linear_data():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = 1.5 * x + rng.normal(size=500)
    data = np.column_stack([x, y])
    empty = nx.DiGraph()
    empty.add_nodes_from([0, 1])
    true = empty.copy()
    true.add_edge(0, 1)
    # Lower is better.
    assert gaussian_bic_score(data, true) < gaussian_bic_score(data, empty)
    assert gaussian_bic_score(data, nx.DiGraph()) == float("inf")
