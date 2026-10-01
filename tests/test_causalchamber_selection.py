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
        "ari": ari, "atom_dir_f1": f1, "atom_dir_precision": precision,
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
    all_edge_metrics,
    baseline_env_lists,
    gaussian_bic_score,
    quotient_graph,
    select_utigsp_oracle_row,
    select_utigsp_score_row,
    subsample_rows,
)


def _true_chain():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


def test_quotient_graph_keeps_both_directions_and_drops_within_block_edges():
    # 0 -> 1 directed, 1 - 2 undirected (both entries set), 2 -> 3
    adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]])
    edges = list(zip(*np.nonzero(adj)))
    singletons = quotient_graph(edges, [{0}, {1}, {2}, {3}])
    assert set(singletons.edges) == {((0,), (1,)), ((1,), (2,)), ((2,), (1,)), ((2,), (3,))}
    coarse = quotient_graph(edges, [frozenset({0}), frozenset({1, 2}), frozenset({3})])
    assert set(coarse.nodes) == {(0,), (1, 2), (3,)}
    assert set(coarse.edges) == {((0,), (1, 2)), ((1, 2), (3,))}


def test_all_edge_metrics_on_a_partition_differing_from_the_truth():
    truth = _true_chain()
    true_blocks = [{0}, {1, 2}, {3}]
    est = nx.DiGraph()
    est.add_nodes_from([(0, 1), (2,), (3,)])
    est.add_edges_from([((0, 1), (2,)), ((3,), (2,))])
    m = all_edge_metrics(est, truth, true_blocks)
    # Truth on the estimate's blocks: (0,1)->(2,), (2,)->(3,).
    assert (m["blk_dir_precision"], m["blk_dir_recall"]) == (0.5, 0.5)
    assert (m["blk_skel_precision"], m["blk_skel_recall"]) == (1.0, 1.0)
    # Atom pairs: estimate {02, 12, 32}; G/Π {01, 02, 13, 23}.
    assert m["atom_dir_precision"] == pytest.approx(1 / 3)
    assert m["atom_dir_recall"] == pytest.approx(1 / 4)
    assert m["atom_skel_precision"] == pytest.approx(2 / 3)
    assert m["atom_skel_recall"] == pytest.approx(1 / 2)
    # Unsorted tuple nodes (RePaRe's convention) score the same.
    unsorted = nx.relabel_nodes(est, {(0, 1): (1, 0)})
    assert all_edge_metrics(unsorted, truth, true_blocks) == m


def _ut_row(alpha_ci, alpha_inv, bic, f1=0.0, precision=0.0):
    return {
        "alpha_ci": alpha_ci, "alpha_inv": alpha_inv, "bic": bic,
        "atom_dir_f1": f1, "atom_dir_precision": precision,
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


from _common import block_dag_prf  # noqa: E402
from _causalchamber_common import build_block_data_dict  # noqa: E402


def test_block_dag_prf_counts_forward_block_edges():
    est = nx.DiGraph()
    est.add_nodes_from([(0,), (1,), (2, 3)])
    est.add_edge((0,), (1,))
    est.add_edge((2, 3), (1,))
    truth = _true_chain()  # 0->1, 1->2, 2->3
    # Reference over forward pairs in est node order: (0,)->(1,) and (1,)->(2,3) are true block
    # edges; est has 1 TP of 2 edges, reference has 2 edges.
    assert block_dag_prf(est, truth) == (0.5, 0.5, 0.5)
    empty = nx.DiGraph()
    empty.add_nodes_from([(0, 1, 2, 3)])
    assert block_dag_prf(empty, truth) == (1, 1, 1)


def test_build_block_data_dict_keys_and_types():
    blocks = {"obs": np.zeros((3, 2)), "rgb": np.ones((3, 2))}
    out = build_block_data_dict(blocks, {"rgb": {0, 1}})
    assert out["obs"] == (blocks["obs"], set(), "obs")
    assert out["rgb"][1] == {0, 1} and out["rgb"][2] == "soft"
