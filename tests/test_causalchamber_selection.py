"""Tests for the causalchamber grid-selection and edge-metric helpers."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from coarse.partition import infer_partition

from _causalchamber_common import (
    atomic_graph,
    baseline_env_lists,
    collapse,
    exact_nc,
    gaussian_bic_score,
    graph_edge_metrics,
    ground_truth_partition,
    icpdag,
    random_expanded,
    random_icpdag,
    select_oracle_row,
    select_score_row,
    select_utigsp_oracle_row,
    select_utigsp_score_row,
    singleton_dag,
    skeleton_metrics,
    variable_metrics,
)


def test_grid_selection_rules():
    """Score row: min score, ties to smaller α then β. Oracle row: max ARI, then F1, then smaller α
    and β (β optional). UT-IGSP: min BIC / max F1·precision, ties to the first cell in grid order."""
    def row(alpha, beta, score, ari=0.0, f1=0.0):
        return {"alpha": alpha, "beta": beta, "score": score, "ari": ari, "f1": f1}

    assert select_score_row([
        row(1e-4, 1e-4, 2.0, ari=1.0), row(1e-2, 1e-4, 1.0, ari=1.0),
        row(1e-3, 1e-1, 1.0), row(1e-3, 1e-3, 1.0),
    ]) == row(1e-3, 1e-3, 1.0)
    assert select_oracle_row([
        row(1e-4, 1e-4, 1.0, ari=0.2, f1=1.0), row(1e-4, 1e-4, 9.0, ari=0.9, f1=0.1),
        row(1e-2, 1e-4, 9.0, ari=0.9, f1=0.9), row(1e-3, 1e-2, 9.0, ari=0.9, f1=0.9),
        row(1e-3, 1e-3, 9.0, ari=0.9, f1=0.9),
    ]) == row(1e-3, 1e-3, 9.0, ari=0.9, f1=0.9)
    no_beta = [{"alpha": 1e-2, "ari": 1.0, "f1": 0.5}, {"alpha": 1e-3, "ari": 1.0, "f1": 0.5}]
    assert select_oracle_row(no_beta) == no_beta[1]

    def ut(alpha_ci, alpha_inv, bic, f1=0.0, precision=0.0):
        return {"alpha_ci": alpha_ci, "alpha_inv": alpha_inv, "bic": bic, "f1": f1, "precision": precision}

    assert select_utigsp_score_row([
        ut(1e-2, 1e-2, 10.0, f1=1.0), ut(1e-3, 1e-1, 5.0), ut(1e-3, 1e-2, 5.0), ut(1e-4, 1e-1, 5.0),
    ]) == ut(1e-4, 1e-1, 5.0)
    assert select_utigsp_oracle_row([
        ut(1e-4, 1e-4, 1.0, f1=0.9, precision=0.5),
        ut(1e-2, 1e-2, 2.0, f1=0.8, precision=0.7),
        ut(1e-3, 1e-3, 9.0, f1=0.8, precision=0.7),
    ]) == ut(1e-3, 1e-3, 9.0, f1=0.8, precision=0.7)


def _true_chain():
    g = nx.DiGraph()
    g.add_nodes_from(range(4))
    g.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return g


def test_graph_and_skeleton_metrics_on_an_atomic_cpdag():
    # 0 -> 1, 1 - 2 undirected (both entries set), 3 -> 2 reversed.
    adj = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    est = atomic_graph(adj)
    assert set(est.edges) == {(0, 1), (1, 2), (2, 1), (3, 2)}
    m = graph_edge_metrics(est, _true_chain())
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)
    s = skeleton_metrics(est, _true_chain())
    assert (s["precision"], s["recall"]) == (1.0, 1.0)
    assert set(singleton_dag(est).nodes) == {(0,), (1,), (2,), (3,)}


def test_ground_truth_partition_matches_infer_partition_blocks():
    truth = nx.DiGraph([(0, 1), (1, 2), (3, 2), (2, 4)])
    parts = [{i} for i in range(5)]
    targets = {"a": {0}, "b": {3}}
    partition, labels = ground_truth_partition(targets, parts, truth)
    M = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)
    assert {frozenset(b) for b in partition} == set(infer_partition(M))
    assert all(labels[i] == labels[j] for b in partition for i in b for j in b)


def test_baseline_env_lists_is_the_ungrouped_regime():
    blocks = {k: np.full((2, 3), i, dtype=float)
              for i, k in enumerate(["obs", "rgb", "pol", "red", "pol_1"])}
    name_to_idx = {"red": 0, "pol_1": 1, "green": 2}
    data, targets, labels = baseline_env_lists(blocks, ["red", "pol_1"], name_to_idx)
    assert labels == ["obs", "red", "pol_1"]
    assert targets == [[], [0], [1]]
    assert [d[0, 0] for d in data] == [0.0, 3.0, 4.0]


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


def test_icpdag_hand_cases_and_gnies_convention():
    chain = {(0, 1), (1, 2)}
    assert icpdag(chain, [[0]], 3) == (chain, set())
    assert icpdag(chain, [], 3) == (set(), {frozenset((0, 1)), frozenset((1, 2))})
    assert icpdag({(0, 2), (1, 2)}, [], 3) == ({(0, 2), (1, 2)}, set())
    from gnies.utils import dag_to_icpdag

    arcs = {(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)}
    adj = np.zeros((5, 5), dtype=int)
    for u, v in arcs:
        adj[u, v] = 1
    ref = dag_to_icpdag(adj, {1, 2})
    ref_arcs = {(i, j) for i, j in zip(*np.nonzero(ref)) if not ref[j, i]}
    assert icpdag(arcs, [[1], [2]], 5)[0] == ref_arcs


def test_collapse_drops_within_block_and_merges_opposite_arcs():
    block_of = {0: "a", 1: "a", 2: "b", 3: "c", 4: "c"}
    g = ({(0, 1), (0, 2), (2, 1), (3, 0)}, {frozenset((2, 4))})
    arcs, und = collapse(g, block_of)
    assert arcs == {("c", "a")}
    assert und == {frozenset(("a", "b")), frozenset(("b", "c"))}


def test_variable_metrics_counts():
    truth = {(0, 1), (1, 2), (2, 3)}
    est = ({(0, 1), (2, 1)}, {frozenset((2, 3))})  # correct, reversed, undirected
    m = variable_metrics(est, truth)
    assert (m["adj_precision"], m["adj_recall"], m["m_est"]) == (1.0, 1.0, 3)
    assert m["dir_precision"] == pytest.approx(2 / 4)
    assert m["dir_recall"] == pytest.approx(2 / 3)


def test_exact_nc_mean_and_p():
    res = exact_nc(190, 39, 29, 0)
    assert res["adj_f1"][0] == pytest.approx(2 * 29 * 39 / (190 * (29 + 39)))
    assert res["adj_f1"][3] == pytest.approx(1.0)


def test_random_draws_keep_size_and_block_structure():
    rng = np.random.default_rng(0)
    arcs, und = random_icpdag(rng, 8, 10, [[0]])
    assert len(arcs) + len(und) == 10
    blocks = [(0, 1), (2,), (3, 4, 5)]
    arcs, _ = random_expanded(rng, blocks, 2)
    block_of = {v: b for b in blocks for v in b}
    assert all(block_of[u] != block_of[v] for u, v in arcs)
    assert len({(block_of[u], block_of[v]) for u, v in arcs}) == 2
