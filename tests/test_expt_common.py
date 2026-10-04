"""Tests for the label and metric helpers in workflow/scripts/_common.py."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from _common import (
    block_parents,
    build_oracle_partition,
    correct,
    family_size,
    interventional_partition,
    partition_edge_metrics,
    partition_labels,
    prf,
    welch_pvalues,
)
from coarse.hypothesis_tests import compute_M
from conftest import sample_chain_dataset


def test_interventional_partition_single_vs_multi_target():
    """Chain 0→1→2→3: single-target envs {0}, {2} give the env partition; env {0, 2} merges
    {0, 1} with {2, 3} in the env partition but not in the interventional one."""
    W = np.zeros((4, 4))
    W[0, 1] = W[1, 2] = W[2, 3] = 1.0
    blocks = {frozenset({0, 1}), frozenset({2, 3})}
    assert set(interventional_partition(W, [[0], [2]], 4)) == blocks
    assert set(build_oracle_partition(W, [[0], [2]], 4)[3]) == blocks
    assert set(interventional_partition(W, [[0, 2]], 4)) == blocks
    assert set(build_oracle_partition(W, [[0, 2]], 4)[3]) == {frozenset({0, 1, 2, 3})}


def test_partition_labels_indexes_blocks_in_iteration_order():
    labels = partition_labels([frozenset({2}), (0, 3), {1}], 4)
    assert labels.tolist() == [1, 2, 0, 1]
    assert labels.dtype.kind == "i"
    dag = nx.DiGraph()
    dag.add_nodes_from([(1, 2), (0,)])
    assert partition_labels(dag.nodes, 3).tolist() == [1, 0, 0]


def test_partition_edge_metrics_scores_forward_pairs_in_node_order():
    """Truth 0→1, 1→2, 2→3; blocks (0,), (1, 2), (3,); the backward edge is a false positive."""
    truth = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    est = nx.DiGraph()
    est.add_nodes_from([(0,), (1, 2), (3,)])
    est.add_edges_from([((0,), (1, 2)), ((3,), (1, 2))])
    assert partition_edge_metrics(est, truth) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    empty = nx.DiGraph()
    empty.add_nodes_from([(0, 1, 2, 3)])
    assert partition_edge_metrics(empty, truth) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_block_parents_and_family_size():
    """Blocks {0, 1}, {2}, {3, 4, 5}; edges 0→1 (within), 1→3, 2→4, 0→5: block 2 has parents
    blocks 0 and 1. Family sizes in variables and with at most k per block."""
    truth = nx.DiGraph([(0, 1), (1, 3), (2, 4), (0, 5)])
    partition = [frozenset({0, 1}), frozenset({2}), frozenset({3, 4, 5})]
    parents = block_parents(truth, partition)
    assert parents == [set(), set(), {0, 1}]
    assert family_size(partition, parents, 0) == 2
    assert family_size(partition, parents, 2) == 6
    assert family_size(partition, parents, 2, k=1) == 3
    assert family_size(partition, parents, 2, k=3) == 6
    assert family_size(partition, parents, 2, k=2) == 5


def test_prf_empty_conventions():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(0, 2, 0) == (0.0, 1.0, 0.0)


def test_correct_matches_hand_computed_rejections():
    """N = 4, q = 0.05: Bonferroni cutoff 0.0125; BH thresholds k·0.0125; BY divides q by H_4."""
    P = np.array([[0.001, 0.02], [0.03, 0.5]])
    assert correct(P, "bonferroni", 0.05).tolist() == [[True, False], [False, False]]
    assert correct(P, "bh", 0.05).tolist() == [[True, True], [True, False]]
    assert correct(P, "by", 0.05).tolist() == [[True, False], [False, False]]
    # Step-up: the largest passing rank rejects every smaller p-value.
    assert correct(np.array([[0.04, 0.045]]), "bh", 0.05).all()
    with pytest.raises(ValueError):
        correct(P, "holm", 0.05)


def test_welch_pvalues_threshold_equals_compute_M():
    rng = np.random.default_rng(0)
    data = {"obs": sample_chain_dataset(300, rng),
            "1": sample_chain_dataset(300, rng, (0,)),
            "0": sample_chain_dataset(300, rng, (3,))}
    M, _ = compute_M(data, alpha=1e-3)
    assert np.array_equal(welch_pvalues(data) < 1e-3, M)
