"""Tests for the label and metric helpers in workflow/scripts/_common.py."""
from __future__ import annotations

import networkx as nx

from _common import partition_edge_metrics, partition_labels, prf


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


def test_prf_empty_conventions():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)
    assert prf(0, 2, 0) == (0.0, 1.0, 0.0)
