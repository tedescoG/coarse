"""Tests for the synthetic-harness helpers in workflow/scripts/_common.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "expt" / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import partition_labels  # noqa: E402


def test_partition_labels_indexes_blocks_in_iteration_order():
    labels = partition_labels([frozenset({2}), (0, 3), {1}], 4)
    assert labels.tolist() == [1, 2, 0, 1]
    assert labels.dtype.kind == "i"


def test_partition_labels_accepts_dag_nodes():
    import networkx as nx

    dag = nx.DiGraph()
    dag.add_nodes_from([(1, 2), (0,)])
    assert partition_labels(dag.nodes, 3).tolist() == [1, 0, 0]


import networkx as nx  # noqa: E402

from _common import block_dag_prf, prf_from_counts, project_true_dag  # noqa: E402


def _truth_and_estimate():
    """Atom DAG 0→1, 0→2, 2→3; blocks (0,), (1, 2), (3,); estimate (0,)→(1,2), (0,)→(3,)."""
    true_dag = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    est = nx.DiGraph()
    est.add_nodes_from([(0,), (1, 2), (3,)])
    est.add_edges_from([((0,), (1, 2)), ((0,), (3,))])
    return true_dag, est


def test_project_true_dag_and_prf_match_hand_counts():
    true_dag, est = _truth_and_estimate()
    ref = project_true_dag(est.nodes, true_dag)
    assert set(ref.edges) == {((0,), (1, 2)), ((1, 2), (3,))}
    precision, recall, f = block_dag_prf(est, true_dag)
    assert (precision, recall, f) == (0.5, 0.5, 0.5)
    assert prf_from_counts(0, 0, 0) == (1.0, 1.0, 1.0)
    assert prf_from_counts(0, 2, 0) == (0.0, 1.0, 0.0)
