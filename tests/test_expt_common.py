"""Tests for the synthetic-harness helpers in workflow/scripts/_common.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
