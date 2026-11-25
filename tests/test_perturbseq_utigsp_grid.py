"""Tests for the UT-IGSP tuning-grid check: target metrics and stored-DAG comparison."""

import importlib
import sys
from pathlib import Path

SCRIPTS = str(Path(__file__).resolve().parents[1] / "src/expt/workflow/scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
grid = importlib.import_module("perturbseq_utigsp_grid")

BLOCKS = [["A"], ["B"], ["C"], ["D"]]
# Block edges take the first gene of each block, exact for UT-IGSP's singletons.
EDGES = [["A"], ["B"]], [["B"], ["C"]], [["C"], ["D"]]


def gene_graph():
    return grid._gene_dag(BLOCKS, EDGES)


def test_gene_dag_preserves_direction():
    g = gene_graph()
    assert set(g.edges) == {("A", "B"), ("B", "C"), ("C", "D")}
    assert g.number_of_nodes() == 4


def test_target_metrics_propagate_descendants():
    g = gene_graph()
    # Seeding A asserts A's whole downstream cone.
    out_degree, asserted = grid._target_metrics(g, {"A"})
    assert out_degree == 1
    assert asserted == 4
    # Seeding the sink asserts only itself.
    out_degree, asserted = grid._target_metrics(g, {"D"})
    assert out_degree == 0
    assert asserted == 1
    # A gene outside the graph asserts nothing (targets missing from a panel).
    out_degree, asserted = grid._target_metrics(g, {"Z"})
    assert out_degree == 0 and asserted == 0


def test_stored_edges_match_ignores_block_order():
    rec = {"edges": [[["B", "C"], ["A"]]]}
    assert grid._stored_edges(rec) == frozenset({(("B", "C"), ("A",))})
