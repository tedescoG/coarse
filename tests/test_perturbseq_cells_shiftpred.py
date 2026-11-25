"""Tests for held-out intervention prediction: seed blocks, DAG propagation, scoring."""

import importlib
import sys
from pathlib import Path

import numpy as np

SCRIPTS = str(Path(__file__).resolve().parents[1] / "src/expt/workflow/scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
shiftpred = importlib.import_module("perturbseq_cells_shiftpred")

BLOCKS = [["A"], ["B", "C"], ["D"], ["E", "F"]]
# Block DAG: {A} -> {B,C} -> {D}; {E,F} isolated.
EDGES = [[["A"], ["B", "C"]], [["B", "C"], ["D"]]]


def test_seed_blocks_find_target_blocks():
    assert shiftpred._seed_blocks(BLOCKS, {"A"}) == {0}
    assert shiftpred._seed_blocks(BLOCKS, {"C", "D"}) == {1, 2}
    assert shiftpred._seed_blocks(BLOCKS, {"Z"}) == set()


def test_descendant_genes_propagate_downstream():
    # A's block plus every descendant block; E/F and the isolated pair stay out.
    got = shiftpred._descendant_genes(
        EDGES, BLOCKS, shiftpred._seed_blocks(BLOCKS, {"A"})
    )
    assert got == {"A", "B", "C", "D"}
    # A downstream target seeds its own block only (no ancestors).
    got = shiftpred._descendant_genes(
        EDGES, BLOCKS, shiftpred._seed_blocks(BLOCKS, {"D"})
    )
    assert got == {"D"}
    # Edges whose blocks are listed in either order resolve via the sorted-name index.
    assert shiftpred._descendant_genes(
        [[["B", "C"], ["A"]]], BLOCKS, shiftpred._seed_blocks(BLOCKS, {"B"})
    ) == {"A", "B", "C"}


def test_prf_brackets_and_empty_sets():
    # Perfect prediction, no false positives.
    assert shiftpred._prf({"A", "B"}, {"A"}) == (0.5, 1.0, 2 / 3)
    # Predicting nothing: precision is undefined, not zero.
    p, r, f1 = shiftpred._prf(set(), {"A"})
    assert np.isnan(p) and r == 0.0 and np.isnan(f1)
    # Observing nothing: recall undefined, precision still meaningful.
    p, r, f1 = shiftpred._prf({"A"}, set())
    assert p == 0.0 and np.isnan(r) and np.isnan(f1)
