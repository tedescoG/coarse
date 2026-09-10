"""Coarse-level SHD via the meet (intersection) of the true and estimated partitions.

Produces a multi-page PDF: one intro page (definition + normalisation) and one page per
example, each showing (1) the true DAG with the true partition, (2) the estimated block
DAG with the estimated partition, (3) the meet cells with the two projected graphs
overlaid and every cell pair classified, (4) the score breakdown.

Green = agreement, red = disagreement. Line style carries the same information so it
does not rely on colour alone: solid = present in the estimate, dashed = only in truth.

Usage:  uv run python src/expt/diagnostics/shd_examples.py [out.pdf]
"""

from __future__ import annotations

import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------------------
# SHD on the meet of two partitions
# --------------------------------------------------------------------------------------

Partition = list[frozenset[int]]


def meet(p: Partition, q: Partition) -> Partition:
    """All non-empty intersections A ∩ B, A ∈ p, B ∈ q — the coarsest common refinement."""
    cells = [a & b for a in p for b in q if a & b]
    return sorted(cells, key=lambda c: min(c))


def project_true(true_dag: nx.DiGraph, cells: Partition) -> set[tuple[int, int]]:
    """Cell X → cell Y iff some atomic edge u → v has u ∈ X, v ∈ Y."""
    out = set()
    for i, x in enumerate(cells):
        for j, y in enumerate(cells):
            if i != j and any(true_dag.has_edge(u, v) for u in x for v in y):
                out.add((i, j))
    return out


def project_est(est_dag: nx.DiGraph, cells: Partition) -> set[tuple[int, int]]:
    """Cell X → cell Y iff blk(X) → blk(Y) in the estimated block DAG.

    `est_dag` nodes are tuple(sorted(block)) — the convention used by `_materialize_dag`.
    Cells inside the same estimated block get no edge: the model treats them as one unit.
    """
    blk_of = {}
    for node in est_dag.nodes:
        for atom in node:
            blk_of[atom] = node
    out = set()
    for i, x in enumerate(cells):
        for j, y in enumerate(cells):
            if i == j:
                continue
            bx, by = blk_of[min(x)], blk_of[min(y)]
            if bx != by and est_dag.has_edge(bx, by):
                out.add((i, j))
    return out


def coarse_shd(true_dag: nx.DiGraph, est_dag: nx.DiGraph, true_partition: Partition) -> dict:
    est_partition = [frozenset(n) for n in est_dag.nodes]
    cells = meet(est_partition, true_partition)
    t_edges = project_true(true_dag, cells)
    e_edges = project_est(est_dag, cells)

    pairs = {}  # (i, j) with i < j -> status
    for i, j in combinations(range(len(cells)), 2):
        t = {d for d in ((i, j), (j, i)) if d in t_edges}
        e = {d for d in ((i, j), (j, i)) if d in e_edges}
        if t == e:
            status = "correct" if t else "absent"
        elif not e:
            status = "missing"
        elif not t:
            status = "spurious"
        else:
            status = "reversed"
        pairs[(i, j)] = status

    shd = sum(s in ("missing", "spurious", "reversed") for s in pairs.values())
    n_pairs = len(pairs)
    return {
        "cells": cells,
        "true_edges": t_edges,
        "est_edges": e_edges,
        "pairs": pairs,
        "shd": shd,
        "n_pairs": n_pairs,
        "shd_norm": shd / n_pairs if n_pairs else 0.0,
        "counts": {k: sum(s == k for s in pairs.values()) for k in ("correct", "missing", "spurious", "reversed")},
    }


# --------------------------------------------------------------------------------------
# Helpers for building examples
# --------------------------------------------------------------------------------------


def dag_from_edges(edges) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def true_partition_from_targets(true_dag: nx.DiGraph, targets, nodes) -> Partition:
    """Row-classes of M_true (same rule as build_oracle_partition / infer_partition)."""
    rows = {}
    for v in nodes:
        row = tuple(v == t or v in nx.descendants(true_dag, t) for t in targets)
        rows.setdefault(row, set()).add(v)
    return sorted((frozenset(s) for s in rows.values()), key=lambda b: (sum(1 for _ in b), min(b)))


def block_dag(blocks, edges) -> nx.DiGraph:
    """Estimated DAG in the `_materialize_dag` convention (tuple-sorted block nodes)."""
    g = nx.DiGraph()
    nodes = [tuple(sorted(b)) for b in blocks]
    g.add_nodes_from(nodes)
    lookup = {frozenset(n): n for n in nodes}
    for a, b in edges:
        g.add_edge(lookup[frozenset(a)], lookup[frozenset(b)])
    return g


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------

GREEN = "#2e8b57"
RED = "#c0392b"
INK = "#222222"
MUTED = "#8a8a8a"
BLOCK_FILL = "#dfe9f3"
BLOCK_EDGE = "#4a6fa5"


def atom_positions(true_dag: nx.DiGraph, order_hint=None) -> dict:
    g = true_dag.copy()
    for layer, gen in enumerate(nx.topological_generations(g)):
        for v in gen:
            g.nodes[v]["layer"] = layer
    pos = nx.multipartite_layout(g, subset_key="layer", align="vertical")
    return {v: np.asarray(p) for v, p in pos.items()}


def group_positions(groups, atom_pos) -> dict:
    return {i: np.mean([atom_pos[a] for a in grp], axis=0) for i, grp in enumerate(groups)}


def label(cell) -> str:
    return "{" + ",".join(str(a) for a in sorted(cell)) + "}"


def _frame(ax, atom_pos, title):
    xs = np.array([p[0] for p in atom_pos.values()])
    ax.set_xlim(xs.min() - 0.45, xs.max() + 0.45)
    ax.set_ylim(-1.4, 1.4)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_axis_off()


def draw_boxes(ax, groups, atom_pos, pad=0.16):
    patches = []
    for grp in groups:
        pts = np.array([atom_pos[a] for a in grp])
        lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
        patch = FancyBboxPatch(lo, *(hi - lo), boxstyle="round,pad=0.03", fc=BLOCK_FILL, ec=BLOCK_EDGE, lw=1.0, zorder=0)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def arrow(ax, p, q, color, style="solid", rad=0.0, width=2.0, shrink=0, patchA=None, patchB=None):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=16, color=color, linestyle=style, linewidth=width,
        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink, patchA=patchA, patchB=patchB, zorder=1.5,
    ))


NODE_R_PT = 12  # circle radius in points for atoms / cells


def draw_nodes(ax, pos, labels, size_pt=NODE_R_PT, color=INK, font=10):
    for k, p in pos.items():
        ax.scatter(*p, s=(2 * size_pt) ** 2, facecolor="white", edgecolor=color, lw=1.2, zorder=2)
        ax.text(*p, labels[k], ha="center", va="center", fontsize=font, color=color, zorder=3)


def draw_atom_graph(ax, dag, atom_pos, partition, title):
    draw_boxes(ax, partition, atom_pos)
    draw_nodes(ax, atom_pos, {a: str(a) for a in atom_pos})
    for u, v in dag.edges:
        arrow(ax, atom_pos[u], atom_pos[v], INK, width=1.6, shrink=NODE_R_PT + 1)
    _frame(ax, atom_pos, title)


def draw_block_graph(ax, est_dag, atom_pos, title):
    blocks = [frozenset(n) for n in est_dag.nodes]
    boxes = draw_boxes(ax, blocks, atom_pos)
    draw_nodes(ax, atom_pos, {a: str(a) for a in atom_pos}, color=MUTED)
    gp = group_positions(blocks, atom_pos)
    idx = {n: i for i, n in enumerate(est_dag.nodes)}
    for u, v in est_dag.edges:
        arrow(ax, gp[idx[u]], gp[idx[v]], BLOCK_EDGE, width=2.2, rad=0.15, patchA=boxes[idx[u]], patchB=boxes[idx[v]])
    _frame(ax, atom_pos, title)


def draw_meet(ax, res, atom_pos, title):
    cells = res["cells"]
    gp = group_positions(cells, atom_pos)
    r = NODE_R_PT + 4
    draw_nodes(ax, gp, {i: label(c) for i, c in enumerate(cells)}, size_pt=r, font=8)
    t, e = res["true_edges"], res["est_edges"]
    for d in e & t:
        arrow(ax, gp[d[0]], gp[d[1]], GREEN, width=2.4, shrink=r + 1)
    for d in t - e:
        arrow(ax, gp[d[0]], gp[d[1]], RED, style="dashed", rad=0.25, shrink=r + 1)
    for d in e - t:
        arrow(ax, gp[d[0]], gp[d[1]], RED, rad=0.25, shrink=r + 1)
    _frame(ax, atom_pos, title)


def draw_score(ax, res, notes):
    ax.set_axis_off()
    c = res["counts"]
    lines = [
        f"cells (meet): {len(res['cells'])}   pairs: {res['n_pairs']}",
        "",
        f"correct edges   {c['correct']}",
        f"missing         {c['missing']}   (truth only, dashed red)",
        f"spurious        {c['spurious']}   (estimate only, solid red)",
        f"reversed        {c['reversed']}",
        "",
        f"SHD = {res['shd']}",
        f"SHD_norm = {res['shd']} / {res['n_pairs']} = {res['shd_norm']:.3f}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9, transform=ax.transAxes)
    wrapped = "\n".join(textwrap.wrap(" ".join(notes.split()), 62))
    ax.text(0.0, 0.34, wrapped, va="top", ha="left", fontsize=8.5, transform=ax.transAxes, color=INK)


def example_page(pdf, name, true_dag, true_partition, est_dag, notes, targets=()):
    res = coarse_shd(true_dag, est_dag, true_partition)
    atom_pos = atom_positions(true_dag)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.3, 1.05]})
    fig.suptitle(name, fontsize=12, x=0.01, ha="left")
    draw_atom_graph(axes[0], true_dag, atom_pos, true_partition, f"1. truth + true partition  (interventions on {sorted(targets)})")
    draw_block_graph(axes[1], est_dag, atom_pos, "2. estimate: block DAG + est. partition")
    draw_meet(axes[2], res, atom_pos, "3. meet cells: truth vs estimate")
    draw_score(axes[3], res, notes)
    fig.text(
        0.01, 0.01,
        "green solid = edge in both   |   red dashed = missing (truth only)   |   red solid = spurious (estimate only)   |   "
        "reversed = both a red dashed and a red solid arrow on the same pair",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)
    return res


def intro_page(pdf):
    fig = plt.figure(figsize=(15, 4.2))
    fig.text(
        0.03, 0.95,
        "Coarse-level structural Hamming distance (SHD) on the meet of two partitions",
        fontsize=13, va="top",
    )
    text = (
        "Standard SHD (Tsamardinos et al. 2006) counts edge additions, deletions and reversals needed to turn the\n"
        "estimated graph into the true one. It requires both graphs to share a node set. At the coarse level they do not:\n"
        "the truth lives on atoms / true blocks, the estimate on estimated blocks.\n\n"
        "Fix: build the MEET of the two partitions, i.e. every non-empty intersection (estimated block) ∩ (true block).\n"
        "Each atom lands in exactly one cell. Both graphs are projected onto the cells:\n"
        "    truth:     X → Y  iff some atomic edge u → v has u ∈ X, v ∈ Y\n"
        "    estimate:  X → Y  iff blk(X) → blk(Y) in the estimated block DAG; no edge when X, Y share an estimated block\n"
        "Then for every unordered pair of cells {X, Y} compare what the two graphs say (nothing / X→Y / Y→X / both).\n"
        "A pair costs 1 if they disagree (missing, spurious or reversed edge), 0 otherwise.  SHD = number of disagreeing pairs.\n\n"
        "Normalisation:  SHD_norm = SHD / C(n_cells, 2), the fraction of cell pairs on which the two graphs disagree.\n"
        "It lies in [0, 1]; 0 means the projected graphs are identical. It makes runs with different numbers of cells\n"
        "comparable, at the price that with very few cells one wrong pair is a large fraction (see the 2-cell example).\n\n"
        "Limits worth knowing:  all-singleton estimate  ⇒  ordinary SHD on the true DAG.   Correct partition  ⇒  FP + FN on block edges."
    )
    fig.text(0.03, 0.85, text, fontsize=9.5, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------


def main(out="shd_examples.pdf"):
    results = []
    with PdfPages(out) as pdf:
        intro_page(pdf)

        # Small truth: chain 1→2→3→4 ; interventions on 1, 2, 3 → true blocks {1},{2},{3,4}
        # (3 and 4 share a row of M: both are descendants of 3 and 4 itself is never targeted)
        T = dag_from_edges([(1, 2), (2, 3), (3, 4)])
        tT = [1, 2, 3]
        P = true_partition_from_targets(T, targets=tT, nodes=[1, 2, 3, 4])

        results.append(example_page(
            pdf, "Ex 1 — correct partition, one edge missing", T, P,
            block_dag([{1}, {2}, {3, 4}], [({1}, {2})]),
            "Partition is exact, so the cells are the true blocks. The missing {2}→{3,4} edge is the only disagreement.\n"
            "This is what F1 already sees (recall = 1/2).", targets=tT,
        ))

        results.append(example_page(
            pdf, "Ex 2 — partition too coarse (2 merged with 3,4)", T, P,
            block_dag([{1}, {2, 3, 4}], [({1}, {2, 3, 4})]),
            "The merge splits back into cells {2} and {3,4}. The block edge {1}→{2,3,4} expands to 1→{2} (right) and\n"
            "1→{3,4} (spurious); the edge {2}→{3,4} is missing because both cells sit in one estimated block.\n"
            "The existing F1 would report 1.0 here.", targets=tT,
        ))

        results.append(example_page(
            pdf, "Ex 3 — partition too fine (3,4 split), edges right", T, P,
            block_dag([{1}, {2}, {3}, {4}], [({1}, {2}), ({2}, {3}), ({3}, {4})]),
            "Over-splitting a block costs nothing as long as the edges are right: the estimate is a valid refinement of the\n"
            "truth. ARI still penalises it, which is why SHD should sit next to ARI, not replace it.", targets=tT,
        ))

        results.append(example_page(
            pdf, "Ex 4 — atomic baseline (all singletons): one reversed, one spurious", T, P,
            block_dag([{1}, {2}, {3}, {4}], [({2}, {1}), ({2}, {3}), ({3}, {4}), ({1}, {4})]),
            "An atomic method has a singleton partition, so the meet is just the atoms and this is textbook SHD on the true\n"
            "DAG: 1↔2 reversed (1), 1→4 spurious (1). Coarse and atomic methods are scored by the same yardstick.", targets=tT,
        ))

        # Chain with two big blocks: 1→2→3→4→5, interventions on 1 and 3 → {1,2}, {3,4,5}
        C = dag_from_edges([(1, 2), (2, 3), (3, 4), (4, 5)])
        tC = [1, 3]
        PC = true_partition_from_targets(C, targets=tC, nodes=[1, 2, 3, 4, 5])

        results.append(example_page(
            pdf, "Ex 5a — two true blocks, the single edge is missing", C, PC,
            block_dag([{1, 2}, {3, 4, 5}], []),
            "Only 2 cells → 1 pair. Missing the only edge gives SHD = 1 but SHD_norm = 1.0: with few cells the normalised\n"
            "score is very coarse-grained. Report both numbers.", targets=tC,
        ))

        results.append(example_page(
            pdf, "Ex 5b — everything merged into one block", C, PC,
            block_dag([{1, 2, 3, 4, 5}], []),
            "Same SHD as 5a: merging the two blocks also loses the edge (cells in one estimated block get no edge).\n"
            "SHD cannot tell 'merged' from 'edge missing'; ARI can (ARI = 0 here vs 1 in 5a).", targets=tC,
        ))

        # Diamond: 1→2, 1→3, 2→4, 3→4, 4→5, 5→6 ; interventions on 1..5 → {1},{2},{3},{4},{5,6}
        D = dag_from_edges([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6)])
        tD = [1, 2, 3, 4, 5]
        PD = true_partition_from_targets(D, targets=tD, nodes=[1, 2, 3, 4, 5, 6])

        results.append(example_page(
            pdf, "Ex 6a — non-adjacent siblings 2,3 merged: free under SHD", D, PD,
            block_dag([{1}, {2, 3}, {4}, {5, 6}], [({1}, {2, 3}), ({2, 3}, {4}), ({4}, {5, 6})]),
            "2 and 3 are not adjacent in truth, so merging them and drawing block edges reproduces exactly the true\n"
            "projected edges: SHD = 0 even though the partition is wrong. Only ARI notices.", targets=tD,
        ))

        results.append(example_page(
            pdf, "Ex 6b — same merge, plus edge {2,3}→{4} missing and {1}→{5,6} spurious", D, PD,
            block_dag([{1}, {2, 3}, {4}, {5, 6}], [({1}, {2, 3}), ({1}, {5, 6}), ({4}, {5, 6})]),
            "One missing block edge into a merged block costs TWO cell pairs (2→4 and 3→4), because the block claimed two\n"
            "cells at once. A missing edge is charged per meet-cell pair it should have covered, not per block edge.", targets=tD,
        ))

        results.append(example_page(
            pdf, "Ex 6c — mixed: 2,3 merged AND 5,6 split, one edge reversed", D, PD,
            block_dag([{1}, {2, 3}, {4}, {5}, {6}], [({1}, {2, 3}), ({2, 3}, {4}), ({5}, {4}), ({5}, {6})]),
            "Cells are {1},{2},{3},{4},{5},{6}. The split of {5,6} is free (edges still right); the reversed 4↔5 edge\n"
            "costs 1 — the only disagreement.", targets=tD,
        ))

    return results


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "shd_examples.pdf"
    res = main(out)
    for r in res:
        print(f"SHD={r['shd']:>2}  pairs={r['n_pairs']:>2}  norm={r['shd_norm']:.3f}  counts={r['counts']}")
    print("wrote", out)
