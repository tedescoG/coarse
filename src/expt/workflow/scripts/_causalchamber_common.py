"""Shared helpers for the CausalChamber experiment scripts.

Snakemake's `script:` directive adds the script's directory to sys.path before
execution, so sibling imports like `from _causalchamber_common import ...` work
without extra wiring. The leading underscore signals "module-local; do not
import from outside `src/expt/workflow/scripts/`."

All helpers operate on `nx.DiGraph` objects whose nodes are `tuple[int, ...]`
(sorted indices into the kept-feature list) — the convention shared by
`coarse.COARSE.dag` (see `coarse/coarse.py:248`) and
`repare.repare.PartitionDagModelIvn.dag`.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from coarse.partition import infer_partition


def build_data_dict(blocks: dict, targets: dict[str, set[int]]) -> dict:
    """Construct the (X, targets, type) input dict shared by COARSE and RePaRe.

    Both methods accept `{"obs": (X, set(), "obs"), env_key: (X, targets, "soft")}`.
    """
    data: dict = {"obs": (blocks["obs"], set(), "obs")}
    for label, tgt in targets.items():
        data[label] = (blocks[label], set(int(t) for t in tgt), "soft")
    return data


def partition_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph) -> dict:
    """Precision/recall/F1 of `model_dag`'s edges against the ground-truth
    partition graph, computed by collapsing `true_graph`'s atomic edges to the
    coarsened nodes of `model_dag`. Verbatim port of
    `repare-0.2.0/src/expt/workflow/scripts/causalchamber_repare.py:21-33`.
    """
    true_edge_partition = nx.create_empty_copy(model_dag)
    node_list = list(true_edge_partition.nodes)
    for i, pa in enumerate(node_list[:-1]):
        for ch in node_list[i + 1 :]:
            if any(true_graph.has_edge(u, v) for u in pa for v in ch):
                true_edge_partition.add_edge(pa, ch)
    tp = sum(1 for edge in model_dag.edges if edge in true_edge_partition.edges)
    precision = tp / len(model_dag.edges) if model_dag.edges else 1.0
    recall = tp / len(true_edge_partition.edges) if true_edge_partition.edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def labeled_summary(
    model_dag: nx.DiGraph, feature_cols: list[str]
) -> tuple[list[tuple[int, tuple[str, ...]]], list[tuple[int, int]]]:
    """Extract a JSON-friendly partition + edge summary.

    Returns ([(idx, (col_name, ...)), ...], [(u_idx, v_idx), ...]) — node
    iteration order is preserved (networkx DiGraph stores insertion order), so
    `idx` lines up across calls on the same DAG.
    """
    node_to_idx = {node: idx for idx, node in enumerate(model_dag.nodes)}
    parts = [
        (idx, tuple(feature_cols[i] for i in part))
        for part, idx in node_to_idx.items()
    ]
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in model_dag.edges]
    return parts, edges


def draw_dag(
    model_dag: nx.DiGraph,
    feature_cols: list[str],
    ax,
    title: str | None = None,
) -> None:
    """Draw `model_dag` onto a matplotlib `ax` with column-name tuples as node
    labels. Shared by `save_dag_plot` (single-panel PNG) and the aggregator's
    3-panel figure.
    """
    labeled = nx.relabel_nodes(
        model_dag,
        {node: tuple(feature_cols[idx] for idx in node) for node in model_dag.nodes},
        copy=True,
    )
    pos = nx.spring_layout(labeled, seed=0)
    nx.draw_networkx(
        labeled,
        pos=pos,
        ax=ax,
        node_color="#8fbcd4",
        edgecolors="#1f4b73",
        linewidths=1.0,
        font_size=8,
    )
    if title is not None:
        ax.set_title(title)


def save_dag_plot(
    model_dag: nx.DiGraph,
    feature_cols: list[str],
    path,
    title: str | None = None,
) -> None:
    """Render `model_dag` to a single-panel PNG."""
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_dag(model_dag, feature_cols, ax, title=title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def dag_from_labeled_parts(
    parts: Iterable[tuple[int, Iterable[str]]],
    edges: Iterable[tuple[int, int]],
    name_to_idx: dict[str, int],
) -> nx.DiGraph:
    """Reconstruct an `nx.DiGraph` (atomic-index nodes) from the JSON-friendly
    summary produced by `labeled_summary`. Used by the aggregator to redraw
    DAGs without unpickling models.
    """
    idx_to_node: dict[int, tuple[int, ...]] = {}
    dag = nx.DiGraph()
    for idx, labels in parts:
        node = tuple(sorted(name_to_idx[label] for label in labels))
        idx_to_node[idx] = node
        dag.add_node(node)
    for u_idx, v_idx in edges:
        dag.add_edge(idx_to_node[u_idx], idx_to_node[v_idx])
    return dag


def descendant_mask(atom_indices, parts, true_dag_full: nx.DiGraph) -> np.ndarray:
    """Boolean mask over `parts` (atomic partition) marking every part whose
    atom set intersects the descendant closure of `atom_indices` in
    `true_dag_full`. Used to build ground-truth intervention masks.
    """
    closure = set(atom_indices)
    for idx in list(atom_indices):
        closure.update(nx.descendants(true_dag_full, idx))
    mask = np.zeros(len(parts), dtype=bool)
    for part_idx, atoms in enumerate(parts):
        if atoms & closure:
            mask[part_idx] = True
    return mask


def ground_truth_partition(target_dict, parts, true_dag_full: nx.DiGraph):
    """Compute the ground-truth partition + atomic label vector for a given
    intervention regime.

    Builds the M_true mask matrix (num_parts × num_envs) by descendant-closure,
    then feeds it to `coarse.partition.infer_partition` — same pattern as
    `scripts/_common.build_oracle_partition` already uses.
    """
    masks = []
    for label in sorted(target_dict):
        atom_union = set().union(*[parts[idx] for idx in target_dict[label]])
        masks.append(descendant_mask(atom_union, parts, true_dag_full))
    if not masks:
        return [], np.zeros(len(parts), dtype=int)
    M_true = np.column_stack(masks).astype(bool)
    partition = infer_partition(M_true)
    labels = np.zeros(len(parts), dtype=int)
    for label_idx, block in enumerate(partition):
        labels[list(block)] = label_idx
    return partition, labels


def partition_labels_from_dag(model_dag: nx.DiGraph, num_atoms: int) -> np.ndarray:
    """Project a partition DAG onto a length-`num_atoms` label vector.

    Each atom (column index in the kept-feature list) gets the integer index of
    its enclosing partition block, matching the convention used to compute ARI
    against the ground-truth labels.
    """
    labels = np.zeros(num_atoms, dtype=int)
    for block_idx, block in enumerate(model_dag.nodes):
        labels[list(block)] = block_idx
    return labels
