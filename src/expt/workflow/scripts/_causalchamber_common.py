"""Shared helpers for the CausalChamber experiment scripts.

Graphs are `nx.DiGraph`s whose nodes are sorted atom-index tuples (the `COARSE.dag`
convention). Atomic baselines (GIES / GnIES / UT-IGSP) are scored after projection onto the
true coarsening Π (`quotient_graph`), so every method is compared at the coarse level.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from coarse.partition import infer_partition

from _common import partition_labels


def build_block_data_dict(blocks: dict, targets: dict[str, set[int]]) -> dict:
    """Construct the (X, targets, type) input dict shared by COARSE and RePaRe.

    Both methods accept `{"obs": (X, set(), "obs"), env_key: (X, targets, "soft")}`.
    """
    data: dict = {"obs": (blocks["obs"], set(), "obs")}
    for label, tgt in targets.items():
        data[label] = (blocks[label], set(int(t) for t in tgt), "soft")
    return data



def _prf(tp: int, n_est: int, n_true: int) -> dict:
    precision = tp / n_est if n_est else 1.0
    recall = tp / n_true if n_true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}



def quotient_graph(atom_edges, blocks) -> nx.DiGraph:
    """Atomic edges collapsed onto `blocks` (tuple nodes): A → B iff some u → v has u ∈ A,
    v ∈ B, A ≠ B. Every ordered pair counts, so the quotient may be cyclic."""
    nodes = [tuple(sorted(int(u) for u in block)) for block in blocks]
    block_of = {u: node for node in nodes for u in node}
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for u, v in atom_edges:
        a, b = block_of[int(u)], block_of[int(v)]
        if a != b:
            graph.add_edge(a, b)
    return graph


def _edge_prf(est_edges, true_edges, prefix: str) -> dict:
    out = {}
    est, true = set(est_edges), set(true_edges)
    for kind, (e, t) in {
        "dir": (est, true),
        "skel": ({frozenset(x) for x in est}, {frozenset(x) for x in true}),
    }.items():
        out.update({f"{prefix}_{kind}_{k}": v for k, v in _prf(len(e & t), len(e), len(t)).items()})
    return out


def _atom_pairs(block_dag: nx.DiGraph) -> set:
    return {(u, v) for a, b in block_dag.edges for u in a for v in b}


def all_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph, true_blocks) -> dict:
    """Direction (`dir`) and skeleton (`skel`) P/R/F1 of a block graph.
    `blk_*`: against the atomic truth collapsed onto the estimate's blocks.
    `atom_*`: estimate and true coarse graph G/Π both expanded to atom pairs."""
    model_dag = quotient_graph(_atom_pairs(model_dag), model_dag.nodes)  # sorted-tuple nodes
    blk = quotient_graph(true_graph.edges, model_dag.nodes)
    true_coarse = quotient_graph(true_graph.edges, true_blocks)
    return {
        **_edge_prf(model_dag.edges, blk.edges, "blk"),
        **_edge_prf(_atom_pairs(model_dag), _atom_pairs(true_coarse), "atom"),
    }


def gaussian_bic_score(data: np.ndarray, graph: nx.DiGraph) -> float:
    """Gaussian BIC of `graph` (atomic integer nodes) on `data`; lower is
    better. Lifted from RePaRe's `causalchamber_utigsp.py` and used only for
    UT-IGSP's data-driven grid selection on the observational sample."""
    if data.size == 0 or graph.number_of_nodes() == 0:
        return float("inf")
    n_samples = data.shape[0]
    total_ll = 0.0
    total_params = 0
    eps = 1e-12
    for node in sorted(graph.nodes):
        parents = list(graph.predecessors(node))
        y = data[:, node]
        if parents:
            X_aug = np.column_stack([np.ones(n_samples), data[:, parents]])
            beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
            resid = y - X_aug @ beta
            params = len(parents) + 1
        else:
            resid = y - y.mean()
            params = 1
        sigma2 = max(float(np.mean(resid**2)), eps)
        total_ll += -0.5 * n_samples * (np.log(2 * np.pi * sigma2) + 1)
        total_params += params
    return float(-2 * total_ll + total_params * np.log(n_samples))


def subsample_rows(arr: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Return `arr` unchanged if it has at most `max_rows` rows, else a
    seeded uniform row subsample (RePaRe's `subsample_env`, used for GnIES)."""
    if arr.shape[0] <= max_rows:
        return arr
    rng = np.random.default_rng(seed)
    take = rng.choice(arr.shape[0], size=max_rows, replace=False)
    return arr[take]


def select_targets(
    mode: str,
    group_targets: dict[str, set[int]],
    single_env_labels: list[str],
    name_to_idx: dict[str, int],
) -> dict[str, set[int]]:
    """Intervention regime → `{env_label: target atom indices}`.

    "grouped": the pooled rgb / pol blocks with their multi-target sets.
    "ungrouped": one env per single-variable experiment, targeting itself.
    """
    if mode == "grouped":
        return {label: set(t) for label, t in group_targets.items()}
    if mode == "ungrouped":
        return {label: {name_to_idx[label]} for label in single_env_labels}
    raise ValueError(f"Unknown mode: {mode!r}")


def baseline_env_lists(
    mode: str,
    blocks: dict,
    group_targets: dict[str, set[int]],
    single_env_labels: list[str],
    name_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[list[int]], list[str]]:
    """List-form input for the atomic baselines (GIES / GnIES / UT-IGSP).

    Returns `(data_list, target_lists, env_labels)` with the observational
    sample first and an empty target list for it — the layout `gies.fit_bic`,
    `gnies.fit` and `ut_igsp.fit(obs_idx=0)` expect.
    """
    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    env_labels = ["obs", *targets]
    data_list = [blocks[label] for label in env_labels]
    target_lists = [[], *(sorted(int(t) for t in targets[l]) for l in targets)]
    return data_list, target_lists, env_labels


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
    labels = partition_labels(partition, len(parts))
    return partition, labels


def select_oracle_row(records: list[dict]) -> dict:
    """Ground-truth pick over the whole grid: max ARI, then atom direction F1, then its
    precision, then score."""
    return max(
        records,
        key=lambda r: (r["ari"], r["atom_dir_f1"], r["atom_dir_precision"], r["score"]),
    )


def select_utigsp_score_row(records: list[dict]) -> dict:
    """UT-IGSP data-driven pick: min observational Gaussian BIC. Ties go to
    the first cell in grid order (smaller `alpha_ci`, then `alpha_inv`) —
    RePaRe's `df["bic"].idxmin()`. Ground truth never enters."""
    return min(records, key=lambda r: (r["bic"], r["alpha_ci"], r["alpha_inv"]))


def select_utigsp_oracle_row(records: list[dict]) -> dict:
    """UT-IGSP ground-truth pick: max atom direction `f1 * precision` (RePaRe's rule), ties
    to the first cell in grid order."""
    return max(
        records,
        key=lambda r: (
            r["atom_dir_f1"] * r["atom_dir_precision"], -r["alpha_ci"], -r["alpha_inv"]
        ),
    )


METRIC_KEYS = tuple(
    f"{ref}_{kind}_{stat}"
    for ref in ("blk", "atom")
    for kind in ("dir", "skel")
    for stat in ("precision", "recall", "f1")
)
PAYLOAD_KEYS = (
    "alpha", "second_hp", "second_hp_name", "ari", *METRIC_KEYS,
    "score", "fit_time", "search_runtime_sec", "num_parts", "num_edges",
)


def params_payload(row: dict, parts, edges, **extra) -> dict:
    """JSON payload for `score_params.json` / `oracle_params.json`.

    Every method emits the same `PAYLOAD_KEYS` (missing ones are `None`) plus
    the `labeled_summary` `parts` / `edges` and any method-specific `extra`
    (e.g. `k`, `n_folds`, `estimated_targets`).
    """
    payload = {key: row.get(key) for key in PAYLOAD_KEYS}
    payload.update(extra)
    payload["parts"] = parts
    payload["edges"] = edges
    return payload


METHOD_LABEL = {
    "coarse": "COARSE",
    "repare": "RePaRe",
    "gies": "GIES",
    "gnies": "GnIES",
    "utigsp": "UT-IGSP",
}
