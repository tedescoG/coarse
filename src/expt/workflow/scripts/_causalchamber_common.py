"""Shared helpers for the CausalChamber experiment scripts.

Snakemake's `script:` directive adds the script's directory to sys.path before
execution, so sibling imports like `from _causalchamber_common import ...` work
without extra wiring. The leading underscore signals "module-local; do not
import from outside `src/expt/workflow/scripts/`."

All helpers operate on `nx.DiGraph` objects whose nodes are `tuple[int, ...]`
(sorted indices into the kept-feature list) — the `coarse.COARSE.dag`
convention (see `_materialize_dag` in `coarse/coarse.py`), which every method
in the experiment is coerced to before evaluation. Atomic-output baselines
(GIES / GnIES / UT-IGSP) are coerced through `adjacency_to_singleton_dag`, whose
singleton blocks make `partition_edge_metrics` collapse to RePaRe's plain
directed-edge metric and `skeleton_edge_metrics` to its skeleton metric.
"""
from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from coarse.partition import infer_partition

from _common import block_dag_prf, partition_labels


def build_block_data_dict(blocks: dict, targets: dict[str, set[int]]) -> dict:
    """Construct the (X, targets, type) input dict shared by COARSE and RePaRe.

    Both methods accept `{"obs": (X, set(), "obs"), env_key: (X, targets, "soft")}`.
    """
    data: dict = {"obs": (blocks["obs"], set(), "obs")}
    for label, tgt in targets.items():
        data[label] = (blocks[label], set(int(t) for t in tgt), "soft")
    return data


def partition_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph) -> dict:
    """Precision/recall/F1 of `model_dag`'s edges against the ground-truth partition graph:
    `true_graph`'s atomic edges collapsed onto `model_dag`'s blocks over forward node pairs.
    Delegates to `_common.block_dag_prf` so the synthetic and chamber halves share one
    definition of the metric.
    """
    precision, recall, f1 = block_dag_prf(model_dag, true_graph)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _prf(tp: int, n_est: int, n_true: int) -> dict:
    precision = tp / n_est if n_est else 1.0
    recall = tp / n_true if n_true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def directed_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph) -> dict:
    """Order-independent directed block-edge precision/recall/F1.

    Unlike `partition_edge_metrics`, which only inspects block pairs that go
    *forward* in node insertion order (valid for a topologically sorted
    partition DAG, but it silently drops every true edge running backward in
    that order), this checks every ordered pair `(pa, ch)`, `pa != ch`: a true
    block edge exists iff some atomic `u -> v`, `u ∈ pa`, `v ∈ ch`, is in
    `true_graph`. For singleton blocks this is exactly RePaRe's
    `graph_edge_metrics` (the GIES / UT-IGSP native metric): every atomic true
    edge is in the recall denominator and every estimated edge is matched
    regardless of index order.
    """
    nodes = list(model_dag.nodes)
    true_edges = {
        (pa, ch)
        for pa in nodes
        for ch in nodes
        if pa != ch and any(true_graph.has_edge(u, v) for u in pa for v in ch)
    }
    est_edges = set(model_dag.edges)
    return _prf(len(est_edges & true_edges), len(est_edges), len(true_edges))


def skeleton_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph) -> dict:
    """Orientation-free counterpart of `partition_edge_metrics`.

    Estimated and true edges are collapsed to unordered block pairs
    `frozenset({pa, ch})`; a true pair exists iff some atomic edge crosses the
    two blocks in either direction. For singleton blocks this is exactly
    RePaRe's `skeleton_metrics` (GnIES' native metric).
    """
    nodes = list(model_dag.nodes)
    true_pairs = set()
    for i, a in enumerate(nodes[:-1]):
        for b in nodes[i + 1 :]:
            if any(
                true_graph.has_edge(u, v) or true_graph.has_edge(v, u)
                for u in a
                for v in b
            ):
                true_pairs.add(frozenset((a, b)))
    est_pairs = {frozenset(edge) for edge in model_dag.edges}
    return _prf(len(est_pairs & true_pairs), len(est_pairs), len(true_pairs))


NATIVE_METRIC_TYPES = ("partition", "edges", "skeleton")


def all_edge_metrics(model_dag: nx.DiGraph, true_graph: nx.DiGraph, native: str) -> dict:
    """Directed + skeleton block-collapsed metrics for one estimate.

    `native` is RePaRe's `metric_type` label for the method and selects which
    of RePaRe's metric functions fills the plain `precision` / `recall` / `f1`
    keys, so the summary reproduces RePaRe's Table 1 numbers:
        "partition" (COARSE / RePaRe) → `partition_edge_metrics` (forward pairs only)
        "edges"     (GIES / UT-IGSP)  → `directed_edge_metrics` (== `graph_edge_metrics`)
        "skeleton"  (GnIES)           → `skeleton_edge_metrics` (== `skeleton_metrics`)
    `dir_*` / `skel_*` are the order-independent directed and skeleton
    collapses for every method — the cross-method comparison columns. For
    partition methods `dir_*` can be below the native triple when the
    estimated block order runs against a true edge (forward-only never
    counts that edge as missed; the all-pairs collapse does).
    """
    if native not in NATIVE_METRIC_TYPES:
        raise ValueError(f"native must be one of {NATIVE_METRIC_TYPES}, got {native!r}")
    directed = directed_edge_metrics(model_dag, true_graph)
    skeleton = skeleton_edge_metrics(model_dag, true_graph)
    native_triple = {
        "partition": lambda: partition_edge_metrics(model_dag, true_graph),
        "edges": lambda: directed,
        "skeleton": lambda: skeleton,
    }[native]()
    out = {"metric_type": native}
    out.update({f"dir_{k}": v for k, v in directed.items()})
    out.update({f"skel_{k}": v for k, v in skeleton.items()})
    out.update(native_triple)
    return out


def adjacency_to_singleton_dag(adj: np.ndarray) -> nx.DiGraph:
    """Coerce a p×p adjacency matrix to the block-DAG convention with
    singleton blocks `(i,)`.

    Every nonzero `adj[i, j]` becomes the edge `(i,) -> (j,)`. GIES / GnIES
    return I-essential graphs in which an undirected edge is stored as both
    `adj[i, j]` and `adj[j, i]`; both directed edges are kept, matching how
    RePaRe's wrappers score them (so one of the two is a false positive under
    the directed metric — use `skeleton_edge_metrics` for the orientation-free
    view).
    """
    adj = np.asarray(adj)
    dag = nx.DiGraph()
    dag.add_nodes_from((i,) for i in range(adj.shape[0]))
    for i, j in zip(*np.nonzero(adj)):
        dag.add_edge((int(i),), (int(j),))
    return dag


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
    """Ground-truth pick over the whole grid: max ARI, then F1, then
    precision, then score (upper bound on what any selection could reach)."""
    return max(
        records, key=lambda r: (r["ari"], r["f1"], r["precision"], r["score"])
    )


def select_utigsp_score_row(records: list[dict]) -> dict:
    """UT-IGSP data-driven pick: min observational Gaussian BIC. Ties go to
    the first cell in grid order (smaller `alpha_ci`, then `alpha_inv`) —
    RePaRe's `df["bic"].idxmin()`. Ground truth never enters."""
    return min(records, key=lambda r: (r["bic"], r["alpha_ci"], r["alpha_inv"]))


def select_utigsp_oracle_row(records: list[dict]) -> dict:
    """UT-IGSP ground-truth pick: max `f1 * precision`, ties to the first cell
    in grid order — RePaRe's `(df["f1"] * df["precision"]).idxmax()`."""
    return max(
        records,
        key=lambda r: (r["f1"] * r["precision"], -r["alpha_ci"], -r["alpha_inv"]),
    )


PAYLOAD_KEYS = (
    "alpha", "second_hp", "second_hp_name", "metric_type", "ari",
    "precision", "recall", "f1",
    "dir_precision", "dir_recall", "dir_f1",
    "skel_precision", "skel_recall", "skel_f1",
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
