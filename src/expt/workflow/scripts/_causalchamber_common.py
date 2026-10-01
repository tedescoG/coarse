# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""Helpers shared by the Causal Chamber scripts."""
from __future__ import annotations

from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _common import PRF_KEYS, prf


def build_block_data_dict(blocks: dict, targets: dict[str, set[int]]) -> dict:
    """`{"obs": (X, set(), "obs"), env: (X, targets, "soft")}` for COARSE and RePaRe."""
    data: dict = {"obs": (blocks["obs"], set(), "obs")}
    for label, tgt in targets.items():
        data[label] = (blocks[label], set(int(t) for t in tgt), "soft")
    return data


def graph_edge_metrics(est_graph, true_graph):
    """Directed-edge P/R/F1."""
    tp = sum(1 for edge in est_graph.edges if true_graph.has_edge(*edge))
    return dict(zip(PRF_KEYS, prf(tp, len(est_graph.edges), len(true_graph.edges))))


def skeleton_metrics(est_graph, true_graph):
    """Undirected-edge P/R/F1."""
    est_edges = {frozenset(edge) for edge in est_graph.edges}
    true_edges = {frozenset(edge) for edge in true_graph.edges}
    tp = len(est_edges & true_edges)
    return dict(zip(PRF_KEYS, prf(tp, len(est_edges), len(true_edges))))


def atomic_graph(adj: np.ndarray) -> nx.DiGraph:
    """Edge i → j for every nonzero `adj[i, j]` (an undirected edge becomes both)."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(adj.shape[0]))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.nonzero(adj)))
    return graph


def singleton_dag(graph: nx.DiGraph) -> nx.DiGraph:
    """`graph` with `(i,)` tuple nodes."""
    return nx.relabel_nodes(graph, {node: (node,) for node in graph.nodes})


def gaussian_bic_score(data: np.ndarray, graph: nx.DiGraph) -> float:
    """Gaussian BIC of `graph` on `data`; lower is better."""
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


def select_targets(
    mode: str,
    group_targets: dict[str, set[int]],
    single_env_labels: list[str],
    name_to_idx: dict[str, int],
) -> dict[str, set[int]]:
    """`{env: targets}`: "grouped" pools rgb / pol, "ungrouped" has one env per variable."""
    if mode == "grouped":
        return {label: set(t) for label, t in group_targets.items()}
    if mode == "ungrouped":
        return {label: {name_to_idx[label]} for label in single_env_labels}
    raise ValueError(f"Unknown mode: {mode!r}")


def baseline_env_lists(
    blocks: dict,
    single_env_labels: list[str],
    name_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[list[int]], list[str]]:
    """`(data_list, target_lists, env_labels)` of the ungrouped regime, observational first."""
    targets = select_targets("ungrouped", {}, single_env_labels, name_to_idx)
    env_labels = ["obs", *targets]
    data_list = [blocks[label] for label in env_labels]
    target_lists = [[], *(sorted(int(t) for t in targets[l]) for l in targets)]
    return data_list, target_lists, env_labels


def labeled_summary(
    model_dag: nx.DiGraph, feature_cols: list[str]
) -> tuple[list[tuple[int, tuple[str, ...]]], list[tuple[int, int]]]:
    """JSON-friendly `([(idx, names), ...], [(u_idx, v_idx), ...])` in node order."""
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
    """Draw `model_dag` on `ax`, nodes labelled by column names."""
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
    """Inverse of `labeled_summary`: tuple-of-index nodes."""
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
    """Mask of the parts hit by `atom_indices` or their descendants."""
    closure = set(atom_indices)
    for idx in list(atom_indices):
        closure.update(nx.descendants(true_dag_full, idx))
    mask = np.zeros(len(parts), dtype=bool)
    for part_idx, atoms in enumerate(parts):
        if atoms & closure:
            mask[part_idx] = True
    return mask


def ground_truth_partition(target_dict, parts, true_dag_full: nx.DiGraph):
    """Ground-truth partition and labels for an intervention regime."""
    from repare.repare import _get_totally_ordered_partition

    ordered_masks = {}
    for order, label in enumerate(sorted(target_dict)):
        atom_union = set().union(*[parts[idx] for idx in target_dict[label]])
        ordered_masks[str(order)] = descendant_mask(atom_union, parts, true_dag_full)
    partition = _get_totally_ordered_partition(ordered_masks)
    labels = np.zeros(len(parts), dtype=int)
    for label, block in enumerate(partition):
        labels[list(block)] = label
    return partition, labels


def select_score_row(records: list[dict]) -> dict:
    """Min score; ties to the smaller α, then β."""
    return min(records, key=lambda r: (r["score"], r["alpha"], r["beta"]))


def select_oracle_row(records: list[dict]) -> dict:
    """Max ARI, then F1; ties to the smaller α, then β."""
    return max(
        records, key=lambda r: (r["ari"], r["f1"], -r["alpha"], -r.get("beta", 0.0))
    )


def select_utigsp_score_row(records: list[dict]) -> dict:
    """Min observational BIC; ties to the first cell in grid order."""
    return min(records, key=lambda r: (r["bic"], r["alpha_ci"], r["alpha_inv"]))


def select_utigsp_oracle_row(records: list[dict]) -> dict:
    """Max `f1 * precision`; ties to the first cell in grid order."""
    return max(
        records,
        key=lambda r: (r["f1"] * r["precision"], -r["alpha_ci"], -r["alpha_inv"]),
    )


PAYLOAD_KEYS = (
    "metric_type", "ari", "precision", "recall", "f1", "fit_time", "search_runtime_sec",
    "score", "alpha", "beta", "alpha_ci", "alpha_inv", "lambda", "num_parts", "num_edges",
)


def params_payload(row: dict, parts, edges, **extra) -> dict:
    """`PAYLOAD_KEYS` (missing → None), `extra`, then `parts` / `edges`."""
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


# Variable- and block-level metrics with negative controls. A graph is (arcs, undirected).
LEVELS = ("variable", "block")
VARIABLE_METRICS = ("adj_precision", "adj_recall", "adj_f1", "dir_precision", "dir_recall",
                    "dir_f1")


def split_edges(pairs):
    pairs = set(pairs)
    arcs = {e for e in pairs if (e[1], e[0]) not in pairs}
    return arcs, {frozenset(e) for e in pairs if (e[1], e[0]) in pairs}


def expand_blocks(block_edges):
    return {(u, v) for a, b in block_edges for u in a for v in b}, set()


def collapse(graph, block_of):
    arcs, und = graph
    pairs = {(block_of[u], block_of[v]) for u, v in arcs}
    for u, v in map(tuple, und):
        pairs |= {(block_of[u], block_of[v]), (block_of[v], block_of[u])}
    return split_edges((a, b) for a, b in pairs if a != b)


def icpdag(arcs, family, p):
    from causaldag import DAG

    dag = DAG(nodes=set(range(p)), arcs=set(arcs))
    out = dag.interventional_cpdag([set(t) for t in family], cpdag=dag.cpdag())
    return set(out.arcs), {frozenset(e) for e in out.edges}


def variable_metrics(est, true_arcs) -> dict:
    """Adjacency and directed (undirected = both ways) metrics."""
    arcs, und = est
    est_adj = {frozenset(e) for e in arcs} | und
    true_adj = {frozenset(e) for e in true_arcs}
    est_dir = arcs | {(u, v) for e in und for u, v in (tuple(e), tuple(e)[::-1])}
    tp_adj = len(est_adj & true_adj)
    out = dict(zip(VARIABLE_METRICS[:3], prf(tp_adj, len(est_adj), len(true_adj))))
    out.update(zip(VARIABLE_METRICS[3:],
                   prf(len(est_dir & true_arcs), len(est_dir), len(true_arcs))))
    out["m_est"], out["tp_adj"] = len(est_adj), tp_adj
    return out


def exact_nc(m_max, m_true, m_est, tp) -> dict:
    """Hypergeometric null of the adjacency metrics: (mean, 2.5%, 97.5%, P(TP ≥ tp))."""
    from scipy.stats import hypergeom

    rv = hypergeom(m_max, m_true, m_est)
    lo, hi, p = rv.ppf(0.025), rv.ppf(0.975), float(rv.sf(tp - 1))
    scale = {"adj_precision": m_est, "adj_recall": m_true, "adj_f1": (m_est + m_true) / 2}
    return {k: (m_est * m_true / m_max / s, lo / s, hi / s, p) for k, s in scale.items()}


def _forward(order):
    return [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]


def random_icpdag(rng, p, m_adj, family):
    fwd = _forward([int(v) for v in rng.permutation(p)])
    return icpdag((fwd[i] for i in rng.choice(len(fwd), m_adj, replace=False)), family, p)


def random_expanded(rng, blocks, k):
    fwd = _forward([blocks[i] for i in rng.permutation(len(blocks))])
    return expand_blocks(fwd[i] for i in rng.choice(len(fwd), k, replace=False))


def sim_nc(draw, n, rng, truths, block_of, observed) -> dict:
    """Simulated null per (level, metric): (mean, 2.5%, 97.5%, P(NC ≥ observed))."""
    vals = {(lv, k): [] for lv in LEVELS for k in VARIABLE_METRICS}
    for _ in range(n):
        graph = draw(rng)
        for lv, g in zip(LEVELS, (graph, collapse(graph, block_of))):
            m = variable_metrics(g, truths[lv])
            for k in VARIABLE_METRICS:
                vals[(lv, k)].append(m[k])
    return {
        (lv, k): (np.mean(v), *np.quantile(v, [0.025, 0.975]),
                  float(np.mean(np.asarray(v) >= observed[lv][k])))
        for (lv, k), v in vals.items()
    }
