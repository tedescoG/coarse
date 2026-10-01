"""Helpers shared by the Snakemake scripts (importable, unlike `script:` files)."""
from __future__ import annotations

import re

import networkx as nx
import numpy as np

from coarse.partition import infer_partition


def normalize_intervention_type(value: str) -> str:
    """COARSE supports only soft (shift) interventions; anything else raises."""
    lowered = str(value).lower()
    if lowered == "soft":
        return "soft"
    raise ValueError(f"COARSE supports only soft interventions; got {value!r}")


def parse_targets_per_interv(token):
    """`targets_per_interv` wildcard → sempler's target `size`: None → 1, "k" → k,
    "AtoB" → (A, B)."""
    if token is None:
        return 1
    s = str(token)
    if "to" in s:
        lo, hi = s.split("to", 1)
        return (int(lo), int(hi))
    return int(s)


def build_data_dict(data, targets, intervention_type: str) -> dict:
    """`{env: (data, targets, type)}` from a generate.py archive."""
    data_dict = {"obs": (data["obs"], set(), "obs")}
    for idx, target in enumerate(targets):
        tgt = set(np.atleast_1d(target).astype(int))
        data_dict[str(idx)] = (data[str(idx)], tgt, intervention_type)
    return data_dict


def build_oracle_partition(weights: np.ndarray, targets, num_nodes: int):
    """(true_dag, M_true, env_order, partition) from the true weights; column e of M_true
    marks the targets of env e and their descendants."""
    true_dag = nx.DiGraph(weights.astype(bool))
    masks = []
    for target in targets:
        target_nodes = [int(n) for n in np.atleast_1d(target)]
        affected: set[int] = set(target_nodes)
        for node in target_nodes:
            affected.update(nx.descendants(true_dag, node))
        mask = np.zeros(num_nodes, dtype=bool)
        mask[list(affected)] = True
        masks.append(mask)
    M_true = np.column_stack(masks).astype(bool)
    env_order = [str(idx) for idx in range(len(targets))]
    partition = infer_partition(M_true)
    return true_dag, M_true, env_order, partition


def partition_labels(blocks, num_nodes: int) -> np.ndarray:
    """Block index of every node, in `blocks` iteration order (for ARI)."""
    labels = np.zeros(num_nodes, dtype=int)
    for label, block in enumerate(blocks):
        labels[list(block)] = label
    return labels


PRF_KEYS = ("precision", "recall", "f1")


def prf(tp, n_est, n_true):
    """Precision, recall, F1; P = 1 if nothing estimated, R = 1 if nothing true."""
    precision = tp / n_est if n_est else 1.0
    recall = tp / n_true if n_true else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def partition_edge_metrics(model_dag, true_graph) -> dict:
    """Block-DAG P/R/F1. A forward pair (in node order) is a true edge if any of its
    variable pairs is an edge of `true_graph`."""
    nodes = list(model_dag.nodes)
    reference = {
        (pa, ch)
        for i, pa in enumerate(nodes)
        for ch in nodes[i + 1 :]
        if any(true_graph.has_edge(u, v) for u in pa for v in ch)
    }
    tp = sum(1 for edge in model_dag.edges if edge in reference)
    return dict(zip(PRF_KEYS, prf(tp, len(model_dag.edges), len(reference))))


# iSCM (Ormaniec et al., 2024): every observational variable has unit variance. Interventions
# reuse the observational standardization; interventional envs are not re-standardized.
NOISE_FAMILIES = ("gaussian", "uniform", "laplace", "t3", "t4")


def iscm_standardize(W: np.ndarray, noise_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale `(W, noise_var)` to unit population variance: in causal order, node j's
    incoming weights become w / s and its noise variance σ²/s², with s² = wᵀ Σ_pa w + σ²."""
    W = np.asarray(W, dtype=float).copy()
    noise_var = np.asarray(noise_var, dtype=float).copy()
    p = len(W)
    Sigma = np.zeros((p, p))
    dag = nx.from_numpy_array((W != 0).astype(int), create_using=nx.DiGraph)
    for j in nx.topological_sort(dag):
        parents = np.flatnonzero(W[:, j])
        if parents.size:
            w = W[parents, j]
            s2 = w @ Sigma[np.ix_(parents, parents)] @ w + noise_var[j]
            W[parents, j] = w / np.sqrt(s2)
            noise_var[j] = noise_var[j] / s2
            Sigma[j, :] = Sigma[:, j] = W[parents, j] @ Sigma[parents, :]
        else:
            noise_var[j] = 1.0
        Sigma[j, j] = 1.0
    return W, noise_var


def noise_distribution(family: str, mean: float, var: float):
    """Noise callable `f(n)` with the given mean and variance, drawing from the global
    `np.random` state. `t<ν>` is Student-t with ν > 2."""
    import sempler.noise as noise  # lazy: sempler warns about rpy2 on import

    sd = float(np.sqrt(var))
    match family:
        case "gaussian":
            return noise.normal(mean, var)
        case "uniform":
            return noise.uniform(mean - np.sqrt(3) * sd, mean + np.sqrt(3) * sd)
        case "laplace":
            return noise.laplace(mean, sd / np.sqrt(2))
        case _:
            m = re.fullmatch(r"t(\d+)", family)
            if m is None:
                raise ValueError(f"unknown noise family {family!r}; expected one of {NOISE_FAMILIES}")
            df = int(m.group(1))
            if df <= 2:
                raise ValueError(f"Student-t noise needs ν > 2 for a finite variance, got {family!r}")
            scale = sd * np.sqrt((df - 2) / df)
            return lambda n: mean + scale * np.random.standard_t(df, n)


def draw_noise_params(p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Noise variances U(0.5, 2) and means U(-2, 2)."""
    rng = np.random.default_rng(seed)
    noise_var = rng.uniform(0.5, 2, size=p)
    means = rng.uniform(-2, 2, size=p)
    return noise_var, means


def build_model(W: np.ndarray, noise_var: np.ndarray, means: np.ndarray, family: str = "gaussian"):
    """Linear SCM `X_j = Σ_i W_ij X_i + ε_j`: `sempler.LGANM` if Gaussian, else `sempler.ANM`."""
    from sempler import ANM, LGANM

    W = np.asarray(W, dtype=float)
    noise_var = np.asarray(noise_var, dtype=float)
    means = np.asarray(means, dtype=float)
    if family == "gaussian":
        return LGANM(W, means=means, variances=noise_var)
    if family not in NOISE_FAMILIES:
        raise ValueError(f"unknown noise family {family!r}; expected one of {NOISE_FAMILIES}")
    assignments = []
    for j in range(len(W)):
        parents = np.flatnonzero(W[:, j])
        # ANM hands the assignment X[:, parents] with parents in increasing index order.
        assignments.append((lambda X, w=W[parents, j]: X @ w) if parents.size else None)
    noises = [noise_distribution(family, means[j], noise_var[j]) for j in range(len(W))]
    return ANM((W != 0).astype(int), assignments, noises)


def sample_shift(model, n: int, nodes, family: str = "gaussian", shift=(2.0, 1.0), random_state=None):
    """One environment with additive `(mean, var) = shift` noise, of the model's family, on
    every node in `nodes`."""
    from sempler import LGANM

    nodes = [int(v) for v in np.atleast_1d(nodes)]
    if isinstance(model, LGANM):
        return model.sample(n, shift_interventions={v: tuple(shift) for v in nodes}, random_state=random_state)
    return model.sample(
        n,
        shift_interventions={v: noise_distribution(family, *shift) for v in nodes},
        random_state=random_state,
    )


def make_weights(graph: str, p: int, avg_deg: float, seed: int) -> np.ndarray:
    """Random DAG weights, |w| ~ U(0.5, 2) with random signs: `er` (sempler) or `sf`
    (Barabási–Albert oriented by a random order)."""
    from sempler.generators import dag_avg_deg

    rng = np.random.default_rng(seed)
    if graph == "er":
        W = dag_avg_deg(p, avg_deg, w_min=0.5, w_max=2, return_ordering=False, random_state=seed)
    elif graph == "sf":
        m = max(1, min(p - 1, int(round(max(avg_deg / 2, 1)))))
        base = nx.barabasi_albert_graph(p, m, seed=seed)
        rank = {node: i for i, node in enumerate(rng.permutation(p))}
        W = np.zeros((p, p))
        for u, v in base.edges():
            src, dst = (u, v) if rank[u] < rank[v] else (v, u)
            W[src, dst] = rng.uniform(0.5, 2.0)
    else:
        raise ValueError(f"unknown graph family {graph!r}; expected 'er' or 'sf'")
    edges = np.flatnonzero(W)
    negative = edges[np.random.default_rng(seed).choice([True, False], len(edges))]
    W[np.unravel_index(negative, (p, p))] *= -1
    return W


def make_dataset(graph: str, p: int, num_intervs: int, avg_deg: float, seed: int, n: int,
                 targets_size=1, noise: str = "gaussian", shift=(2.0, 1.0)):
    """Observational and shift-interventional samples from an iSCM.

    Returns `(W, targets, data)`: the standardized weights and `{"obs", "0", "1", …}` in target
    order. `targets_size` is sempler's target `size` (int or (min, max)).
    """
    from sempler.generators import intervention_targets

    W = make_weights(graph, p, avg_deg, seed)
    noise_var, means = draw_noise_params(p, seed)
    W, noise_var = iscm_standardize(W, noise_var)
    model = build_model(W, noise_var, means, noise)
    max_size = targets_size if isinstance(targets_size, int) else max(targets_size)
    targets = intervention_targets(p, num_intervs, targets_size, replace=max_size > 1, random_state=seed)
    np.random.seed(seed)  # LGANM and ANM sample from the global state
    data = {"obs": model.sample(n)}
    for idx, target in enumerate(targets):
        data[str(idx)] = sample_shift(model, n, target, noise, shift)
    return W, targets, data
