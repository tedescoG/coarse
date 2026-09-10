"""Shared helpers for snakemake scripts. Not part of the coarse public API.

Snakemake's `script:` directive adds the script's directory to sys.path before
execution, so sibling imports like `from _common import ...` work without any
extra wiring. The leading underscore signals "module-local; do not import from
elsewhere in the codebase."
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from coarse.partition import infer_partition


def normalize_intervention_type(value: str) -> str:
    """COARSE supports only soft (shift) interventions; reject anything
    else explicitly.

    Replaces the silent hard/do fallthrough from the previous four `_map_type`
    copies — COARSE has no hard-intervention pathway (Algorithm 1's two-sample
    test detects shift, not structural cuts), so a 'hard' string would silently
    mislead the algorithm. A raise surfaces the misconfiguration at fit time.
    """
    lowered = str(value).lower()
    if lowered == "soft":
        return "soft"
    raise ValueError(f"COARSE supports only soft interventions; got {value!r}")


def parse_targets_per_interv(token):
    """Parse the optional ``targets_per_interv`` snakemake wildcard.

    Returns the ``size`` argument for ``sempler.generators.intervention_targets``:
    an ``int`` for fixed-size targets, or a ``(min, max)`` tuple for
    heterogeneous (random-size) targets.

    Token grammar:
      - ``None`` (wildcard absent)  → ``1``  (single-target, backward-compatible default)
      - ``"k"`` (integer string)    → ``int(k)``
      - ``"AtoB"`` (e.g. ``"1to5"``) → ``(A, B)``

    Lives in _common.py rather than inline in generate.py so the parser is
    importable from the test suite (generate.py is a snakemake script and
    therefore not importable as a plain Python module).
    """
    if token is None:
        return 1
    s = str(token)
    if "to" in s:
        lo, hi = s.split("to", 1)
        return (int(lo), int(hi))
    return int(s)


def build_data_dict(data, targets, intervention_type: str) -> dict:
    """Build the (data, targets, type) dict from a generate.py .npz archive.

    Used by fit.py and fit_oracle.py — keeping the construction in one place
    means a future change to the env-tuple shape only needs one edit.
    """
    data_dict = {"obs": (data["obs"], set(), "obs")}
    for idx, target in enumerate(targets):
        tgt = set(np.atleast_1d(target).astype(int))
        data_dict[str(idx)] = (data[str(idx)], tgt, intervention_type)
    return data_dict


def build_oracle_partition(weights: np.ndarray, targets, num_nodes: int):
    """Reconstruct (true_dag, M_true, env_order, partition) from a ground-truth
    weight matrix and intervention targets.

    Each entry of ``targets`` is an iterable of node indices — typically a
    single int (the canonical single-target case) but possibly a set/tuple of
    several ints (multi-target interventions). The affected mask for an
    environment is the union of each target node together with all of its
    descendants in the true DAG.

    For singleton targets this reduces to ``{t} ∪ descendants(t)`` — identical
    to the historical single-target behavior — so cached results from existing
    single-target sweeps are unchanged.
    """
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
    """Integer block label per node: the index of the block containing it, in ``blocks``
    iteration order. ``blocks`` is any iterable of node-index collections (``frozenset``
    blocks from ``infer_partition`` or the tuple nodes of a block DAG). The one ARI label
    construction shared by evaluate.py and the CausalChamber scripts.
    """
    labels = np.zeros(num_nodes, dtype=int)
    for label, block in enumerate(blocks):
        labels[list(block)] = label
    return labels


def block_dag_prf(est_dag: nx.DiGraph, true_dag: nx.DiGraph) -> tuple[float, float, float]:
    """Precision, recall and F-score of the estimated block DAG.

    The reference is the true DAG projected onto the *estimated* blocks: an edge
    between two blocks exists iff some atom of the first has a true edge into
    some atom of the second, taken over forward pairs in ``est_dag`` node order.
    Shared by evaluate.py and the diagnostics so the metric has one definition.
    """

    def _is_adj(pa, ch):
        for atom in pa:
            for chatom in ch:
                if true_dag.has_edge(atom, chatom):
                    return True
        return False

    true_edge_est_partition = nx.create_empty_copy(est_dag)
    node_list = list(true_edge_est_partition.nodes)
    for idx, pa in enumerate(node_list[:-1]):
        for ch in node_list[idx + 1 :]:
            if _is_adj(pa, ch):
                true_edge_est_partition.add_edge(pa, ch)

    true_positive = sum(
        (1 for edge in est_dag.edges if edge in true_edge_est_partition.edges)
    )
    try:
        precision = true_positive / len(est_dag.edges)
    except ZeroDivisionError:
        precision = 1
    try:
        recall = true_positive / len(true_edge_est_partition.edges)
    except ZeroDivisionError:
        recall = 1
    try:
        f_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f_score = 0
    return precision, recall, f_score


# --------------------------------------------------------------------------------------
# Internally standardized SCM (iSCM), Ormaniec et al. 2024, arXiv:2406.11601.
#
# Every variable of the observational distribution has unit variance by construction, so
# marginal variances cannot snowball with depth or in-degree and the data is neither
# var- nor R²-sortable. The standardization is a post-processing step on (W, σ²): it uses
# only second moments, so it is valid for any finite-variance noise family. Interventions
# are applied to the standardized model with the standardization constants fixed from the
# observational model (their Appendix B); interventional environments must not be
# re-standardized.
# --------------------------------------------------------------------------------------

NOISE_FAMILIES = ("gaussian", "uniform", "laplace")


def iscm_standardize(W: np.ndarray, noise_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale ``(W, noise_var)`` so that every variable has population variance 1.

    Walks the DAG in causal order; for node ``j`` with parents ``pa`` and incoming weights
    ``w``, ``s² = wᵀ Σ_pa w + σ_j²`` is the variance of the pre-standardized variable, and
    ``w̃ = w / s``, ``σ̃_j² = σ_j² / s²``. Signs and ratios of the incoming weights are
    preserved; only their common scale and the noise variance change. Same closed form as
    ``closed_form_data_generator.py`` in the authors' repository.
    """
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
    """A sempler-style noise callable ``f(n)`` with the requested mean and variance.

    Uses ``sempler.noise`` (which draws from the global ``np.random`` state, the one
    ``sempler.ANM.sample(random_state=...)`` seeds). Variance-matching parameters:
    uniform half-width ``√3·sd``, Laplace scale ``sd/√2``.
    """
    import sempler.noise as noise  # local: sempler warns about rpy2 at import time

    sd = float(np.sqrt(var))
    match family:
        case "gaussian":
            return noise.normal(mean, var)
        case "uniform":
            return noise.uniform(mean - np.sqrt(3) * sd, mean + np.sqrt(3) * sd)
        case "laplace":
            return noise.laplace(mean, sd / np.sqrt(2))
        case _:
            raise ValueError(f"unknown noise family {family!r}; expected one of {NOISE_FAMILIES}")


def draw_noise_params(p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(noise_var, means)`` drawn exactly as ``LGANM(W, means=(-2, 2), variances=(0.5, 2),
    random_state=seed)`` draws them internally (variances first, then means, one generator),
    so a standardized model rescales precisely the noise variances of the raw model."""
    rng = np.random.default_rng(seed)
    noise_var = rng.uniform(0.5, 2, size=p)
    means = rng.uniform(-2, 2, size=p)
    return noise_var, means


def build_model(W: np.ndarray, noise_var: np.ndarray, means: np.ndarray, family: str = "gaussian"):
    """Linear SCM ``X_j = Σ_i W_ij X_i + ε_j`` with ``ε_j ~ family(mean_j, noise_var_j)``.

    Gaussian → ``sempler.LGANM`` (closed-form multivariate-normal sampling, the path the
    suite uses today). Other families → ``sempler.ANM`` with linear assignments and
    variance-matched noise callables. Both sample the same model given the same parameters.
    """
    from sempler import ANM, LGANM  # local: sempler warns about rpy2 at import time

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
    """One interventional environment: additive noise with ``(mean, var) = shift`` on every
    node in ``nodes``. The shift noise is drawn from the target's own family, so an
    environment stays within one noise family. Standardization constants are untouched."""
    from sempler import LGANM  # local: sempler warns about rpy2 at import time

    nodes = [int(v) for v in np.atleast_1d(nodes)]
    if isinstance(model, LGANM):
        return model.sample(n, shift_interventions={v: tuple(shift) for v in nodes}, random_state=random_state)
    return model.sample(
        n,
        shift_interventions={v: noise_distribution(family, *shift) for v in nodes},
        random_state=random_state,
    )


# --------------------------------------------------------------------------------------
# The synthetic data generator: one function used by generate.py (Snakemake) and by the
# diagnostics, so both sample exactly the same DGP.
# --------------------------------------------------------------------------------------


def make_weights(graph: str, p: int, avg_deg: float, seed: int) -> np.ndarray:
    """Raw weight matrix of a random DAG: ``er`` via ``sempler.generators.dag_avg_deg``,
    ``sf`` via a Barabási–Albert graph oriented by a random permutation. Weights are
    ``U(0.5, 2)`` in magnitude with an independent random sign per edge. The suite passes
    ``avg_deg = density * (p - 1)`` for both graph types."""
    from sempler.generators import dag_avg_deg  # local: sempler warns about rpy2 at import time

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

    Draws the raw DAG and noise parameters, standardizes them with ``iscm_standardize`` (every
    variable has population variance 1 in the observational distribution), then samples one
    observational environment and one environment per intervention target with additive
    ``shift = (mean, var)`` noise on the target (the standardization constants are fixed from
    the observational model; interventional data is not re-standardized).

    ``targets_size`` is the ``size`` argument of ``sempler.generators.intervention_targets``:
    an int, or a ``(min, max)`` tuple for heterogeneous target sets (``replace=True`` in that
    case, since distinct targets across environments may be infeasible; single-target keeps
    sempler's ``replace=False``).

    Returns ``(W, targets, data)`` with ``W`` the *standardized* weight matrix (same support and
    signs as the raw draw) and ``data = {"obs": ..., "0": ..., "1": ...}`` in target order.
    """
    from sempler.generators import intervention_targets  # local: sempler warns about rpy2

    W = make_weights(graph, p, avg_deg, seed)
    noise_var, means = draw_noise_params(p, seed)
    W, noise_var = iscm_standardize(W, noise_var)
    model = build_model(W, noise_var, means, noise)
    max_size = targets_size if isinstance(targets_size, int) else max(targets_size)
    targets = intervention_targets(p, num_intervs, targets_size, replace=max_size > 1, random_state=seed)
    np.random.seed(seed)  # both LGANM and ANM sample from the global state; one seed for all environments
    data = {"obs": model.sample(n)}
    for idx, target in enumerate(targets):
        data[str(idx)] = sample_shift(model, n, target, noise, shift)
    return W, targets, data
