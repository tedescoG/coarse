"""Score-based partition learning: per-cell two-sample evidence, pooled segment scores, and
greedy agglomerative merging (guarantees in `paper/rebuttal.tex`)."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from coarse.hypothesis_tests import _normalize_env_data
from coarse.types import Block, EnvKey

_VAR_FLOOR = 1e-12
_LOG_2PI = float(np.log(2.0 * np.pi))


def _gaussian_lr_stats(X0: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    """Column-wise `2(ll_sep − ll_same)` for `H0: x, y ~ same N(mu, w2)` vs separate MLE fits."""
    n0 = X0.shape[0]
    ne = Xe.shape[0]
    n = n0 + ne
    mu0 = X0.mean(axis=0)
    mue = Xe.mean(axis=0)
    s20 = np.maximum(((X0 - mu0) ** 2).mean(axis=0), _VAR_FLOOR)
    s2e = np.maximum(((Xe - mue) ** 2).mean(axis=0), _VAR_FLOOR)
    mup = (n0 * mu0 + ne * mue) / n
    s2p = np.maximum(
        (n0 * (s20 + (mu0 - mup) ** 2) + ne * (s2e + (mue - mup) ** 2)) / n,
        _VAR_FLOOR,
    )
    ll_same = -0.5 * n * (_LOG_2PI + np.log(s2p) + 1.0)
    ll_sep = -0.5 * n0 * (_LOG_2PI + np.log(s20) + 1.0) - 0.5 * ne * (
        _LOG_2PI + np.log(s2e) + 1.0
    )
    return np.maximum(2.0 * (ll_sep - ll_same), 0.0)


def _welch_t2_stats(X0: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    """Column-wise squared Welch statistics: mean-shift evidence that cannot see variance
    or shape differences (robust to non-Gaussian marginals, unlike the Gaussian LR)."""
    n0 = X0.shape[0]
    ne = Xe.shape[0]
    mu0 = X0.mean(axis=0)
    mue = Xe.mean(axis=0)
    s20 = np.maximum(((X0 - mu0) ** 2).mean(axis=0), _VAR_FLOOR)
    s2e = np.maximum(((Xe - mue) ** 2).mean(axis=0), _VAR_FLOOR)
    se = np.sqrt(s20 / n0 + s2e / ne)
    return np.maximum(((mue - mu0) / se) ** 2, 0.0)


def evidence_matrix(
    data_dict: Mapping[EnvKey, Any],
    baseline_key: EnvKey = "obs",
    mean_only: bool = False,
) -> tuple[np.ndarray, list[EnvKey]]:
    """`L[v, j]` = two-sample evidence for env j vs the baseline (rebuttal.tex, Def. 2.1):
    the Gaussian mean-and-variance LR, or the squared Welch statistic when `mean_only`.
    Returns `(L, env_order)` with columns in string-sorted key order, matching
    `compute_M`. Runs in O(d n) with one pass per (variable, env).

    Use `mean_only` when marginals are non-Gaussian: the LR conflates mean shifts with
    variance/shape differences, and heavy tails then inflate evidence unevenly across
    (variable, env) cells — pooled evidence inherits that noise (synthetic check:
    lognormal marginals drop the scored partition from Welch-parity to ARI 0.64; the
    mean-only statistic restores parity)."""
    if baseline_key not in data_dict:
        raise ValueError(
            f"data_dict must contain a baseline env keyed {baseline_key!r}"
        )
    X0 = _normalize_env_data(data_dict[baseline_key])
    env_order: list[EnvKey] = sorted(
        (k for k in data_dict if k != baseline_key),
        key=lambda k: str(k),
    )
    p = X0.shape[1]
    L = np.zeros((p, len(env_order)), dtype=np.float64)
    for col, key in enumerate(env_order):
        Xe = _normalize_env_data(data_dict[key])
        if Xe.shape[1] != p:
            raise ValueError(f"env {key!r} has {Xe.shape[1]} columns; baseline has {p}")
        L[:, col] = _welch_t2_stats(X0, Xe) if mean_only else _gaussian_lr_stats(X0, Xe)
    return L, env_order


def null_calibrated_evidence(
    data_dict: Mapping[EnvKey, Any],
    baseline_key: EnvKey = "obs",
    num_splits: int = 32,
    rng: np.random.Generator | None = None,
    mean_only: bool = False,
) -> tuple[np.ndarray, list[EnvKey]]:
    """Evidence standardized by an empirical baseline-vs-baseline null: `z[v, j] =
    (L[v, j] − med_j) / MAD_j`, where med_j and MAD_j come from LR statistics of disjoint
    same-size halves of the baseline, pooled across variables (rebuttal discussion).
    Raw LR magnitudes grow with sample size and inherit tail inflation from
    within-environment dependence, so absolute bars and merge tolerances are not
    comparable across environments or datasets; after calibration one evidence unit is
    one null MAD. Values are clipped at zero from below (the segment score only needs
    distance to the bar, which lies far above zero). Uses `B = min(num_splits, floor(n_0
    / (2 n_j)))` split pairs per env, so the baseline must have at least `2 n_j` rows.
    Per-variable nulls are a possible refinement; pooled per-env nulls stay robust when
    envs are small. When the baseline has fewer than `2 n_j` rows, the null pairs are
    bootstrapped (2n rows with replacement, split in half); overlapping halves
    under-disperse the null, so the calibration errs conservative. The null statistic
    always matches the realized one (mean-only or Gaussian LR)."""
    if rng is None:
        rng = np.random.default_rng(0)
    L, env_order = evidence_matrix(
        data_dict, baseline_key=baseline_key, mean_only=mean_only
    )
    X0 = _normalize_env_data(data_dict[baseline_key])
    n0 = X0.shape[0]
    z = np.zeros_like(L)
    for col, key in enumerate(env_order):
        ne = _normalize_env_data(data_dict[key]).shape[0]
        stat = _welch_t2_stats if mean_only else _gaussian_lr_stats
        if n0 >= 2 * ne:
            b = min(int(num_splits), n0 // (2 * ne))
            perm = rng.permutation(n0)
            pairs = [
                (
                    X0[perm[(2 * i) * ne : (2 * i + 1) * ne]],
                    X0[perm[(2 * i + 1) * ne : (2 * i + 2) * ne]],
                )
                for i in range(b)
            ]
        else:
            b = int(num_splits)
            pairs = []
            for _ in range(b):
                idx = rng.choice(n0, size=2 * ne, replace=True)
                pairs.append((X0[idx[:ne]], X0[idx[ne:]]))
        null_vals = np.empty(b * L.shape[0], dtype=np.float64)
        for i, (h1, h2) in enumerate(pairs):
            null_vals[i * L.shape[0] : (i + 1) * L.shape[0]] = stat(h1, h2)
        med = float(np.median(null_vals))
        mad = max(float(np.median(np.abs(null_vals - med))), 1e-12)
        z[:, col] = np.maximum((L[:, col] - med) / mad, 0.0)
    return z, env_order


def _kappa(n_e: Sequence[float], n_0: float, lam: float) -> np.ndarray:
    """Per-env declaration cost `2 lam log(n_e + n_0)` per variable of the segment."""
    return 2.0 * lam * np.log(np.asarray(n_e, dtype=np.float64) + float(n_0))


def _segment_score(g: np.ndarray, size: float, kappa: np.ndarray) -> float:
    """`S(A) = sum_e max(0, g_e − kappa_e |A|)` (rebuttal.tex, Def. 2.3)."""
    return float(np.sum(np.maximum(0.0, g - kappa * size)))


def _merge_delta(
    g_a: np.ndarray,
    size_a: float,
    score_a: float,
    g_b: np.ndarray,
    size_b: float,
    score_b: float,
    kappa: np.ndarray,
) -> float:
    """`S(A ∪ B) − S(A) − S(B)`; always ≤ 0 (rebuttal.tex, Lem. 2.6)."""
    return _segment_score(g_a + g_b, size_a + size_b, kappa) - score_a - score_b


@dataclass(frozen=True)
class ScoredPartition:
    """Output of the score-based partition phase; `M` is the signature matrix in the
    `compute_M` convention, so `COARSEOracle` accepts `(partition, M, env_order)` directly."""

    partition: list[Block]
    signatures: dict[Block, frozenset[int]]
    env_order: list[EnvKey]
    M: np.ndarray
    score: float
    merge_deltas: tuple[float, ...]


def _finalize(
    members: list[list[int]],
    g_list: list[np.ndarray],
    size_list: list[np.ndarray],
    kappa: np.ndarray,
    active: list[bool],
    env_order: list[EnvKey],
    score: float,
    deltas: list[float],
    kappa_0: float = 0.0,
) -> ScoredPartition:
    kept = [i for i, ok in enumerate(active) if ok]
    # Canonical order: by (|signature|, min member), a linear extension of support inclusion.
    # Readout bar (rebuttal.tex, Rem. 5): kappa|A| - kappa_0, floored at zero; a discount
    # widens selected signatures without touching any merge decision.
    sigs = {
        i: frozenset(
            int(e)
            for e in np.flatnonzero(
                g_list[i] > np.maximum(kappa * size_list[i] - kappa_0, 0.0)
            )
        )
        for i in kept
    }
    kept.sort(key=lambda i: (len(sigs[i]), members[i][0]))
    partition = [frozenset(members[i]) for i in kept]
    signatures = {frozenset(members[i]): sigs[i] for i in kept}
    p = sum(len(members[i]) for i in kept)
    M = np.zeros((p, kappa.size), dtype=bool)
    for i in kept:
        rows = np.asarray(members[i], dtype=np.int64)[:, None]
        cols = np.asarray(sorted(sigs[i]), dtype=np.int64)[None, :]
        M[rows, cols] = True
    return ScoredPartition(
        partition=partition,
        signatures=signatures,
        env_order=env_order,
        M=M,
        score=score,
        merge_deltas=tuple(deltas),
    )


def scored_partition(
    L: np.ndarray,
    n_e: Sequence[float],
    n_0: float,
    lam: float = 1.0,
    c: float = 1.0,
    kappa_0: float = 0.0,
    size_cap: float | None = None,
) -> ScoredPartition:
    """Greedy agglomerative partition from the evidence matrix `L` (rebuttal.tex, Alg. 3.1):
    merge the pair with the largest score delta while it exceeds `−c`; deterministic
    tie-breaking. O(d^2 m + d^2 log d) time after O(d n) evidence.

    `kappa_0` is a signature-readout discount (rebuttal.tex, Rem. 5): the readout bar
    becomes `kappa_e |A| - kappa_0` (floored at 0), widening selected supports without
    changing any merge decision.

    `size_cap` bounds the number of genes per block: pairs whose merge would exceed it
    are never offered, so evidence-null genes stay in capped same-signature blocks
    instead of one blob the score phase cannot fit (the per-env BIC needs `n_e > r_j +
    s_j`). Set it below the smallest environment's sample size; feed `partition` — not
    the M-row canonicalization of `M` — to `COARSEOracle`, which accepts refinements of
    the row classes."""
    if kappa_0 < 0.0:
        raise ValueError(f"kappa_0 must be nonnegative, got {kappa_0!r}")
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2:
        raise ValueError(f"L must be 2-D, got shape {L.shape}")
    d, m = L.shape
    if np.any(~np.isfinite(L)) or np.any(L < 0.0):
        raise ValueError("L must be finite and nonnegative")
    if len(n_e) != m:
        raise ValueError(f"n_e has length {len(n_e)}; L has {m} columns")
    if np.any(np.asarray(n_e, dtype=np.float64) < 1.0) or n_0 < 1.0:
        raise ValueError("every env needs at least one sample")
    if c < 0.0:
        raise ValueError(f"c must be nonnegative, got {c!r}")
    if size_cap is not None and size_cap < 1.0:
        raise ValueError(f"size_cap must be at least 1, got {size_cap!r}")
    kappa = _kappa(n_e, n_0, lam)

    members: list[list[int]] = [[v] for v in range(d)]
    g_list: list[np.ndarray] = [L[v, :].copy() for v in range(d)]
    size_list: list[float] = [1.0] * d
    score_list: list[float] = [_segment_score(g, 1.0, kappa) for g in g_list]
    active: list[bool] = [True] * d
    deltas: list[float] = []

    heap: list[tuple[float, int, int, int, int]] = []

    def _try_pair(a: int, b: int) -> None:
        if size_cap is not None and size_list[a] + size_list[b] > size_cap:
            return
        delta = _merge_delta(
            g_list[a],
            size_list[a],
            score_list[a],
            g_list[b],
            size_list[b],
            score_list[b],
            kappa,
        )
        if delta > -c:
            lo, hi = (a, b) if a < b else (b, a)
            heapq.heappush(heap, (-delta, lo, hi, a, b))

    for a in range(d):
        for b in range(a + 1, d):
            _try_pair(a, b)

    while heap:
        neg_delta, _, _, a, b = heapq.heappop(heap)
        if not (active[a] and active[b]):
            continue
        deltas.append(-neg_delta)
        members.append(members[a] + members[b])
        g_list.append(g_list[a] + g_list[b])
        size_list.append(size_list[a] + size_list[b])
        score_list.append(_segment_score(g_list[-1], size_list[-1], kappa))
        active.append(True)
        active[a] = False
        active[b] = False
        new = len(members) - 1
        for o in range(new):
            if active[o]:
                _try_pair(new, o)

    kept_score = sum(score_list[i] for i, ok in enumerate(active) if ok)
    total = kept_score - c * (sum(active) - 1)
    return _finalize(
        members,
        g_list,
        size_list,
        kappa,
        active,
        env_order=[],
        score=total,
        deltas=deltas,
        kappa_0=kappa_0,
    )


def dp_segmentation(
    L: np.ndarray,
    order: Sequence[int],
    n_e: Sequence[float],
    n_0: float,
    lam: float = 1.0,
    c: float = 1.0,
    kappa_0: float = 0.0,
) -> ScoredPartition:
    """Exact best contiguous segmentation of `order` by DP: `F(i) = max_k F(k) + S(k..i) − c`
    (rebuttal.tex, Rem. 3.4). O(d^2 m) time."""
    L = np.asarray(L, dtype=np.float64)
    d, m = L.shape
    order = list(order)
    if sorted(order) != list(range(d)):
        raise ValueError("order must be a permutation of range(d)")
    kappa = _kappa(n_e, n_0, lam)
    prefix = np.zeros((d + 1, m), dtype=np.float64)
    for i, v in enumerate(order):
        prefix[i + 1] = prefix[i] + L[v]

    F = np.full(d + 1, -np.inf)
    F[0] = 0.0
    back = np.full(d + 1, -1, dtype=np.int64)
    for i in range(1, d + 1):
        for k in range(i):
            seg = _segment_score(prefix[i] - prefix[k], float(i - k), kappa)
            cand = F[k] + seg - c
            if cand > F[i]:
                F[i] = cand
                back[i] = k
    total = F[d] + c  # the DP charges −c per block; the score charges −c per boundary

    cuts: list[int] = []
    i = d
    while i > 0:
        cuts.append(i)
        i = int(back[i])
    cuts.append(0)
    cuts.reverse()
    partition = [frozenset(order[cuts[k] : cuts[k + 1]]) for k in range(len(cuts) - 1)]

    # Canonical block order and signatures from the found partition.
    g_by_block = {blk: L[sorted(blk), :].sum(axis=0) for blk in partition}
    sigs = {
        blk: frozenset(
            int(e)
            for e in np.flatnonzero(
                g_by_block[blk] > np.maximum(kappa * len(blk) - kappa_0, 0.0)
            )
        )
        for blk in partition
    }
    ordered = sorted(partition, key=lambda blk: (len(sigs[blk]), min(blk)))
    signatures = {blk: sigs[blk] for blk in ordered}
    M = np.zeros((d, m), dtype=bool)
    for blk in ordered:
        M[np.ix_(sorted(blk), np.asarray(sorted(sigs[blk]), dtype=np.int64))] = True
    return ScoredPartition(
        partition=ordered,
        signatures=signatures,
        env_order=[],
        M=M,
        score=total,
        merge_deltas=(),
    )


__all__ = [
    "ScoredPartition",
    "dp_segmentation",
    "evidence_matrix",
    "null_calibrated_evidence",
    "scored_partition",
]
