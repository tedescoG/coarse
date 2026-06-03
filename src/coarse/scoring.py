"""Block-level multi-environment BIC scoring.

The MLE convention here is `S = X^T X / n`.

Centering is performed once per env in `coarse._run_score_phase` (coarse.py)
before any scoring call; the functions in this module assume column-centered
input.

Sign convention: BIC is **maximized**. `pooled_block_bic_from_sigma` returns
`2·ℓ̂ − λ·log(n_e)·d_j` so larger means better. Grow-shrink uses strict `>`.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np
from scipy import linalg as sla

from coarse.types import Block, EnvKey


def parameter_count_d_j(r_j: int, s_j: int) -> int:
    """Equation 20 — total free parameters per block.

    d_j = r_j · s_j + r_j(r_j + 1) / 2
        = (regression entries in B_j^e)  +  (symmetric covariance entries in Σ_j^e)
    """
    return r_j * s_j + r_j * (r_j + 1) // 2


def _block_indices(block: Block) -> np.ndarray:
    return np.asarray(sorted(block), dtype=np.int64)


def _parents_indices(parents: Iterable[Block]) -> np.ndarray:
    """Concatenated sorted indices across all parent blocks. Returns shape (0,)
    when `parents` is empty."""
    parts = [_block_indices(p) for p in parents]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _parents_indices_cached(
    parents: Iterable[Block], idx_cache: dict[Block, np.ndarray]
) -> np.ndarray:
    """Like `_parents_indices` but looks up pre-computed per-block index arrays
    instead of re-sorting each frozenset on every call."""
    parts = [idx_cache[p] for p in parents]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


class EnvStats(NamedTuple):
    """Per-environment summary statistics cached once per fit.

    `sigma` is the (p, p) sample covariance `(X.T @ X) / n_e` on column-centered
    `X`; `n_e` is the row count of the underlying `X` (NOT `sigma.shape[0]`).
    `log_n_e` is `log(n_e)`, precomputed to avoid redundant calls in the hot loop.
    The caller is responsible for centering `X` before computing `sigma` — this
    container does not verify the centering invariant. See `compute_env_stats`.
    """
    sigma: np.ndarray
    n_e: int
    log_n_e: float


def compute_env_stats(
    data_dict: dict[EnvKey, np.ndarray],
) -> dict[EnvKey, EnvStats]:
    """Build per-env `EnvStats` from centered arrays. Called once per fit by
    `_run_score_phase` after centering (and optional Z-scoring); the result is
    then sliced by `_block_*_from_sigma` for every grow-shrink probe.
    """
    return {
        k: EnvStats(
            sigma=(v.T @ v) / v.shape[0],
            n_e=v.shape[0],
            log_n_e=float(np.log(v.shape[0])),
        )
        for k, v in data_dict.items()
    }


def _block_regression_from_sigma(
    block_idx: np.ndarray,
    parent_idx: np.ndarray,
    sigma: np.ndarray,
    *,
    S_jj: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Schur-complement extraction of (B, Σ_residual) from precomputed `sigma`.

    Identity used:
        Y = Σ_PaPa^{-1} Σ_jPa^T          (shape (s_j, r_j) — i.e. Bᵀ)
        Σ_residual = Σ_jj − Σ_jPa Y      (Schur complement)

    Returns
    -------
    B : ndarray, shape (r_j, s_j)
        Regression coefficients with X_block ≈ X_parents @ B.T. Shape
        ``(r_j, 0)`` when ``parent_idx.size == 0``.
    Sigma : ndarray, shape (r_j, r_j)
        Residual covariance. Falls back to ``S_jj`` when ``parent_idx`` is empty.

    Notes
    -----
    Same Cholesky path as the previous inline computation in
    ``_block_log_det_residual_from_sigma`` (BIC is byte-identical).  Surfacing
    ``(B, Σ)`` here lets the CV evaluator reuse one well-tested code path
    instead of duplicating the math; the BIC scorer just throws ``B`` away
    and pipes ``Sigma`` into ``slogdet``.
    """
    if S_jj is None:
        S_jj = sigma[np.ix_(block_idx, block_idx)]
    if parent_idx.size == 0:
        return np.empty((block_idx.size, 0)), S_jj
    S_jPa = sigma[np.ix_(block_idx, parent_idx)]
    S_PaPa = sigma[np.ix_(parent_idx, parent_idx)]
    c, low = sla.cho_factor(S_PaPa, lower=True, check_finite=False)
    Y = sla.cho_solve((c, low), S_jPa.T, check_finite=False)
    Sigma = S_jj - S_jPa @ Y
    return Y.T, Sigma


def _block_log_det_residual_from_sigma(
    block_idx: np.ndarray,
    parent_idx: np.ndarray,
    sigma: np.ndarray,
    *,
    S_jj: np.ndarray | None = None,
) -> float:
    """log|Sigma_hat_{j|XP}| via slicing of precomputed Sigma_hat^e. Same
    Schur-complement math and `LinAlgError` semantics as the raw-array
    `block_log_det_residual` (now in tests/test_coarse.py); the only difference
    is the source of `S_jj`/`S_jPa`/`S_PaPa` (slices of `sigma` instead of
    `(X_block.T @ X_parents) / n_e` from raw data).

    When ``S_jj`` is provided (Tier-2 cache), the ``sigma[np.ix_(...)]`` slice
    for the target block is skipped. Mutation-safe: ``slogdet`` is non-mutating,
    and ``S_jj - S_jPa @ Y`` creates a fresh array via ``__sub__``."""
    _, Sigma = _block_regression_from_sigma(
        block_idx, parent_idx, sigma, S_jj=S_jj,
    )
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(logdet):
        raise sla.LinAlgError(
            f"residual covariance has non-positive determinant (sign={sign})"
        )
    return float(logdet)


def _block_bic_env_from_sigma(
    block_idx: np.ndarray,
    parent_idx: np.ndarray,
    sigma: np.ndarray,
    n_e: int,
    lambda_pen: float = 1.0,
    *,
    d_j: int | None = None,
    log_n_e: float | None = None,
    S_jj: np.ndarray | None = None,
) -> float:
    """Per-env BIC from precomputed `sigma` + `n_e`. `n_e` MUST travel
    separately — it is NOT inferable from `sigma.shape[0]` (which is `p`,
    not the row count).

    Optional keyword args ``d_j``, ``log_n_e``, and ``S_jj`` accept
    precomputed values from the caller's hot loop, avoiding per-env
    recomputation of quantities that are invariant across environments."""
    r_j, s_j = block_idx.size, parent_idx.size
    if n_e <= 0 or r_j <= 0:
        return -np.inf
    if n_e <= s_j + r_j:
        return -np.inf
    if d_j is None:
        d_j = parameter_count_d_j(r_j, s_j)
    if log_n_e is None:
        log_n_e = float(np.log(n_e))
    try:
        logdet = _block_log_det_residual_from_sigma(
            block_idx, parent_idx, sigma, S_jj=S_jj,
        )
    except sla.LinAlgError:
        return -np.inf
    log_lik = -0.5 * n_e * logdet
    return 2.0 * log_lik - lambda_pen * log_n_e * d_j


def pooled_block_bic_from_sigma(
    block: Block,
    parents: list[Block],
    env_stats: dict[EnvKey, EnvStats],
    lambda_pen: float = 1.0,
    *,
    block_idx: np.ndarray | None = None,
    idx_cache: dict[Block, np.ndarray] | None = None,
    S_jj_cache: dict[EnvKey, np.ndarray] | None = None,
) -> float:
    """Cached-path scoring. Sums per-env BICs from precomputed `EnvStats`,
    short-circuiting to -inf on the first non-finite contribution.

    Optional keyword args (Tier-2 caches) allow the grow-shrink hot loop to
    pass pre-computed invariants: ``block_idx`` for the target block's sorted
    index array, ``idx_cache`` mapping each candidate block to its sorted
    indices, and ``S_jj_cache`` mapping each env key to the pre-sliced
    ``sigma[block_idx, block_idx]``. All default to ``None`` (recomputed
    on the spot), so call sites outside the hot loop work unchanged."""
    if block_idx is None:
        block_idx = _block_indices(block)
    if idx_cache is not None:
        parent_idx = _parents_indices_cached(parents, idx_cache)
    else:
        parent_idx = _parents_indices(parents)
    r_j, s_j = block_idx.size, parent_idx.size
    d_j = parameter_count_d_j(r_j, s_j)
    total = 0.0
    for k, stats in env_stats.items():
        S_jj = S_jj_cache[k] if S_jj_cache is not None else None
        contrib = _block_bic_env_from_sigma(
            block_idx, parent_idx, stats.sigma, stats.n_e, lambda_pen,
            d_j=d_j, log_n_e=stats.log_n_e, S_jj=S_jj,
        )
        if not np.isfinite(contrib):
            return -np.inf
        total += contrib
    return total


def pooled_block_bic(
    block: Block,
    parents: list[Block],
    data_dict: dict[EnvKey, np.ndarray],
    lambda_pen: float = 1.0,
) -> float:
    """Equation 23 — sum-of-environment block BIC.

    BIC_j(pi_j, Pa_j) = Sum_{e in E} BIC_j^e(pi_j, Pa_j)

    Internally routes through `compute_env_stats` + `pooled_block_bic_from_sigma`
    so a single call rebuilds the per-env sample covariances once instead of
    once per environment-probe.
    """
    env_stats = compute_env_stats(data_dict)
    return pooled_block_bic_from_sigma(block, parents, env_stats, lambda_pen)


__all__ = [
    "EnvStats",
    "compute_env_stats",
    "parameter_count_d_j",
    "pooled_block_bic",
    "pooled_block_bic_from_sigma",
]
