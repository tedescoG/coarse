"""Pooled multi-environment block BIC, `2·ℓ̂ − λ·log(n_e)·d_j` (higher is better).
Inputs must be column-centered; `S = XᵀX / n`."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np
from scipy import linalg as sla

from coarse.types import Block, EnvKey


def parameter_count_d_j(r_j: int, s_j: int) -> int:
    """d_j = r_j·s_j (regression) + r_j(r_j + 1)/2 (residual covariance)."""
    return r_j * s_j + r_j * (r_j + 1) // 2


def _block_indices(block: Block) -> np.ndarray:
    return np.asarray(sorted(block), dtype=np.int64)


def _parents_indices(parents: Iterable[Block]) -> np.ndarray:
    parts = [_block_indices(p) for p in parents]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _parents_indices_cached(
    parents: Iterable[Block], idx_cache: dict[Block, np.ndarray]
) -> np.ndarray:
    parts = [idx_cache[p] for p in parents]
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


class EnvStats(NamedTuple):
    """Per-env `sigma = XᵀX / n_e` of centered X, and its row count (sigma is p × p)."""
    sigma: np.ndarray
    n_e: int
    log_n_e: float


def compute_env_stats(
    data_dict: dict[EnvKey, np.ndarray],
) -> dict[EnvKey, EnvStats]:
    """`EnvStats` per env, from centered arrays."""
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
    """`(B, Σ)` with `B = Σ_jPa Σ_PaPa⁻¹` (r_j × s_j) and `Σ = Σ_jj − Σ_jPa Bᵀ`. Raises
    `LinAlgError` if Σ_PaPa is not positive definite. Shared by the BIC and CV scorers."""
    if S_jj is None:
        S_jj = sigma[np.ix_(block_idx, block_idx)]
    if parent_idx.size == 0:
        return np.empty((block_idx.size, 0)), S_jj
    S_jPa = sigma[np.ix_(block_idx, parent_idx)]
    S_PaPa = sigma[np.ix_(parent_idx, parent_idx)]
    c, low = sla.cho_factor(S_PaPa, lower=True, check_finite=False)
    Bt = sla.cho_solve((c, low), S_jPa.T, check_finite=False)
    Sigma = S_jj - S_jPa @ Bt
    return Bt.T, Sigma


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
    """Per-env BIC; -inf if n_e <= s_j + r_j or a covariance is not positive definite."""
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
        _, Sigma = _block_regression_from_sigma(
            block_idx, parent_idx, sigma, S_jj=S_jj,
        )
    except sla.LinAlgError:
        return -np.inf
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(logdet):
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
    """Sum of per-env BICs; -inf as soon as one is."""
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
    """`pooled_block_bic_from_sigma` from raw centered arrays."""
    env_stats = compute_env_stats(data_dict)
    return pooled_block_bic_from_sigma(block, parents, env_stats, lambda_pen)


__all__ = [
    "EnvStats",
    "compute_env_stats",
    "parameter_count_d_j",
    "pooled_block_bic",
    "pooled_block_bic_from_sigma",
]
