"""Score-based non-greed GrowShrink implementation for coarse nodes."""

from __future__ import annotations

import numpy as np

from coarse.scoring import (
    _block_indices,
    compute_env_stats,
    EnvStats,
    pooled_block_bic_from_sigma,
)
from coarse.types import Block, EnvKey


def _shuffled(items: list[Block], rng: np.random.Generator) -> list[Block]:
    """Return a new list with `items` in a random order. Index-based shuffle
    so we avoid any object-dtype quirks with `rng.shuffle` on frozensets."""
    if not items:
        return []
    order = rng.permutation(len(items))
    return [items[int(i)] for i in order]


def grow_shrink(
    block: Block,
    candidate_pool: list[Block],
    data_dict: dict[EnvKey, np.ndarray],
    lambda_pen: float = 1.0,
    rng: np.random.Generator | None = None,
    *,
    env_stats: dict[EnvKey, EnvStats] | None = None,
) -> list[Block]:
    """Algorithm 4 — return the estimated parent set P̂a_j ⊆ candidate_pool.

    Parameters
    ----------
    block
        The target block π_j.
    candidate_pool
        Z_j ⊆ Π_E \\ {π_j} — typically `Pa⋆(π_j)` from
        `partition.compute_candidate_pools`.
    data_dict
        Mapping env_key → (n_e, p) ndarray. Both `block` and every candidate
        index into the columns of these arrays. Ignored when `env_stats` is
        provided (the cached path bypasses raw arrays entirely).
    lambda_pen
        BIC penalty coefficient λ (Eq. 21). Default 1.0 matches standard BIC.
    rng
        Random generator for shuffling the inner-sweep iteration order.
        Defaults to `np.random.default_rng(0)` for reproducibility.
    env_stats
        Keyword-only Tier-1 optimisation hook. When `None` (the default), each
        call rebuilds the per-env sample covariance cache from `data_dict` on
        entry — saving the per-probe recomputation cost but still paying a
        one-time O(n_e · p²) per environment per `grow_shrink` invocation.
        When provided, the cache is reused as-is (zero per-invocation
        rebuild); this is the path `_run_score_phase` takes to amortise the
        cache across every block's grow-shrink call. The cache MUST have been
        built from the same centered (and optionally Z-scored) arrays the
        caller would otherwise pass in `data_dict`.

    Returns
    -------
    list[Block]
        Estimated parent blocks. Order is meaningless (a set semantically).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if env_stats is None:
        env_stats = compute_env_stats(data_dict)

    # pre-compute invariants for the loop.
    # block_idx and S_jj are constant across all probes (the target block
    # never changes); idx_cache avoids re-sorting each candidate's frozenset.
    block_idx = _block_indices(block)
    idx_cache: dict[Block, np.ndarray] = {block: block_idx}
    idx_cache.update({b: _block_indices(b) for b in candidate_pool})
    S_jj_cache = {
        k: s.sigma[np.ix_(block_idx, block_idx)]
        for k, s in env_stats.items()
    }

    def _score(parents: list[Block]) -> float:
        return pooled_block_bic_from_sigma(
            block, parents, env_stats, lambda_pen,
            block_idx=block_idx, idx_cache=idx_cache, S_jj_cache=S_jj_cache,
        )

    P: list[Block] = []
    non_parents: list[Block] = list(candidate_pool)
    current_bic = _score(P)

    # --- Grow phase --------------------------------------------------------
    while True:
        done = True
        for pi_k in _shuffled(non_parents, rng):
            candidate_bic = _score(P + [pi_k])
            if candidate_bic > current_bic:
                P.append(pi_k)
                non_parents.remove(pi_k)
                current_bic = candidate_bic
                done = False
        if done:
            break

    # --- Shrink phase ------------------------------------------------------
    while True:
        done = True
        for pi_k in _shuffled(P, rng):
            P_minus = [p for p in P if p != pi_k]
            candidate_bic = _score(P_minus)
            if candidate_bic > current_bic:
                P = P_minus
                current_bic = candidate_bic
                done = False
        if done:
            break

    return P


__all__ = ["grow_shrink"]
