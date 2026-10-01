"""Score-based grow-shrink parent search for one block."""

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
    """`items` in random order."""
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
    """Parent blocks of `block` from `candidate_pool`: add, then remove, candidates while
    the pooled BIC strictly improves, in a random order each sweep. `data_dict` (centered)
    is ignored when `env_stats` is given."""
    if rng is None:
        rng = np.random.default_rng(0)
    if env_stats is None:
        env_stats = compute_env_stats(data_dict)

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

    # Grow.
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

    # Shrink.
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
