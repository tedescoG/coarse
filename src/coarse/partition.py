"""Partition refinement, supports supp(π) and candidate parent pools Pa⋆(π) from M."""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from coarse.types import Block


def _split_once(part: Block, M: np.ndarray) -> tuple[Block, Block] | None:
    """Split `part` by M-row equality with its smallest member; None if it doesn't split."""
    if len(part) < 2:
        return None
    u = min(part)
    M_u = M[u, :]
    pi_a_members = [v for v in part if np.array_equal(M[v, :], M_u)]
    pi_a = frozenset(pi_a_members)
    pi_b = part - pi_a
    if not pi_b:
        return None
    return pi_a, pi_b


def infer_partition(M: np.ndarray) -> list[Block]:
    """Row-class partition of M, sorted by (|supp|, min): a linear extension of ⊆ on supports."""
    M = np.asarray(M, dtype=bool)
    p = M.shape[0]
    if p == 0:
        return []
    queue: deque[Block] = deque([frozenset(range(p))])
    final: list[Block] = []
    while queue:
        part = queue.popleft()
        split = _split_once(part, M)
        if split is None:
            final.append(part)
        else:
            queue.extend(split)
    return sorted(final, key=lambda b: (int(M[min(b), :].sum()), min(b)))


def compute_supports(
    M: np.ndarray, partition: list[Block]
) -> dict[Block, frozenset[int]]:
    """supp(π) = {e : M[v, e]} for any v ∈ π (rows within a block are equal)."""
    M = np.asarray(M, dtype=bool)
    return {
        block: frozenset(int(e) for e in np.flatnonzero(M[min(block), :]))
        for block in partition
    }


def compute_candidate_pools(
    supports: Mapping[Block, frozenset[int]],
) -> dict[Block, list[Block]]:
    """Pa⋆(π) = {π' ≠ π : supp(π') ⊆ supp(π)}, in the order of `supports`."""
    blocks = list(supports.keys())
    pools: dict[Block, list[Block]] = {}
    for pi in blocks:
        pi_supp = supports[pi]
        candidates = [
            other for other in blocks if other != pi and supports[other] <= pi_supp
        ]
        pools[pi] = candidates
    return pools


__all__ = [
    "compute_candidate_pools",
    "compute_supports",
    "infer_partition",
]
