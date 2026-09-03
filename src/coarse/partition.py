"""Compute partitions, support and candidate parents pool from M.

`infer_partition` returns the row-class partition Π_E of M .
`compute_supports` reads supp(π) from Definition 10 .
`compute_candidate_pools` builds Pa⋆(π) from Corollary 1 .
`linear_extension` (now in tests/) returns a τ ∈ T(≤_supp) .
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from coarse.types import Block


def _split_once(part: Block, M: np.ndarray) -> tuple[Block, Block] | None:
    """One iteration of Algorithm 2's `RefineTest` for a single block.

    Picks u = min(part) as the deterministic tie-break — Proposition 2 of the
    draft guarantees the eventual partition is unique regardless of the choice.
    Returns (π_a, π_b) if a split occurs, else None.
    """
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
    """Algorithm 2 — row-class refinement starting from the trivial partition {V}.

    Returns the partition as a list of frozenset[int]. The list is sorted by
    (|supp|, min) so downstream callers get a deterministic order, but the
    partition itself is unordered as a mathematical object.
    """
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
    # Deterministic ordering: by support cardinality, then min element.
    # (Same ordering linear_extension would produce; cheap to compute here too.)
    return sorted(final, key=lambda b: (int(M[min(b), :].sum()), min(b)))


def compute_supports(
    M: np.ndarray, partition: list[Block]
) -> dict[Block, frozenset[int]]:
    """Definition 10 — supp(π) = {e : M[v, e] == 1} for any v ∈ π.

    The choice of v is well-defined because all rows of M restricted to π are
    identical by construction of `infer_partition`. We pick min(π).
    """
    M = np.asarray(M, dtype=bool)
    return {
        block: frozenset(int(e) for e in np.flatnonzero(M[min(block), :]))
        for block in partition
    }


def compute_candidate_pools(
    supports: Mapping[Block, frozenset[int]],
) -> dict[Block, list[Block]]:
    """Corollary 1 (p. 12) — Pa⋆(π) = {π' ∈ Π \\ {π} : supp(π') ⊆ supp(π)}.

    Distinct partition blocks have distinct supports, so on real inputs ⊆ and ⊊ coincide.
    """
    blocks = list(supports.keys())
    pools: dict[Block, list[Block]] = {}
    for pi in blocks:
        pi_supp = supports[pi]
        candidates = [
            other for other in blocks if other != pi and supports[other] <= pi_supp
        ]
        # Sorted for deterministic iteration in grow-shrink
        pools[pi] = sorted(candidates, key=lambda b: (len(supports[b]), min(b)))
    return pools


__all__ = [
    "compute_candidate_pools",
    "compute_supports",
    "infer_partition",
]
