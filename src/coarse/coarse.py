# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""COARSE (partition and block DAG from data) and COARSEOracle (block DAG given the partition)."""

from __future__ import annotations

import time
from typing import Any

import networkx as nx
import numpy as np

from coarse.growshrink import grow_shrink
from coarse.hypothesis_tests import _normalize_env_data, compute_M
from coarse.partition import (
    compute_candidate_pools,
    compute_supports,
    infer_partition,
)
from coarse.scoring import (
    compute_env_stats,
    pooled_block_bic_from_sigma,
)
from coarse.types import Block, EnvKey


def _project_blocks(
    partition: list[Block],
    env_arrays: dict[EnvKey, np.ndarray],
    k: int | None,
    baseline_key: EnvKey,
) -> tuple[dict[EnvKey, np.ndarray], list[Block], dict[Block, Block]]:
    """Project each block onto its top-k right singular vectors in the baseline env (kPC), or onto
    the block mean when `k=None` (avg)."""
    offset = 0
    proj_partition: list[Block] = []
    block_map: dict[Block, Block] = {}
    projections: list[tuple[np.ndarray, np.ndarray]] = []

    for block in partition:
        orig_idx = np.asarray(sorted(block), dtype=np.int64)
        if k is None:
            V_j = np.full((len(orig_idx), 1), 1.0 / len(orig_idx))
        else:
            _, _, Vt = np.linalg.svd(
                env_arrays[baseline_key][:, orig_idx], full_matrices=False
            )
            # The SVD has only min(n_ref, r_j) vectors.
            V_j = Vt[: min(k, Vt.shape[0]), :].T  # (r_j, k_j)
        k_j = V_j.shape[1]

        proj_block = frozenset(range(offset, offset + k_j))
        proj_partition.append(proj_block)
        block_map[block] = proj_block
        projections.append((orig_idx, V_j))
        offset += k_j

    proj_arrays: dict[EnvKey, np.ndarray] = {}
    for env_key, arr in env_arrays.items():
        parts = [arr[:, idx] @ V_j for idx, V_j in projections]
        proj_arrays[env_key] = np.hstack(parts)

    return proj_arrays, proj_partition, block_map


def _normalize_data_dict(
    data_dict: dict[EnvKey, Any], baseline_key: EnvKey = "obs"
) -> tuple[dict[EnvKey, np.ndarray], EnvKey]:
    """Float64 (n_e, p) arrays. An env is an array, a `(data, targets, type)` tuple
    or a `{"data": ...}` dict; only the data is used."""
    if baseline_key not in data_dict:
        raise ValueError(
            f"data_dict must contain a baseline env keyed {baseline_key!r}"
        )
    out: dict[EnvKey, np.ndarray] = {}
    p_ref: int | None = None
    for key, value in data_dict.items():
        arr = _normalize_env_data(value)
        if p_ref is None:
            p_ref = arr.shape[1]
        elif arr.shape[1] != p_ref:
            raise ValueError(
                f"env {key!r} has {arr.shape[1]} columns; expected {p_ref}"
            )
        out[key] = arr
    return out, baseline_key


def _check_data_dict(data_dict: dict[EnvKey, Any]) -> None:
    """Input validation, run before the timed fit: every env has at least 2 finite rows and
    non-degenerate columns."""
    for key, value in data_dict.items():
        arr = _normalize_env_data(value)
        if arr.shape[0] < 2:
            raise ValueError(f"env {key!r} needs at least 2 rows, got {arr.shape[0]}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"env {key!r} contains NaN or inf")
        _check_columns_non_degenerate(key, arr)


def _check_columns_non_degenerate(key: EnvKey, arr: np.ndarray) -> None:
    """Reject constant, identical or (when n > p) linearly dependent columns."""
    constant = np.flatnonzero(np.ptp(arr, axis=0) == 0.0)
    if constant.size:
        raise ValueError(f"env {key!r}: column(s) {constant.tolist()} are constant")
    p = arr.shape[1]
    if p < 2:
        return
    order = np.lexsort(arr[::-1])
    cols = arr[:, order]
    dup = np.flatnonzero(np.all(cols[:, 1:] == cols[:, :-1], axis=0))
    if dup.size:
        i, j = sorted((int(order[dup[0]]), int(order[dup[0] + 1])))
        raise ValueError(f"env {key!r}: columns {i} and {j} are identical")
    n = arr.shape[0]
    if n > p:
        # Default tolerance: only exact dependence fails. n <= p is left to the -inf path.
        centered = arr - arr.mean(axis=0, keepdims=True)
        rank = int(np.linalg.matrix_rank(centered))
        if rank < p:
            raise ValueError(
                f"env {key!r}: columns are linearly dependent (rank {rank} < {p})"
            )


def _materialize_dag(
    partition: list[Block], parent_sets: dict[Block, list[Block]]
) -> nx.DiGraph:
    """Block DAG with `tuple(sorted(block))` nodes."""
    dag = nx.DiGraph()
    for block in partition:
        dag.add_node(tuple(sorted(block)))
    for child, parents in parent_sets.items():
        child_node = tuple(sorted(child))
        for parent in parents:
            dag.add_edge(tuple(sorted(parent)), child_node)
    return dag


def _run_score_phase(
    partition: list[Block],
    M: np.ndarray,
    env_arrays: dict[EnvKey, np.ndarray],
    lambda_pen: float,
    rng: np.random.Generator,
    k: int | None = None,
    baseline_key: EnvKey = "obs",
    avg: bool = False,
) -> tuple[
    dict[Block, frozenset[int]],
    dict[Block, list[Block]],
    list[Block],
    dict[Block, list[Block]],
    float,
]:
    """Supports, candidate pools, τ, parent sets and total BIC, shared by both classes.
    Centers each env here, once; kPC (`k` set) and `avg` also scale by the baseline std."""
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1
    ):
        raise ValueError(f"k must be a positive integer, got {k!r}")
    if avg and k is not None:
        raise ValueError("avg and k are mutually exclusive")
    if lambda_pen < 0:
        raise ValueError(f"lambda_pen must be >= 0, got {lambda_pen!r}")
    reduce = k is not None or avg
    centered_env_arrays: dict[EnvKey, np.ndarray] = {
        ek: v - v.mean(axis=0, keepdims=True) for ek, v in env_arrays.items()
    }
    if reduce:
        # obs_std > 0 assumed: constant columns are rejected by `check_input`.
        obs_std = centered_env_arrays[baseline_key].std(axis=0, keepdims=True)
        centered_env_arrays = {ek: v / obs_std for ek, v in centered_env_arrays.items()}

    supports = compute_supports(M, partition)
    candidate_pools = compute_candidate_pools(supports)
    tau = list(partition)
    # Blocks may only inherit from τ-predecessors. For canonical partitions this is a
    # no-op: distinct row classes have distinct supports, so every strictly-smaller
    # candidate is τ-earlier already. For finer partitions it gives same-signature
    # siblings a consistent direction, keeping the block DAG acyclic.
    position = {block: i for i, block in enumerate(tau)}
    candidate_pools = {
        block: [c for c in pool if position[c] < position[block]]
        for block, pool in candidate_pools.items()
    }

    if reduce:
        score_data, score_partition, block_map = _project_blocks(
            partition,
            centered_env_arrays,
            k,
            baseline_key,
        )
        rev_map = {v: orig for orig, v in block_map.items()}
        score_pools: dict[Block, list[Block]] = {
            block_map[b]: [block_map[c] for c in candidate_pools[b]] for b in partition
        }
        score_tau = [block_map[b] for b in tau]
    else:
        score_data = centered_env_arrays
        score_pools = candidate_pools
        score_tau = tau
        block_map = rev_map = None  # type: ignore[assignment]

    env_stats = compute_env_stats(score_data)

    parent_sets_scored: dict[Block, list[Block]] = {}
    for pi_j in score_tau:
        parent_sets_scored[pi_j] = grow_shrink(
            pi_j,
            score_pools[pi_j],
            score_data,
            lambda_pen=lambda_pen,
            rng=rng,
            env_stats=env_stats,
        )
    total_score = sum(
        pooled_block_bic_from_sigma(
            pi_j, parent_sets_scored[pi_j], env_stats, lambda_pen
        )
        for pi_j in score_tau
    )

    if block_map is not None:
        parent_sets = {
            rev_map[pi_j]: [rev_map[p] for p in parents]
            for pi_j, parents in parent_sets_scored.items()
        }
    else:
        parent_sets = parent_sets_scored

    return supports, candidate_pools, tau, parent_sets, float(total_score)


class COARSE:
    """Partition and block DAG from soft interventions. `fit` sets `partition`, `M`, `env_order`,
    `parent_sets`, `dag` (nodes `tuple(sorted(block))`), `score` and `fit_runtime_sec`."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(
        self,
        data_dict: dict[EnvKey, Any],
        alpha: float = 1e-4,
        lambda_pen: float = 1.0,
        refine_test: str = "welch",
        baseline_key: EnvKey = "obs",
        k: int | None = None,
        check_input: bool = False,
        avg: bool = False,
    ) -> "COARSE":
        check_start = time.perf_counter()
        if check_input:
            _check_data_dict(data_dict)
        start = time.perf_counter()

        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        rng_M, rng_gs = self.rng.spawn(2)

        M, env_order = compute_M(
            env_arrays,
            alpha=alpha,
            test_name=refine_test,
            rng=rng_M,
            baseline_key=baseline_key,
        )

        partition = infer_partition(M)

        supports, candidate_pools, tau, parent_sets, score = _run_score_phase(
            partition,
            M,
            env_arrays,
            lambda_pen,
            rng_gs,
            k=k,
            baseline_key=baseline_key,
            avg=avg,
        )

        dag = _materialize_dag(partition, parent_sets)

        self.partition = partition
        self.M = M
        self.env_order = env_order
        self.supports = supports
        self.candidate_pools = candidate_pools
        self.linear_extension_ = tau
        self.parent_sets = parent_sets
        self.dag = dag
        self.score = score
        self.num_features = env_arrays[baseline_key].shape[1]
        self.fit_metadata = {
            "alpha": alpha,
            "lambda_pen": lambda_pen,
            "refine_test": refine_test,
            "k": k,
            "avg": avg,
            "num_parts": len(partition),
            "num_edges": dag.number_of_edges(),
            "score": score,
        }
        self.check_runtime_sec = start - check_start
        self.fit_runtime_sec = time.perf_counter() - start
        return self

    def expand_coarsened_dag(self, fully_connected: bool = False) -> np.ndarray:
        """p × p adjacency of the block DAG; `fully_connected` also links each block internally."""
        num_features = self.num_features
        adjacency = np.zeros((num_features, num_features), dtype=int)

        if fully_connected:
            for part in self.dag.nodes:
                ordered = sorted(part)
                for idx, src in enumerate(ordered):
                    for dst in ordered[idx + 1 :]:
                        adjacency[src, dst] = 1

        for src_part, dst_part in self.dag.edges:
            for src in src_part:
                for dst in dst_part:
                    adjacency[src, dst] = 1

        return adjacency


class COARSEOracle:
    """COARSE's score step with the partition and M given."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(
        self,
        partition: list[Block],
        M: np.ndarray,
        env_order: list[EnvKey],
        data_dict: dict[EnvKey, Any],
        lambda_pen: float = 1.0,
        baseline_key: EnvKey = "obs",
        k: int | None = None,
        check_input: bool = False,
        avg: bool = False,
    ) -> "COARSEOracle":
        check_start = time.perf_counter()
        if check_input:
            _check_data_dict(data_dict)
        start = time.perf_counter()
        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        M = np.asarray(M, dtype=bool)
        p = env_arrays[baseline_key].shape[1]
        if M.ndim != 2 or M.shape[0] != p:
            raise ValueError(
                f"M has shape {M.shape}; expected ({p}, |E|-1) for {p} variables"
            )
        if len(env_order) != M.shape[1]:
            raise ValueError(
                f"env_order has {len(env_order)} entries; M has {M.shape[1]} columns"
            )
        canonical = infer_partition(M)
        if set(partition) == set(canonical):
            partition = canonical
        else:
            # A refinement of the row classes is legitimate score-phase input: a
            # size-capped merge may split a signature class into several blocks with
            # identical supports (which can never be ordered by signature, so τ and
            # the pool filter decide direction). Refinements must still be disjoint,
            # cover every variable, and carry a constant signature per block.
            partition = [frozenset(blk) for blk in partition]
            seen: set[int] = set()
            for blk in partition:
                rows = sorted(blk)
                if not rows or seen.intersection(rows):
                    raise ValueError(
                        "partition blocks must be nonempty, disjoint, and cover every variable"
                    )
                seen.update(rows)
                block_M = M[rows]
                if not bool((block_M == block_M[0]).all()):
                    raise ValueError("every block must have a constant signature in M")
            if seen != set(range(p)):
                raise ValueError(
                    "partition blocks must be nonempty, disjoint, and cover every variable"
                )
            # Canonical order, so τ ignores the caller's order.
            partition = sorted(
                partition,
                key=lambda blk: (int(M[min(blk), :].sum()), min(blk)),
            )
        rng_gs = self.rng.spawn(1)[0]
        supports, candidate_pools, tau, parent_sets, score = _run_score_phase(
            partition,
            M,
            env_arrays,
            lambda_pen,
            rng_gs,
            k=k,
            baseline_key=baseline_key,
            avg=avg,
        )
        dag = _materialize_dag(partition, parent_sets)

        self.partition = partition
        self.M = M
        self.env_order = list(env_order)
        self.supports = supports
        self.candidate_pools = candidate_pools
        self.linear_extension_ = tau
        self.parent_sets = parent_sets
        self.dag = dag
        self.score = score
        self.num_features = env_arrays[baseline_key].shape[1]
        self.fit_metadata = {
            "lambda_pen": lambda_pen,
            "k": k,
            "avg": avg,
            "num_parts": len(partition),
            "num_edges": dag.number_of_edges(),
            "score": score,
        }
        self.check_runtime_sec = start - check_start
        self.fit_runtime_sec = time.perf_counter() - start
        return self

    expand_coarsened_dag = COARSE.expand_coarsened_dag


__all__ = ["COARSE", "COARSEOracle"]
