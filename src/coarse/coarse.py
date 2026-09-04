"""COARSE and COARSEOracle classes.

- COARSE: main class implementation of the algorithm. Estimate partition and coarse DAG from interventional data.
- COARSEOracle: used for experiment on known partition. Recover edges by using the block level score-based grow-shrink.
"""

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
    k: int,
    baseline_key: EnvKey,
) -> tuple[dict[EnvKey, np.ndarray], list[Block], dict[Block, Block]]:
    """Per-block PCA projection for kPC-COARSE.

    For each block π_j, computes the top-k_j right singular vectors from the
    observational environment, then projects every environment's
    block data onto that basis.  Returns projected env arrays
    (n_e, K), the remapped partition, and the original→projected block map.
    """
    offset = 0
    proj_partition: list[Block] = []
    block_map: dict[Block, Block] = {}
    # (column indices into the *original* p-dimensional arrays, V_j matrix)
    projections: list[tuple[np.ndarray, np.ndarray]] = []

    for block in partition:
        orig_idx = np.asarray(sorted(block), dtype=np.int64)
        r_j = len(orig_idx)
        k_j = min(k, r_j)

        ref = env_arrays[baseline_key][:, orig_idx]

        _, _, Vt = np.linalg.svd(ref, full_matrices=False)
        # SVD returns only min(n_ref, r_j) right singular vectors when the
        # reference matrix has fewer rows than columns. Clamping k_j here
        # keeps proj_block, offset, and V_j.shape[1] in lock-step.
        k_j = min(k_j, Vt.shape[0])
        V_j = Vt[:k_j, :].T  # (r_j, k_j)

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
    """Coerce every entry to a contiguous float64 (n_e, p) ndarray and
    validate that the baseline env is present.

    Each env value may be a bare 2-D array, a ``(data, targets, type)`` tuple,
    or a ``{"data": ...}`` dict. Intervention targets and intervention-type
    strings are accepted-and-ignored: COARSE reads only the data array.
    """
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


def _materialize_dag(
    partition: list[Block], parent_sets: dict[Block, list[Block]]
) -> nx.DiGraph:
    """Build a directed graph where each node is `tuple(sorted(block))` and
    each edge corresponds to a learned parent → child relationship.
    """
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
) -> tuple[
    dict[Block, frozenset[int]],
    dict[Block, list[Block]],
    list[Block],
    dict[Block, list[Block]],
    float,
]:
    """Derive signature + grow-shrink loop for both COARSE and
    COARSEOracle.

    Computes supports, candidate pools, τ, parent sets, and the
    total score Σ_j BIC_j(π_j, P̂a_j).

    Centers each env's data once per fit before scoring: `block_residual_covariance`
    (and everything downstream of it) assumes column-centered input.

    When ``k`` is not None, additionally Z-scores each env's columns:
    without it, high-variance variables dominate the singular vectors.
    """
    centered_env_arrays: dict[EnvKey, np.ndarray] = {
        ek: v - v.mean(axis=0, keepdims=True) for ek, v in env_arrays.items()
    }
    if k is not None:
        for ek, v in centered_env_arrays.items():
            sigma = v.std(axis=0, keepdims=True)
            sigma = np.where(sigma > 0.0, sigma, 1.0)
            centered_env_arrays[ek] = v / sigma

    # Supports & candidate pools always from original partition + M.
    supports = compute_supports(M, partition)
    candidate_pools = compute_candidate_pools(supports)
    # infer_partition already sorts blocks by (|supp|, min)
    # so `list(partition)` is a valid τ ∈ T(≤_supp).
    # See partition.py
    tau = list(partition)

    # --- kPC-COARSE: per-block PCA projection ---------------------------------
    if k is not None:
        score_data, score_partition, block_map = _project_blocks(
            partition, centered_env_arrays, k, baseline_key,
        )
        rev_map = {v: orig for orig, v in block_map.items()}
        score_pools: dict[Block, list[Block]] = {
            block_map[b]: [block_map[c] for c in candidate_pools[b]]
            for b in partition
        }
        score_tau = [block_map[b] for b in tau]
    else:
        score_data = centered_env_arrays
        score_pools = candidate_pools
        score_tau = tau
        block_map = rev_map = None  # type: ignore[assignment]

    # Precompute and cache per-env Σ̂^e.
    # Avoid re-computing the covariance matrix for each evaluation.
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

    # Remap parent sets back to original blocks for DAG assembly.
    if block_map is not None:
        parent_sets = {
            rev_map[pi_j]: [rev_map[p] for p in parents]
            for pi_j, parents in parent_sets_scored.items()
        }
    else:
        parent_sets = parent_sets_scored

    return supports, candidate_pools, tau, parent_sets, float(total_score)


class COARSE:
    """Block-level score-based causal discovery from soft interventions.

    Attributes set by `.fit`
    ------------------------
    partition : list[Block]                        — Π_E
    M : ndarray (bool)                             — interventional descendant matrix
    env_order : list[EnvKey]                       — column order of M
    supports : dict[Block, frozenset[int]]         — supp(π) per block
    candidate_pools : dict[Block, list[Block]]     — Pa⋆(π) per block
    linear_extension_ : list[Block]                — τ ∈ T(≤_supp)
    parent_sets : dict[Block, list[Block]]         — P̂a_j per block
    dag : nx.DiGraph                               — nodes = tuple(sorted(block))
    score : float                                  — total Σ_j BIC_j
    num_features : int                             — feature count p
    fit_metadata : dict[str, Any]
    fit_runtime_sec : float
    """

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
    ) -> "COARSE":
        start = time.perf_counter()

        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        # Two independent sub-streams so the partition step and the score step
        # can't influence each other through shared RNG state (e.g., when
        # refine_test='energy' uses resampling).
        rng_M, rng_gs = self.rng.spawn(2)

        # DescendantTest: compute the matrix M by marginal two-sample tests
        M, env_order = compute_M(
            env_arrays,
            alpha=alpha,
            test_name=refine_test,
            rng=rng_M,
            baseline_key=baseline_key,
        )

        # RefineSplit: Refine partition by splitting over different supports
        partition = infer_partition(M)

        # Compute support, candidate parent pool and support ordering (linear extension)
        supports, candidate_pools, tau, parent_sets, score = _run_score_phase(
            partition, M, env_arrays, lambda_pen, rng_gs,
            k=k, baseline_key=baseline_key,
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
            "num_parts": len(partition),
            "num_edges": dag.number_of_edges(),
            "score": score,
        }
        self.fit_runtime_sec = time.perf_counter() - start
        return self

    def expand_coarsened_dag(self, fully_connected: bool = False) -> np.ndarray:
        """Project the coarse partition DAG into a full p × p adjacency matrix.
        """
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
    """Run COARSE with both partition and M provided as inputs.
    Used for experiments focused solely on edge recovery.
    """

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
    ) -> "COARSEOracle":
        start = time.perf_counter()
        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        M = np.asarray(M, dtype=bool)
        if set(partition) != set(infer_partition(M)):
            raise ValueError("partition must be the row-class partition of M")
        rng_gs = self.rng.spawn(1)[0]
        supports, candidate_pools, tau, parent_sets, score = _run_score_phase(
            partition, M, env_arrays, lambda_pen, rng_gs,
            k=k, baseline_key=baseline_key,
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
            "num_parts": len(partition),
            "num_edges": dag.number_of_edges(),
            "score": score,
        }
        self.fit_runtime_sec = time.perf_counter() - start
        return self

    expand_coarsened_dag = COARSE.expand_coarsened_dag


__all__ = ["COARSE", "COARSEOracle"]
