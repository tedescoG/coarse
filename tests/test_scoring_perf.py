"""Microbenchmark for the Tier-1 scoring refactor.

Compares `pooled_block_bic` (old recompute path — now an internal wrapper
that builds `_EnvStats` per call) against `_pooled_block_bic_from_sigma`
(cached path that reuses pre-built `_EnvStats` across probes). Same
mathematical result; the difference is whether the per-env `(X.T @ X) / n`
is computed once outside the hot loop or once per call inside it.

The test samples data ONCE (sempler-free; pure numpy via the chain helper)
and runs K probes through each path with a warm-up. Soft-asserts a 1.67x
speedup floor — the microbench-on-laptop measurement was 3.5x-21x across
scenarios, so 1.67x leaves ~2x safety margin for CI hardware variance.
"""

from __future__ import annotations

import time

import numpy as np

from coarse.scoring import (
    compute_env_stats,
    pooled_block_bic_from_sigma,
    pooled_block_bic,
)


def _sample_chain(n: int, rng: np.random.Generator, shift: tuple[int, ...] = ()):
    """Local copy of `tests/test_coarse.py::_sample_chain_dataset` to avoid
    cross-test imports. Linear-Gaussian chain DAG: {v0,v1} -> {v2,v3} -> {v4,v5}."""
    s = np.zeros(6)
    for i in shift:
        s[i] = 2.0
    v0 = rng.standard_normal(n) + s[0]
    v1 = rng.standard_normal(n) + s[1]
    v2 = 0.8 * v0 + 0.5 * v1 + 0.3 * rng.standard_normal(n) + s[2]
    v3 = -0.7 * v0 + 0.6 * v1 + 0.3 * rng.standard_normal(n) + s[3]
    v4 = 0.9 * v2 + 0.4 * v3 + 0.3 * rng.standard_normal(n) + s[4]
    v5 = -0.6 * v2 + 0.8 * v3 + 0.3 * rng.standard_normal(n) + s[5]
    return np.column_stack([v0, v1, v2, v3, v4, v5])


def test_pooled_block_bic_from_sigma_is_faster():
    """Tier-1 microbench. Sample ONCE, time both paths over K reps with
    warm-up. Print the per-call latency and the speedup. Soft-assert >1.67x."""
    rng = np.random.default_rng(2026)
    n_per_env = 1500
    data_dict = {
        f"e{i}": _sample_chain(n_per_env, rng, shift=(i % 6,) if i else ())
        for i in range(4)
    }
    centered = {k: v - v.mean(axis=0, keepdims=True) for k, v in data_dict.items()}
    env_stats = compute_env_stats(centered)

    block = frozenset({4, 5})
    parents = [frozenset({0, 1}), frozenset({2, 3})]
    K = 200

    # Warm-up — first call to each path eats numpy/scipy cold-start overhead.
    for _ in range(5):
        pooled_block_bic(block, parents, centered, lambda_pen=1.0)
        pooled_block_bic_from_sigma(block, parents, env_stats, 1.0)

    t0 = time.perf_counter()
    for _ in range(K):
        pooled_block_bic(block, parents, centered, lambda_pen=1.0)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(K):
        pooled_block_bic_from_sigma(block, parents, env_stats, 1.0)
    t_new = time.perf_counter() - t0

    ratio = t_old / t_new
    print(
        f"\n[T1 microbench] K={K} envs={len(centered)} "
        f"n_per_env={n_per_env} block_size=2 parents=2x2"
    )
    print(
        f"  pooled_block_bic (recompute):       "
        f"{t_old * 1000:>8.2f} ms total | {t_old / K * 1e6:>7.1f} us/call"
    )
    print(
        f"  _pooled_block_bic_from_sigma:       "
        f"{t_new * 1000:>8.2f} ms total | {t_new / K * 1e6:>7.1f} us/call"
    )
    print(f"  speedup:                            {ratio:>8.2f}x")
    # Sanity: values must agree (this is the correctness side-channel of the
    # timing test) — but at rtol=1e-10 they always will.
    a = pooled_block_bic(block, parents, centered, lambda_pen=1.0)
    b = pooled_block_bic_from_sigma(block, parents, env_stats, 1.0)
    np.testing.assert_allclose(a, b, rtol=1e-10)
    assert ratio > 1.67, (
        f"expected >=1.67x speedup, got {ratio:.2f}x -- "
        "likely a regression or hardware-shared CI noise"
    )
