"""Tests for the COARSE package.

Each test maps to a specific Algorithm or section of the thesis draft so the
implementation can be verified against the math piece-by-piece. The sempler
integration test mirrors `repare-0.2.0/tests/test_repare.py::test_intervention`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg as sla

from coarse.hypothesis_tests import compute_M
from coarse.partition import (
    compute_candidate_pools,
    compute_supports,
    infer_partition,
)
from coarse.coarse import COARSE, COARSEOracle
from coarse.growshrink import grow_shrink
from coarse.scoring import (
    parameter_count_d_j,
    pooled_block_bic,
)
import networkx as nx

from conftest import sample_chain_dataset


# ---------------------------------------------------------------------------
# Reference implementations moved from scoring.py / partition.py.
# These operate on raw (n_e, p) arrays and serve as correctness oracles for
# the production cached-covariance path.
# ---------------------------------------------------------------------------
def linear_extension(supports):
    """Definition 11 — sort blocks by |supp| ascending (ties by min(block))."""
    return sorted(supports.keys(), key=lambda b: (len(supports[b]), min(b)))


def block_residual_covariance(X_block, X_parents):
    """Equation 11 — block residual covariance from raw arrays."""
    n_e = X_block.shape[0]
    S_jj = (X_block.T @ X_block) / n_e
    if X_parents.size == 0 or X_parents.shape[1] == 0:
        return S_jj
    S_jPa = (X_block.T @ X_parents) / n_e
    S_PaPa = (X_parents.T @ X_parents) / n_e
    c, low = sla.cho_factor(S_PaPa, lower=True, check_finite=False)
    Y = sla.cho_solve((c, low), S_jPa.T, check_finite=False)
    return S_jj - S_jPa @ Y


def block_log_det_residual(X_block, X_parents):
    """log|Sigma_hat_j^e|. Raises LinAlgError if non-PD."""
    Sigma = block_residual_covariance(X_block, X_parents)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(logdet):
        raise sla.LinAlgError(
            f"residual covariance has non-positive determinant (sign={sign})"
        )
    return float(logdet)


def block_bic_env(X_block, X_parents, lambda_pen=1.0):
    """Per-environment BIC for one block (Equations 12 + 21)."""
    n_e = X_block.shape[0]
    r_j = X_block.shape[1]
    s_j = X_parents.shape[1] if X_parents.size else 0
    if n_e <= 0 or r_j <= 0:
        return -np.inf
    if n_e <= s_j + r_j:
        return -np.inf
    try:
        logdet = block_log_det_residual(X_block, X_parents)
    except sla.LinAlgError:
        return -np.inf
    log_lik = -0.5 * n_e * logdet
    d_j = parameter_count_d_j(r_j, s_j)
    return 2.0 * log_lik - lambda_pen * np.log(n_e) * d_j


# ---------------------------------------------------------------------------
# Test 1 — Algorithm 1 (RefineAux), draft p. 10
# ---------------------------------------------------------------------------
def test_algorithm_1_refineaux_population():
    """Welch must catch mean shifts but miss pure variance shifts;
    KS must catch both."""
    rng = np.random.default_rng(0)
    n = 1000

    # Three variables, three envs (one baseline + two interventional)
    # env 'obs': all N(0, 1)
    # env '1':   var0 mean-shifted to N(2, 1); var1, var2 unchanged
    # env '2':   var1 variance-inflated to N(0, sigma=2); var0, var2 unchanged
    obs = rng.standard_normal((n, 3))

    env1 = rng.standard_normal((n, 3))
    env1[:, 0] += 2.0  # mean shift on v0

    env2 = rng.standard_normal((n, 3))
    env2[:, 1] *= 2.0  # variance inflation on v1 (mean stays 0)

    data_dict = {"obs": obs, "1": env1, "2": env2}

    # alpha=1e-4 keeps the test robust against ~1%-tail random fluctuations at
    # n=1000 while leaving both the mean shift (~44 sigma_mean) and the variance
    # change (population KS distance ~0.15 vs threshold ~0.07 at this n) well
    # above their respective rejection thresholds.
    M_welch, env_order = compute_M(data_dict, alpha=1e-4, test_name="welch")
    assert env_order == ["1", "2"]
    # Welch — mean shifts only
    assert M_welch[0, 0]                   # v0 in env 1: mean shift detected
    assert not M_welch[1, 0]               # v1 unchanged in env 1
    assert not M_welch[2, 0]
    assert not M_welch[0, 1]               # v0 unchanged in env 2
    assert not M_welch[1, 1]               # v1 variance-only shift: Welch misses
    assert not M_welch[2, 1]

    M_ks, _ = compute_M(data_dict, alpha=1e-4, test_name="ks")
    # KS — detects both mean and variance shifts
    assert M_ks[0, 0]                      # v0 in env 1
    assert not M_ks[1, 0]
    assert not M_ks[2, 0]
    assert not M_ks[0, 1]
    assert M_ks[1, 1]                      # KS catches the variance change Welch missed
    assert not M_ks[2, 1]


def test_compute_M_requires_obs_baseline():
    """Mirrors RePaRe's contract (repare.py:84): 'obs' is mandatory."""
    with pytest.raises(ValueError, match="baseline"):
        compute_M({"1": np.zeros((10, 2))}, alpha=0.05)


def test_compute_M_rejects_inconsistent_p():
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": rng.standard_normal((20, 3)),
        "1": rng.standard_normal((20, 4)),  # wrong column count
    }
    with pytest.raises(ValueError, match="columns"):
        compute_M(data_dict, alpha=0.05)


def test_ttest_alias_matches_welch():
    """The 'ttest' alias should produce identical M to 'welch'."""
    rng = np.random.default_rng(1)
    data_dict = {
        "obs": rng.standard_normal((300, 2)),
        "1": rng.standard_normal((300, 2)) + np.array([1.0, 0.0]),
    }
    M_welch, _ = compute_M(data_dict, alpha=0.05, test_name="welch")
    M_alias, _ = compute_M(data_dict, alpha=0.05, test_name="ttest")
    assert np.array_equal(M_welch, M_alias)


# ---------------------------------------------------------------------------
# Test 2 — Algorithm 2 (RefineTest), draft p. 11
# ---------------------------------------------------------------------------
_M_HANDCODED = np.array(
    [
        [1, 0, 1],  # v0 — row class A
        [1, 0, 1],  # v1 — row class A
        [0, 1, 1],  # v2 — row class B
        [0, 0, 0],  # v3 — row class C
    ],
    dtype=bool,
)


def test_algorithm_2_refinetest_handcoded_M():
    partition = infer_partition(_M_HANDCODED)
    assert set(partition) == {frozenset({0, 1}), frozenset({2}), frozenset({3})}


def test_infer_partition_permutation_invariance():
    """The recovered partition does not depend on the row ordering of M."""
    rng = np.random.default_rng(0)
    base = infer_partition(_M_HANDCODED)
    base_set = {block for block in base}
    for _ in range(5):
        perm = rng.permutation(_M_HANDCODED.shape[0])
        M_perm = _M_HANDCODED[perm]
        perm_partition = infer_partition(M_perm)
        # Relabel each block back to original indices
        relabeled = {
            frozenset(int(perm[v]) for v in block) for block in perm_partition
        }
        assert relabeled == base_set


def test_compute_supports_handcoded():
    partition = infer_partition(_M_HANDCODED)
    supports = compute_supports(_M_HANDCODED, partition)
    expected = {
        frozenset({0, 1}): frozenset({0, 2}),  # row [1, 0, 1] → cols 0 and 2
        frozenset({2}): frozenset({1, 2}),
        frozenset({3}): frozenset(),
    }
    assert supports == expected


def test_compute_candidate_pools_handcoded():
    partition = infer_partition(_M_HANDCODED)
    supports = compute_supports(_M_HANDCODED, partition)
    pools = compute_candidate_pools(supports)
    # block {3} has empty supp → only blocks with empty supp can be its parents
    # block {0,1} has supp {0,2} → only blocks whose supp ⊆ {0,2}
    # block {2}   has supp {1,2} → only blocks whose supp ⊆ {1,2}
    assert pools[frozenset({3})] == []
    assert pools[frozenset({0, 1})] == [frozenset({3})]
    assert pools[frozenset({2})] == [frozenset({3})]


def test_linear_extension_respects_supp_inclusion():
    """A valid τ must place π_a before π_b whenever supp(π_a) ⊊ supp(π_b)."""
    partition = infer_partition(_M_HANDCODED)
    supports = compute_supports(_M_HANDCODED, partition)
    tau = linear_extension(supports)
    # block {3} (|supp|=0) must come first; the two |supp|=2 blocks come after,
    # in either order.
    assert tau[0] == frozenset({3})
    assert set(tau[1:]) == {frozenset({0, 1}), frozenset({2})}

    # Verify the contract for an arbitrary chain pattern too.
    chain_supports = {
        frozenset({0}): frozenset({1, 2}),
        frozenset({1}): frozenset({1}),
        frozenset({2}): frozenset(),
    }
    chain_tau = linear_extension(chain_supports)
    pos = {b: i for i, b in enumerate(chain_tau)}
    # supp({2}) ⊊ supp({1}) ⊊ supp({0}), so τ must respect this ordering
    assert pos[frozenset({2})] < pos[frozenset({1})] < pos[frozenset({0})]


# ---------------------------------------------------------------------------
# Scoring tests — Equations 11, 12, 20, 21, 23 (draft pp. 16, 17, 21, 23)
# ---------------------------------------------------------------------------
def test_parameter_count_eq_20():
    """d_j = r_j·s_j + r_j(r_j+1)/2 per Equation 20."""
    # r_j=2 block, s_j=3 parents total:
    # regression entries = 2*3 = 6; covariance entries = 2*3/2 = 3 → d_j = 9
    assert parameter_count_d_j(2, 3) == 9
    # Singleton block with no parents: only variance parameter → d_j = 1
    assert parameter_count_d_j(1, 0) == 1
    # Singleton block with 4 parents: d_j = 1*4 + 1*2/2 = 5
    assert parameter_count_d_j(1, 4) == 5


def test_block_residual_covariance_no_parents_is_sample_cov():
    """With s_j = 0 and column-centered input, Σ̂_j^e = X^T X / n_e."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3))
    Xc = X - X.mean(axis=0, keepdims=True)
    Sigma = block_residual_covariance(Xc, np.empty((500, 0)))
    expected = (Xc.T @ Xc) / 500
    np.testing.assert_allclose(Sigma, expected, rtol=1e-10)


def test_sign_convention_true_parents_score_higher():
    """Sanity check: in a known linear-Gaussian setup, BIC of the true parent
    set must exceed BIC of the empty parent set. This is the load-bearing
    sign check — if it's wrong, every grow-shrink decision is inverted."""
    rng = np.random.default_rng(42)
    n = 2000
    X_parent = rng.standard_normal((n, 2))
    # block depends linearly on the two parent columns + noise
    B = np.array([[1.5, -0.7], [0.4, 1.1]])
    noise = 0.3 * rng.standard_normal((n, 2))
    X_block = X_parent @ B.T + noise

    bic_with_parent = block_bic_env(X_block, X_parent, lambda_pen=1.0)
    bic_empty = block_bic_env(X_block, np.empty((n, 0)), lambda_pen=1.0)
    # True parents must score strictly higher (we maximize)
    assert bic_with_parent > bic_empty


def test_pooled_block_bic_sums_envs():
    """Equation 23: BIC_j(π_j, Pa_j) = Σ_e BIC_j^e(π_j, Pa_j)."""
    rng = np.random.default_rng(7)
    n_per_env = 500
    X_pa = rng.standard_normal((2 * n_per_env, 1))
    noise = 0.5 * rng.standard_normal((2 * n_per_env, 1))
    X_block = 2.0 * X_pa + noise

    env0 = np.column_stack([X_block[:n_per_env, 0], X_pa[:n_per_env, 0]])
    env1 = np.column_stack(
        [X_block[n_per_env:, 0] + 1.0, X_pa[n_per_env:, 0] + 1.0]  # mean shifts
    )
    block = frozenset({0})
    parents = [frozenset({1})]
    data_dict = {"obs": env0, "1": env1}

    pooled = pooled_block_bic(block, parents, data_dict, lambda_pen=1.0)
    # Compute per-env BIC manually and sum
    summed = sum(
        block_bic_env(env[:, [0]], env[:, [1]], lambda_pen=1.0)
        for env in data_dict.values()
    )
    np.testing.assert_allclose(pooled, summed, rtol=1e-12)


def test_numerical_stability_near_singular():
    """Test 6: near-rank-deficient parent covariance.

    Two parent blocks that are essentially the same data (with tiny noise) give
    a numerically singular parent-block covariance. block_bic_env /
    pooled_block_bic must either return a finite value (when bare Cholesky
    still succeeds despite the conditioning) or -inf (when Cholesky fails and
    LinAlgError is caught and translated). Either is acceptable; what matters
    is that the call does not crash.
    """
    rng = np.random.default_rng(0)
    n = 200
    X_pa0 = rng.standard_normal((n, 1))
    X_pa1 = X_pa0 + 1e-12 * rng.standard_normal((n, 1))  # near-perfectly collinear
    X_block = X_pa0 + 0.1 * rng.standard_normal((n, 1))

    # Should not raise; result is either finite or -inf
    X_pa = np.column_stack([X_pa0[:, 0], X_pa1[:, 0]])
    bic = block_bic_env(X_block, X_pa)
    assert np.isfinite(bic) or bic == -np.inf

    # Same via pooled API
    data_dict = {"obs": np.column_stack([X_block[:, 0], X_pa0[:, 0], X_pa1[:, 0]])}
    pooled = pooled_block_bic(
        block=frozenset({0}),
        parents=[frozenset({1}), frozenset({2})],
        data_dict=data_dict,
    )
    assert np.isfinite(pooled) or pooled == -np.inf


def test_block_bic_env_undersized_env_returns_minus_inf():
    """Remark 6 (p. 21): if n_e < s_j + r_j the model is ill-posed.
    Should yield -inf rather than crashing."""
    rng = np.random.default_rng(0)
    # n=3 observations, r_j=2, s_j=5 → 3 < 7
    X_block = rng.standard_normal((3, 2))
    X_pa = rng.standard_normal((3, 5))
    bic = block_bic_env(X_block, X_pa, lambda_pen=1.0)
    assert bic == -np.inf


def test_pooled_block_bic_from_sigma_matches_public_path():
    """Tier-1 refactor parity: the cached-covariance path
    (`_pooled_block_bic_from_sigma`) must agree with the original
    direct-array path (`pooled_block_bic`) at `rtol=1e-10` across every
    grow-shrink-relevant branch — no parents, one parent block, two parent
    blocks, singleton target — plus the -inf short-circuit when
    n_e <= s_j + r_j. Catches any slice-vs-resum BLAS divergence."""
    from coarse.scoring import compute_env_stats, pooled_block_bic_from_sigma

    rng = np.random.default_rng(2026)
    data_dict = {
        "obs": sample_chain_dataset(600, rng),
        "1":   sample_chain_dataset(600, rng, shift_targets=(0, 1)),
        "2":   sample_chain_dataset(600, rng, shift_targets=(2, 3)),
    }
    centered = {k: v - v.mean(axis=0, keepdims=True) for k, v in data_dict.items()}
    env_stats = compute_env_stats(centered)

    configs = [
        (frozenset({2, 3}), []),                              # no parents
        (frozenset({2, 3}), [frozenset({0, 1})]),             # one parent block
        (frozenset({4, 5}), [frozenset({0, 1}), frozenset({2, 3})]),
        (frozenset({4}),    [frozenset({0, 1}), frozenset({2, 3})]),
        (frozenset({0}),    []),                              # singleton, no parents
    ]
    for block, parents in configs:
        # pooled_block_bic now ROUTES through the cached path internally
        # (see scoring.py); to test that the wrapper itself is correct we
        # compare against a hand-summed loop of the lower-level
        # `block_bic_env`, which is unchanged.
        expected_summed = sum(
            block_bic_env(
                env[:, sorted(block)],
                (env[:, sorted({i for p in parents for i in p})]
                 if parents else np.empty((env.shape[0], 0), dtype=env.dtype)),
                lambda_pen=1.0,
            )
            for env in centered.values()
        )
        actual = pooled_block_bic_from_sigma(block, parents, env_stats, 1.0)
        np.testing.assert_allclose(
            actual, expected_summed, rtol=1e-10,
            err_msg=f"mismatch at block={block} parents={parents}",
        )

    # -inf propagation: tiny env where n_e < s_j + r_j
    small = {k: v[:3] for k, v in centered.items()}
    small_stats = compute_env_stats(small)
    block = frozenset({0, 1})
    parents = [frozenset({2, 3}), frozenset({4, 5})]   # s_j=4, r_j=2, n=3
    expected = pooled_block_bic(block, parents, small, lambda_pen=1.0)
    actual = pooled_block_bic_from_sigma(block, parents, small_stats, 1.0)
    assert expected == -np.inf and actual == -np.inf


def test_coarse_oracle_score_invariant_under_refactor():
    """Integration guard for the Tier-1 refactor. `COARSEOracle().fit(...).score`
    must remain finite, agree across the two `scale` modes, and the inferred
    DAG must still recover the chain A → B → C — confirming the
    cache-derivation in `_run_score_phase` is consistent for both centered-only
    and centered+Z-scored inputs."""
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(1500, rng),
        "1":   sample_chain_dataset(1500, rng, shift_targets=(0, 1)),
        "2":   sample_chain_dataset(1500, rng, shift_targets=(2, 3)),
        "3":   sample_chain_dataset(1500, rng, shift_targets=(4, 5)),
    }
    partition_true = [frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})]
    M_true = np.array(
        [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]],
        dtype=bool,
    )
    A = tuple(sorted({0, 1}))
    B = tuple(sorted({2, 3}))
    C = tuple(sorted({4, 5}))
    for scale in (False, True):
        m = COARSEOracle().fit(
            partition=partition_true,
            M=M_true,
            env_order=["1", "2", "3"],
            data_dict=data_dict,
            scale=scale,
        )
        assert np.isfinite(m.score), f"score not finite at scale={scale}"
        assert set(m.dag.nodes) == {A, B, C}, f"DAG nodes drift at scale={scale}"
        assert set(m.dag.edges) == {(A, B), (B, C)}, (
            f"DAG edges drift at scale={scale}: got {set(m.dag.edges)}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Algorithm 4 (GrowShrink), draft p. 28
# ---------------------------------------------------------------------------
def test_algorithm_4_growshrink_unit():
    """grow_shrink must return the true parent blocks {{2}, {3}} and reject the
    distractor block {4} that depends on the target only through {2}.

    Setup (linear-Gaussian, single env, n=2000):
        X2 ~ N(0, 1), X3 ~ N(0, 1)              (true parents of target)
        X4 = 1.5*X2 + 0.3*ε                     (correlated with target only via X2)
        target0 = 0.8*X2 + 0.5*X3 + 0.2*ε
        target1 = -0.6*X2 + 0.7*X3 + 0.2*ε

    Conditional on {2}, target ⊥⊥ {4}; conditional on {3}, target ⊥⊥ {4} as well.
    Grow may transiently add {4} depending on shuffle order, but shrink must
    eliminate it once {2, 3} are present.
    """
    rng_data = np.random.default_rng(42)
    n = 2000
    X2 = rng_data.standard_normal(n)
    X3 = rng_data.standard_normal(n)
    X4 = 1.5 * X2 + 0.3 * rng_data.standard_normal(n)
    eps0 = 0.2 * rng_data.standard_normal(n)
    eps1 = 0.2 * rng_data.standard_normal(n)
    X0 = 0.8 * X2 + 0.5 * X3 + eps0
    X1 = -0.6 * X2 + 0.7 * X3 + eps1
    data = np.column_stack([X0, X1, X2, X3, X4])

    block = frozenset({0, 1})
    candidate_pool = [frozenset({2}), frozenset({3}), frozenset({4})]

    # Try multiple seeds to confirm shuffle order doesn't break correctness
    for seed in (0, 1, 7, 123, 2026):
        parents = grow_shrink(
            block,
            candidate_pool,
            {"obs": data},
            lambda_pen=1.0,
            rng=np.random.default_rng(seed),
        )
        assert set(parents) == {frozenset({2}), frozenset({3})}, (
            f"seed={seed}: got {set(parents)}"
        )


def test_growshrink_empty_pool_returns_empty():
    rng_data = np.random.default_rng(0)
    data = rng_data.standard_normal((100, 2))
    parents = grow_shrink(frozenset({0, 1}), [], {"obs": data})
    assert parents == []


def test_growshrink_pooled_across_envs():
    """The grow-shrink decision should consider summed-over-env BIC, not
    per-env. We construct two envs where neither one alone has enough power to
    distinguish the right candidate, but together they do."""
    rng_data = np.random.default_rng(0)
    n = 1000
    X_pa_obs = rng_data.standard_normal(n)
    X_pa_env = rng_data.standard_normal(n) + 1.5  # mean shift
    X_block_obs = 0.9 * X_pa_obs + 0.2 * rng_data.standard_normal(n)
    X_block_env = 0.9 * X_pa_env + 0.2 * rng_data.standard_normal(n)
    distractor_obs = rng_data.standard_normal(n)
    distractor_env = rng_data.standard_normal(n)

    data_obs = np.column_stack([X_block_obs, X_pa_obs, distractor_obs])
    data_env = np.column_stack([X_block_env, X_pa_env, distractor_env])
    data_dict = {"obs": data_obs, "1": data_env}

    parents = grow_shrink(
        block=frozenset({0}),
        candidate_pool=[frozenset({1}), frozenset({2})],
        data_dict=data_dict,
        rng=np.random.default_rng(0),
    )
    assert parents == [frozenset({1})]


# ---------------------------------------------------------------------------
# Test 3 — Algorithm 3 (COARSE driver) end-to-end, oracle inputs
# ---------------------------------------------------------------------------
def test_algorithm_3_coarse_oracle_chain():
    """Three-block chain A → B → C. With M and partition fed as ground truth,
    COARSEOracle must return exactly the two edges A → B and B → C and reject
    the indirect A → C edge that would violate the chain Markov property."""
    rng = np.random.default_rng(0)
    n_per_env = 1500
    data_dict = {
        "obs": sample_chain_dataset(n_per_env, rng),
        "1": sample_chain_dataset(n_per_env, rng, shift_targets=(0, 1)),  # shift A
        "2": sample_chain_dataset(n_per_env, rng, shift_targets=(2, 3)),  # shift B
        "3": sample_chain_dataset(n_per_env, rng, shift_targets=(4, 5)),  # shift C
    }
    partition_true = [
        frozenset({0, 1}),  # A
        frozenset({2, 3}),  # B
        frozenset({4, 5}),  # C
    ]
    # Supports: A intervened in env 1 (and is ancestor of B, C — they shift too)
    # B intervened in env 2 (and is ancestor of C)
    # C intervened only in env 3
    M_true = np.array(
        [
            [1, 0, 0],  # v0 — block A
            [1, 0, 0],  # v1
            [1, 1, 0],  # v2 — block B
            [1, 1, 0],  # v3
            [1, 1, 1],  # v4 — block C
            [1, 1, 1],  # v5
        ],
        dtype=bool,
    )
    model = COARSEOracle().fit(
        partition=partition_true,
        M=M_true,
        env_order=["1", "2", "3"],
        data_dict=data_dict,
    )

    A = tuple(sorted({0, 1}))
    B = tuple(sorted({2, 3}))
    C = tuple(sorted({4, 5}))
    assert set(model.dag.nodes) == {A, B, C}
    assert set(model.dag.edges) == {(A, B), (B, C)}


# ---------------------------------------------------------------------------
# Test 5 — sempler integration, mirrors repare/tests/test_repare.py:73-107
# ---------------------------------------------------------------------------
def test_intervention_sempler():
    """End-to-end smoke test on sempler-generated synthetic data. Uses the
    same parameters as RePaRe's test_intervention so any drift is comparable."""
    from sempler import LGANM
    from sempler.generators import dag_avg_deg, intervention_targets
    from sklearn.metrics import adjusted_rand_score

    seed = 0
    num_nodes = 20
    num_intervs = 5
    density = 0.1
    deg = density * (num_nodes - 1)

    weights = dag_avg_deg(
        num_nodes,
        deg,
        w_min=0.5,
        w_max=2,
        return_ordering=False,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    edge_idcs = np.flatnonzero(weights)
    to_neg = edge_idcs[rng.choice([True, False], len(edge_idcs))]
    weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1

    model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
    targets = intervention_targets(num_nodes, num_intervs, 1, random_state=seed)

    obs_dataset = model.sample(1000)
    data_dict = {"obs": obs_dataset}
    for idx, target in enumerate(targets):
        sample = model.sample(1000, shift_interventions={target[0]: (2, 1)})
        data_dict[str(idx)] = sample

    coarse_model = COARSE().fit(
        data_dict, alpha=1e-4, lambda_pen=1.0, refine_test="welch"
    )

    # --- Sanity asserts -----------------------------------------------------
    # 1. Fit completed
    assert coarse_model.dag.number_of_nodes() >= 1
    assert coarse_model.fit_runtime_sec > 0

    # 2. Partition ARI vs ground truth
    true_dag = nx.DiGraph(weights.astype(bool))
    target_des_masks = []
    for target in targets:
        mask = np.zeros(num_nodes, dtype=bool)
        descendants = nx.descendants(true_dag, target[0])
        mask[list(descendants) + [target[0]]] = True
        target_des_masks.append(mask)
    M_true = np.column_stack(target_des_masks)
    from coarse.partition import infer_partition

    true_partition = infer_partition(M_true)
    true_labels = np.zeros(num_nodes, dtype=int)
    for label, part in enumerate(true_partition):
        true_labels[list(part)] = label
    est_labels = np.zeros(num_nodes, dtype=int)
    for label, part in enumerate(coarse_model.dag.nodes):
        est_labels[list(part)] = label
    ari = adjusted_rand_score(true_labels, est_labels)
    assert ari >= 0.5, f"Partition ARI {ari} below threshold"


# ---------------------------------------------------------------------------
# Test 7 — expand_coarsened_dag (port of RePaRe's test, repare.py:349-375)
# ---------------------------------------------------------------------------
def test_expand_coarsened_dag():
    model = COARSE()
    model.num_features = 6
    model.dag = nx.DiGraph()
    block_a = (3, 0, 2)
    block_b = (1,)
    block_c = (4, 5)
    model.dag.add_node(block_a)
    model.dag.add_node(block_b)
    model.dag.add_node(block_c)
    model.dag.add_edge(block_a, block_c)
    model.dag.add_edge(block_b, block_c)
    adjacency = model.expand_coarsened_dag()

    expected = np.zeros((6, 6), dtype=int)
    for src in block_a:
        for dst in block_c:
            expected[src, dst] = 1
    for src in block_b:
        for dst in block_c:
            expected[src, dst] = 1
    assert np.array_equal(adjacency, expected)


def test_expand_coarsened_dag_fully_connected():
    model = COARSE()
    model.num_features = 5
    model.dag = nx.DiGraph()
    block_a = (0, 1, 2)
    block_b = (3, 4)
    model.dag.add_node(block_a)
    model.dag.add_node(block_b)
    model.dag.add_edge(block_a, block_b)

    adjacency = model.expand_coarsened_dag(fully_connected=True)

    expected = np.zeros((5, 5), dtype=int)
    for src, dst in ((0, 1), (0, 2), (1, 2), (3, 4)):
        expected[src, dst] = 1
    for src in block_a:
        for dst in block_b:
            expected[src, dst] = 1
    assert np.array_equal(adjacency, expected)


# ---------------------------------------------------------------------------
# Test — kPC-COARSE (per-block PCA projection)
# ---------------------------------------------------------------------------

def _chain_oracle_fixtures():
    """Shared data + ground truth for kPC-COARSE oracle tests."""
    rng = np.random.default_rng(42)
    n = 1500
    data_dict = {
        "obs": sample_chain_dataset(n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }
    partition = [frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})]
    M = np.array(
        [[1, 0, 0], [1, 0, 0],
         [1, 1, 0], [1, 1, 0],
         [1, 1, 1], [1, 1, 1]], dtype=bool,
    )
    env_order = ["1", "2", "3"]
    return data_dict, partition, M, env_order


def test_kpc_coarse_backward_compat():
    """k=None must give identical DAG and score to standard COARSE."""
    data_dict, partition, M, env_order = _chain_oracle_fixtures()
    std = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, scale=True,
    )
    pca = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, scale=True, k=None,
    )
    assert set(std.dag.edges) == set(pca.dag.edges)
    np.testing.assert_allclose(std.score, pca.score, rtol=1e-12)


def test_kpc_coarse_full_rank_matches_standard():
    """When k >= max block size, PCA is a rotation and BIC is invariant."""
    data_dict, partition, M, env_order = _chain_oracle_fixtures()
    max_block = max(len(b) for b in partition)
    std = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, scale=True,
    )
    pca = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, scale=True, k=max_block,
    )
    assert set(std.dag.edges) == set(pca.dag.edges)
    np.testing.assert_allclose(std.score, pca.score, rtol=1e-10)


def test_kpc_coarse_k1_finite_scores():
    """k=1 (single PC per block) must produce finite scores, not -Inf."""
    data_dict, partition, M, env_order = _chain_oracle_fixtures()
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, k=1,
    )
    assert np.isfinite(model.score), f"score is {model.score}"
    assert model.dag.number_of_nodes() == 3


def test_kpc_coarse_oracle_chain_recovery():
    """COARSEOracle with k=2 recovers the A→B→C chain."""
    data_dict, partition, M, env_order = _chain_oracle_fixtures()
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, k=2,
    )
    A = (0, 1)
    B = (2, 3)
    C = (4, 5)
    assert set(model.dag.edges) == {(A, B), (B, C)}


def test_kpc_coarse_pooled_runs():
    """pca_pooled=True must run and produce a valid DAG."""
    data_dict, partition, M, env_order = _chain_oracle_fixtures()
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        partition, M, env_order, data_dict, k=1, pca_pooled=True,
    )
    assert np.isfinite(model.score)
    assert model.dag.number_of_nodes() == 3


def test_kpc_project_blocks_clamps_to_svd_rank():
    """When the reference matrix has fewer rows than the requested k_j,
    np.linalg.svd returns only min(n_ref, r_j) right singular vectors.
    _project_blocks must clamp k_j to that rank so proj_block, offset, and
    V_j.shape[1] stay consistent — otherwise the projected array ends up
    shorter than `offset` and downstream sigma slices read the wrong block.
    """
    from coarse.coarse import _project_blocks

    rng = np.random.default_rng(0)
    # Obs ref has only n=2 rows → SVD returns at most 2 right singular vectors
    # per 4-column block, so requesting k=3 must collapse to k_j=2.
    env_arrays = {
        "obs": rng.standard_normal((2, 8)),
        "1": rng.standard_normal((100, 8)),
    }
    partition = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})]
    proj_arrays, proj_partition, block_map = _project_blocks(
        partition, env_arrays, k=3, baseline_key="obs", pca_pooled=False,
    )

    # Each block should have collapsed to 2 dims (rank of obs ref), not 3.
    assert [len(b) for b in proj_partition] == [2, 2]

    # Total projected dim must equal the union of block index ranges and
    # match the actual column count of every env's projected array.
    total_dims = sum(len(b) for b in proj_partition)
    assert total_dims == 4
    for env_key, arr in proj_arrays.items():
        assert arr.shape[1] == total_dims, (
            f"env {env_key}: arr has {arr.shape[1]} cols, "
            f"partition claims {total_dims}"
        )

    # Index contiguity: each block occupies a disjoint contiguous range
    # starting where the previous block ended.
    offset = 0
    for b in proj_partition:
        assert b == frozenset(range(offset, offset + len(b)))
        offset += len(b)

    # And the pooled-mode path with the same shapes — there n_ref = 102, so
    # SVD has full rank 4 and k_j hits the requested k=3 ceiling instead.
    proj_arrays_p, proj_partition_p, _ = _project_blocks(
        partition, env_arrays, k=3, baseline_key="obs", pca_pooled=True,
    )
    assert [len(b) for b in proj_partition_p] == [3, 3]
    for arr in proj_arrays_p.values():
        assert arr.shape[1] == 6

