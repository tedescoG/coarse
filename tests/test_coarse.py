"""Tests for the COARSE package."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy import linalg as sla

import coarse.growshrink as growshrink_mod
from coarse.coarse import COARSE, COARSEOracle
from coarse.growshrink import grow_shrink
from coarse.hypothesis_tests import _normalize_env_data, compute_M
from coarse.partition import (
    compute_candidate_pools,
    compute_supports,
    infer_partition,
)
from coarse.scoring import (
    _block_bic_env_from_sigma,
    _block_indices,
    compute_env_stats,
    parameter_count_d_j,
    pooled_block_bic,
    pooled_block_bic_from_sigma,
)

from conftest import sample_chain_dataset


# Raw-array reference scorer, used to cross-check the cached covariance path.
def block_residual_covariance(X_block, X_parents):
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
    sign, logdet = np.linalg.slogdet(block_residual_covariance(X_block, X_parents))
    if sign <= 0 or not np.isfinite(logdet):
        raise sla.LinAlgError("residual covariance is not positive definite")
    return float(logdet)


def block_bic_env(X_block, X_parents, lambda_pen=1.0):
    n_e, r_j = X_block.shape
    s_j = X_parents.shape[1] if X_parents.size else 0
    if n_e <= 0 or r_j <= 0 or n_e <= s_j + r_j:
        return -np.inf
    try:
        logdet = block_log_det_residual(X_block, X_parents)
    except sla.LinAlgError:
        return -np.inf
    return -n_e * logdet - lambda_pen * np.log(n_e) * parameter_count_d_j(r_j, s_j)


def _chain_data_dict(n, seed):
    rng = np.random.default_rng(seed)
    return {
        "obs": sample_chain_dataset(n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }


_CHAIN_PARTITION = [frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})]
_CHAIN_M = np.array(
    [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool
)
_CHAIN_ENV_ORDER = ["1", "2", "3"]
_CHAIN_EDGES = {((0, 1), (2, 3)), ((2, 3), (4, 5))}


# ---------------------------------------------------------------------------
# compute_M
# ---------------------------------------------------------------------------
def test_compute_M_welch_vs_ks():
    """Welch catches the mean shift but misses the variance shift; KS catches both."""
    rng = np.random.default_rng(0)
    n = 1000
    obs = rng.standard_normal((n, 3))
    env1 = rng.standard_normal((n, 3))
    env1[:, 0] += 2.0
    env2 = rng.standard_normal((n, 3))
    env2[:, 1] *= 2.0
    data_dict = {"obs": obs, "1": env1, "2": env2}

    # alpha=1e-4: both shifts sit far above threshold at n=1000, unshifted cells do not.
    M_welch, env_order = compute_M(data_dict, alpha=1e-4, test_name="welch")
    assert env_order == ["1", "2"]
    assert M_welch.tolist() == [[True, False], [False, False], [False, False]]

    M_ks, _ = compute_M(data_dict, alpha=1e-4, test_name="ks")
    assert M_ks.tolist() == [[True, False], [False, True], [False, False]]


def test_compute_M_requires_obs_baseline():
    with pytest.raises(ValueError, match="baseline"):
        compute_M({"1": np.zeros((10, 2))}, alpha=0.05)


def test_compute_M_rejects_inconsistent_p():
    rng = np.random.default_rng(0)
    data_dict = {"obs": rng.standard_normal((20, 3)), "1": rng.standard_normal((20, 4))}
    with pytest.raises(ValueError, match="columns"):
        compute_M(data_dict, alpha=0.05)


def test_compute_M_columns_follow_env_order():
    """Keys sort as strings; the shifted env's column sits at its env_order position."""
    rng = np.random.default_rng(0)
    n = 500
    data_dict = {
        "obs": rng.standard_normal((n, 2)),
        "2": rng.standard_normal((n, 2)),
        "10": rng.standard_normal((n, 2)),
    }
    data_dict["10"][:, 1] += 3.0
    M, env_order = compute_M(data_dict, alpha=1e-4)
    assert env_order == ["10", "2"]
    assert M[:, env_order.index("10")].tolist() == [False, True]
    assert M[:, env_order.index("2")].tolist() == [False, False]


def test_compute_M_other_tests_and_names():
    """gaussian_lrt, energy, and case-insensitive names all detect a mean shift; energy is
    reproducible under the same rng; unknown names and alpha outside (0, 1) raise."""
    rng = np.random.default_rng(0)
    n = 150
    data_dict = {"obs": rng.standard_normal((n, 2)), "1": rng.standard_normal((n, 2))}
    data_dict["1"][:, 0] += 3.0
    expected = [[True], [False]]
    for name in ("gaussian_lrt", "WELCH", "Ks"):
        M, _ = compute_M(data_dict, alpha=1e-3, test_name=name)
        assert M.tolist() == expected, name
    M_a, _ = compute_M(data_dict, alpha=1e-2, test_name="energy", rng=np.random.default_rng(1))
    M_b, _ = compute_M(data_dict, alpha=1e-2, test_name="energy", rng=np.random.default_rng(1))
    assert M_a.tolist() == expected
    assert np.array_equal(M_a, M_b)
    with pytest.raises(ValueError, match="unknown"):
        compute_M(data_dict, alpha=1e-2, test_name="foo")
    for alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            compute_M(data_dict, alpha=alpha)


def test_normalize_env_data_forms():
    """Dict, tuple, and list inputs are accepted; malformed forms raise."""
    X = [[1.0, 2.0], [3.0, 4.5], [0.5, -1.0]]
    for value in ({"data": X}, (X, {0}, "soft"), X):
        arr = _normalize_env_data(value)
        assert arr.dtype == np.float64 and arr.shape == (3, 2)
    for bad, match in (
        ({"targets": {0}}, "'data'"),
        ((), "empty"),
        (np.arange(4.0), "2-D"),
        (np.ones((1, 2)), "2 rows"),
    ):
        with pytest.raises(ValueError, match=match):
            _normalize_env_data(bad)


def test_vectorized_gaussian_lrt_matches_scalar_large_offset():
    """The column-wise fast path stays accurate when the data has a huge common offset."""
    from coarse.hypothesis_tests import _vectorized_gaussian_lrt, gaussian_lrt_p

    rng = np.random.default_rng(0)
    x = 1e7 + 1e-3 * rng.standard_normal(500)
    y = 1e7 + 1e-3 * rng.standard_normal(500)
    p_vec = _vectorized_gaussian_lrt(x[:, None], y[:, None])[0]
    p_scalar = gaussian_lrt_p(x, y)
    assert np.isclose(p_vec, p_scalar, rtol=1e-4), (p_vec, p_scalar)


# ---------------------------------------------------------------------------
# partition, supports, candidate pools
# ---------------------------------------------------------------------------
_M_HANDCODED = np.array(
    [
        [1, 0, 1],  # v0
        [1, 0, 1],  # v1
        [0, 1, 1],  # v2
        [0, 0, 0],  # v3
    ],
    dtype=bool,
)


def test_partition_supports_pools_handcoded():
    partition = infer_partition(_M_HANDCODED)
    assert set(partition) == {frozenset({0, 1}), frozenset({2}), frozenset({3})}

    supports = compute_supports(_M_HANDCODED, partition)
    assert supports == {
        frozenset({0, 1}): frozenset({0, 2}),
        frozenset({2}): frozenset({1, 2}),
        frozenset({3}): frozenset(),
    }

    pools = compute_candidate_pools(supports)
    assert pools[frozenset({3})] == []
    assert pools[frozenset({0, 1})] == [frozenset({3})]
    assert pools[frozenset({2})] == [frozenset({3})]


def test_infer_partition_permutation_invariance():
    """Permuting the rows of M permutes the blocks but does not change them."""
    rng = np.random.default_rng(0)
    base = set(infer_partition(_M_HANDCODED))
    for _ in range(5):
        perm = rng.permutation(_M_HANDCODED.shape[0])
        relabeled = {
            frozenset(int(perm[v]) for v in block)
            for block in infer_partition(_M_HANDCODED[perm])
        }
        assert relabeled == base


def test_infer_partition_order_respects_supp_inclusion():
    """Blocks come out ordered so that strictly smaller supports precede larger ones."""
    tau = infer_partition(_M_HANDCODED)
    assert tau[0] == frozenset({3})
    assert set(tau[1:]) == {frozenset({0, 1}), frozenset({2})}

    M_chain = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    pos = {b: i for i, b in enumerate(infer_partition(M_chain))}
    assert pos[frozenset({2})] < pos[frozenset({1})] < pos[frozenset({0})]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_parameter_count():
    assert parameter_count_d_j(2, 3) == 9
    assert parameter_count_d_j(1, 0) == 1
    assert parameter_count_d_j(1, 4) == 5


def test_true_parents_score_higher():
    rng = np.random.default_rng(42)
    n = 2000
    X_parent = rng.standard_normal((n, 2))
    B = np.array([[1.5, -0.7], [0.4, 1.1]])
    X_block = X_parent @ B.T + 0.3 * rng.standard_normal((n, 2))
    data = np.column_stack([X_block, X_parent])
    data -= data.mean(axis=0)
    data_dict = {"obs": data}

    block, parents = frozenset({0, 1}), [frozenset({2, 3})]
    assert pooled_block_bic(block, parents, data_dict) > pooled_block_bic(block, [], data_dict)


def test_block_bic_minus_inf_branches():
    """Non-PD S_PaPa, non-PD S_jj, and n_e <= s_j + r_j each give -inf; one more row is finite."""
    pd_sigma = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    block, parents = np.array([0]), np.array([1, 2])
    bad_papa = pd_sigma.copy()
    bad_papa[1:, 1:] = [[1.0, 2.0], [2.0, 1.0]]  # eigenvalues 3, -1
    assert _block_bic_env_from_sigma(block, parents, bad_papa, n_e=100) == -np.inf
    bad_jj = pd_sigma.copy()
    bad_jj[0, 0] = -1.0
    assert _block_bic_env_from_sigma(block, np.array([], dtype=np.int64), bad_jj, n_e=100) == -np.inf
    assert _block_bic_env_from_sigma(block, parents, pd_sigma, n_e=3) == -np.inf
    assert np.isfinite(_block_bic_env_from_sigma(block, parents, pd_sigma, n_e=4))


_PARITY_CONFIGS = [
    (frozenset({2, 3}), []),
    (frozenset({2, 3}), [frozenset({0, 1})]),
    (frozenset({4, 5}), [frozenset({0, 1}), frozenset({2, 3})]),
    (frozenset({4}), [frozenset({0, 1}), frozenset({2, 3})]),
    (frozenset({0}), []),
]


def _unequal_centered_envs(seed=2026):
    rng = np.random.default_rng(seed)
    data_dict = {
        "obs": sample_chain_dataset(600, rng),
        "1": sample_chain_dataset(300, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(900, rng, shift_targets=(2, 3)),
    }
    return {k: v - v.mean(axis=0, keepdims=True) for k, v in data_dict.items()}


def test_pooled_bic_matches_raw_reference():
    """Cached-covariance scorer agrees with the raw-array reference, summed over envs of
    different sizes, for several lambda_pen values."""
    centered = _unequal_centered_envs()
    env_stats = compute_env_stats(centered)
    for lambda_pen in (0.0, 1.0, 2.0):
        for block, parents in _PARITY_CONFIGS:
            pa_idx = sorted({i for p in parents for i in p})
            expected = sum(
                block_bic_env(env[:, sorted(block)], env[:, pa_idx], lambda_pen=lambda_pen)
                for env in centered.values()
            )
            actual = pooled_block_bic_from_sigma(block, parents, env_stats, lambda_pen)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-10, err_msg=f"{block} {parents} lambda={lambda_pen}"
            )

    # n_e <= s_j + r_j in every env -> -inf on both paths
    small = {k: v[:3] for k, v in centered.items()}
    block, parents = frozenset({0, 1}), [frozenset({2, 3}), frozenset({4, 5})]
    assert pooled_block_bic(block, parents, small) == -np.inf
    assert pooled_block_bic_from_sigma(block, parents, compute_env_stats(small)) == -np.inf


def test_pooled_bic_cached_kwargs_match_uncached():
    """The precomputed block_idx / idx_cache / S_jj_cache path used by grow_shrink gives the
    same value as the plain call."""
    env_stats = compute_env_stats(_unequal_centered_envs())
    all_blocks = {b for cfg in _PARITY_CONFIGS for b in (cfg[0], *cfg[1])}
    idx_cache = {b: _block_indices(b) for b in all_blocks}
    for block, parents in _PARITY_CONFIGS:
        block_idx = idx_cache[block]
        S_jj_cache = {k: s.sigma[np.ix_(block_idx, block_idx)] for k, s in env_stats.items()}
        plain = pooled_block_bic_from_sigma(block, parents, env_stats, 1.0)
        cached = pooled_block_bic_from_sigma(
            block, parents, env_stats, 1.0,
            block_idx=block_idx, idx_cache=idx_cache, S_jj_cache=S_jj_cache,
        )
        assert cached == plain, (block, parents)


def test_near_singular_parents_do_not_raise():
    """Near-collinear parent blocks give a finite value or -inf, never an exception."""
    rng = np.random.default_rng(0)
    n = 200
    X_pa0 = rng.standard_normal(n)
    X_pa1 = X_pa0 + 1e-12 * rng.standard_normal(n)
    X_block = X_pa0 + 0.1 * rng.standard_normal(n)
    data_dict = {"obs": np.column_stack([X_block, X_pa0, X_pa1])}
    bic = pooled_block_bic(frozenset({0}), [frozenset({1}), frozenset({2})], data_dict)
    assert not np.isnan(bic) and bic < np.inf


# ---------------------------------------------------------------------------
# grow_shrink
# ---------------------------------------------------------------------------
def test_grow_shrink_rejects_distractor():
    """True parents {2}, {3} are kept; {4}, which depends on the target only via {2}, is dropped
    regardless of the candidate shuffle order."""
    rng = np.random.default_rng(42)
    n = 2000
    X2 = rng.standard_normal(n)
    X3 = rng.standard_normal(n)
    X4 = 1.5 * X2 + 0.3 * rng.standard_normal(n)
    X0 = 0.8 * X2 + 0.5 * X3 + 0.2 * rng.standard_normal(n)
    X1 = -0.6 * X2 + 0.7 * X3 + 0.2 * rng.standard_normal(n)
    data = np.column_stack([X0, X1, X2, X3, X4])

    block = frozenset({0, 1})
    candidate_pool = [frozenset({2}), frozenset({3}), frozenset({4})]
    for seed in (0, 1, 7, 123, 2026):
        parents = grow_shrink(
            block, candidate_pool, {"obs": data}, lambda_pen=1.0, rng=np.random.default_rng(seed)
        )
        assert set(parents) == {frozenset({2}), frozenset({3})}, f"seed={seed}: {set(parents)}"


_A, _B = frozenset({1}), frozenset({2})


@pytest.mark.parametrize(
    "table, expected",
    [
        # second grow pass: a is rejected alone but accepted once b is in
        ({frozenset(): 0.0, _A: -1.0, _B: 1.0, _A | _B: 3.0}, {_A, _B}),
        # shrink drops a parent that grow added
        ({frozenset(): 0.0, _A: 1.0, _B: 2.0, _A | _B: 1.5}, {_B}),
        # ties are not accepted
        ({frozenset(): 0.0, _A: 0.0}, set()),
    ],
)
def test_grow_shrink_control_flow(monkeypatch, table, expected):
    """Grow-shrink follows the score table alone: multi-pass grow, shrink, and strict >."""
    scored: list[frozenset] = []

    def fake_score(block, parents, env_stats, lambda_pen, **kwargs):
        key = frozenset().union(*parents)
        scored.append(key)
        return table[key]

    monkeypatch.setattr(growshrink_mod, "pooled_block_bic_from_sigma", fake_score)
    pool = [b for b in (_A, _B) if b in table]
    shrink_seen = False
    for seed in range(10):
        scored.clear()
        parents = grow_shrink(frozenset({0}), pool, {}, env_stats={}, rng=np.random.default_rng(seed))
        assert set(parents) == expected, seed
        # {b} scored after {a, b} only happens when shrink drops a from a grown {a, b}
        if _A | _B in scored and _B in scored[scored.index(_A | _B):]:
            shrink_seen = True
    if expected == {_B}:
        assert shrink_seen


def test_growshrink_empty_pool_returns_empty():
    data = np.random.default_rng(0).standard_normal((100, 2))
    assert grow_shrink(frozenset({0, 1}), [], {"obs": data}) == []


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def test_oracle_recovers_chain():
    """With the true partition and M, the oracle returns exactly A -> B -> C."""
    data_dict = _chain_data_dict(1500, 0)
    model = COARSEOracle().fit(_CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict)
    assert np.isfinite(model.score)
    assert set(model.dag.nodes) == {(0, 1), (2, 3), (4, 5)}
    assert set(model.dag.edges) == _CHAIN_EDGES
    centered = {k: v - v.mean(axis=0, keepdims=True) for k, v in data_dict.items()}
    expected = sum(pooled_block_bic(b, model.parent_sets[b], centered) for b in model.partition)
    assert model.score == pytest.approx(expected, rel=1e-12)


def test_oracle_unequal_env_sizes_recovers_chain():
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(2000, rng),
        "1": sample_chain_dataset(400, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(900, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(250, rng, shift_targets=(4, 5)),
    }
    model = COARSEOracle().fit(_CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict)
    assert set(model.dag.edges) == _CHAIN_EDGES
    assert np.isfinite(model.score)


def test_fit_lambda_extremes():
    """A huge penalty selects no parents; a zero penalty selects the whole candidate pool."""
    data_dict = _chain_data_dict(1500, 0)
    none = COARSE().fit(data_dict, alpha=1e-4, lambda_pen=1e6)
    assert none.dag.number_of_edges() == 0
    full = COARSE().fit(data_dict, alpha=1e-4, lambda_pen=0.0)
    for block, parents in full.parent_sets.items():
        assert set(parents) == set(full.candidate_pools[block]), block
    assert full.dag.number_of_edges() == 3


def test_fit_invariant_to_column_rescale_and_offset():
    """Per-column rescaling keeps M and the edges; a global offset keeps the score too."""
    rng = np.random.default_rng(3)
    data_dict = _chain_data_dict(1500, 0)
    base = COARSE().fit(data_dict, alpha=1e-4)
    scale = rng.uniform(0.2, 5.0, size=6)
    scaled = COARSE().fit({k: v * scale for k, v in data_dict.items()}, alpha=1e-4)
    assert np.array_equal(scaled.M, base.M)
    assert set(scaled.dag.edges) == set(base.dag.edges)
    shifted = COARSE().fit({k: v + 1e3 for k, v in data_dict.items()}, alpha=1e-4)
    assert np.array_equal(shifted.M, base.M)
    assert set(shifted.dag.edges) == set(base.dag.edges)
    assert shifted.score == pytest.approx(base.score, rel=1e-9)


def test_fit_recovers_chain():
    """Same chain, with M and the partition estimated from data."""
    model = COARSE().fit(_chain_data_dict(1500, 0), alpha=1e-4, refine_test="welch")
    assert set(model.partition) == set(_CHAIN_PARTITION)
    assert set(model.dag.edges) == _CHAIN_EDGES
    assert np.isfinite(model.score)


def test_coarse_oracle_rejects_partition_inconsistent_with_M():
    merged = [frozenset({0, 1, 2, 3}), frozenset({4, 5})]
    with pytest.raises(ValueError, match="row-class partition"):
        COARSEOracle().fit(merged, _CHAIN_M, _CHAIN_ENV_ORDER, _chain_data_dict(300, 0))


def test_coarse_oracle_rejects_M_row_mismatch():
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(300, rng),
        "1": sample_chain_dataset(300, rng, shift_targets=(0, 1)),
    }
    M_short = np.array([[1], [1], [0], [0]], dtype=bool)
    M_long = np.array([[1], [1], [0], [0], [0], [0], [0], [0]], dtype=bool)
    for M in (M_short, M_long):
        with pytest.raises(ValueError, match="rows|shape"):
            COARSEOracle().fit(infer_partition(M), M, ["1"], data_dict)


def test_coarse_oracle_rejects_env_order_mismatch():
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(300, rng),
        "1": sample_chain_dataset(300, rng, shift_targets=(0, 1)),
    }
    M = np.array([[1], [1], [0], [0], [0], [0]], dtype=bool)
    with pytest.raises(ValueError, match="env_order"):
        COARSEOracle().fit(infer_partition(M), M, ["1", "2"], data_dict)


def test_coarse_oracle_canonicalises_partition_order():
    """The caller's block order does not affect the partition, linear extension, or DAG."""
    data_dict = _chain_data_dict(1500, 42)
    canonical = infer_partition(_CHAIN_M)
    fwd = COARSEOracle(rng=np.random.default_rng(0)).fit(
        list(_CHAIN_PARTITION), _CHAIN_M, _CHAIN_ENV_ORDER, data_dict
    )
    rev = COARSEOracle(rng=np.random.default_rng(0)).fit(
        list(reversed(_CHAIN_PARTITION)), _CHAIN_M, _CHAIN_ENV_ORDER, data_dict
    )
    assert fwd.partition == rev.partition == canonical
    assert fwd.linear_extension_ == rev.linear_extension_ == canonical
    assert set(fwd.dag.edges) == set(rev.dag.edges)
    assert fwd.score == rev.score


@pytest.mark.parametrize("corrupt", ["nan", "one_row"])
def test_fit_rejects_bad_input(corrupt):
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(200, rng),
        "1": sample_chain_dataset(200, rng, shift_targets=(0, 1)),
    }
    if corrupt == "nan":
        data_dict["1"][3, 2] = np.nan
    else:
        data_dict["1"] = data_dict["1"][:1]
    with pytest.raises(ValueError):
        COARSE().fit(data_dict)


@pytest.mark.parametrize("corrupt", ["constant", "duplicate"])
def test_fit_rejects_constant_and_duplicate_columns(corrupt):
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(200, rng),
        "1": sample_chain_dataset(200, rng, shift_targets=(0, 1)),
    }
    if corrupt == "constant":
        data_dict["1"][:, 2] = 3.0
        match = "constant"
    else:
        data_dict["1"][:, 4] = data_dict["1"][:, 0]
        match = "identical"
    with pytest.raises(ValueError, match=match):
        COARSE().fit(data_dict)


@pytest.mark.parametrize("dependent", ["target_block", "parents"])
def test_fit_rejects_linearly_dependent_columns(dependent):
    """Exact (affine) linear dependence within an env is rejected before scoring."""
    data_dict = _chain_data_dict(300, 0)
    for v in data_dict.values():
        if dependent == "target_block":
            v[:, 1] = 2.0 * v[:, 0]
        else:
            v[:, 2] = 2.0 * v[:, 3] + 3.0
    with pytest.raises(ValueError, match="linearly dependent"):
        COARSE().fit(data_dict)


def test_fit_accepts_noisy_linear_combination():
    """A column that is a linear combination plus small noise passes the rank check."""
    noise_rng = np.random.default_rng(99)  # not the fixture seed: the noise must be fresh
    data_dict = _chain_data_dict(300, 0)
    for v in data_dict.values():
        v[:, 1] = 2.0 * v[:, 0] + 3.0 + 1e-6 * noise_rng.standard_normal(v.shape[0])
    model = COARSE().fit(data_dict)
    assert np.isfinite(model.score)


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_fit_rejects_bad_k(k):
    data_dict = _chain_data_dict(200, 0)
    with pytest.raises(ValueError, match="k must be"):
        COARSE().fit(data_dict, k=k)
    with pytest.raises(ValueError, match="k must be"):
        COARSEOracle().fit(_CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict, k=k)


def test_fit_rejects_bad_alpha_and_lambda():
    data_dict = _chain_data_dict(200, 0)
    for alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            COARSE().fit(data_dict, alpha=alpha)
    with pytest.raises(ValueError, match="lambda_pen"):
        COARSE().fit(data_dict, lambda_pen=-1.0)
    with pytest.raises(ValueError, match="lambda_pen"):
        COARSEOracle().fit(_CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict, lambda_pen=-1.0)


def test_fit_custom_baseline_key():
    """A non-"obs" baseline works end to end, plain and with k; the default key then raises."""
    data_dict = {("ctrl" if k == "obs" else k): v for k, v in _chain_data_dict(1500, 0).items()}
    plain = COARSE().fit(data_dict, alpha=1e-4, baseline_key="ctrl")
    assert plain.env_order == _CHAIN_ENV_ORDER
    assert set(plain.dag.edges) == _CHAIN_EDGES
    pca = COARSE().fit(data_dict, alpha=1e-4, baseline_key="ctrl", k=2)
    assert set(pca.dag.edges) == _CHAIN_EDGES
    with pytest.raises(ValueError, match="baseline"):
        COARSE().fit(data_dict)


def test_fit_single_variable():
    rng = np.random.default_rng(0)
    data_dict = {"obs": rng.standard_normal((200, 1)), "1": 2.0 + rng.standard_normal((200, 1))}
    model = COARSE().fit(data_dict)
    assert list(model.dag.nodes) == [(0,)]
    assert model.dag.number_of_edges() == 0
    assert np.isfinite(model.score)


def test_fit_degenerate_tiny_n_returns_minus_inf_no_crash():
    """n_e <= p in every env: nothing is scorable, so the fit is an empty DAG with -inf score."""
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(5, rng),
        "1": sample_chain_dataset(5, rng, shift_targets=(0, 1)),
    }
    model = COARSE().fit(data_dict)
    assert model.score == -np.inf
    assert model.dag.number_of_edges() == 0


def test_fit_baseline_only_single_block():
    """No interventional env: M has zero columns and every variable lands in one block."""
    model = COARSE().fit({"obs": sample_chain_dataset(500, np.random.default_rng(0))})
    assert model.M.shape == (6, 0)
    assert model.partition == [frozenset(range(6))]
    assert model.dag.number_of_edges() == 0
    assert np.isfinite(model.score)


def test_fit_on_sempler_dag_smoke():
    """Fit runs on a 20-node sempler DAG and roughly recovers the true partition."""
    from sempler import LGANM
    from sempler.generators import dag_avg_deg, intervention_targets
    from sklearn.metrics import adjusted_rand_score

    seed, num_nodes, num_intervs, density = 0, 20, 5, 0.1
    weights = dag_avg_deg(
        num_nodes, density * (num_nodes - 1), w_min=0.5, w_max=2,
        return_ordering=False, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    edge_idcs = np.flatnonzero(weights)
    to_neg = edge_idcs[rng.choice([True, False], len(edge_idcs))]
    weights[np.unravel_index(to_neg, (num_nodes, num_nodes))] *= -1

    model = LGANM(weights, means=(-2, 2), variances=(0.5, 2), random_state=seed)
    targets = intervention_targets(num_nodes, num_intervs, 1, random_state=seed)
    data_dict = {"obs": model.sample(1000)}
    for idx, target in enumerate(targets):
        data_dict[str(idx)] = model.sample(1000, shift_interventions={target[0]: (2, 1)})

    coarse_model = COARSE().fit(data_dict, alpha=1e-4, lambda_pen=1.0, refine_test="welch")
    assert coarse_model.dag.number_of_nodes() >= 1
    assert coarse_model.fit_runtime_sec > 0
    assert nx.is_directed_acyclic_graph(coarse_model.dag)

    true_dag = nx.DiGraph(weights.astype(bool))
    M_true = np.column_stack([
        np.isin(np.arange(num_nodes), list(nx.descendants(true_dag, t[0])) + [t[0]])
        for t in targets
    ])
    true_labels = np.zeros(num_nodes, dtype=int)
    for label, part in enumerate(infer_partition(M_true)):
        true_labels[list(part)] = label
    est_labels = np.zeros(num_nodes, dtype=int)
    for label, part in enumerate(coarse_model.dag.nodes):
        est_labels[list(part)] = label
    ari = adjusted_rand_score(true_labels, est_labels)
    assert ari >= 0.5, f"partition ARI {ari}"


# ---------------------------------------------------------------------------
# expand_coarsened_dag
# ---------------------------------------------------------------------------
def test_expand_coarsened_dag():
    model = COARSE()
    model.num_features = 6
    model.dag = nx.DiGraph()
    block_a, block_b, block_c = (3, 0, 2), (1,), (4, 5)
    model.dag.add_nodes_from([block_a, block_b, block_c])
    model.dag.add_edges_from([(block_a, block_c), (block_b, block_c)])

    expected = np.zeros((6, 6), dtype=int)
    for src in block_a + block_b:
        for dst in block_c:
            expected[src, dst] = 1
    assert np.array_equal(model.expand_coarsened_dag(), expected)


def test_expand_coarsened_dag_fully_connected():
    model = COARSE()
    model.num_features = 5
    model.dag = nx.DiGraph()
    block_a, block_b = (0, 1, 2), (3, 4)
    model.dag.add_nodes_from([block_a, block_b])
    model.dag.add_edge(block_a, block_b)

    expected = np.zeros((5, 5), dtype=int)
    for src, dst in ((0, 1), (0, 2), (1, 2), (3, 4)):
        expected[src, dst] = 1
    for src in block_a:
        for dst in block_b:
            expected[src, dst] = 1
    assert np.array_equal(model.expand_coarsened_dag(fully_connected=True), expected)


# ---------------------------------------------------------------------------
# kPC-COARSE
# ---------------------------------------------------------------------------
def _scaled_by_obs_std(data_dict):
    obs_std = data_dict["obs"].std(0)
    return {ek: (X - X.mean(0)) / obs_std for ek, X in data_dict.items()}


def test_kpc_coarse_full_rank_matches_standard():
    """k >= block size is a rotation: same DAG and score as plain COARSE on obs-std-scaled data."""
    data_dict = _chain_data_dict(1500, 42)
    std = COARSEOracle(rng=np.random.default_rng(0)).fit(
        _CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, _scaled_by_obs_std(data_dict)
    )
    pca = COARSEOracle(rng=np.random.default_rng(0)).fit(
        _CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict, k=2
    )
    assert set(pca.dag.edges) == set(std.dag.edges) == _CHAIN_EDGES
    np.testing.assert_allclose(std.score, pca.score, rtol=1e-10)


def test_kpc_scaling_uses_observational_std():
    """The kPC scale is the observational std even when another env has inflated variance."""
    data_dict = _chain_data_dict(1500, 42)
    data_dict["2"][:, [2, 3]] *= 3.0
    std = COARSEOracle(rng=np.random.default_rng(0)).fit(
        _CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, _scaled_by_obs_std(data_dict)
    )
    pca = COARSEOracle(rng=np.random.default_rng(0)).fit(
        _CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, data_dict, k=2
    )
    assert set(std.dag.edges) == set(pca.dag.edges)
    np.testing.assert_allclose(std.score, pca.score, rtol=1e-10)


def test_kpc_coarse_k1_finite_scores():
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        _CHAIN_PARTITION, _CHAIN_M, _CHAIN_ENV_ORDER, _chain_data_dict(1500, 42), k=1
    )
    assert np.isfinite(model.score), model.score
    assert model.dag.number_of_nodes() == 3


def test_kpc_project_blocks_clamps_to_svd_rank():
    """With fewer obs rows than k, k_j is clamped to the SVD rank and the projected
    arrays, partition, and block offsets stay consistent."""
    from coarse.coarse import _project_blocks

    rng = np.random.default_rng(0)
    env_arrays = {"obs": rng.standard_normal((2, 8)), "1": rng.standard_normal((100, 8))}
    partition = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})]
    proj_arrays, proj_partition, _ = _project_blocks(partition, env_arrays, k=3, baseline_key="obs")

    assert [len(b) for b in proj_partition] == [2, 2]
    for env_key, arr in proj_arrays.items():
        assert arr.shape[1] == 4, env_key
    offset = 0
    for b in proj_partition:
        assert b == frozenset(range(offset, offset + len(b)))
        offset += len(b)
