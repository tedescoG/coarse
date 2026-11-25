"""Tests for the COARSE package."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy import linalg as sla

import coarse.coarse as coarse_mod
import coarse.growshrink as growshrink_mod
from coarse.coarse import COARSE, COARSEOracle, _project_blocks
from coarse.growshrink import grow_shrink
from coarse.hypothesis_tests import (
    _normalize_env_data,
    _vectorized_gaussian_lrt,
    compute_M,
    gaussian_lrt_p,
)
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

from conftest import (
    CHAIN_EDGES,
    CHAIN_ENV_ORDER,
    CHAIN_M,
    CHAIN_PARTITION,
    chain_data_dict,
    sample_chain_dataset,
)


# Raw-array reference scorer, used to cross-check the cached covariance path.
def block_bic_env(X_block, X_parents, lambda_pen=1.0):
    n_e, r_j = X_block.shape
    s_j = X_parents.shape[1]
    if n_e <= s_j + r_j:
        return -np.inf
    S_jj = (X_block.T @ X_block) / n_e
    if s_j:
        S_jPa = (X_block.T @ X_parents) / n_e
        S_PaPa = (X_parents.T @ X_parents) / n_e
        S_jj = S_jj - S_jPa @ sla.solve(S_PaPa, S_jPa.T, assume_a="pos")
    sign, logdet = np.linalg.slogdet(S_jj)
    if sign <= 0:
        return -np.inf
    return -n_e * logdet - lambda_pen * np.log(n_e) * parameter_count_d_j(r_j, s_j)


def _oracle(data_dict, partition=CHAIN_PARTITION, **kw):
    return COARSEOracle(rng=np.random.default_rng(0)).fit(
        list(partition), CHAIN_M, CHAIN_ENV_ORDER, data_dict, **kw
    )


# ---------------------------------------------------------------------------
# compute_M
# ---------------------------------------------------------------------------
def test_compute_M_welch_vs_ks():
    """Welch flags the mean shift only; KS flags mean and variance shifts."""
    rng = np.random.default_rng(0)
    n = 1000
    obs = rng.standard_normal((n, 3))
    env1 = rng.standard_normal((n, 3))
    env1[:, 0] += 2.0
    env2 = rng.standard_normal((n, 3))
    env2[:, 1] *= 2.0
    data_dict = {"obs": obs, "1": env1, "2": env2}

    M_welch, env_order = compute_M(data_dict, alpha=1e-4, test_name="welch")
    assert env_order == ["1", "2"]
    assert M_welch.tolist() == [[True, False], [False, False], [False, False]]
    M_ks, _ = compute_M(data_dict, alpha=1e-4, test_name="ks")
    assert M_ks.tolist() == [[True, False], [False, True], [False, False]]


def test_compute_M_columns_follow_env_order():
    """Env keys sort as strings and M's columns follow that order."""
    rng = np.random.default_rng(0)
    data_dict = {k: rng.standard_normal((500, 2)) for k in ("obs", "2", "10")}
    data_dict["10"][:, 1] += 3.0
    M, env_order = compute_M(data_dict, alpha=1e-4)
    assert env_order == ["10", "2"]
    assert M.tolist() == [[False, False], [True, False]]


def test_compute_M_other_tests():
    """gaussian_lrt, energy and case-insensitive names detect a mean shift; energy is seeded."""
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": rng.standard_normal((150, 2)),
        "1": rng.standard_normal((150, 2)),
    }
    data_dict["1"][:, 0] += 3.0
    expected = [[True], [False]]
    for name in ("gaussian_lrt", "WELCH", "Ks"):
        assert (
            compute_M(data_dict, alpha=1e-3, test_name=name)[0].tolist() == expected
        ), name
    M_a, _ = compute_M(
        data_dict, alpha=1e-2, test_name="energy", rng=np.random.default_rng(1)
    )
    M_b, _ = compute_M(
        data_dict, alpha=1e-2, test_name="energy", rng=np.random.default_rng(1)
    )
    assert M_a.tolist() == M_b.tolist() == expected


def test_compute_M_rejects_bad_input():
    rng = np.random.default_rng(0)
    good = {"obs": rng.standard_normal((20, 3)), "1": rng.standard_normal((20, 3))}
    with pytest.raises(ValueError, match="baseline"):
        compute_M({"1": good["1"]}, alpha=0.05)
    with pytest.raises(ValueError, match="columns"):
        compute_M({**good, "2": rng.standard_normal((20, 4))}, alpha=0.05)
    with pytest.raises(ValueError, match="unknown"):
        compute_M(good, alpha=0.05, test_name="foo")
    for alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            compute_M(good, alpha=alpha)


def test_normalize_env_data_forms():
    """Dict, tuple and array forms give the same float64 array; malformed forms raise."""
    X = [[1.0, 2.0], [3.0, 4.5], [0.5, -1.0]]
    for value in ({"data": X}, (X, {0}, "soft"), X):
        arr = _normalize_env_data(value)
        assert arr.dtype == np.float64 and arr.shape == (3, 2)
    for bad, match in (
        ({"targets": {0}}, "'data'"),
        ((), "empty"),
        (np.arange(4.0), "2-D"),
    ):
        with pytest.raises(ValueError, match=match):
            _normalize_env_data(bad)


def test_vectorized_gaussian_lrt_matches_scalar_large_offset():
    """The column-wise LRT matches the scalar one under a large common offset."""
    rng = np.random.default_rng(0)
    x = 1e7 + 1e-3 * rng.standard_normal(500)
    y = 1e7 + 1e-3 * rng.standard_normal(500)
    p_vec = _vectorized_gaussian_lrt(x[:, None], y[:, None])[0]
    assert np.isclose(p_vec, gaussian_lrt_p(x, y), rtol=1e-4)


# ---------------------------------------------------------------------------
# partition, supports, candidate pools
# ---------------------------------------------------------------------------
_M_HANDCODED = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=bool)


def test_partition_supports_pools_handcoded():
    partition = infer_partition(_M_HANDCODED)
    assert partition[0] == frozenset({3})
    assert set(partition) == {frozenset({0, 1}), frozenset({2}), frozenset({3})}
    supports = compute_supports(_M_HANDCODED, partition)
    assert supports == {
        frozenset({0, 1}): frozenset({0, 2}),
        frozenset({2}): frozenset({1, 2}),
        frozenset({3}): frozenset(),
    }
    pools = compute_candidate_pools(supports)
    assert pools == {
        frozenset({3}): [],
        frozenset({0, 1}): [frozenset({3})],
        frozenset({2}): [frozenset({3})],
    }


def test_infer_partition_order_and_permutation_invariance():
    """Smaller supports come first; permuting M's rows relabels the blocks only."""
    M_chain = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    assert infer_partition(M_chain) == [frozenset({2}), frozenset({1}), frozenset({0})]
    rng = np.random.default_rng(0)
    base = set(infer_partition(_M_HANDCODED))
    for _ in range(5):
        perm = rng.permutation(_M_HANDCODED.shape[0])
        relabeled = {
            frozenset(int(perm[v]) for v in b)
            for b in infer_partition(_M_HANDCODED[perm])
        }
        assert relabeled == base


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_block_bic_minus_inf_branches():
    """Non-PD S_PaPa, non-PD S_jj and n_e <= s_j + r_j give -inf; one more row is finite."""
    pd_sigma = np.array([[2.0, 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.0]])
    block, parents = np.array([0]), np.array([1, 2])
    bad_papa = pd_sigma.copy()
    bad_papa[1:, 1:] = [[1.0, 2.0], [2.0, 1.0]]
    assert _block_bic_env_from_sigma(block, parents, bad_papa, n_e=100) == -np.inf
    bad_jj = pd_sigma.copy()
    bad_jj[0, 0] = -1.0
    assert (
        _block_bic_env_from_sigma(block, np.array([], dtype=np.int64), bad_jj, n_e=100)
        == -np.inf
    )
    assert _block_bic_env_from_sigma(block, parents, pd_sigma, n_e=3) == -np.inf
    assert np.isfinite(_block_bic_env_from_sigma(block, parents, pd_sigma, n_e=4))


def test_near_singular_parents_do_not_raise():
    """Near-collinear parents give a finite value or -inf, never an exception."""
    rng = np.random.default_rng(0)
    X_pa0 = rng.standard_normal(200)
    X_pa1 = X_pa0 + 1e-12 * rng.standard_normal(200)
    X_block = X_pa0 + 0.1 * rng.standard_normal(200)
    data_dict = {"obs": np.column_stack([X_block, X_pa0, X_pa1])}
    bic = pooled_block_bic(frozenset({0}), [frozenset({1}), frozenset({2})], data_dict)
    assert not np.isnan(bic) and bic < np.inf


def test_pooled_bic_matches_raw_reference():
    """Cached-covariance scorer, with and without the grow-shrink caches, equals the raw-array
    reference summed over unequal-size envs, for several λ; too few rows give -inf."""
    rng = np.random.default_rng(2026)
    sizes = {"obs": 600, "1": 300, "2": 900}
    shifts = {"obs": (), "1": (0, 1), "2": (2, 3)}
    centered = {}
    for k, n in sizes.items():
        X = sample_chain_dataset(n, rng, shift_targets=shifts[k])
        centered[k] = X - X.mean(axis=0)
    env_stats = compute_env_stats(centered)
    configs = [
        (frozenset({2, 3}), []),
        (frozenset({2, 3}), [frozenset({0, 1})]),
        (frozenset({4, 5}), [frozenset({0, 1}), frozenset({2, 3})]),
        (frozenset({4}), [frozenset({0, 1}), frozenset({2, 3})]),
        (frozenset({0}), []),
    ]
    idx_cache = {b: _block_indices(b) for cfg in configs for b in (cfg[0], *cfg[1])}
    for lambda_pen in (0.0, 1.0, 2.0):
        for block, parents in configs:
            pa_idx = sorted({i for p in parents for i in p})
            expected = sum(
                block_bic_env(X[:, sorted(block)], X[:, pa_idx], lambda_pen)
                for X in centered.values()
            )
            plain = pooled_block_bic_from_sigma(block, parents, env_stats, lambda_pen)
            np.testing.assert_allclose(plain, expected, rtol=1e-10)
            b_idx = idx_cache[block]
            cached = pooled_block_bic_from_sigma(
                block,
                parents,
                env_stats,
                lambda_pen,
                block_idx=b_idx,
                idx_cache=idx_cache,
                S_jj_cache={
                    k: s.sigma[np.ix_(b_idx, b_idx)] for k, s in env_stats.items()
                },
            )
            assert cached == plain

    small = {k: v[:3] for k, v in centered.items()}
    block, parents = frozenset({0, 1}), [frozenset({2, 3}), frozenset({4, 5})]
    assert pooled_block_bic(block, parents, small) == -np.inf


# ---------------------------------------------------------------------------
# grow_shrink
# ---------------------------------------------------------------------------
def test_grow_shrink_rejects_distractor():
    """True parents {2}, {3} are kept and the distractor {4} dropped, for any shuffle order."""
    rng = np.random.default_rng(42)
    n = 2000
    X2 = rng.standard_normal(n)
    X3 = rng.standard_normal(n)
    X4 = 1.5 * X2 + 0.3 * rng.standard_normal(n)
    X0 = 0.8 * X2 + 0.5 * X3 + 0.2 * rng.standard_normal(n)
    X1 = -0.6 * X2 + 0.7 * X3 + 0.2 * rng.standard_normal(n)
    data = np.column_stack([X0, X1, X2, X3, X4])
    pool = [frozenset({2}), frozenset({3}), frozenset({4})]
    for seed in (0, 1, 7, 123, 2026):
        parents = grow_shrink(
            frozenset({0, 1}), pool, {"obs": data}, rng=np.random.default_rng(seed)
        )
        assert set(parents) == {frozenset({2}), frozenset({3})}, seed


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
        # empty pool
        ({frozenset(): 0.0}, set()),
    ],
)
def test_grow_shrink_control_flow(monkeypatch, table, expected):
    """The result follows the score table: multi-pass grow, shrink, strict >."""
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
        parents = grow_shrink(
            frozenset({0}), pool, {}, env_stats={}, rng=np.random.default_rng(seed)
        )
        assert set(parents) == expected, seed
        # {b} scored after {a, b} only when shrink drops a from a grown {a, b}
        if _A | _B in scored and _B in scored[scored.index(_A | _B) :]:
            shrink_seen = True
    if expected == {_B}:
        assert shrink_seen


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sizes", [(1500,) * 4, (2000, 400, 900, 250)])
def test_oracle_recovers_chain(sizes):
    """Given the true partition and M, the oracle returns A -> B -> C and the summed block BIC."""
    rng = np.random.default_rng(0)
    data_dict = {
        k: sample_chain_dataset(n, rng, shift_targets=s)
        for k, n, s in zip(("obs", "1", "2", "3"), sizes, ((), (0, 1), (2, 3), (4, 5)))
    }
    model = _oracle(data_dict)
    assert set(model.dag.nodes) == {(0, 1), (2, 3), (4, 5)}
    assert set(model.dag.edges) == CHAIN_EDGES
    centered = {k: v - v.mean(axis=0) for k, v in data_dict.items()}
    expected = sum(
        pooled_block_bic(b, model.parent_sets[b], centered) for b in model.partition
    )
    assert model.score == pytest.approx(expected, rel=1e-12)


def test_fit_recovers_chain():
    """With M and the partition estimated from data."""
    model = COARSE().fit(chain_data_dict(1500, 0), alpha=1e-4)
    assert set(model.partition) == set(CHAIN_PARTITION)
    assert set(model.dag.edges) == CHAIN_EDGES
    assert np.isfinite(model.score)


def test_fit_lambda_extremes():
    """A huge penalty selects no parents; zero penalty selects every candidate."""
    data_dict = chain_data_dict(1500, 0)
    assert (
        COARSE().fit(data_dict, alpha=1e-4, lambda_pen=1e6).dag.number_of_edges() == 0
    )
    full = COARSE().fit(data_dict, alpha=1e-4, lambda_pen=0.0)
    for block, parents in full.parent_sets.items():
        assert set(parents) == set(full.candidate_pools[block]), block
    assert full.dag.number_of_edges() == 3


def test_fit_invariant_to_column_rescale_and_offset():
    """Column rescaling keeps M and the edges; a global offset also keeps the score."""
    data_dict = chain_data_dict(1500, 0)
    base = COARSE().fit(data_dict, alpha=1e-4)
    scale = np.random.default_rng(3).uniform(0.2, 5.0, size=6)
    scaled = COARSE().fit({k: v * scale for k, v in data_dict.items()}, alpha=1e-4)
    shifted = COARSE().fit({k: v + 1e3 for k, v in data_dict.items()}, alpha=1e-4)
    for other in (scaled, shifted):
        assert np.array_equal(other.M, base.M)
        assert set(other.dag.edges) == set(base.dag.edges)
    assert shifted.score == pytest.approx(base.score, rel=1e-9)


def test_coarse_oracle_canonicalises_partition_order():
    """The caller's block order does not change the partition, τ, DAG or score."""
    data_dict = chain_data_dict(1500, 42)
    fwd = _oracle(data_dict)
    rev = _oracle(data_dict, partition=reversed(CHAIN_PARTITION))
    canonical = infer_partition(CHAIN_M)
    assert fwd.partition == rev.partition == canonical
    assert fwd.linear_extension_ == rev.linear_extension_ == canonical
    assert set(fwd.dag.edges) == set(rev.dag.edges)
    assert fwd.score == rev.score


def test_coarse_oracle_rejects_inconsistent_inputs():
    """Partition not a refinement of M, M with the wrong row count, env_order of the wrong length."""
    data_dict = chain_data_dict(300, 0)
    with pytest.raises(ValueError, match="constant signature"):
        COARSEOracle().fit(
            [frozenset(range(4)), frozenset({4, 5})],
            CHAIN_M,
            CHAIN_ENV_ORDER,
            data_dict,
        )
    for rows in (4, 8):
        M = np.zeros((rows, 1), dtype=bool)
        M[:2] = True
        with pytest.raises(ValueError, match="shape"):
            COARSEOracle().fit(infer_partition(M), M, ["1"], data_dict)
    M = CHAIN_M[:, :1]
    with pytest.raises(ValueError, match="env_order"):
        COARSEOracle().fit(infer_partition(M), M, ["1", "2"], data_dict)


def test_coarse_oracle_accepts_finer_partition():
    """A refinement of a row class is valid: same-signature blocks, disjoint, covered,
    with the DAG kept acyclic and blocks preserved."""
    import networkx as nx

    data_dict = chain_data_dict(300, 0)
    finer = [frozenset({0}), frozenset({1}), frozenset({2, 3}), frozenset({4, 5})]
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        finer, CHAIN_M, CHAIN_ENV_ORDER, data_dict
    )
    assert set(map(frozenset, model.partition)) == set(map(frozenset, finer))
    assert np.isfinite(model.score)
    assert nx.is_directed_acyclic_graph(model.dag)
    # Same-signature siblings never parent each other: only τ-earlier blocks do.
    for child, parents in model.parent_sets.items():
        for parent in parents:
            assert model.supports[parent] <= model.supports[child]


def test_coarse_oracle_rejects_overlapping_refinement():
    """A refinement must cover every variable exactly once."""
    data_dict = chain_data_dict(300, 0)
    with pytest.raises(ValueError, match="disjoint"):
        COARSEOracle().fit(
            [frozenset({0}), frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})],
            CHAIN_M,
            CHAIN_ENV_ORDER,
            data_dict,
        )


def _set(col, value, envs=("1",)):
    def corrupt(d):
        for e in envs:
            d[e][:, col] = value(d[e])

    return corrupt


_ALL = ("obs", "1", "2", "3")


@pytest.mark.parametrize(
    "corrupt, match",
    [
        (_set(2, lambda X: np.where(np.arange(len(X)) == 3, np.nan, X[:, 2])), "NaN"),
        (lambda d: d.update({"1": d["1"][:1]}), "2 rows"),
        (_set(2, lambda X: 3.0), "constant"),
        (_set(4, lambda X: X[:, 0]), "identical"),
        (_set(1, lambda X: 2.0 * X[:, 0], _ALL), "linearly dependent"),
        (_set(2, lambda X: 2.0 * X[:, 3] + 3.0, _ALL), "linearly dependent"),
    ],
    ids=[
        "nan",
        "one_row",
        "constant",
        "identical",
        "dependent_block",
        "dependent_parents",
    ],
)
def test_check_input_rejects_bad_data(corrupt, match):
    data_dict = chain_data_dict(300, 0)
    corrupt(data_dict)
    with pytest.raises(ValueError, match=match):
        COARSE().fit(data_dict, check_input=True)


def test_check_input_accepts_noisy_linear_combination():
    """A linear combination plus small noise passes the rank check."""
    noise_rng = np.random.default_rng(99)
    data_dict = chain_data_dict(300, 0)
    for v in data_dict.values():
        v[:, 1] = 2.0 * v[:, 0] + 3.0 + 1e-6 * noise_rng.standard_normal(v.shape[0])
    assert np.isfinite(COARSE().fit(data_dict, check_input=True).score)


@pytest.mark.parametrize("cls", ["COARSE", "COARSEOracle"])
def test_check_input_gates_data_validation(cls, monkeypatch):
    """Data validation runs only with `check_input=True`."""
    calls = []
    monkeypatch.setattr(coarse_mod, "_check_data_dict", lambda d: calls.append(d))
    data_dict = chain_data_dict(300, 0)
    fit = (
        (lambda **kw: COARSE().fit(data_dict, **kw))
        if cls == "COARSE"
        else (lambda **kw: _oracle(data_dict, **kw))
    )
    fit()
    assert calls == []
    fit(check_input=True)
    assert len(calls) == 1


def test_fit_rejects_bad_hyperparameters():
    """k not a positive int, alpha outside (0, 1), negative λ, avg together with k."""
    data_dict = chain_data_dict(200, 0)
    fits = (
        lambda **kw: COARSE().fit(data_dict, **kw),
        lambda **kw: _oracle(data_dict, **kw),
    )
    for fit in fits:
        for k in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="k must be"):
                fit(k=k)
        with pytest.raises(ValueError, match="lambda_pen"):
            fit(lambda_pen=-1.0)
        with pytest.raises(ValueError, match="avg"):
            fit(k=1, avg=True)
    for alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            COARSE().fit(data_dict, alpha=alpha)


def test_fit_custom_baseline_key():
    """A non-"obs" baseline works plain and with k; the default key then raises."""
    data_dict = {
        ("ctrl" if k == "obs" else k): v for k, v in chain_data_dict(1500, 0).items()
    }
    plain = COARSE().fit(data_dict, alpha=1e-4, baseline_key="ctrl")
    assert plain.env_order == CHAIN_ENV_ORDER
    assert set(plain.dag.edges) == CHAIN_EDGES
    pca = COARSE().fit(data_dict, alpha=1e-4, baseline_key="ctrl", k=2)
    assert set(pca.dag.edges) == CHAIN_EDGES
    with pytest.raises(ValueError, match="baseline"):
        COARSE().fit(data_dict)


def test_fit_trivial_inputs():
    """One variable, and no interventional env: a single-block, edgeless, finite fit."""
    rng = np.random.default_rng(0)
    one_var = COARSE().fit(
        {"obs": rng.standard_normal((200, 1)), "1": 2.0 + rng.standard_normal((200, 1))}
    )
    assert list(one_var.dag.nodes) == [(0,)]
    obs_only = COARSE().fit({"obs": sample_chain_dataset(500, rng)})
    assert obs_only.M.shape == (6, 0)
    assert obs_only.partition == [frozenset(range(6))]
    for model in (one_var, obs_only):
        assert model.dag.number_of_edges() == 0
        assert np.isfinite(model.score)


def test_fit_degenerate_tiny_n_returns_minus_inf():
    """n_e <= p in every env: an empty DAG with score -inf, no exception."""
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(5, rng),
        "1": sample_chain_dataset(5, rng, shift_targets=(0, 1)),
    }
    model = COARSE().fit(data_dict)
    assert model.score == -np.inf
    assert model.dag.number_of_edges() == 0


@pytest.mark.parametrize("fully_connected", [False, True])
def test_expand_coarsened_dag(fully_connected):
    """Every variable pair across an edge, plus forward pairs within a block if requested."""
    model = COARSE()
    model.num_features = 6
    model.dag = nx.DiGraph()
    a, b, c = (3, 0, 2), (1,), (4, 5)
    model.dag.add_nodes_from([a, b, c])
    model.dag.add_edges_from([(a, c), (b, c)])
    expected = np.zeros((6, 6), dtype=int)
    for src in a + b:
        expected[src, list(c)] = 1
    if fully_connected:
        for src, dst in ((0, 2), (0, 3), (2, 3), (4, 5)):
            expected[src, dst] = 1
    assert np.array_equal(model.expand_coarsened_dag(fully_connected), expected)


# ---------------------------------------------------------------------------
# kPC-COARSE and avg-COARSE
# ---------------------------------------------------------------------------
def _scaled_by_obs_std(data_dict):
    obs_std = data_dict["obs"].std(0)
    return {ek: (X - X.mean(0)) / obs_std for ek, X in data_dict.items()}


@pytest.mark.parametrize("inflate", [False, True])
def test_kpc_full_rank_matches_plain_on_obs_scaled_data(inflate):
    """k >= block size equals plain COARSE on obs-std-scaled data, also when a non-obs env has
    inflated variance."""
    data_dict = chain_data_dict(1500, 42)
    if inflate:
        data_dict["2"][:, [2, 3]] *= 3.0
    plain = _oracle(_scaled_by_obs_std(data_dict))
    pca = _oracle(data_dict, k=2)
    assert set(pca.dag.edges) == set(plain.dag.edges) == CHAIN_EDGES
    np.testing.assert_allclose(plain.score, pca.score, rtol=1e-10)


def test_avg_matches_plain_on_block_means():
    """avg equals plain COARSE on the obs-std-scaled block means."""
    data_dict = chain_data_dict(1500, 42)
    reduced = {
        ek: np.column_stack([X[:, sorted(b)].mean(1) for b in CHAIN_PARTITION])
        for ek, X in _scaled_by_obs_std(data_dict).items()
    }
    plain = COARSEOracle(rng=np.random.default_rng(0)).fit(
        [frozenset({i}) for i in range(3)], CHAIN_M[[0, 2, 4]], CHAIN_ENV_ORDER, reduced
    )
    avg = _oracle(data_dict, avg=True)
    node = [tuple(sorted(b)) for b in CHAIN_PARTITION]
    assert {(node[i], node[j]) for (i,), (j,) in plain.dag.edges} == set(avg.dag.edges)
    np.testing.assert_allclose(plain.score, avg.score, rtol=1e-10)


def test_kpc_project_blocks_clamps_to_svd_rank():
    """With fewer obs rows than k, k_j is the SVD rank and blocks stay contiguous."""
    rng = np.random.default_rng(0)
    env_arrays = {
        "obs": rng.standard_normal((2, 8)),
        "1": rng.standard_normal((100, 8)),
    }
    partition = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})]
    proj_arrays, proj_partition, _ = _project_blocks(
        partition, env_arrays, k=3, baseline_key="obs"
    )
    assert proj_partition == [frozenset({0, 1}), frozenset({2, 3})]
    assert all(arr.shape[1] == 4 for arr in proj_arrays.values())
