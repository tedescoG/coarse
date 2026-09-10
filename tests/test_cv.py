"""Tests for COARSECV: K-fold selection of alpha, splitter, held-out likelihood, refit RNG."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg as sla
from scipy.stats import multivariate_normal

from conftest import sample_chain_dataset

from coarse.coarse import COARSE
from coarse.cv import (
    COARSECV,
    _evaluate_fold,
    _heldout_block_log_lik,
    _kfold_split_env,
    cv_coarse,
)
from coarse.scoring import (
    _block_indices,
    _block_regression_from_sigma,
    _parents_indices,
    compute_env_stats,
)


def _chain_data_dict(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "obs": sample_chain_dataset(n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def test_cv_smoke_returns_best_alpha_in_grid():
    data_dict = _chain_data_dict(n=800, seed=0)
    grid = (1e-4, 1e-2, 0.1)
    n_folds = 3
    cv = COARSECV(rng=np.random.default_rng(1)).fit(data_dict, alpha_grid=grid, n_folds=n_folds)

    assert cv.best_alpha in grid
    assert cv.cv_per_fold_log_lik.shape == (len(grid), n_folds)
    assert set(cv.cv_log_lik.keys()) == set(grid)
    assert np.isfinite(cv.cv_log_lik[cv.best_alpha])
    assert cv.cv_log_lik[cv.best_alpha] == max(cv.cv_log_lik.values())

    # Forwarded attributes match the final refit.
    assert set(cv.dag.nodes) == set(cv.final_model.dag.nodes)
    assert set(cv.dag.edges) == set(cv.final_model.dag.edges)
    assert cv.score == cv.final_model.score
    assert cv.fit_runtime_sec == cv.final_model.fit_runtime_sec
    assert cv.fit_metadata["best_alpha"] == cv.best_alpha
    assert cv.fit_metadata["alpha_grid"] == list(grid)
    assert cv.fit_metadata["n_folds"] == n_folds


def test_cv_coarse_matches_class_with_default_rng():
    """`cv_coarse` is `COARSECV().fit(...)` with the default RNG."""
    data_dict = _chain_data_dict(n=600, seed=2)
    kwargs = dict(alpha_grid=(1e-3, 1e-2), n_folds=2)
    cv_class = COARSECV(rng=np.random.default_rng(0)).fit(data_dict, **kwargs)
    cv_func = cv_coarse(data_dict, **kwargs)
    assert isinstance(cv_func, COARSECV)
    assert cv_func.best_alpha == cv_class.best_alpha
    assert set(cv_func.dag.edges) == set(cv_class.dag.edges)
    assert cv_func.score == cv_class.score


def test_cv_fold_log_lik_independent_of_grid_size():
    """The per-fold log-lik of one alpha does not change when other alphas join the grid."""
    data_dict = _chain_data_dict(n=600, seed=4)
    one = COARSECV(rng=np.random.default_rng(7)).fit(data_dict, alpha_grid=(1e-3,), n_folds=3)
    three = COARSECV(rng=np.random.default_rng(7)).fit(
        data_dict, alpha_grid=(1e-3, 1e-2, 0.1), n_folds=3
    )
    np.testing.assert_array_equal(one.cv_per_fold_log_lik[0], three.cv_per_fold_log_lik[0])


def test_cv_partial_fold_failure_excludes_alpha(monkeypatch):
    """An alpha with one -inf fold is excluded even if its other folds score best."""
    import coarse.cv as cv_mod

    calls: dict[float, int] = {}

    def fake(tr, te, alpha, *args, **kwargs):
        calls[alpha] = calls.get(alpha, 0) + 1
        if alpha == 1e-2:
            return -np.inf if calls[alpha] == 2 else 100.0
        return 1.0

    monkeypatch.setattr(cv_mod, "_evaluate_fold", fake)
    cv = COARSECV().fit(_chain_data_dict(n=200, seed=0), alpha_grid=(1e-3, 1e-2), n_folds=3)
    assert cv.best_alpha == 1e-3
    assert cv.cv_log_lik[1e-2] == -np.inf
    assert cv.cv_log_lik[1e-3] == 3.0


def test_cv_fit_validates_arguments():
    data_dict = _chain_data_dict(n=200, seed=0)
    with pytest.raises(ValueError, match="alpha_grid"):
        COARSECV().fit(data_dict, alpha_grid=(), n_folds=3)
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit(data_dict, alpha_grid=(1e-3,), n_folds=1)


def test_cv_all_folds_fail_raises():
    rng = np.random.default_rng(0)
    data_dict = {
        "obs": sample_chain_dataset(8, rng),
        "1": sample_chain_dataset(8, rng, shift_targets=(0, 1)),
    }
    with pytest.raises(RuntimeError, match="all"):
        COARSECV().fit(data_dict, alpha_grid=(1e-3,), n_folds=2)


def test_cv_tiebreak_prefers_smaller_alpha(monkeypatch):
    import coarse.cv as cv_mod

    monkeypatch.setattr(cv_mod, "_evaluate_fold", lambda *a, **k: -1.0)
    grid = (1e-3, 1e-2, 1e-4, 0.05)
    cv = COARSECV(rng=np.random.default_rng(0)).fit(
        _chain_data_dict(n=200, seed=0), alpha_grid=grid, n_folds=2
    )
    assert cv.best_alpha == min(grid)


# ---------------------------------------------------------------------------
# splitter
# ---------------------------------------------------------------------------
def test_cv_splitter_disjoint_complete():
    """Per env and fold, train and test are disjoint and sum to n_e; test folds cover every row."""
    rng_data = np.random.default_rng(0)
    env_arrays = {"obs": rng_data.standard_normal((20, 3)), "1": rng_data.standard_normal((15, 3))}
    n_folds = 5
    pairs = list(_kfold_split_env(env_arrays, n_folds, np.random.default_rng(42)))
    assert len(pairs) == n_folds

    for ek, X in env_arrays.items():
        all_test_rows: set[tuple[float, ...]] = set()
        for tr, te in pairs:
            train_rows = {tuple(r) for r in tr[ek]}
            test_rows = {tuple(r) for r in te[ek]}
            assert train_rows.isdisjoint(test_rows)
            assert tr[ek].shape[0] + te[ek].shape[0] == X.shape[0]
            all_test_rows |= test_rows
        assert all_test_rows == {tuple(r) for r in X}


def test_cv_splitter_handles_non_divisible_row_counts():
    env_arrays = {"obs": np.random.default_rng(0).standard_normal((23, 2))}
    pairs = list(_kfold_split_env(env_arrays, n_folds=5, rng=np.random.default_rng(0)))
    test_sizes = sorted(te["obs"].shape[0] for _, te in pairs)
    assert test_sizes == [4, 4, 5, 5, 5]


def test_cv_raises_when_env_smaller_than_n_folds():
    with pytest.raises(ValueError, match="n_folds"):
        list(_kfold_split_env({"obs": np.zeros((3, 2))}, n_folds=5, rng=np.random.default_rng(0)))
    rng = np.random.default_rng(0)
    data_dict = {"obs": rng.standard_normal((3, 2)), "1": rng.standard_normal((3, 2))}
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit(data_dict, alpha_grid=(1e-4,), n_folds=5)


# ---------------------------------------------------------------------------
# held-out log-likelihood
# ---------------------------------------------------------------------------
def test_heldout_log_lik_no_parents_matches_scipy():
    rng = np.random.default_rng(0)
    p = 3
    X = rng.standard_normal((400, p))
    Xtr, Xte = X[:300], X[300:]
    mu_tr = Xtr.mean(axis=0, keepdims=True)
    Xtr_c, Xte_c = Xtr - mu_tr, Xte - mu_tr
    Sigma_train = (Xtr_c.T @ Xtr_c) / Xtr_c.shape[0]

    ll = _heldout_block_log_lik(
        np.arange(p, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        B_train=np.empty((p, 0)),
        Sigma_train=Sigma_train,
        X_test_block_centered=Xte_c,
        X_test_parents_centered=np.empty((Xte_c.shape[0], 0)),
    )
    ref = multivariate_normal(mean=np.zeros(p), cov=Sigma_train).logpdf(Xte_c).sum()
    assert ll == pytest.approx(ref, rel=1e-10)


def test_heldout_log_lik_with_parents_matches_scipy():
    """Equals the summed Gaussian log-density at the train-fit conditional mean."""
    rng = np.random.default_rng(1)
    n_train, n_test = 500, 200
    r_j, s_j = 2, 3
    L = np.array([[0.5, 0.0], [0.1, 0.4]])
    B_true = rng.standard_normal((r_j, s_j))
    Xp_train = rng.standard_normal((n_train, s_j))
    Xb_train = Xp_train @ B_true.T + rng.standard_normal((n_train, r_j)) @ L

    mu_p = Xp_train.mean(axis=0, keepdims=True)
    mu_b = Xb_train.mean(axis=0, keepdims=True)
    Xp_train_c, Xb_train_c = Xp_train - mu_p, Xb_train - mu_b

    # OLS from sufficient statistics.
    Sxx = (Xp_train_c.T @ Xp_train_c) / n_train
    Syx = (Xb_train_c.T @ Xp_train_c) / n_train
    Syy = (Xb_train_c.T @ Xb_train_c) / n_train
    c, low = sla.cho_factor(Sxx, lower=True)
    Y = sla.cho_solve((c, low), Syx.T)
    B_fit = Y.T
    Sigma_fit = Syy - Syx @ Y

    Xp_test = rng.standard_normal((n_test, s_j))
    Xb_test = Xp_test @ B_true.T + rng.standard_normal((n_test, r_j)) @ L
    Xp_test_c, Xb_test_c = Xp_test - mu_p, Xb_test - mu_b

    ll = _heldout_block_log_lik(
        block_idx=np.arange(r_j, dtype=np.int64),
        parent_idx=np.arange(s_j, dtype=np.int64),
        B_train=B_fit,
        Sigma_train=Sigma_fit,
        X_test_block_centered=Xb_test_c,
        X_test_parents_centered=Xp_test_c,
    )
    means = Xp_test_c @ B_fit.T
    ref = sum(
        multivariate_normal(mean=means[i], cov=Sigma_fit).logpdf(Xb_test_c[i])
        for i in range(n_test)
    )
    assert ll == pytest.approx(ref, rel=1e-10)


def test_heldout_log_lik_returns_minus_inf_on_non_pd_sigma():
    Xte = np.random.default_rng(0).standard_normal((50, 2))
    ll = _heldout_block_log_lik(
        np.array([0, 1], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        B_train=np.empty((2, 0)),
        Sigma_train=np.array([[1.0, 2.0], [2.0, 1.0]]),  # eigenvalues 3, -1
        X_test_block_centered=Xte,
        X_test_parents_centered=np.empty((50, 0)),
    )
    assert ll == -np.inf


def test_evaluate_fold_centers_test_with_train_mean():
    """The fold value is the summed held-out log-lik with both folds centered on the train mean,
    so shifting the test fold lowers it."""
    data_dict = _chain_data_dict(n=600, seed=5)
    train, test = next(_kfold_split_env(data_dict, 3, np.random.default_rng(0)))
    args = (1e-3, 1.0, "welch", "obs")
    ll = _evaluate_fold(train, test, *args, np.random.default_rng(0))

    model = COARSE(rng=np.random.default_rng(0)).fit(train, alpha=1e-3)
    means = {ek: v.mean(axis=0, keepdims=True) for ek, v in train.items()}
    stats = compute_env_stats({ek: v - means[ek] for ek, v in train.items()})
    expected = 0.0
    for block in model.partition:
        b_idx, pa_idx = _block_indices(block), _parents_indices(model.parent_sets[block])
        for ek, st in stats.items():
            B, Sigma = _block_regression_from_sigma(b_idx, pa_idx, st.sigma)
            Xt = test[ek] - means[ek]
            expected += _heldout_block_log_lik(b_idx, pa_idx, B, Sigma, Xt[:, b_idx], Xt[:, pa_idx])
    assert ll == pytest.approx(expected, rel=1e-12)

    shifted = {ek: v + 5.0 for ek, v in test.items()}
    assert _evaluate_fold(train, shifted, *args, np.random.default_rng(0)) < ll


# ---------------------------------------------------------------------------
# refit RNG contract
# ---------------------------------------------------------------------------
def test_cv_refit_matches_fresh_fit_at_best_alpha():
    """`rng.spawn(3)` is [splitter, refit, inner]; a fresh fit with the refit RNG and the same
    kwargs reproduces the final model."""
    data_dict = _chain_data_dict(n=600, seed=3)
    seed = 42
    kwargs = dict(lambda_pen=2.0, refine_test="ks")
    cv = COARSECV(rng=np.random.default_rng(seed)).fit(
        data_dict, alpha_grid=(1e-3, 1e-2), n_folds=2, **kwargs
    )
    _, refit_rng, _ = np.random.default_rng(seed).spawn(3)
    fresh = COARSE(rng=refit_rng).fit(data_dict, alpha=cv.best_alpha, **kwargs)

    assert set(cv.final_model.dag.nodes) == set(fresh.dag.nodes)
    assert set(cv.final_model.dag.edges) == set(fresh.dag.edges)
    assert cv.final_model.score == pytest.approx(fresh.score, rel=1e-12)
