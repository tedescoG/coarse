"""Tests for COARSECV: alpha selection, splitter, held-out likelihood, RNG contract."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import coarse.coarse as coarse_mod
import coarse.cv as cv_mod
from coarse.coarse import COARSE
from coarse.cv import COARSECV, _evaluate_fold, _heldout_block_log_lik, _kfold_split_env
from coarse.scoring import (
    _block_indices,
    _block_regression_from_sigma,
    _parents_indices,
    compute_env_stats,
)

from conftest import chain_data_dict, sample_chain_dataset


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def test_cv_picks_best_alpha_and_forwards_refit():
    data_dict = chain_data_dict(800, 0)
    grid, n_folds = (1e-4, 1e-2, 0.1), 3
    cv = COARSECV(rng=np.random.default_rng(1)).fit(data_dict, alpha_grid=grid, n_folds=n_folds)

    assert cv.cv_per_fold_log_lik.shape == (len(grid), n_folds)
    assert set(cv.cv_log_lik) == set(grid)
    assert cv.cv_log_lik[cv.best_alpha] == max(cv.cv_log_lik.values())
    assert np.isfinite(cv.cv_log_lik[cv.best_alpha])

    assert set(cv.dag.edges) == set(cv.final_model.dag.edges)
    assert cv.score == cv.final_model.score
    assert cv.fit_runtime_sec == cv.final_model.fit_runtime_sec
    assert cv.fit_metadata["best_alpha"] == cv.best_alpha
    assert cv.fit_metadata["alpha_grid"] == list(grid)
    assert cv.fit_metadata["n_folds"] == n_folds


def test_cv_selection_excludes_failed_alpha_and_breaks_ties_to_smaller(monkeypatch):
    """An alpha with one -inf fold is excluded even if its other folds score best; equal sums
    go to the smaller alpha."""
    calls: dict[float, int] = {}

    def partial_failure(tr, te, alpha, *args, **kwargs):
        calls[alpha] = calls.get(alpha, 0) + 1
        if alpha == 1e-2:
            return (-np.inf if calls[alpha] == 2 else 100.0), 0.0
        return 1.0, 0.0

    data_dict = chain_data_dict(200, 0)
    monkeypatch.setattr(cv_mod, "_evaluate_fold", partial_failure)
    cv = COARSECV().fit(data_dict, alpha_grid=(1e-3, 1e-2), n_folds=3)
    assert cv.best_alpha == 1e-3
    assert cv.cv_log_lik == {1e-3: 3.0, 1e-2: -np.inf}

    monkeypatch.setattr(cv_mod, "_evaluate_fold", lambda *a, **k: (-1.0, 0.0))
    cv = COARSECV().fit(data_dict, alpha_grid=(1e-3, 1e-2, 1e-4, 0.05), n_folds=2)
    assert cv.best_alpha == 1e-4


def test_cv_fold_log_lik_independent_of_grid_size():
    """One alpha's per-fold log-lik does not change when other alphas join the grid."""
    data_dict = chain_data_dict(600, 4)
    one = COARSECV(rng=np.random.default_rng(7)).fit(data_dict, alpha_grid=(1e-3,), n_folds=3)
    three = COARSECV(rng=np.random.default_rng(7)).fit(data_dict, alpha_grid=(1e-3, 1e-2, 0.1), n_folds=3)
    np.testing.assert_array_equal(one.cv_per_fold_log_lik[0], three.cv_per_fold_log_lik[0])


def test_cv_rejects_bad_arguments_and_unscorable_data():
    """Empty grid, n_folds < 2, an env smaller than n_folds, and every fold failing."""
    data_dict = chain_data_dict(200, 0)
    with pytest.raises(ValueError, match="alpha_grid"):
        COARSECV().fit(data_dict, alpha_grid=(), n_folds=3)
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit(data_dict, alpha_grid=(1e-3,), n_folds=1)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit({"obs": rng.standard_normal((3, 2)), "1": rng.standard_normal((3, 2))}, n_folds=5)
    tiny = {"obs": sample_chain_dataset(8, rng), "1": sample_chain_dataset(8, rng, shift_targets=(0, 1))}
    with pytest.raises(RuntimeError, match="all"):
        COARSECV().fit(tiny, alpha_grid=(1e-3,), n_folds=2)


def test_cv_check_input_gates_data_validation(monkeypatch):
    """`check_input=True` rejects NaN and validates the input, every fold fit and the refit;
    the default validates nothing."""
    data_dict = chain_data_dict(300, 0)
    bad = {k: v.copy() for k, v in data_dict.items()}
    bad["obs"][3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        COARSECV().fit(bad, alpha_grid=(1e-3,), n_folds=2, check_input=True)

    calls = []
    monkeypatch.setattr(cv_mod, "_check_data_dict", calls.append)
    monkeypatch.setattr(coarse_mod, "_check_data_dict", calls.append)
    COARSECV().fit(data_dict, alpha_grid=(1e-3, 1e-2), n_folds=2)
    assert calls == []
    COARSECV().fit(data_dict, alpha_grid=(1e-3, 1e-2), n_folds=2, check_input=True)
    assert len(calls) == 1 + 2 * 2 + 1


def test_cv_refit_matches_fresh_fit_at_best_alpha():
    """`rng.spawn(3)` is [splitter, refit, inner]: a fresh fit with the refit RNG reproduces the
    final model."""
    data_dict = chain_data_dict(600, 3)
    kwargs = dict(lambda_pen=2.0, refine_test="ks")
    cv = COARSECV(rng=np.random.default_rng(42)).fit(data_dict, alpha_grid=(1e-3, 1e-2), n_folds=2, **kwargs)
    _, refit_rng, _ = np.random.default_rng(42).spawn(3)
    fresh = COARSE(rng=refit_rng).fit(data_dict, alpha=cv.best_alpha, **kwargs)
    assert set(cv.final_model.dag.edges) == set(fresh.dag.edges)
    assert cv.final_model.score == pytest.approx(fresh.score, rel=1e-12)


# ---------------------------------------------------------------------------
# splitter and held-out likelihood
# ---------------------------------------------------------------------------
def test_cv_splitter_partitions_every_env():
    """Per env and fold, train and test are disjoint and complete; test folds cover every row
    and differ in size by at most one."""
    rng = np.random.default_rng(0)
    env_arrays = {"obs": rng.standard_normal((23, 3)), "1": rng.standard_normal((15, 3))}
    pairs = list(_kfold_split_env(env_arrays, 5, np.random.default_rng(42)))
    assert len(pairs) == 5
    assert sorted(te["obs"].shape[0] for _, te in pairs) == [4, 4, 5, 5, 5]
    for ek, X in env_arrays.items():
        seen: set[tuple[float, ...]] = set()
        for tr, te in pairs:
            train_rows, test_rows = {tuple(r) for r in tr[ek]}, {tuple(r) for r in te[ek]}
            assert train_rows.isdisjoint(test_rows)
            assert len(tr[ek]) + len(te[ek]) == len(X)
            seen |= test_rows
        assert seen == {tuple(r) for r in X}


def test_heldout_log_lik_matches_scipy():
    """Equals the summed Gaussian log-density at the train-fit conditional mean, with and
    without parents; a non-PD Σ gives -inf."""
    rng = np.random.default_rng(1)
    r_j, s_j = 2, 3
    L = np.array([[0.5, 0.0], [0.1, 0.4]])
    B_true = rng.standard_normal((r_j, s_j))
    Xp = rng.standard_normal((700, s_j))
    Xb = Xp @ B_true.T + rng.standard_normal((700, r_j)) @ L
    X = np.column_stack([Xb, Xp])
    mu = X[:500].mean(axis=0)
    Xtr, Xte = X[:500] - mu, X[500:] - mu
    sigma = Xtr.T @ Xtr / 500
    b_idx = np.arange(r_j)
    for pa_idx in (np.empty(0, dtype=np.int64), np.arange(r_j, r_j + s_j)):
        B, Sigma = _block_regression_from_sigma(b_idx, pa_idx, sigma)
        ll = _heldout_block_log_lik(b_idx, pa_idx, B, Sigma, Xte[:, b_idx], Xte[:, pa_idx])
        means = Xte[:, pa_idx] @ B.T
        ref = sum(
            multivariate_normal(mean=m, cov=Sigma).logpdf(x) for m, x in zip(means, Xte[:, b_idx])
        )
        assert ll == pytest.approx(ref, rel=1e-10)

    non_pd = np.array([[1.0, 2.0], [2.0, 1.0]])
    no_pa = np.empty(0, dtype=np.int64)
    assert _heldout_block_log_lik(b_idx, no_pa, np.empty((2, 0)), non_pd, Xte[:, b_idx], Xte[:, no_pa]) == -np.inf


def test_evaluate_fold_centers_test_with_train_mean():
    """The fold value is the summed held-out log-lik with both folds centered on the train mean;
    shifting the test fold lowers it."""
    data_dict = chain_data_dict(600, 5)
    train, test = next(_kfold_split_env(data_dict, 3, np.random.default_rng(0)))
    args = (1e-3, 1.0, "welch", "obs")
    ll, _ = _evaluate_fold(train, test, *args, np.random.default_rng(0))

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
    assert _evaluate_fold(train, shifted, *args, np.random.default_rng(0))[0] < ll
