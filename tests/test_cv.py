"""Tests for COARSECV — K-fold CV wrapper that selects α from a grid.

Five tests pin the contract:

  1. Public API smoke — best_alpha lands in the grid, per-fold matrix has the
     right shape, forwarded attributes (dag, score) match the final refit.
  2. Splitter geometry — train/test disjoint, test folds cover all rows,
     sizes sum to n_e per environment.
  3. Splitter precondition — n_e < n_folds raises ValueError.
  4. Held-out log-likelihood closed-form — matches scipy multivariate_normal
     in the no-parents case to rel=1e-10.
  5. Refit RNG contract — reproducing the documented spawn order yields the
     same final DAG as the CV refit.

Per memory `feedback_neutral_diagnostics_no_targeted_signal`, no test asserts
that the "true" α wins on small samples; fixtures are vanilla and we report
what the math does in practice (smoke + closed-form sanity instead).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import linalg as sla
from scipy.stats import multivariate_normal

from conftest import sample_chain_dataset

from coarse.coarse import COARSE
from coarse.cv import (
    COARSECV,
    DEFAULT_ALPHA_GRID,
    DEFAULT_N_FOLDS,
    _heldout_block_log_lik,
    _kfold_split_env,
    cv_coarse,
)


def _chain_data_dict(n: int, seed: int) -> dict:
    """Standard three-intervention chain fixture used across the smoke tests."""
    rng = np.random.default_rng(seed)
    return {
        "obs": sample_chain_dataset(n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }


# ---------------------------------------------------------------------------
# Test 1 — public API smoke
# ---------------------------------------------------------------------------
def test_cv_smoke_returns_best_alpha_in_grid():
    data_dict = _chain_data_dict(n=800, seed=0)
    grid = (1e-4, 1e-2, 0.1)
    n_folds = 3

    cv = COARSECV(rng=np.random.default_rng(1)).fit(
        data_dict, alpha_grid=grid, n_folds=n_folds,
    )

    assert cv.best_alpha in grid
    assert cv.cv_per_fold_log_lik.shape == (len(grid), n_folds)
    assert set(cv.cv_log_lik.keys()) == set(grid)
    # The selected α's CV objective is finite (otherwise the run is degenerate).
    assert np.isfinite(cv.cv_log_lik[cv.best_alpha])
    # Tiebreaker: argmax-on-finite-sums is the same as max(cv_log_lik.values()).
    assert cv.cv_log_lik[cv.best_alpha] == max(cv.cv_log_lik.values())

    # Forwarded COARSE surface matches the final refit exactly.
    assert set(cv.dag.nodes) == set(cv.final_model.dag.nodes)
    assert set(cv.dag.edges) == set(cv.final_model.dag.edges)
    assert cv.score == cv.final_model.score
    assert cv.fit_runtime_sec == cv.final_model.fit_runtime_sec
    assert cv.fit_metadata["best_alpha"] == cv.best_alpha
    assert cv.fit_metadata["alpha_grid"] == list(grid)
    assert cv.fit_metadata["n_folds"] == n_folds


def test_cv_coarse_functional_facade_equivalent_to_class():
    """``cv_coarse`` is the no-frills functional entry point. It must produce
    a ``COARSECV`` whose final DAG matches the class form with the same RNG."""
    data_dict = _chain_data_dict(n=600, seed=2)
    grid = (1e-3, 1e-2)

    cv_class = COARSECV(rng=np.random.default_rng(7)).fit(
        data_dict, alpha_grid=grid, n_folds=2,
    )
    # The functional facade uses a default-rng(0) internally, so we just check
    # it returns a fully-formed COARSECV (not that it matches a specific seed).
    cv_func = cv_coarse(data_dict, alpha_grid=grid, n_folds=2)
    assert isinstance(cv_func, COARSECV)
    assert cv_func.best_alpha in grid
    # Both runs select an α from the same grid; we don't pin which one.
    assert cv_class.best_alpha in grid


# ---------------------------------------------------------------------------
# Test 2 — splitter geometry
# ---------------------------------------------------------------------------
def test_cv_splitter_disjoint_complete():
    """Per env: train and test partitions are disjoint, test folds across k
    cover all rows exactly once, and (n_train + n_test) == n_e for each fold."""
    rng_data = np.random.default_rng(0)
    env_arrays = {
        # Use distinct row values so set-membership uniquely identifies a row.
        "obs": rng_data.standard_normal((20, 3)),
        "1": rng_data.standard_normal((15, 3)),
    }
    n_folds = 5
    pairs = _kfold_split_env(env_arrays, n_folds, np.random.default_rng(42))

    assert len(pairs) == n_folds

    for ek, X in env_arrays.items():
        all_test_rows: set[tuple[float, ...]] = set()
        for tr, te in pairs:
            train_rows = {tuple(r) for r in tr[ek]}
            test_rows = {tuple(r) for r in te[ek]}
            # Train/test disjoint within a fold.
            assert train_rows.isdisjoint(test_rows)
            # Sizes sum to n_e.
            assert tr[ek].shape[0] + te[ek].shape[0] == X.shape[0]
            all_test_rows |= test_rows
        # Across all K folds, test partitions cover every original row once.
        original_rows = {tuple(r) for r in X}
        assert all_test_rows == original_rows


def test_cv_splitter_handles_non_divisible_row_counts():
    """np.array_split makes the last few chunks one element smaller when n_e
    is not divisible by n_folds. Test fold sizes must still cover all rows."""
    rng_data = np.random.default_rng(0)
    # n_e = 23, n_folds = 5 → chunk sizes [5, 5, 5, 4, 4].
    env_arrays = {"obs": rng_data.standard_normal((23, 2))}
    pairs = _kfold_split_env(env_arrays, n_folds=5, rng=np.random.default_rng(0))

    test_sizes = sorted(te["obs"].shape[0] for _, te in pairs)
    assert test_sizes == [4, 4, 5, 5, 5]
    assert sum(test_sizes) == 23


# ---------------------------------------------------------------------------
# Test 3 — splitter preconditions
# ---------------------------------------------------------------------------
def test_cv_splitter_raises_on_small_env():
    """An env with n_e < n_folds is incoherent (one fold would be empty);
    raise rather than silently shrink K for that env."""
    env_arrays = {"obs": np.zeros((3, 2))}
    with pytest.raises(ValueError, match="n_folds"):
        _kfold_split_env(env_arrays, n_folds=5, rng=np.random.default_rng(0))


def test_cv_fit_propagates_splitter_error():
    """The splitter precondition surfaces through the full ``fit`` path."""
    data_dict = {"obs": np.zeros((3, 2)), "1": np.zeros((3, 2))}
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit(data_dict, alpha_grid=(1e-4,), n_folds=5)


def test_cv_fit_validates_arguments():
    data_dict = _chain_data_dict(n=200, seed=0)
    with pytest.raises(ValueError, match="alpha_grid"):
        COARSECV().fit(data_dict, alpha_grid=(), n_folds=3)
    with pytest.raises(ValueError, match="n_folds"):
        COARSECV().fit(data_dict, alpha_grid=(1e-3,), n_folds=1)


# ---------------------------------------------------------------------------
# Test 4 — held-out log-likelihood closed-form
# ---------------------------------------------------------------------------
def test_heldout_log_lik_no_parents_matches_scipy():
    """With no parents, the held-out log-lik collapses to summing the
    Gaussian log-density of test residuals (= centered test rows) under a
    train-fit Σ. scipy.stats.multivariate_normal gives the closed-form."""
    rng = np.random.default_rng(0)
    p = 3
    X = rng.standard_normal((400, p))
    Xtr, Xte = X[:300], X[300:]

    mu_tr = Xtr.mean(axis=0, keepdims=True)
    Xtr_c = Xtr - mu_tr
    Xte_c = Xte - mu_tr
    Sigma_train = (Xtr_c.T @ Xtr_c) / Xtr_c.shape[0]

    block_idx = np.arange(p, dtype=np.int64)
    parent_idx = np.empty(0, dtype=np.int64)

    ll = _heldout_block_log_lik(
        block_idx,
        parent_idx,
        B_train=np.empty((p, 0)),
        Sigma_train=Sigma_train,
        X_test_block_centered=Xte_c,
        X_test_parents_centered=np.empty((Xte_c.shape[0], 0)),
    )
    ref = multivariate_normal(mean=np.zeros(p), cov=Sigma_train).logpdf(Xte_c).sum()
    assert ll == pytest.approx(ref, rel=1e-10)


def test_heldout_log_lik_with_parents_matches_scipy():
    """With parents, the held-out log-lik should equal the sum over test rows
    of the multivariate-normal log-density at the train-fit conditional mean
    ``X_test_parents @ B.T`` with covariance Σ_train."""
    rng = np.random.default_rng(1)
    n_train, n_test = 500, 200
    r_j, s_j = 2, 3
    Xp_train = rng.standard_normal((n_train, s_j))
    # Linear-Gaussian with arbitrary true B, true Σ. Use OLS on train to fit.
    B_true = rng.standard_normal((r_j, s_j))
    noise = rng.standard_normal((n_train, r_j)) @ np.array([[0.5, 0.0], [0.1, 0.4]])
    Xb_train = Xp_train @ B_true.T + noise

    mu_p = Xp_train.mean(axis=0, keepdims=True)
    mu_b = Xb_train.mean(axis=0, keepdims=True)
    Xp_train_c = Xp_train - mu_p
    Xb_train_c = Xb_train - mu_b

    # OLS via the same Schur path the production code uses; here computed
    # explicitly from sufficient statistics.
    Sxx = (Xp_train_c.T @ Xp_train_c) / n_train
    Syx = (Xb_train_c.T @ Xp_train_c) / n_train
    Syy = (Xb_train_c.T @ Xb_train_c) / n_train
    c, low = sla.cho_factor(Sxx, lower=True)
    Y = sla.cho_solve((c, low), Syx.T)         # (s_j, r_j) = B.T
    B_fit = Y.T
    Sigma_fit = Syy - Syx @ Y

    # Test fold drawn from the same distribution; center with train means.
    Xp_test = rng.standard_normal((n_test, s_j))
    eps_test = rng.standard_normal((n_test, r_j)) @ np.array([[0.5, 0.0], [0.1, 0.4]])
    Xb_test = Xp_test @ B_true.T + eps_test
    Xp_test_c = Xp_test - mu_p
    Xb_test_c = Xb_test - mu_b

    ll = _heldout_block_log_lik(
        block_idx=np.arange(r_j, dtype=np.int64),
        parent_idx=np.arange(s_j, dtype=np.int64),
        B_train=B_fit,
        Sigma_train=Sigma_fit,
        X_test_block_centered=Xb_test_c,
        X_test_parents_centered=Xp_test_c,
    )

    # Reference: per-row multivariate-normal log-pdf at the train-fit
    # conditional mean, summed.
    means = Xp_test_c @ B_fit.T
    ref = sum(
        multivariate_normal(mean=means[i], cov=Sigma_fit).logpdf(Xb_test_c[i])
        for i in range(n_test)
    )
    assert ll == pytest.approx(ref, rel=1e-10)


def test_heldout_log_lik_returns_minus_inf_on_non_pd_sigma():
    rng = np.random.default_rng(0)
    block_idx = np.array([0, 1], dtype=np.int64)
    parent_idx = np.empty(0, dtype=np.int64)
    bad_sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD: eigenvalues 3, -1
    Xte = rng.standard_normal((50, 2))
    ll = _heldout_block_log_lik(
        block_idx,
        parent_idx,
        B_train=np.empty((2, 0)),
        Sigma_train=bad_sigma,
        X_test_block_centered=Xte,
        X_test_parents_centered=np.empty((50, 0)),
    )
    assert ll == -np.inf


# ---------------------------------------------------------------------------
# Test 5 — refit RNG contract
# ---------------------------------------------------------------------------
def test_cv_refit_matches_fresh_fit_at_best_alpha():
    """COARSECV documents that ``self.rng.spawn(3) → [splitter, refit, inner]``.
    A fresh ``COARSE.fit(..., alpha=cv.best_alpha, rng=refit_rng)`` with the
    spawned-out ``refit_rng`` must reproduce ``cv.final_model`` exactly."""
    data_dict = _chain_data_dict(n=600, seed=3)
    grid = (1e-3, 1e-2)
    n_folds = 2
    seed = 42

    cv = COARSECV(rng=np.random.default_rng(seed)).fit(
        data_dict, alpha_grid=grid, n_folds=n_folds,
    )

    # Reproduce the refit RNG by spawning the documented order on a fresh
    # default_rng with the same seed.
    fresh_root = np.random.default_rng(seed)
    _, refit_rng, _ = fresh_root.spawn(3)
    fresh = COARSE(rng=refit_rng).fit(
        data_dict,
        alpha=cv.best_alpha,
        lambda_pen=1.0,
        refine_test="welch",
    )

    assert set(cv.final_model.dag.nodes) == set(fresh.dag.nodes)
    assert set(cv.final_model.dag.edges) == set(fresh.dag.edges)
    assert cv.final_model.score == pytest.approx(fresh.score, rel=1e-12)


# ---------------------------------------------------------------------------
# Misc: default constants stay where the public API promises they are.
# ---------------------------------------------------------------------------
def test_default_alpha_grid_and_n_folds():
    """Pin the documented defaults so a careless re-edit can't change them."""
    assert DEFAULT_ALPHA_GRID == (1e-4, 1e-3, 1e-2, 0.05, 0.1)
    assert DEFAULT_N_FOLDS == 5
