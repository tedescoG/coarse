"""K-fold CV of COARSE's test threshold α: fit on the train folds, score the held-out Gaussian
log-likelihood (test folds centered with the train means), refit at the best α. The full
likelihood (with the log 2π and trace terms) is kept since blocks change across α."""

from __future__ import annotations

import time
from typing import Any, Iterator, Sequence

import numpy as np
from scipy import linalg as sla

from coarse.coarse import COARSE, _check_data_dict, _normalize_data_dict
from coarse.scoring import (
    _block_indices,
    _block_regression_from_sigma,
    _parents_indices,
    compute_env_stats,
)
from coarse.types import EnvKey

DEFAULT_ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.05, 0.1)
# Large K keeps n_train close to n, where the chosen α is applied.
DEFAULT_N_FOLDS: int = 10


def _kfold_split_env(
    env_arrays: dict[EnvKey, np.ndarray],
    n_folds: int,
    rng: np.random.Generator,
) -> Iterator[tuple[dict[EnvKey, np.ndarray], dict[EnvKey, np.ndarray]]]:
    """Per-env K-fold split, yielding `(train, test)` dicts one fold at a time."""
    chunks: dict[EnvKey, list[np.ndarray]] = {}
    for ek, X in env_arrays.items():
        n_e = X.shape[0]
        if n_e < n_folds:
            raise ValueError(
                f"env {ek!r} has n_e={n_e} < n_folds={n_folds}"
            )
        chunks[ek] = np.array_split(rng.permutation(n_e), n_folds)

    for k in range(n_folds):
        train_dict: dict[EnvKey, np.ndarray] = {}
        test_dict: dict[EnvKey, np.ndarray] = {}
        for ek, X in env_arrays.items():
            test_idx = chunks[ek][k]
            train_idx = np.concatenate(
                [chunks[ek][j] for j in range(n_folds) if j != k]
            )
            train_dict[ek] = X[train_idx]
            test_dict[ek] = X[test_idx]
        yield train_dict, test_dict


def _heldout_block_log_lik(
    block_idx: np.ndarray,
    parent_idx: np.ndarray,
    B_train: np.ndarray,
    Sigma_train: np.ndarray,
    X_test_block_centered: np.ndarray,
    X_test_parents_centered: np.ndarray,
) -> float:
    """`Σ_i log N(x_i | B pa_i, Σ)` over the test rows; -inf if empty or Σ is not PD."""
    n_test = X_test_block_centered.shape[0]
    r_j = block_idx.size
    if n_test <= 0:
        return -np.inf

    if parent_idx.size:
        R = X_test_block_centered - X_test_parents_centered @ B_train.T
    else:
        R = X_test_block_centered

    try:
        c, low = sla.cho_factor(Sigma_train, lower=True, check_finite=False)
    except sla.LinAlgError:
        return -np.inf

    sign, logdet = np.linalg.slogdet(Sigma_train)
    if sign <= 0 or not np.isfinite(logdet):
        return -np.inf

    # tr(R Σ⁻¹ Rᵀ) without forming Σ⁻¹.
    Z = sla.cho_solve((c, low), R.T, check_finite=False)
    trace_term = float(np.einsum("ij,ji->", R, Z))

    log_2pi = float(np.log(2.0 * np.pi))
    return -0.5 * (n_test * r_j * log_2pi + n_test * logdet + trace_term)


def _evaluate_fold(
    train_dict: dict[EnvKey, np.ndarray],
    test_dict: dict[EnvKey, np.ndarray],
    alpha: float,
    lambda_pen: float,
    refine_test: str,
    baseline_key: EnvKey,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """(held-out log-likelihood, input-check seconds) of COARSE fit on `train_dict` at
    `alpha`; the log-likelihood is -inf if any block fails."""
    model = COARSE(rng=rng).fit(
        train_dict,
        alpha=alpha,
        lambda_pen=lambda_pen,
        refine_test=refine_test,
        baseline_key=baseline_key,
    )

    # Both folds centered with the train means.
    env_means = {ek: v.mean(axis=0, keepdims=True) for ek, v in train_dict.items()}
    centered_train = {ek: v - env_means[ek] for ek, v in train_dict.items()}
    env_stats_train = compute_env_stats(centered_train)

    total = 0.0
    for block in model.partition:
        block_idx = _block_indices(block)
        parent_idx = _parents_indices(model.parent_sets[block])
        for ek, stats in env_stats_train.items():
            try:
                B, Sigma = _block_regression_from_sigma(
                    block_idx, parent_idx, stats.sigma,
                )
            except sla.LinAlgError:
                return -np.inf, model.check_runtime_sec

            Xt_c = test_dict[ek] - env_means[ek]
            n_test = Xt_c.shape[0]
            X_test_parents = (
                Xt_c[:, parent_idx]
                if parent_idx.size
                else np.empty((n_test, 0))
            )
            ll = _heldout_block_log_lik(
                block_idx,
                parent_idx,
                B,
                Sigma,
                Xt_c[:, block_idx],
                X_test_parents,
            )
            if not np.isfinite(ll):
                return -np.inf, model.check_runtime_sec
            total += ll
    return total, model.check_runtime_sec


class COARSECV:
    """COARSE with α chosen by K-fold CV. After `fit`: `best_alpha`, `cv_log_lik` (α → sum
    over folds), `cv_per_fold_log_lik`, `final_model`, `cv_runtime_sec` (excludes input
    validation), plus every COARSE attribute of `final_model`."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)

    def fit(
        self,
        data_dict: dict[EnvKey, Any],
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        n_folds: int = DEFAULT_N_FOLDS,
        lambda_pen: float = 1.0,
        refine_test: str = "welch",
        baseline_key: EnvKey = "obs",
    ) -> "COARSECV":
        _check_data_dict(data_dict)
        # Input checks are not timed: fold and refit check time is subtracted below.
        cv_start = time.perf_counter()

        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        alpha_grid = tuple(float(a) for a in alpha_grid)
        if len(alpha_grid) == 0:
            raise ValueError("alpha_grid must contain at least one threshold")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        splitter_rng, refit_rng, inner_root_rng = self.rng.spawn(3)

        inner_rngs = np.asarray(
            inner_root_rng.spawn(len(alpha_grid) * n_folds),
            dtype=object,
        ).reshape(len(alpha_grid), n_folds)

        # One fold in memory at a time; the (α, fold) RNG grid makes loop order irrelevant.
        per_fold = np.full((len(alpha_grid), n_folds), -np.inf, dtype=float)
        check_sec = 0.0
        for k, (tr, te) in enumerate(
            _kfold_split_env(env_arrays, n_folds, splitter_rng)
        ):
            for i, alpha in enumerate(alpha_grid):
                per_fold[i, k], fold_check_sec = _evaluate_fold(
                    tr,
                    te,
                    alpha,
                    lambda_pen,
                    refine_test,
                    baseline_key,
                    inner_rngs[i, k],
                )
                check_sec += fold_check_sec

        sums = per_fold.sum(axis=1)
        cv_log_lik = {alpha_grid[i]: float(sums[i]) for i in range(len(alpha_grid))}

        if not np.any(np.isfinite(sums)):
            raise RuntimeError(
                "all (α, fold) pairs failed; check data sufficiency "
                "(n_e per env, |block| relative to test-fold size)"
            )

        # Ties go to the smaller α.
        best_i = int(
            max(
                (i for i in range(len(alpha_grid)) if np.isfinite(sums[i])),
                key=lambda i: (sums[i], -alpha_grid[i]),
            )
        )
        best_alpha = alpha_grid[best_i]

        final_model = COARSE(rng=refit_rng).fit(
            data_dict,
            alpha=best_alpha,
            lambda_pen=lambda_pen,
            refine_test=refine_test,
            baseline_key=baseline_key,
        )

        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.cv_per_fold_log_lik = per_fold
        self.cv_log_lik = cv_log_lik
        self.best_alpha = best_alpha
        self.final_model = final_model
        self.cv_runtime_sec = (
            time.perf_counter() - cv_start - check_sec - final_model.check_runtime_sec
        )

        for attr in (
            "partition",
            "M",
            "env_order",
            "supports",
            "candidate_pools",
            "linear_extension_",
            "parent_sets",
            "dag",
            "score",
            "num_features",
            "fit_runtime_sec",
            "check_runtime_sec",
        ):
            setattr(self, attr, getattr(final_model, attr))
        self.fit_metadata = dict(final_model.fit_metadata)
        self.fit_metadata.update(
            {
                "alpha_grid": list(alpha_grid),
                "n_folds": n_folds,
                "best_alpha": best_alpha,
                "cv_log_lik_at_best": cv_log_lik[best_alpha],
                "cv_runtime_sec": self.cv_runtime_sec,
            }
        )
        return self

    expand_coarsened_dag = COARSE.expand_coarsened_dag


def cv_coarse(data_dict: dict[EnvKey, Any], **kwargs: Any) -> COARSECV:
    """`COARSECV().fit(data_dict, **kwargs)`."""
    return COARSECV().fit(data_dict, **kwargs)


__all__ = [
    "COARSECV",
    "cv_coarse",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_N_FOLDS",
]
