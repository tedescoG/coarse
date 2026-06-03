"""K-fold cross-validation for the M-step threshold α in COARSE.

For each α ∈ alpha_grid, for each fold k ∈ {1..K}:
  1. Fit COARSE on the (-k) train data at α → (partition, parent_sets).
  2. Build train-fold per-env Σ̂ via `compute_env_stats`; extract (B, Σ) per
     (env, block) via the shared Schur-complement helper from `scoring.py`.
  3. Center the test fold with the *train* per-env mean (not the test mean —
     centering with the test mean would mechanically shrink residuals and
     bias L_CV(α)).
  4. Sum log N(X_test_block | X_test_parents @ B^T, Σ) over (env, block).

Aggregate L_CV(α) = Σ_k L^{(k)}(α). Pick α̂ = argmax; refit COARSE on the full
data at α̂. Forward COARSE attributes onto the COARSECV instance so downstream
consumers that read `dag`, `score`, `fit_runtime_sec`, etc. work unchanged.

The Gaussian log-likelihood retains the trace term (it is *not* constant
across α here — the train-fitted Σ is not the test-fold MLE) and the
``|block|·log(2π)`` constant (|block| varies with the partition, so dropping
it would bias selection toward partitions with larger blocks).

RNG contract (documented for reproducibility — tests pin this exactly):
  self.rng.spawn(3) → [splitter_rng, refit_rng, inner_root_rng]
    - splitter_rng draws the per-env row permutations.
    - refit_rng feeds the final full-data COARSE refit.
    - inner_root_rng.spawn(|A|·K) reshape (|A|, K) → per-(α, fold) inner COARSE
      RNGs. Spawned eagerly so that changing the α grid size does not reshuffle
      folds or the refit.

"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from scipy import linalg as sla

from coarse.coarse import COARSE, _normalize_data_dict
from coarse.scoring import (
    _block_indices,
    _block_regression_from_sigma,
    _parents_indices,
    compute_env_stats,
)
from coarse.types import EnvKey

DEFAULT_ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 0.05, 0.1)
DEFAULT_N_FOLDS: int = 5


def _kfold_split_env(
    env_arrays: dict[EnvKey, np.ndarray],
    n_folds: int,
    rng: np.random.Generator,
) -> list[tuple[dict[EnvKey, np.ndarray], dict[EnvKey, np.ndarray]]]:
    """Per-environment K-fold split.

    Each environment's rows are permuted independently and partitioned into
    ``n_folds`` contiguous chunks via ``np.array_split`` (which handles
    non-divisible row counts by making the last few chunks one element smaller).

    Raises
    ------
    ValueError
        If any environment has ``n_e < n_folds``. A silent shrink would make
        ``L^{(k)}`` incoherent across environments (envs would contribute
        different numbers of test rows per fold).

    Returns
    -------
    list of (train_dict, test_dict) pairs, length ``n_folds``. Each dict has
    the same keys as ``env_arrays``; the row counts sum to the original
    ``n_e`` per env.
    """
    chunks: dict[EnvKey, list[np.ndarray]] = {}
    for ek, X in env_arrays.items():
        n_e = X.shape[0]
        if n_e < n_folds:
            raise ValueError(
                f"env {ek!r} has n_e={n_e} < n_folds={n_folds}"
            )
        chunks[ek] = np.array_split(rng.permutation(n_e), n_folds)

    pairs: list[tuple[dict[EnvKey, np.ndarray], dict[EnvKey, np.ndarray]]] = []
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
        pairs.append((train_dict, test_dict))
    return pairs


def _heldout_block_log_lik(
    block_idx: np.ndarray,
    parent_idx: np.ndarray,
    B_train: np.ndarray,
    Sigma_train: np.ndarray,
    X_test_block_centered: np.ndarray,
    X_test_parents_centered: np.ndarray,
) -> float:
    """Held-out Gaussian log-density for one (env, block) cell.

    Computes
        log N(X_test_block | X_test_parents @ B_train.T, Sigma_train)
            = -n_test/2 · [ r_j · log(2π) + log|Σ| + tr(Σ^{-1} · Rᵀ R / n_test) ]
            = -1/2 · [ n_test · r_j · log(2π) + n_test · log|Σ| + tr(Σ^{-1} Rᵀ R) ]

    where ``R = X_test_block_centered − X_test_parents_centered @ B_train.T``
    are the test-fold residuals under the train-fit parameters.

    The implementation evaluates ``tr(Σ^{-1} Rᵀ R) = ⟨R, Σ^{-1} R⟩`` via one
    Cholesky solve plus an einsum contraction; ``Σ^{-1}`` is never formed.

    Returns ``-inf`` (no ridge/pseudoinverse rescue) when:
      - the test fold is empty,
      - ``Sigma_train`` is non-PD (Cholesky failure or ``slogdet`` sign ≤ 0).
    """
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

    # tr(Σ^{-1} Rᵀ R) without forming Σ^{-1}: cho_solve gives Z = Σ^{-1} Rᵀ,
    # then ⟨R, Zᵀ⟩ = Σ_{i,p} R[i,p] · Z[p,i] = tr(R Σ^{-1} Rᵀ).
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
    *,
    scale: bool = False,
) -> float:
    """One (α, fold) cell of the CV loop.

    Fits COARSE on the train fold at ``alpha``, then sums the held-out
    Gaussian log-likelihood across (env, block). Returns ``-inf`` if any
    block evaluation fails (singular train Σ_PaPa, non-PD train residual
    Σ, or empty test fold) — the principled-stats short-circuit matches the
    BIC scorer's ``-inf`` posture at ``scoring.py:151-154, 163-164``.

    When ``scale=True`` the train fold is z-scored per env (after centering)
    before building ``env_stats_train``, and the *same* train-derived (μ, σ)
    are applied to the test fold. This is the CV mirror of ``COARSE.fit``'s
    ``scale=True`` path (``coarse.py:171-175``); ``scale`` is also forwarded
    to the inner ``COARSE.fit`` so the partition + parent-sets are produced
    from the scaled BIC. The Jacobian of the per-env z-score is constant
    across α (σ depends on data, not on the partition), so it shifts every
    L_CV(α) by the same amount and argmax-selection is preserved.
    """
    model = COARSE(rng=rng).fit(
        train_dict,
        alpha=alpha,
        lambda_pen=lambda_pen,
        refine_test=refine_test,
        baseline_key=baseline_key,
        scale=scale,
    )

    # Centering contract: train means → applied to both train and test. Using
    # test-fold means here would mechanically shrink residuals and bias the
    # CV objective upward as α changes the partition.
    env_means = {ek: v.mean(axis=0, keepdims=True) for ek, v in train_dict.items()}
    if scale:
        env_sds: dict[EnvKey, np.ndarray] = {}
        centered_train: dict[EnvKey, np.ndarray] = {}
        for ek, v in train_dict.items():
            c = v - env_means[ek]
            sd = c.std(axis=0, keepdims=True)
            env_sds[ek] = np.where(sd > 0.0, sd, 1.0)
            centered_train[ek] = c / env_sds[ek]
    else:
        env_sds = None  # type: ignore[assignment]
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
                return -np.inf

            Xt_c = test_dict[ek] - env_means[ek]
            if scale:
                Xt_c = Xt_c / env_sds[ek]
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
                return -np.inf
            total += ll
    return total


class COARSECV:
    """K-fold CV wrapper around COARSE that selects α from a grid.

    Usage mirrors ``COARSE`` (sklearn-style ``.fit(data_dict) -> self``).
    After ``fit``, the instance exposes:

      - ``alpha_grid``                      tuple of evaluated α's
      - ``n_folds``                         int
      - ``cv_per_fold_log_lik``             ndarray (|A|, K), held-out log-lik
      - ``cv_log_lik``                      dict α → float (sum across folds)
      - ``best_alpha``                      float, argmax of cv_log_lik
      - ``final_model``                     COARSE refit on the full data at α̂
      - ``cv_runtime_sec``                  float, total CV loop wall time
      - ``dag``, ``score``, ``partition``, ``parent_sets``, ``M``,
        ``env_order``, ``supports``, ``candidate_pools``, ``linear_extension_``,
        ``num_features``, ``fit_runtime_sec``, ``fit_metadata``,
        ``expand_coarsened_dag`` — forwarded from ``final_model`` so any
        consumer that reads a COARSE works on a COARSECV unchanged.
    """

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
        scale: bool = False,
    ) -> "COARSECV":
        cv_start = time.perf_counter()

        env_arrays, baseline_key = _normalize_data_dict(data_dict, baseline_key)
        alpha_grid = tuple(float(a) for a in alpha_grid)
        if len(alpha_grid) == 0:
            raise ValueError("alpha_grid must contain at least one threshold")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        splitter_rng, refit_rng, inner_root_rng = self.rng.spawn(3)

        fold_pairs = _kfold_split_env(env_arrays, n_folds, splitter_rng)

        inner_rngs = np.asarray(
            inner_root_rng.spawn(len(alpha_grid) * n_folds),
            dtype=object,
        ).reshape(len(alpha_grid), n_folds)

        per_fold = np.full((len(alpha_grid), n_folds), -np.inf, dtype=float)
        for i, alpha in enumerate(alpha_grid):
            for k, (tr, te) in enumerate(fold_pairs):
                per_fold[i, k] = _evaluate_fold(
                    tr,
                    te,
                    alpha,
                    lambda_pen,
                    refine_test,
                    baseline_key,
                    inner_rngs[i, k],
                    scale=scale,
                )

        sums = per_fold.sum(axis=1)
        cv_log_lik = {alpha_grid[i]: float(sums[i]) for i in range(len(alpha_grid))}

        if not np.any(np.isfinite(sums)):
            raise RuntimeError(
                "all (α, fold) pairs failed; check data sufficiency "
                "(n_e per env, |block| relative to test-fold size)"
            )

        # Argmax over finite sums; tiebreak (exact float equality, which is the
        # common case when several α's produce the same partition) picks the
        # larger α — sparser fit, deterministic.
        best_i = int(
            max(
                (i for i in range(len(alpha_grid)) if np.isfinite(sums[i])),
                key=lambda i: (sums[i], alpha_grid[i]),
            )
        )
        best_alpha = alpha_grid[best_i]

        final_model = COARSE(rng=refit_rng).fit(
            data_dict,
            alpha=best_alpha,
            lambda_pen=lambda_pen,
            refine_test=refine_test,
            baseline_key=baseline_key,
            scale=scale,
        )

        self.alpha_grid = alpha_grid
        self.n_folds = n_folds
        self.cv_per_fold_log_lik = per_fold
        self.cv_log_lik = cv_log_lik
        self.best_alpha = best_alpha
        self.final_model = final_model
        self.cv_runtime_sec = time.perf_counter() - cv_start

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
    """Functional facade — ``COARSECV().fit(data_dict, **kwargs)``."""
    return COARSECV().fit(data_dict, **kwargs)


__all__ = [
    "COARSECV",
    "cv_coarse",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_N_FOLDS",
]
