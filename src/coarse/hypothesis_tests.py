"""Build the interventional descendant indicator matrix M.

Statical hypotheses tests:
- Welch t-test
- Kolmogorov-Smirnov test
- Energy-distance test (dcor.homogeneity.energy_test)
- Gaussian LRT (assumes Gaussian data)

All test are exposed via TEST_REGISTRY and the `compute_M` dispatcher.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

import dcor
import numpy as np
from scipy import stats

from coarse.types import EnvKey


def welch_p(x: np.ndarray, y: np.ndarray) -> float:
    """Welch t-test p-value for `H0: mean(x) == mean(y)`."""
    return float(stats.ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue)


def gaussian_lrt_p(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian likelihood-ratio test p-value for
    `H0: x, y come from the same N(mu, sigma^2)`  vs  different normals.
    """
    m_x = x.size
    m_y = y.size
    mu_x = x.mean()
    mu_y = y.mean()
    s2_x = max(float(np.mean((x - mu_x) ** 2)), 1e-12)
    s2_y = max(float(np.mean((y - mu_y) ** 2)), 1e-12)
    combined = np.concatenate([x, y])
    mu_c = combined.mean()
    s2_c = max(float(np.mean((combined - mu_c) ** 2)), 1e-12)
    ll_sep = -0.5 * m_x * (np.log(2 * np.pi * s2_x) + 1) \
             - 0.5 * m_y * (np.log(2 * np.pi * s2_y) + 1)
    ll_null = -0.5 * (m_x + m_y) * (np.log(2 * np.pi * s2_c) + 1)
    LR = 2.0 * (ll_sep - ll_null)
    return float(stats.chi2.sf(LR, df=2))


def ks_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov p-value for `H0: F_x == F_y`."""
    return float(stats.ks_2samp(x, y, alternative="two-sided", mode="auto").pvalue)


def energy_p(
    x: np.ndarray,
    y: np.ndarray,
    num_resamples: int = 199,
    rng: np.random.Generator | None = None,
) -> float:
    """Energy-distance two-sample test (dcor.homogeneity.energy_test) p-value.

    Higher num_resamples = more stable p-values at the cost of runtime;
    """
    seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
    result = dcor.homogeneity.energy_test(
        x.reshape(-1, 1),
        y.reshape(-1, 1),
        num_resamples=num_resamples,
        random_state=seed,
    )
    return float(result.p_value)


TEST_REGISTRY: dict[str, Callable[..., float]] = {
    "welch": welch_p,
    "ks": ks_p,
    "energy": energy_p,
    "gaussian_lrt": gaussian_lrt_p,  # Gaussian likelihood-ratio test (mean + variance)
}


def _normalize_env_data(env_value: Any) -> np.ndarray:
    """Coerce one env entry to a contiguous float64 (n_e, p) ndarray.

    Targets and types are
    accepted-and-ignored (COARSE doesn't use intervention-target metadata).
    """
    if isinstance(env_value, dict):
        if "data" not in env_value:
            raise ValueError("dict-form env value must contain a 'data' key")
        arr = env_value["data"]
    elif isinstance(env_value, tuple):
        if len(env_value) == 0:
            raise ValueError("tuple-form env value cannot be empty")
        arr = env_value[0]
    else:
        arr = env_value
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"env data must be 2-D, got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"env data needs at least 2 rows, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("env data contains NaN or inf")
    return arr


def _vectorized_welch(X0: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    """Vectorised Welch t-test across columns: returns 1-D array of length p."""
    result = stats.ttest_ind(X0, Xe, equal_var=False, axis=0, nan_policy="omit")
    pvals = np.asarray(result.pvalue, dtype=np.float64)
    # SciPy returns NaN when one column is constant in both samples; treat as p = 1
    # (no evidence of a distributional change).
    return np.where(np.isnan(pvals), 1.0, pvals)


def _vectorized_gaussian_lrt(X0: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    """Vectorised Gaussian LRT across columns.Returns a 1-D p-value array of length p."""
    m_x = X0.shape[0]
    m_y = Xe.shape[0]
    mu_x = X0.mean(axis=0)
    mu_y = Xe.mean(axis=0)
    s2_x = np.maximum(((X0 - mu_x) ** 2).mean(axis=0), 1e-12)
    s2_y = np.maximum(((Xe - mu_y) ** 2).mean(axis=0), 1e-12)
    # Pooled (null) mean and variance — avoid materialising the concatenated
    # array column-wise; use the closed-form combined sample variance instead.
    total = m_x + m_y
    mu_c = (m_x * mu_x + m_y * mu_y) / total
    s2_c = np.maximum(
        (m_x * (s2_x + (mu_x - mu_c) ** 2) + m_y * (s2_y + (mu_y - mu_c) ** 2))
        / total,
        1e-12,
    )
    ll_sep = (
        -0.5 * m_x * (np.log(2 * np.pi * s2_x) + 1)
        - 0.5 * m_y * (np.log(2 * np.pi * s2_y) + 1)
    )
    ll_null = -0.5 * total * (np.log(2 * np.pi * s2_c) + 1)
    LR = 2.0 * (ll_sep - ll_null)
    return np.asarray(stats.chi2.sf(LR, df=2), dtype=np.float64)


def compute_M(
    data_dict: dict[EnvKey, Any],
    alpha: float,
    test_name: str = "welch",
    rng: np.random.Generator | None = None,
    baseline_key: EnvKey = "obs",
) -> tuple[np.ndarray, list[EnvKey]]:
    """Descendant Test. Build M in {0,1}^{|V| x (|E|-1)} by testing
    every variable in every non-baseline environment against the baseline.

    Returns
    -------
    M : ndarray of bool, shape (|V|, |E|-1)
        M[v, e] = 1 iff the test p-value is below alpha.
    env_order : list[EnvKey]
        Non-baseline env keys in M's column order (sorted lex by key).
    """
    if baseline_key not in data_dict:
        raise ValueError(
            f"data_dict must contain a baseline env keyed {baseline_key!r}"
        )
    test_name_lc = test_name.lower()
    if test_name_lc not in TEST_REGISTRY:
        raise ValueError(
            f"unknown test_name {test_name!r}; available: {sorted(TEST_REGISTRY)}"
        )
    test_fn = TEST_REGISTRY[test_name_lc]

    X0 = _normalize_env_data(data_dict[baseline_key])
    p = X0.shape[1]
    env_order: list[EnvKey] = sorted(
        (k for k in data_dict.keys() if k != baseline_key),
        key=lambda k: str(k),
    )

    M = np.zeros((p, len(env_order)), dtype=bool)
    for col, key in enumerate(env_order):
        Xe = _normalize_env_data(data_dict[key])
        if Xe.shape[1] != p:
            raise ValueError(
                f"env {key!r} has {Xe.shape[1]} columns; baseline has {p}"
            )
        if test_name_lc == "welch":
            pvals = _vectorized_welch(X0, Xe)
        elif test_name_lc == "gaussian_lrt":
            pvals = _vectorized_gaussian_lrt(X0, Xe)
        else:
            pvals = np.empty(p, dtype=np.float64)
            for v in range(p):
                if test_name_lc == "energy":
                    pvals[v] = energy_p(X0[:, v], Xe[:, v], rng=rng)
                else:
                    pvals[v] = test_fn(X0[:, v], Xe[:, v])
        M[:, col] = pvals < alpha
    return M, env_order


__all__ = [
    "compute_M",
    "energy_p",
    "ks_p",
    "welch_p",
]
