"""Per-gene inverse-normal transform: nonparanormal (Gaussian-copula) preprocessing.

Reference quantiles are estimated from control cells; every environment is mapped
through the control marginals, so deviations from standard normal on the transformed
scale are exactly env-vs-control distributional differences. Any strictly monotone
per-gene transformation of the data leaves the transformed values unchanged, which is
the property that makes the downstream pipeline assume Gaussianity of the copula
rather than of the marginals (motivation: lognormal marginals break pooled Gaussian-LR
evidence but not mean-only evidence on transformed data)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def inverse_normal_reference(X: np.ndarray) -> np.ndarray:
    """Column-wise sorted reference sample; fit on training control cells only."""
    return np.sort(np.asarray(X, dtype=np.float64), axis=0)


def apply_inverse_normal(ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Map each column of `X` through the reference column's empirical CDF to N(0, 1).

    Ties take the midpoint rank; values outside the reference range clip to the tail
    quantiles, so the output is finite for any input. A constant reference column maps
    every value below/above it to the tails and equal values to zero."""
    ref = np.asarray(ref, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if ref.ndim != 2 or X.ndim != 2 or ref.shape[1] != X.shape[1]:
        raise ValueError(
            f"reference {ref.shape} and input {X.shape} must be 2-D with matching columns"
        )
    n = ref.shape[0]
    out = np.empty_like(X)
    for v in range(X.shape[1]):
        r = ref[:, v]
        pos = np.searchsorted(r, X[:, v], side="left")
        pos_hi = np.searchsorted(r, X[:, v], side="right")
        ranks = (pos + pos_hi) / 2.0
        u = np.clip(ranks, 1.0, float(n)) / (n + 1.0)
        out[:, v] = stats.norm.ppf(u)
    return out


__all__ = ["apply_inverse_normal", "inverse_normal_reference"]
