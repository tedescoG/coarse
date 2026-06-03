"""Shared fixtures for the coarse test suite.

pytest discovers conftest.py automatically and adds its directory to sys.path,
so `from conftest import sample_chain_dataset` resolves cleanly from any test
file in this directory. This is the idiomatic place for shared fixtures —
unlike `from test_coarse import ...`, which IS brittle under pytest collection
order.
"""
from __future__ import annotations

import numpy as np


def sample_chain_dataset(
    n: int,
    rng: np.random.Generator,
    shift_targets: tuple[int, ...] = (),
) -> np.ndarray:
    """Sample from a linear-Gaussian chain DAG:
        A = {v0, v1}  →  B = {v2, v3}  →  C = {v4, v5}

    ``shift_targets`` lists variable indices that receive a soft mean shift of
    +2. Coefficients and noise levels are domain-reasonable, not tuned to make
    any particular partition recovery succeed.
    """
    shift = np.zeros(6)
    for i in shift_targets:
        shift[i] = 2.0
    v0 = rng.standard_normal(n) + shift[0]
    v1 = rng.standard_normal(n) + shift[1]
    v2 = 0.8 * v0 + 0.5 * v1 + 0.3 * rng.standard_normal(n) + shift[2]
    v3 = -0.7 * v0 + 0.6 * v1 + 0.3 * rng.standard_normal(n) + shift[3]
    v4 = 0.9 * v2 + 0.4 * v3 + 0.3 * rng.standard_normal(n) + shift[4]
    v5 = -0.6 * v2 + 0.8 * v3 + 0.3 * rng.standard_normal(n) + shift[5]
    return np.column_stack([v0, v1, v2, v3, v4, v5])
