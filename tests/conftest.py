"""Shared fixtures; puts the experiment scripts on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "expt" / "workflow" / "scripts"))


def sample_chain_dataset(
    n: int,
    rng: np.random.Generator,
    shift_targets: tuple[int, ...] = (),
) -> np.ndarray:
    """Linear-Gaussian chain {v0, v1} → {v2, v3} → {v4, v5}; `shift_targets` get a +2 mean
    shift."""
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


def chain_data_dict(n: int, seed: int) -> dict:
    """Chain data: obs plus one env shifting each block."""
    rng = np.random.default_rng(seed)
    return {
        "obs": sample_chain_dataset(n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }


CHAIN_PARTITION = [frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})]
CHAIN_M = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool)
CHAIN_ENV_ORDER = ["1", "2", "3"]
CHAIN_EDGES = {((0, 1), (2, 3)), ((2, 3), (4, 5))}
