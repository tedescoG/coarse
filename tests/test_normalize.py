"""Tests for the per-gene inverse-normal (nonparanormal) transform."""

from __future__ import annotations

import numpy as np

from coarse.normalize import apply_inverse_normal, inverse_normal_reference


def test_monotone_invariance():
    """Strictly monotone per-gene transforms leave the mapped values unchanged."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 6))
    ref = inverse_normal_reference(X)
    a = rng.uniform(0.2, 5.0, size=6)
    b = rng.normal(0.0, 2.0, size=6)
    z_raw = apply_inverse_normal(ref, X)
    z_exp = apply_inverse_normal(np.sort(X * a + b, axis=0), X * a + b)
    np.testing.assert_allclose(z_raw, z_exp, atol=1e-9)


def test_gaussian_roundtrip_and_shift_survival():
    """Gaussian input maps to approximately standard normal; a large location shift
    lands in the upper tail of the reference scale."""
    rng = np.random.default_rng(1)
    ref_sample = rng.standard_normal((4000, 3))
    ref = inverse_normal_reference(ref_sample)
    z = apply_inverse_normal(ref, rng.standard_normal((400, 3)))
    assert abs(z.mean()) < 0.15
    assert 0.85 < z.std() < 1.15
    shifted = apply_inverse_normal(ref, rng.standard_normal((400, 3)) + 3.0)
    assert shifted.mean() > 1.5
    assert np.isfinite(shifted).all()


def test_constant_column():
    """A constant reference column maps equal values to zero and others to tails."""
    rng = np.random.default_rng(2)
    ref = np.column_stack([np.zeros(300), rng.standard_normal(300)])
    X = np.column_stack([np.zeros(50), rng.standard_normal(50)])
    z = apply_inverse_normal(ref, X)
    # midpoint rank of the tied value: 150 of 300 -> u = 150/301 -> |z| well below 0.05
    assert np.allclose(z[:, 0], 0.0, atol=0.05)
    assert np.isfinite(z).all()


def test_input_validation():
    ref = inverse_normal_reference(np.zeros((10, 2)))
    with pytest_raises():
        apply_inverse_normal(ref, np.zeros((10, 3)))


def pytest_raises():
    import pytest

    return pytest.raises(ValueError)
