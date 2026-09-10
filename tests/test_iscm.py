"""Tests for the iSCM standardization and sampling helpers in workflow/scripts/_common.py."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

warnings.filterwarnings("ignore", message="No module named 'rpy2'.*")

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "expt" / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import (  # noqa: E402
    NOISE_FAMILIES,
    build_model,
    draw_noise_params,
    iscm_standardize,
    make_dataset,
    make_weights,
    noise_distribution,
    sample_shift,
)
import networkx as nx  # noqa: E402
from sempler import ANM, LGANM  # noqa: E402
from sempler.generators import dag_avg_deg  # noqa: E402

P = 15


def _random_scm(seed, p=P, avg_deg=4):
    W = dag_avg_deg(p, avg_deg, w_min=0.5, w_max=2, random_state=seed)
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=W.shape)
    W = W * signs
    noise_var, means = draw_noise_params(p, seed)
    return W, noise_var, means


def _implied_cov(W, noise_var):
    A = np.linalg.inv(np.eye(len(W)) - W.T)
    return A @ np.diag(noise_var) @ A.T


@pytest.mark.parametrize("seed", range(5))
def test_iscm_unit_variance(seed):
    W, noise_var, _ = _random_scm(seed)
    Wt, nvt = iscm_standardize(W, noise_var)
    np.testing.assert_allclose(np.diag(_implied_cov(Wt, nvt)), 1.0, atol=1e-10)


@pytest.mark.parametrize("seed", range(3))
def test_iscm_only_rescales(seed):
    """Per node, incoming weights and noise variance are divided by one common scale s,
    with s^2 = w' Cov(x_tilde_pa) w + sigma^2 computed on the standardized parents."""
    W, noise_var, _ = _random_scm(seed)
    Wt, nvt = iscm_standardize(W, noise_var)
    Sigma_t = _implied_cov(Wt, nvt)
    assert np.array_equal(Wt != 0, W != 0)
    for j in range(P):
        parents = np.flatnonzero(W[:, j])
        if parents.size == 0:
            assert nvt[j] == 1.0
            continue
        w = W[parents, j]
        s2 = w @ Sigma_t[np.ix_(parents, parents)] @ w + noise_var[j]
        np.testing.assert_allclose(Wt[parents, j], w / np.sqrt(s2), rtol=1e-10)
        assert nvt[j] == pytest.approx(noise_var[j] / s2, rel=1e-10)
        assert np.all(Wt[parents, j] / w > 0)


@pytest.mark.parametrize("family", NOISE_FAMILIES)
def test_noise_distribution_moments(family):
    np.random.seed(0)
    x = noise_distribution(family, mean=1.5, var=0.7)(200_000)
    assert x.mean() == pytest.approx(1.5, abs=0.02)
    assert x.var() == pytest.approx(0.7, rel=0.03)


def test_noise_distribution_rejects_unknown_family():
    with pytest.raises(ValueError):
        noise_distribution("weibull", 0.0, 1.0)
    with pytest.raises(ValueError):
        build_model(np.zeros((2, 2)), np.ones(2), np.zeros(2), "weibull")


def test_build_model_types():
    W, noise_var, means = _random_scm(0)
    assert isinstance(build_model(W, noise_var, means, "gaussian"), LGANM)
    assert isinstance(build_model(W, noise_var, means, "laplace"), ANM)


def test_anm_and_lganm_sample_the_same_model():
    W, noise_var, means = _random_scm(1)
    Wt, nvt = iscm_standardize(W, noise_var)
    n = 100_000
    X = build_model(Wt, nvt, means, "gaussian").sample(n, random_state=1)
    Y = build_model(Wt, nvt, means, "uniform").sample(n, random_state=1)  # ANM path, same second moments
    assert np.abs(np.cov(X.T) - np.cov(Y.T)).max() < 0.05
    assert np.abs(X.mean(0) - Y.mean(0)).max() < 0.05


@pytest.mark.parametrize("family", NOISE_FAMILIES)
def test_sample_shift_moves_target_by_shift(family):
    W, noise_var, means = _random_scm(2)
    Wt, nvt = iscm_standardize(W, noise_var)
    model = build_model(Wt, nvt, means, family)
    target = int(np.flatnonzero((Wt != 0).sum(axis=0) > 0)[0])  # a non-root node
    n = 100_000
    np.random.seed(3)
    X = model.sample(n)
    Y = sample_shift(model, n, [target], family, shift=(2.0, 1.0))
    assert Y[:, target].mean() - X[:, target].mean() == pytest.approx(2.0, abs=0.03)
    assert Y[:, target].var() - X[:, target].var() == pytest.approx(1.0, abs=0.05)
    # non-descendants keep their mean
    desc = nx.descendants(nx.DiGraph(Wt != 0), target) | {target}
    others = [j for j in range(P) if j not in desc]
    assert np.abs(Y[:, others].mean(0) - X[:, others].mean(0)).max() < 0.03


# make_weights / make_dataset


@pytest.mark.parametrize("graph", ["er", "sf"])
def test_make_weights_is_a_signed_dag(graph):
    W = make_weights(graph, P, avg_deg=4, seed=3)
    assert W.shape == (P, P)
    assert nx.is_directed_acyclic_graph(nx.DiGraph(W != 0))
    nz = W[W != 0]
    assert nz.size > 0
    assert (np.abs(nz) >= 0.5).all() and (np.abs(nz) <= 2).all()
    assert (nz > 0).any() and (nz < 0).any()
    assert np.array_equal(W, make_weights(graph, P, avg_deg=4, seed=3))
    assert not np.array_equal(W, make_weights(graph, P, avg_deg=4, seed=4))


def test_make_weights_rejects_unknown_graph():
    with pytest.raises(ValueError, match="graph family"):
        make_weights("tree", P, 4, 0)


def test_make_dataset_is_standardized_and_shifted():
    n, k = 20_000, 3
    W, targets, data = make_dataset("er", P, k, avg_deg=4, seed=1, n=n)
    # Same support and signs as the raw draw; population variance 1 everywhere.
    raw = make_weights("er", P, 4, 1)
    assert np.array_equal(np.sign(W), np.sign(raw))
    noise_var, _ = draw_noise_params(P, 1)
    _, nv_t = iscm_standardize(raw, noise_var)
    assert np.allclose(np.diag(_implied_cov(W, nv_t)), 1.0)
    # obs plus one env per target, in target order.
    assert list(data) == ["obs"] + [str(i) for i in range(k)]
    assert len(targets) == k
    assert data["obs"].shape == (n, P)
    assert np.allclose(data["obs"].var(axis=0), 1.0, atol=0.1)
    # Each target gains mean 2 and variance 1 in its own env.
    for idx, target in enumerate(targets):
        t = int(np.atleast_1d(target)[0])
        d = data[str(idx)][:, t].mean() - data["obs"][:, t].mean()
        assert abs(d - 2.0) < 0.1
        assert abs(data[str(idx)][:, t].var() - 2.0) < 0.15


@pytest.mark.parametrize("graph", ["er", "sf"])
@pytest.mark.parametrize("noise", ["gaussian", "uniform", "laplace"])
def test_make_dataset_deterministic_in_seed(graph, noise):
    """Same seed → identical W, targets and every environment, regardless of the global
    ``np.random`` state and of other calls made in between; a different seed differs."""
    a = make_dataset(graph, P, 2, 4, seed=5, n=200, noise=noise, targets_size=(1, 2))
    np.random.seed(12345)
    np.random.normal(size=1000)
    make_dataset(graph, P, 3, 4, seed=6, n=50, noise=noise)
    b = make_dataset(graph, P, 2, 4, seed=5, n=200, noise=noise, targets_size=(1, 2))
    assert np.array_equal(a[0], b[0])
    assert [list(np.atleast_1d(t)) for t in a[1]] == [list(np.atleast_1d(t)) for t in b[1]]
    assert list(a[2]) == list(b[2])
    assert all(np.array_equal(a[2][key], b[2][key]) for key in a[2])

    c = make_dataset(graph, P, 2, 4, seed=7, n=200, noise=noise, targets_size=(1, 2))
    assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(a[2]["obs"], c[2]["obs"])


def test_make_dataset_heterogeneous_targets():
    _, targets, data = make_dataset("er", P, 4, 4, seed=2, n=100, targets_size=(1, 3))
    sizes = [len(np.atleast_1d(t)) for t in targets]
    assert all(1 <= s <= 3 for s in sizes)
    assert len(data) == 5
    ragged = np.array(targets, dtype=object)
    assert len(ragged) == 4
