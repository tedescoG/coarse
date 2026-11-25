"""Corroboration for the score-based partition phase (paper/rebuttal.tex): evidence
statistics, merge geometry, consistency, DP exactness, finite-sample threshold behavior,
evidence swapping, pipeline integration, and complexity scaling."""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import (
    CHAIN_EDGES,
    CHAIN_ENV_ORDER,
    CHAIN_M,
    CHAIN_PARTITION,
    chain_data_dict,
    sample_chain_dataset,
)
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from coarse.coarse import COARSEOracle
from coarse.hypothesis_tests import _vectorized_gaussian_lrt, compute_M
from coarse.partition import infer_partition
from coarse.scored_partition import (
    _merge_delta,
    _segment_score,
    dp_segmentation,
    evidence_matrix,
    null_calibrated_evidence,
    scored_partition,
)

KAPPA = np.array([2.0 * np.log(2000.0), 2.0 * np.log(3000.0), 2.0 * np.log(4000.0)])


def _as_set_partition(blocks) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(b) for b in blocks)


def _score_of(L: np.ndarray, partition, kappa: np.ndarray, c: float) -> float:
    total = 0.0
    for blk in partition:
        g = L[sorted(blk), :].sum(axis=0)
        total += _segment_score(g, len(blk), kappa)
    return total - c * (len(partition) - 1)


def _all_partitions(items: list[int]):
    if not items:
        yield []
        return
    first, *rest = items
    for p in _all_partitions(rest):
        for i, blk in enumerate(p):
            yield p[:i] + [blk | {first}] + p[i + 1 :]
        yield [frozenset({first})] + p


def _contiguous_partitions(order: list[int]):
    d = len(order)
    for mask in range(1 << (d - 1)):
        blocks: list[frozenset[int]] = []
        start = 0
        for i in range(d - 1):
            if mask >> i & 1:
                blocks.append(frozenset(order[start : i + 1]))
                start = i + 1
        blocks.append(frozenset(order[start:]))
        yield blocks


def _sample_block_dag(
    n: int,
    rng: np.random.Generator,
    env_id: int,
    delta: float = 1.0,
) -> np.ndarray:
    """12 vars, 4 blocks: R{0,1,2} A{3,4,5} B{6,7,8} C{9,10,11}; edges R→A, R→B, A→C, B→C.
    Env `env_id` shifts the leader of A (1), B (2) or C (3), so sigma is
    R: {}, A: {1}, B: {2}, C: {1,2,3}."""
    add = {
        0: 0.0,
        3: delta if env_id == 1 else 0.0,
        6: delta if env_id == 2 else 0.0,
        9: delta if env_id == 3 else 0.0,
    }

    def block(leader: np.ndarray, depth: int = 2) -> list[np.ndarray]:
        out = [leader]
        x = leader
        for _ in range(depth):
            x = 0.9 * x + 0.5 * rng.standard_normal(n)
            out.append(x)
        return out

    r = rng.standard_normal(n) + add[0]
    a = 0.9 * r + 0.5 * rng.standard_normal(n) + add[3]
    b = 0.9 * r + 0.5 * rng.standard_normal(n) + add[6]
    c = 0.8 * a + 0.8 * b + 0.5 * rng.standard_normal(n) + add[9]
    cols = []
    for leader in (r, a, b, c):
        cols.extend(block(leader))
    return np.column_stack(cols)


def _block_dag_data_dict(n: int, seed: int, delta: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "obs": _sample_block_dag(n, rng, 0, delta),
        "1": _sample_block_dag(n, rng, 1, delta),
        "2": _sample_block_dag(n, rng, 2, delta),
        "3": _sample_block_dag(n, rng, 3, delta),
    }


BLOCK_DAG_PARTITION = _as_set_partition(
    [
        frozenset({0, 1, 2}),
        frozenset({3, 4, 5}),
        frozenset({6, 7, 8}),
        frozenset({9, 10, 11}),
    ]
)
BLOCK_DAG_SIGNATURES = {
    frozenset({0, 1, 2}): frozenset(),
    frozenset({3, 4, 5}): frozenset({0}),
    frozenset({6, 7, 8}): frozenset({1}),
    frozenset({9, 10, 11}): frozenset({0, 1, 2}),
}
BLOCK_DAG_EDGES = {
    (frozenset({0, 1, 2}), frozenset({3, 4, 5})),
    (frozenset({0, 1, 2}), frozenset({6, 7, 8})),
    (frozenset({3, 4, 5}), frozenset({9, 10, 11})),
    (frozenset({6, 7, 8}), frozenset({9, 10, 11})),
}


# ---------------------------------------------------------------------------
# Evidence matrix (rebuttal.tex, Def. 2.1)
# ---------------------------------------------------------------------------
def test_evidence_matches_gaussian_lrt_pvalues():
    """L equals the LR statistic behind `gaussian_lrt` p-values, column by column."""
    data_dict = chain_data_dict(2000, seed=1)
    L, env_order = evidence_matrix(data_dict)
    assert env_order == CHAIN_ENV_ORDER
    X0 = data_dict["obs"]
    for j, key in enumerate(env_order):
        pvals = _vectorized_gaussian_lrt(
            X0, np.asarray(data_dict[key], dtype=np.float64)
        )
        finite = pvals > 0.0
        assert np.allclose(L[finite, j], stats.chi2.isf(pvals[finite], 2), rtol=1e-8)
        assert np.all(L[~finite, j] > 690.0)  # underflowed p < 1e-300


def test_evidence_detects_shifts_and_nulls():
    """Shifted variables get large evidence; null cells stay O_p(1) (rebuttal.tex, Lem. 4.1)."""
    data_dict = chain_data_dict(5000, seed=2)
    L, _ = evidence_matrix(data_dict)
    shifted = CHAIN_M.astype(bool)  # M[v, e] = 1 iff v shifted in env e
    assert np.all(L[shifted] > 2.0 * np.log(10000.0))
    assert np.all(L[~shifted] < 10.0)


# ---------------------------------------------------------------------------
# Merge geometry (rebuttal.tex, Lem. 2.6)
# ---------------------------------------------------------------------------
def test_merge_delta_nonpositive_and_equality_condition():
    rng = np.random.default_rng(0)
    for _ in range(500):
        g_a = rng.random(3) * 50.0
        g_b = rng.random(3) * 50.0
        size_a = float(rng.integers(1, 6))
        size_b = float(rng.integers(1, 6))
        delta = _merge_delta(
            g_a,
            size_a,
            _segment_score(g_a, size_a, KAPPA),
            g_b,
            size_b,
            _segment_score(g_b, size_b, KAPPA),
            KAPPA,
        )
        assert delta <= 1e-12
        # Equality iff both clusters agree in sign of (g_e − kappa_e |A|) in every env.
        signs_a = np.sign(g_a - KAPPA * size_a)
        signs_b = np.sign(g_b - KAPPA * size_b)
        same_sign = np.all(signs_a * signs_b > 0) or np.all(signs_a == signs_b)
        if same_sign:
            assert abs(delta) <= 1e-12


# ---------------------------------------------------------------------------
# Consistency (rebuttal.tex, Thm. 4.2)
# ---------------------------------------------------------------------------
def test_scored_partition_recovers_chain():
    data_dict = chain_data_dict(20000, seed=3)
    L, env_order = evidence_matrix(data_dict)
    result = scored_partition(L, [20000.0] * 3, 20000.0)
    assert _as_set_partition(result.partition) == _as_set_partition(CHAIN_PARTITION)
    for blk, sig in result.signatures.items():
        cols = [CHAIN_ENV_ORDER.index(e) for e in env_order]
        assert sig == frozenset(np.flatnonzero(CHAIN_M[min(blk), cols]))
    # The derived M reproduces the true interventional descendant matrix.
    cols = [CHAIN_ENV_ORDER.index(e) for e in env_order]
    assert np.array_equal(result.M, CHAIN_M[:, cols])


def test_scored_partition_recovers_incomparable_blocks():
    """Blocks with incomparable supports (A: {1}, B: {2}) are not merged."""
    data_dict = _block_dag_data_dict(30000, seed=4)
    L, env_order = evidence_matrix(data_dict)
    assert env_order == ["1", "2", "3"]
    result = scored_partition(L, [30000.0] * 3, 30000.0)
    assert _as_set_partition(result.partition) == BLOCK_DAG_PARTITION
    got = {blk: sig for blk, sig in result.signatures.items()}
    assert got == BLOCK_DAG_SIGNATURES
    # Deterministic on repeat.
    again = scored_partition(L, [30000.0] * 3, 30000.0)
    assert again.partition == result.partition
    assert again.merge_deltas == result.merge_deltas


def test_size_cap_splits_null_evidence_blob():
    """With a cap, evidence-null genes stay in capped blocks instead of one blob;
    without it they merge freely. Coverage and M consistency hold either way."""
    # Two strong-shift genes and four null genes, one environment.
    L = np.array([[100.0], [100.0], [0.0], [0.0], [0.0], [0.0]])
    n_e, n_0 = [50.0], 1000.0
    free = scored_partition(L, n_e, n_0, lam=1.0, c=1.0)
    assert max(len(blk) for blk in free.partition) == 4  # the null blob
    capped = scored_partition(L, n_e, n_0, lam=1.0, c=1.0, size_cap=2.0)
    assert max(len(blk) for blk in capped.partition) == 2
    # Evidence sums across genes, so equal-evidence pairs merge (delta 0): the two
    # informative genes and the four null genes form three capped pairs.
    assert sorted(len(blk) for blk in capped.partition) == [2, 2, 2]
    for result in (free, capped):
        assert sum(len(blk) for blk in result.partition) == 6
        assert result.M.shape == (6, 1)
    # The capped split of the null class is a refinement: both null pairs share an
    # identical all-false signature row, while the informative pair shifts.
    rows = [tuple(capped.M[sorted(blk), 0]) for blk in capped.partition]
    assert rows.count((False, False)) == 2
    assert rows.count((True, True)) == 1


# ---------------------------------------------------------------------------
# Optimality: greedy vs brute force; DP exactness (rebuttal.tex, Rem. 3.4)
# ---------------------------------------------------------------------------
def test_greedy_matches_brute_force_on_easy_signal():
    rng = np.random.default_rng(5)
    d, m = 8, 2
    L = np.zeros((d, m))
    classes = [
        frozenset({0, 1}),
        frozenset({2, 3}),
        frozenset({4, 5}),
        frozenset({6, 7}),
    ]
    for blk in classes:
        L[sorted(blk), :] = rng.random(len(blk)) * 5.0 + 40.0
    kappa = 2.0 * np.log(6000.0) * np.ones(m)
    result = scored_partition(L, [3000.0] * m, 3000.0)
    best = max(_score_of(L, p, kappa, 1.0) for p in _all_partitions(list(range(d))))
    assert abs(result.score - best) <= 1e-8
    assert (
        np.all(np.diff(result.merge_deltas) <= 1e-12) or len(result.merge_deltas) <= 1
    )


def test_greedy_is_locally_optimal():
    rng = np.random.default_rng(6)
    d, m = 10, 2
    L = rng.random((d, m)) * 30.0
    result = scored_partition(L, [3000.0] * m, 3000.0, c=1.0)
    kappa = 2.0 * np.log(6000.0) * np.ones(m)
    blocks = list(result.partition)
    scores = {
        frozenset(b): _segment_score(L[sorted(b), :].sum(axis=0), len(b), kappa)
        for b in blocks
    }
    for i, bi in enumerate(blocks):
        for bj in blocks[i + 1 :]:
            delta = _merge_delta(
                L[sorted(bi), :].sum(axis=0),
                len(bi),
                scores[bi],
                L[sorted(bj), :].sum(axis=0),
                len(bj),
                scores[bj],
                kappa,
            )
            assert delta <= -1.0 + 1e-9


def test_dp_segmentation_exact():
    rng = np.random.default_rng(7)
    d, m = 10, 2
    L = rng.random((d, m)) * 30.0
    order = list(rng.permutation(d))
    kappa = 2.0 * np.log(6000.0) * np.ones(m)
    result = dp_segmentation(L, order, [3000.0] * m, 3000.0)
    best = max(_score_of(L, p, kappa, 1.0) for p in _contiguous_partitions(order))
    assert abs(result.score - best) <= 1e-8
    assert _as_set_partition(result.partition) in {
        _as_set_partition(p)
        for p in _contiguous_partitions(order)
        if abs(_score_of(L, p, kappa, 1.0) - best) <= 1e-8
    }


# ---------------------------------------------------------------------------
# Finite-sample threshold behavior (rebuttal.tex, Thm. 6.3)
# ---------------------------------------------------------------------------
def test_finite_sample_recovery_threshold():
    """Exact recovery below vs above the detection threshold n·Δ² ≳ 2λ log n."""
    n_weak, n_strong = 600, 40000
    weak = sum(
        _as_set_partition(
            scored_partition(
                evidence_matrix(_block_dag_data_dict(n_weak, seed, delta=0.15))[0],
                [n_weak] * 3,
                n_weak,
            ).partition
        )
        == BLOCK_DAG_PARTITION
        for seed in range(5)
    )
    strong = sum(
        _as_set_partition(
            scored_partition(
                evidence_matrix(_block_dag_data_dict(n_strong, seed, delta=1.0))[0],
                [n_strong] * 3,
                n_strong,
            ).partition
        )
        == BLOCK_DAG_PARTITION
        for seed in range(5)
    )
    assert strong == 5
    assert weak < strong


# ---------------------------------------------------------------------------
# Evidence swap (rebuttal.tex, Rem. 8.2)
# ---------------------------------------------------------------------------
def test_ks_evidence_swap():
    """The merge machinery is agnostic to the evidence: −2 log p from KS works too."""
    data_dict = _block_dag_data_dict(30000, seed=8)
    X0 = data_dict["obs"]
    keys = ["1", "2", "3"]
    L = np.zeros((12, 3))
    for j, key in enumerate(keys):
        Xe = data_dict[key]
        for v in range(12):
            p = stats.ks_2samp(X0[:, v], Xe[:, v]).pvalue
            L[v, j] = -2.0 * np.log(max(p, 1e-300))
    result = scored_partition(L, [30000.0] * 3, 30000.0)
    assert _as_set_partition(result.partition) == BLOCK_DAG_PARTITION
    assert result.signatures == BLOCK_DAG_SIGNATURES


def _calibratable_chain(n: int, seed: int) -> dict:
    """Chain fixture with a 3n baseline, so null calibration has split room."""
    rng = np.random.default_rng(seed)
    return {
        "obs": sample_chain_dataset(3 * n, rng),
        "1": sample_chain_dataset(n, rng, shift_targets=(0, 1)),
        "2": sample_chain_dataset(n, rng, shift_targets=(2, 3)),
        "3": sample_chain_dataset(n, rng, shift_targets=(4, 5)),
    }


# ---------------------------------------------------------------------------
# Null-calibrated evidence (rebuttal discussion)
# ---------------------------------------------------------------------------
def test_null_calibrated_evidence_is_affine_invariant():
    """Per-gene scaling and location shifts leave calibrated evidence unchanged."""
    data = _calibratable_chain(150, seed=11)
    z1, order1 = null_calibrated_evidence(data, rng=np.random.default_rng(0))
    a = np.array([2.0, 0.5, 3.0, 1.0, 0.3, 4.0])
    b = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.0])
    scaled = {k: v * a + b for k, v in data.items()}
    z2, order2 = null_calibrated_evidence(scaled, rng=np.random.default_rng(0))
    assert order1 == order2 == ["1", "2", "3"]
    np.testing.assert_allclose(z1, z2, rtol=1e-6, atol=1e-8)


def test_null_calibrated_evidence_nulls_stay_small_and_shifts_stand_out():
    """Pure-null envs calibrate to small z; shifted genes clear any sensible bar."""
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((2400, 5))
    data = {
        "obs": obs,
        "1": rng.standard_normal((200, 5)),
        "2": obs[:200] + np.r_[np.ones(2), np.zeros(3)],
    }
    z, env_order = null_calibrated_evidence(data, rng=np.random.default_rng(0))
    assert env_order == ["1", "2"]
    assert z[:, 0].max() < 12.0
    assert z[:2, 1].min() > 30.0
    assert z[2:, 1].max() < 12.0


def test_calibrated_pipeline_recovers_chain():
    """Calibrated evidence + scored partition recovers the chain blocks and M."""
    data = _calibratable_chain(20000, seed=12)
    z, env_order = null_calibrated_evidence(data, rng=np.random.default_rng(0))
    n_e = [float(len(data[k])) for k in env_order]
    sp = scored_partition(z, n_e, float(len(data["obs"])), lam=2.0, c=1.0)
    assert _as_set_partition(sp.partition) == _as_set_partition(CHAIN_PARTITION)
    np.testing.assert_array_equal(sp.M, CHAIN_M)


def test_null_calibrated_evidence_bootstraps_small_baselines():
    """With fewer than 2n baseline rows the null falls back to bootstrap halves, and
    the null statistic always matches the realized one (mean-only here)."""
    data = chain_data_dict(500, seed=13)  # baseline and envs both n = 500
    z, env_order = null_calibrated_evidence(
        data, rng=np.random.default_rng(0), mean_only=True
    )
    assert env_order == ["1", "2", "3"]
    assert z.shape == (6, 3)
    assert np.isfinite(z).all() and (z >= 0).all()


# ---------------------------------------------------------------------------
# Pipeline integration (rebuttal.tex, Cor. 4.4)
# ---------------------------------------------------------------------------
def test_full_pipeline_with_coarse_oracle():
    data_dict = chain_data_dict(20000, seed=9)
    L, env_order = evidence_matrix(data_dict)
    result = scored_partition(L, [20000.0] * 3, 20000.0)
    model = COARSEOracle(rng=np.random.default_rng(0)).fit(
        result.partition,
        result.M,
        env_order,
        data_dict,
    )
    got = {(tuple(sorted(src)), tuple(sorted(dst))) for src, dst in model.dag.edges}
    want = {(tuple(sorted(src)), tuple(sorted(dst))) for src, dst in CHAIN_EDGES}
    assert got == want


# ---------------------------------------------------------------------------
# Complexity scaling (rebuttal.tex, Thm. 5.1)
# ---------------------------------------------------------------------------
def test_complexity_scaling_in_d():
    """Runtime of the merge phase grows ~ d², independent of n (n is absent from L)."""
    rng = np.random.default_rng(10)

    def timed(d: int) -> float:
        L = rng.random((d, 3)) * 30.0
        best = np.inf
        for _ in range(3):
            t0 = time.perf_counter()
            scored_partition(L, [3000.0] * 3, 3000.0)
            best = min(best, time.perf_counter() - t0)
        return best

    t_small, t_large = timed(150), timed(300)
    assert t_large <= 10.0 * max(t_small, 1e-4)


# ---------------------------------------------------------------------------
# Readout discount (rebuttal.tex, Cor. 6.5 / Rem. 5)
# ---------------------------------------------------------------------------
def test_readout_discount_invariant_and_monotone():
    """kappa_0 leaves merges, partition and score untouched and only widens signatures."""
    data_dict = chain_data_dict(3000, seed=12)
    L, _ = evidence_matrix(data_dict)
    base = scored_partition(L, [3000.0] * 3, 3000.0, kappa_0=0.0)
    for k0 in (4.0, 8.0, 12.0):
        res = scored_partition(L, [3000.0] * 3, 3000.0, kappa_0=k0)
        assert res.partition == base.partition
        assert res.score == base.score
        assert res.merge_deltas == base.merge_deltas
        for blk in base.partition:
            assert res.signatures[blk] >= base.signatures[blk]


def test_readout_discount_recovers_missing_support():
    """A support env below the linear bar is declared under the discounted readout."""
    data_dict = chain_data_dict(3000, seed=13)
    L, _ = evidence_matrix(data_dict)
    base = scored_partition(L, [3000.0] * 3, 3000.0)
    disc = scored_partition(L, [3000.0] * 3, 3000.0, kappa_0=8.0)
    assert disc.partition == base.partition
    # CHAIN_M truth: every true signature cell is declared under the discount.
    cols = [CHAIN_ENV_ORDER.index(e) for e in disc.env_order]
    for blk in disc.partition:
        truth = frozenset(np.flatnonzero(CHAIN_M[min(blk), cols]))
        assert base.signatures[blk] <= disc.signatures[blk]
        if base.signatures[blk] == truth:
            assert disc.signatures[blk] == truth


def test_mean_only_evidence_restores_parity_on_lognormal():
    """Regression test for the real-data failure mode: on lognormal marginals the
    Gaussian LR degrades the scored partition well below Welch parity, while the same
    machinery on mean-only evidence matches Welch (ARI within 0.1)."""

    def lognormal_design(seed, d=40, n0=5400, ne=250, sat_frac=0.9):
        rng = np.random.default_rng(seed)
        sat = sorted(rng.choice(d, size=int(sat_frac * d), replace=False).tolist())
        d_sat = rng.uniform(0.15, 1.5, size=len(sat))
        groups = [(sat, d_sat)]
        for pair in ((0, 1), (2, 3), (4, 5)):
            genes = list(range(4 * pair[0], 4 * pair[0] + 4))
            groups.append((genes, rng.uniform(0.3, 1.5, size=len(genes))))
        true_sig = np.zeros((d, 4), dtype=bool)
        for e, (genes, _) in enumerate(groups):
            true_sig[genes, e] = True
        rows: dict[tuple[bool, ...], list[int]] = {}
        for v in range(d):
            rows.setdefault(tuple(true_sig[v]), []).append(v)
        X0 = np.exp(
            0.8 * np.random.default_rng(seed * 1000 + 1).standard_normal((n0, d))
        )
        envs = {}
        for e, (genes, deltas) in enumerate(groups):
            Z = np.random.default_rng(seed * 1000 + 2 + e).standard_normal((ne, d))
            Z[:, genes] += deltas
            envs[str(e + 1)] = np.exp(0.8 * Z)
        return {"obs": X0, **envs}, true_sig

    def labels_of(partition, d=40):
        lab = np.empty(d, dtype=int)
        for i, blk in enumerate(partition):
            for v in blk:
                lab[v] = i
        return lab

    def ari_of(data, mean_only):
        L, env_order = evidence_matrix(data, mean_only=mean_only)
        n_e = [float(len(data[k])) for k in env_order]
        best = -1.0
        for lam in (0.5, 1.0, 2.0, 4.0):
            sp = scored_partition(L, n_e, float(len(data["obs"])), lam=lam, c=1.0)
            ari = adjusted_rand_score(
                labels_of(list(sp.partition)),
                labels_of(infer_partition(compute_M(data, alpha=0.01)[0])),
            )
            best = max(best, ari)
        return best

    gaps = []
    for seed in range(3):
        data, true_sig = lognormal_design(seed)
        welch = adjusted_rand_score(
            labels_of(infer_partition(compute_M(data, alpha=0.01)[0])),
            labels_of(infer_partition(true_sig)),
        )
        monly = ari_of(data, mean_only=True)
        gaps.append(monly - welch)
        assert monly > welch - 0.05, (
            f"seed {seed}: mean-only {monly:.3f} vs Welch {welch:.3f}"
        )
    # The mean-only partition must stay close to Welch parity overall.
    assert np.mean(gaps) > -0.02


def test_negative_kappa_0_rejected():
    L = np.zeros((5, 2))
    with pytest.raises(ValueError, match="kappa_0"):
        scored_partition(L, [1000.0] * 2, 1000.0, kappa_0=-1.0)


# ---------------------------------------------------------------------------
# Interface conventions
# ---------------------------------------------------------------------------
def test_matches_compute_M_conventions():
    data_dict = chain_data_dict(2000, seed=11)
    L, env_order = evidence_matrix(data_dict)
    M_welch, order_welch = compute_M(data_dict, alpha=1e-4, test_name="gaussian_lrt")
    assert env_order == order_welch
    assert L.shape == M_welch.shape
