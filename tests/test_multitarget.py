"""Tests for the multi-target intervention extension.

Validates two properties:

1. **Backward compatibility** — the edits to ``_common.build_oracle_partition``
   and ``generate.py`` must be no-ops on the single-target inputs that every
   existing experiment uses. Test #1 locks this in by exact snapshot.

2. **Multi-target correctness** — the oracle partition under a multi-target
   intervention must equal the union of the singletons it decomposes into,
   ``build_data_dict`` must store the full target set, and COARSE / cv_coarse
   must accept and process multi-target ``data_dict`` inputs without raising
   (the algorithm itself ignores targets, but the API contract must hold).

Imports ``_common`` via a ``sys.path`` shim because that file lives outside the
installed package (it is a snakemake-script-local helper).
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from coarse.coarse import COARSE
from coarse.cv import cv_coarse

# _common.py is a sibling of generate.py under src/expt/workflow/scripts/ — it
# is intentionally not part of the installed `coarse` package. Tests import it
# through a path shim, matching the trick that snakemake itself uses (the
# `script:` directive prepends the script's directory to sys.path).
SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "expt" / "workflow" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import (  # noqa: E402
    build_data_dict,
    build_oracle_partition,
    parse_targets_per_interv,
)

from conftest import sample_chain_dataset  # noqa: E402


# A small reference DAG used across the oracle-partition tests:
#   0 → 1, 0 → 2, 1 → 3, 2 → 3, 3 → 4
# Descendants:
#   {0}: {1, 2, 3, 4}    {1}: {3, 4}    {2}: {3, 4}    {3}: {4}    {4}: ∅
_DAG_WEIGHTS = np.array(
    [
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ],
    dtype=float,
)
_NUM_NODES = 5


def test_parse_targets_per_interv_grammar():
    """Cover the three input shapes the wildcard can take."""
    assert parse_targets_per_interv(None) == 1  # absent → single-target default
    assert parse_targets_per_interv("3") == 3  # fixed-size k
    assert parse_targets_per_interv("1to5") == (1, 5)  # heterogeneous (min, max)
    # also accept actual ints (defensive — snakemake usually hands strings)
    assert parse_targets_per_interv(2) == 2


def test_build_oracle_partition_single_target_unchanged():
    """The edit to build_oracle_partition must be a no-op for singleton
    targets — historical single-target sweeps cannot shift by one bit."""
    targets = [[0], [2]]  # single-target, as today's generate.py emits
    _, M_true, env_order, partition = build_oracle_partition(
        _DAG_WEIGHTS, targets, _NUM_NODES
    )
    # env "0" intervenes on node 0 → affected = {0, 1, 2, 3, 4}
    # env "1" intervenes on node 2 → affected = {2, 3, 4}
    expected_M = np.array(
        [
            [True,  False],  # node 0: hit by env 0 only
            [True,  False],  # node 1: descendant of 0 only
            [True,  True ],  # node 2: hit by env 0 (desc) and env 1 (target)
            [True,  True ],  # node 3: descendant of both
            [True,  True ],  # node 4: descendant of both
        ]
    )
    np.testing.assert_array_equal(M_true, expected_M)
    assert env_order == ["0", "1"]
    # Partition: nodes share a block iff their M-rows agree. Rows: {0,1} share
    # [T,F]; {2,3,4} share [T,T]. So expected partition has 2 blocks.
    blocks = [frozenset(b) for b in partition]
    assert frozenset({0, 1}) in blocks
    assert frozenset({2, 3, 4}) in blocks
    assert len(blocks) == 2


def test_build_oracle_partition_multi_target_takes_union():
    """Core correctness property: a multi-target env's M-row equals the OR
    of the M-rows produced by the corresponding singleton envs."""
    _, M_multi, _, _ = build_oracle_partition(
        _DAG_WEIGHTS, [[1, 2]], _NUM_NODES
    )
    _, M_singleton_1, _, _ = build_oracle_partition(
        _DAG_WEIGHTS, [[1]], _NUM_NODES
    )
    _, M_singleton_2, _, _ = build_oracle_partition(
        _DAG_WEIGHTS, [[2]], _NUM_NODES
    )
    expected = M_singleton_1[:, 0] | M_singleton_2[:, 0]
    np.testing.assert_array_equal(M_multi[:, 0], expected)
    # Sanity check the absolute value too:
    # nodes(target={1}) affected = {1, 3, 4}; nodes(target={2}) = {2, 3, 4};
    # union = {1, 2, 3, 4}.
    assert list(M_multi[:, 0]) == [False, True, True, True, True]


def test_build_data_dict_multi_target_set():
    """``build_data_dict`` must store the *full* target set (not just the
    first element) — otherwise the env tuple's intervention info is silently
    truncated to single-target, defeating the experiment."""
    n = 50
    data = {"obs": np.zeros((n, 5)), "0": np.zeros((n, 5))}
    targets = [(0, 2, 4)]  # one env, three-element target set
    out = build_data_dict(data, targets, intervention_type="soft")
    assert out["obs"] == (data["obs"], set(), "obs")
    arr, tgt, kind = out["0"]
    assert arr is data["0"]
    assert tgt == {0, 2, 4}
    assert kind == "soft"


def test_coarse_fit_multi_target_smoke():
    """End-to-end coverage of the 'algorithm is target-agnostic' claim.

    Builds three environments on the 6-node chain dataset, two of which use
    *multi-target* shifts (env_a shifts nodes {0, 2}, env_b shifts {1, 4}).
    COARSE must run to completion and produce a valid partition DAG.
    """
    rng = np.random.default_rng(0)
    n = 2000
    obs = sample_chain_dataset(n, rng, shift_targets=())
    env_a = sample_chain_dataset(n, rng, shift_targets=(0, 2))  # multi-target
    env_b = sample_chain_dataset(n, rng, shift_targets=(1, 4))  # multi-target
    data_dict = {
        "obs": (obs, set(), "obs"),
        "0": (env_a, {0, 2}, "soft"),
        "1": (env_b, {1, 4}, "soft"),
    }
    model = COARSE().fit(
        data_dict, alpha=1e-2, lambda_pen=1.0, refine_test="welch"
    )
    assert isinstance(model.dag, nx.DiGraph)
    # Every node of the partition is a nonempty tuple, and the union covers
    # all 6 atomic variables (the partition is a partition of {0..5}).
    atoms = set()
    for block in model.dag.nodes:
        assert isinstance(block, tuple) and len(block) >= 1
        atoms.update(block)
    assert atoms == set(range(6))


def test_cv_coarse_fit_multi_target_smoke():
    """Same end-to-end check for cv_coarse — the CV wrapper must also accept
    multi-target ``data_dict`` inputs. Uses a tiny α grid and n_folds=2 to
    keep runtime in the sub-second range."""
    rng = np.random.default_rng(1)
    n = 1500
    obs = sample_chain_dataset(n, rng, shift_targets=())
    env_a = sample_chain_dataset(n, rng, shift_targets=(0, 2))
    env_b = sample_chain_dataset(n, rng, shift_targets=(3, 5))
    data_dict = {
        "obs": (obs, set(), "obs"),
        "0": (env_a, {0, 2}, "soft"),
        "1": (env_b, {3, 5}, "soft"),
    }
    cv = cv_coarse(
        data_dict,
        alpha_grid=(1e-3, 1e-2),
        n_folds=2,
        lambda_pen=1.0,
        refine_test="welch",
    )
    atoms = set()
    for block in cv.dag.nodes:
        atoms.update(block)
    assert atoms == set(range(6))
