"""Tests for the multi-target intervention helpers and tuple-form data_dict inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

from coarse.coarse import COARSE
from coarse.cv import cv_coarse

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "expt" / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import (  # noqa: E402
    build_data_dict,
    build_oracle_partition,
    parse_targets_per_interv,
)

from conftest import sample_chain_dataset  # noqa: E402


# 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4
# Descendants: {0}: {1,2,3,4}  {1}: {3,4}  {2}: {3,4}  {3}: {4}  {4}: {}
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
    assert parse_targets_per_interv(None) == 1
    assert parse_targets_per_interv("3") == 3
    assert parse_targets_per_interv("1to5") == (1, 5)
    assert parse_targets_per_interv(2) == 2


def test_build_oracle_partition_single_target():
    _, M_true, env_order, partition = build_oracle_partition(_DAG_WEIGHTS, [[0], [2]], _NUM_NODES)
    expected_M = np.array(
        [[True, False], [True, False], [True, True], [True, True], [True, True]]
    )
    np.testing.assert_array_equal(M_true, expected_M)
    assert env_order == ["0", "1"]
    assert set(frozenset(b) for b in partition) == {frozenset({0, 1}), frozenset({2, 3, 4})}


def test_build_oracle_partition_multi_target_takes_union():
    """A multi-target env's M column is the OR of the corresponding single-target columns."""
    _, M_multi, _, _ = build_oracle_partition(_DAG_WEIGHTS, [[1, 2]], _NUM_NODES)
    _, M_1, _, _ = build_oracle_partition(_DAG_WEIGHTS, [[1]], _NUM_NODES)
    _, M_2, _, _ = build_oracle_partition(_DAG_WEIGHTS, [[2]], _NUM_NODES)
    np.testing.assert_array_equal(M_multi[:, 0], M_1[:, 0] | M_2[:, 0])
    assert list(M_multi[:, 0]) == [False, True, True, True, True]


def test_build_data_dict_keeps_full_target_set():
    n = 50
    data = {"obs": np.zeros((n, 5)), "0": np.zeros((n, 5))}
    out = build_data_dict(data, [(0, 2, 4)], intervention_type="soft")
    assert out["obs"] == (data["obs"], set(), "obs")
    arr, tgt, kind = out["0"]
    assert arr is data["0"]
    assert tgt == {0, 2, 4}
    assert kind == "soft"


def test_fit_accepts_multi_target_tuples():
    """Both entry points accept tuple-form envs with multi-element target sets."""
    rng = np.random.default_rng(0)
    n = 1500
    data_dict = {
        "obs": (sample_chain_dataset(n, rng), set(), "obs"),
        "0": (sample_chain_dataset(n, rng, shift_targets=(0, 2)), {0, 2}, "soft"),
        "1": (sample_chain_dataset(n, rng, shift_targets=(1, 4)), {1, 4}, "soft"),
    }
    model = COARSE().fit(data_dict, alpha=1e-2)
    cv = cv_coarse(data_dict, alpha_grid=(1e-3, 1e-2), n_folds=2)
    for dag in (model.dag, cv.dag):
        assert isinstance(dag, nx.DiGraph)
        blocks = list(dag.nodes)
        assert all(isinstance(b, tuple) and b for b in blocks)
        assert sorted(i for b in blocks for i in b) == list(range(6))
