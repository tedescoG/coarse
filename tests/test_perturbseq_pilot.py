"""Tests for the exploratory batch-split pilot."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = (
    Path(__file__).resolve().parents[1]
    / "src/expt/workflow/scripts/perturbseq_pilot.py"
)
spec = importlib.util.spec_from_file_location("perturbseq_pilot", path)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_batch_alignment_and_invalid_ids():
    blocks = {
        "obs": np.arange(16).reshape(8, 2),
        "env": np.arange(16).reshape(8, 2)[::-1],
    }
    ids = {"obs": list(range(8)), "env": list(reversed(range(8)))}
    aligned, common = pilot.align_batches(blocks, ids)
    assert common == list(range(8))
    np.testing.assert_array_equal(aligned["obs"], aligned["env"])
    ids["env"] = [0] * 8
    with pytest.raises(ValueError, match="invalid batch IDs"):
        pilot.align_batches(blocks, ids)


def test_agreement_and_degenerate_partitions():
    heldout = np.array([[0, 0], [0, 0], [1, 1]])
    observed, baseline, difference = pilot.agreement([{0, 1}, {2}], heldout)
    assert observed == 1
    assert baseline == pytest.approx(1 / 3)
    assert difference == pytest.approx(2 / 3)
    assert pilot.agreement([{0}, {1}, {2}], heldout) == (None, None, None)
    assert pilot.agreement([{0, 1, 2}], heldout) == (None, None, None)


def test_run_pilot_uses_every_batch_split():
    rng = np.random.default_rng(42)
    blocks = {
        "obs": rng.normal(size=(8, 3)),
        "a": rng.normal(size=(8, 3)) + [2, 2, 0],
        "b": rng.normal(size=(8, 3)) + [0, 0, 3],
    }
    batch_ids = {env: list(range(8)) for env in blocks}
    rows, summary = pilot.run_pilot(blocks, batch_ids, ["x", "y", "z"], [0.1])
    assert len(rows) == 70
    assert summary["per_alpha"]["0.1"]["splits"] == 70
    assert all(
        set(r["train_batches"].split(",")).isdisjoint(r["test_batches"].split(","))
        for r in rows
    )
    assert "blocks" in summary["full_data_models"]["0.1"]
