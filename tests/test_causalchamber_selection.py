"""Tests for the causalchamber grid selection helpers.

`select_score_row` picks within the fixed-λ column and does not use the ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "expt" / "workflow" / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))
from _causalchamber_common import (  # noqa: E402
    select_oracle_row,
    select_score_row,
)


def _row(alpha, lam, score, ari=0.0, f1=0.0, precision=0.0):
    return {
        "alpha": alpha, "lambda": lam, "score": score,
        "ari": ari, "f1": f1, "precision": precision,
    }


def test_select_score_row_stays_in_fixed_lambda_column():
    records = [
        _row(1e-2, 0.5, 100.0, ari=1.0),   # higher score, wrong λ
        _row(1e-4, 1.0, 90.0),
        _row(1e-2, 1.0, 95.0),
        _row(1e-2, 2.0, 80.0),
    ]
    assert select_score_row(records, 1.0) == _row(1e-2, 1.0, 95.0)


def test_select_score_row_tiebreak_smaller_alpha():
    records = [_row(1e-2, 1.0, 50.0), _row(1e-4, 1.0, 50.0), _row(1e-3, 1.0, 50.0)]
    assert select_score_row(records, 1.0)["alpha"] == 1e-4


def test_select_score_row_missing_lambda_raises():
    records = [_row(1e-2, 0.5, 1.0), _row(1e-2, 2.0, 1.0)]
    with pytest.raises(ValueError, match="lambda"):
        select_score_row(records, 1.0)


def test_select_oracle_row_uses_ground_truth_over_whole_grid():
    records = [
        _row(1e-4, 1.0, 100.0, ari=0.2),
        _row(1e-2, 4.0, 10.0, ari=0.9, f1=0.5),
        _row(1e-3, 4.0, 20.0, ari=0.9, f1=0.7),
    ]
    assert select_oracle_row(records) == _row(1e-3, 4.0, 20.0, ari=0.9, f1=0.7)
