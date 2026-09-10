"""Tests for the shared figure style in workflow/scripts/_plot_style.py."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.ticker import NullFormatter

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "expt" / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import _plot_style as ps  # noqa: E402


def _visible_labels(axis):
    axis.axes.figure.canvas.draw()
    lo, hi = axis.get_view_interval()
    return [
        t.get_text()
        for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
        if lo <= loc <= hi and t.get_text()
    ]


def test_palette_anchors_on_thesis_colours():
    assert ps.CATEGORICAL[0] == "#901A1E" and ps.CATEGORICAL[1] == "#0A5963"
    ramp = ps.ordinal_palette(3)
    assert len(ramp) == 3
    assert matplotlib.colors.to_hex(ramp[0]).upper() == "#0A5963"
    assert matplotlib.colors.to_hex(ramp[-1]).upper() == "#901A1E"
    assert ps.METHOD_COLOR["COARSE"] == "#901A1E"


def test_metric_labels_use_short_names_with_arrows():
    assert ps.COLUMN_LABEL["ari"] == "ARI ↑"
    assert ps.COLUMN_LABEL["fscore"] == "F-score ↑"
    assert not any("Adjusted" in v for v in ps.COLUMN_LABEL.values())


def test_display_methods_maps_and_rejects_unknown():
    df = pd.DataFrame({"method": ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle", "RePaRe-oracle", "COARSE-CV"]})
    out = ps.display_methods(df)
    assert out["method"].tolist() == ["COARSE", "COARSE-1PC", "COARSE-3PC", "RePaRe", "COARSE-CV"]
    with pytest.raises(ValueError):
        ps.display_methods(pd.DataFrame({"method": ["mystery"]}))


def test_samp_size_axis_shows_decades_only():
    fig, ax = plt.subplots()
    ax.plot([500, 1000], [0.1, 0.2])
    ax.set_xscale("log")
    ps.format_axis(ax.xaxis, "samp_size")
    labels = _visible_labels(ax.xaxis)
    assert labels == ["$\\mathdefault{10^{2}}$", "$\\mathdefault{10^{3}}$"]
    assert isinstance(ax.xaxis.get_minor_formatter(), NullFormatter)
    plt.close(fig)


def test_num_nodes_axis_labels_grid_values():
    fig, ax = plt.subplots()
    ax.plot([10, 20, 500], [1, 2, 3])
    ax.set_xscale("log")
    ps.format_axis(ax.xaxis, "num_nodes", values=[10, 20, 500])
    assert _visible_labels(ax.xaxis) == ["10", "20", "500"]
    assert isinstance(ax.xaxis.get_minor_formatter(), NullFormatter)
    plt.close(fig)


def test_lambda_axis_labels_powers_of_two():
    fig, ax = plt.subplots()
    ax.plot([0.125, 8.0], [1, 2])
    ax.set_xscale("log", base=2)
    ps.format_axis(ax.xaxis, "lambda_pen")
    assert _visible_labels(ax.xaxis)[0] == "$\\mathdefault{2^{-3}}$"
    plt.close(fig)
