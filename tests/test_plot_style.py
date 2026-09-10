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


def test_node_count_column_is_labelled_p():
    assert ps.COLUMN_LABEL["num_nodes"] == "p"


def test_place_legend_puts_a_single_panel_legend_inside_the_axes():
    """On a single Axes the legend's drawn box lies within the axes' drawn box."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="a")
    ax.plot([0, 1], [1, 0], label="b")
    ps.place_legend(ax, "m")
    fig.canvas.draw()
    legend_box = ax.get_legend().get_window_extent()
    axes_box = ax.get_window_extent()
    assert axes_box.x0 <= legend_box.x0 and legend_box.x1 <= axes_box.x1
    assert axes_box.y0 <= legend_box.y0 and legend_box.y1 <= axes_box.y1
    plt.close(fig)


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


def test_line_panel_writes_a_file_and_rejects_hues_outside_the_order(tmp_path):
    """A non-empty PDF is written, and a hue_order missing a present level raises."""
    df = ps.display_methods(
        pd.DataFrame(
            {
                "method": ["COARSE-oracle", "RePaRe-oracle"] * 4,
                "samp_size": [500, 500, 1000, 1000] * 2,
                "seed": [0, 0, 0, 0, 1, 1, 1, 1],
                "fscore": [0.7, 0.5, 0.8, 0.6, 0.72, 0.52, 0.82, 0.62],
            }
        )
    )
    out = tmp_path / "p.pdf"
    ps.line_panel(df, x="samp_size", y="fscore", hue="method", out_path=out)
    assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError):
        ps.line_panel(
            df, x="samp_size", y="fscore", hue="method",
            out_path=tmp_path / "q.pdf", hue_order=["COARSE"],
        )


def test_runtime_y_axis_shows_decades_only_and_keeps_margin():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [200, 300])
    ax.set_yscale("log")
    ps.format_axis(ax.yaxis, "runtime_sec")
    assert _visible_labels(ax.yaxis) == ["$\\mathdefault{10^{2}}$", "$\\mathdefault{10^{3}}$"]
    assert ax.get_ylim()[1] > 300
    plt.close(fig)
