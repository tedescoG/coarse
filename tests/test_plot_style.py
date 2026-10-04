"""Tests for the shared figure style in workflow/scripts/_plot_style.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.ticker import NullFormatter

import _plot_style as ps


def _visible_labels(axis):
    axis.axes.figure.canvas.draw()
    lo, hi = axis.get_view_interval()
    return [
        t.get_text()
        for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
        if lo <= loc <= hi and t.get_text()
    ]


def test_method_hue_uses_okabe_ito_and_other_hues_use_mako():
    assert ps.hue_palette("method", ["COARSE", "RePaRe"]) == {"COARSE": "#0072B2", "RePaRe": "#E69F00"}
    assert set(ps.METHOD_COLOR.values()) <= set(ps.OKABE_ITO)
    assert ps.hue_palette("density", [0.2, 0.5, 0.8]) == sns.color_palette("mako", 3)


def test_place_legend_puts_a_single_panel_legend_inside_the_axes():
    """At the suite's font scale, on a single Axes, the legend's drawn box lies within the
    axes' drawn box."""
    with matplotlib.rc_context():
        ps.apply_style()
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
    df = pd.DataFrame({"method": ["COARSE", "COARSE-oracle", "kPC-k1-oracle", "RePaRe-oracle"]})
    out = ps.display_methods(df)
    assert out["method"].tolist() == ["COARSE", "COARSE", "COARSE-1PC", "RePaRe"]
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


def test_lambda_pen_axis_labels_grid_values():
    fig, ax = plt.subplots()
    ax.plot([0.25, 1.0, 4.0], [1, 2, 3])
    ax.set_xscale("log")
    ps.format_axis(ax.xaxis, "lambda_pen", values=[0.25, 0.5, 1.0, 2.0, 4.0])
    assert _visible_labels(ax.xaxis) == ["0.25", "0.5", "1", "2", "4"]
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
    ps.line_panel(df, x="samp_size", y="fscore", hue="method", out_path=out, vline=700)
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
