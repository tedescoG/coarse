"""Tests for the shared figure style in workflow/scripts/_plot_style.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.ticker import NullFormatter

import _plot_style as ps

DECADES = ["$\\mathdefault{10^{2}}$", "$\\mathdefault{10^{3}}$"]


@pytest.mark.parametrize(
    "column, data, values, expected",
    [
        ("samp_size", [500, 1000], None, DECADES),
        ("num_nodes", [10, 20, 500], [10, 20, 500], ["10", "20", "500"]),
        ("lambda_pen", [0.25, 1.0, 4.0], [0.25, 0.5, 1.0, 2.0, 4.0], ["0.25", "0.5", "1", "2", "4"]),
        ("runtime_sec", [200, 300], None, DECADES),
    ],
)
def test_format_axis_ticks(column, data, values, expected):
    """Grid columns are ticked at the grid values, others at decades only; no minor labels."""
    fig, ax = plt.subplots()
    axis = ax.yaxis if column == "runtime_sec" else ax.xaxis
    if axis is ax.yaxis:
        ax.plot(range(len(data)), data)
        ax.set_yscale("log")
    else:
        ax.plot(data, range(len(data)))
        ax.set_xscale("log")
    ps.format_axis(axis, column, values=values)
    fig.canvas.draw()
    lo, hi = axis.get_view_interval()
    labels = [
        t.get_text()
        for loc, t in zip(axis.get_majorticklocs(), axis.get_majorticklabels())
        if lo <= loc <= hi and t.get_text()
    ]
    assert labels == expected
    assert isinstance(axis.get_minor_formatter(), NullFormatter)
    if column == "runtime_sec":
        assert ax.get_ylim()[1] > 300
    plt.close(fig)


def test_line_panel_maps_methods_and_rejects_unknown_levels(tmp_path):
    """Method labels map to display names (unknown ones raise); a PDF is written; a hue_order
    missing a present level raises."""
    with pytest.raises(ValueError):
        ps.display_methods(pd.DataFrame({"method": ["mystery"]}))
    df = ps.display_methods(
        pd.DataFrame(
            {
                "method": ["COARSE-oracle", "RePaRe-oracle", "kPC-k1-oracle", "avg-oracle"] * 2,
                "samp_size": [500] * 4 + [1000] * 4,
                "fscore": [0.7, 0.5, 0.6, 0.65, 0.8, 0.6, 0.7, 0.75],
            }
        )
    )
    assert df["method"].tolist()[:4] == ["COARSE", "RePaRe", "COARSE-1PC", "COARSE-avg"]
    out = tmp_path / "p.pdf"
    ps.line_panel(df, x="samp_size", y="fscore", hue="method", out_path=out, vline=700)
    assert out.stat().st_size > 0
    with pytest.raises(ValueError):
        ps.line_panel(df, x="samp_size", y="fscore", hue="method", out_path=tmp_path / "q.pdf",
                      hue_order=["COARSE"])
