"""Figure style shared by the synthetic-experiment plots: Okabe–Ito for methods, mako for
other hues, axis labels and log-axis ticks."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogFormatterSciNotation,
    LogLocator,
    NullFormatter,
    NullLocator,
)

OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# evaluate.py labels → display names.
METHOD_DISPLAY = {
    "COARSE": "COARSE",
    "COARSE-oracle": "COARSE",
    "kPC-k1-oracle": "COARSE-1PC",
    "kPC-k3-oracle": "COARSE-3PC",
    "RePaRe-oracle": "RePaRe",
}
METHOD_ORDER = ["COARSE", "COARSE-1PC", "COARSE-3PC", "RePaRe"]
METHOD_COLOR = {
    "COARSE": "#0072B2", "COARSE-1PC": "#009E73", "COARSE-3PC": "#CC79A7", "RePaRe": "#E69F00"
}
METHOD_MARKER = dict(zip(METHOD_ORDER, ["o", "s", "^", "v"]))

COLUMN_LABEL = {
    "samp_size": "sample size (n)",
    "num_nodes": "nodes",
    "lambda_pen": "penalty multiplier λ",
    "density": "density",
    "noise": "noise",
    "method": "method",
    "ari": "ARI ↑",
    "fscore": "F-score ↑",
    "precision": "precision ↑",
    "recall": "recall ↑",
    "runtime_sec": "run time (s)",
}
UNIT_RANGE = {"ari", "fscore", "precision", "recall"}  # y in [0, 1]
LOG_Y = {"runtime_sec"}
GRID_TICKS = {"num_nodes", "lambda_pen"}  # log x ticked at the grid values

def label(column: str) -> str:
    return COLUMN_LABEL.get(column, column)


def apply_style() -> None:
    """Seaborn paper style; no TeX needed."""
    sns.set_theme(style="ticks", context="paper", font_scale=2.3, palette=OKABE_ITO)


def hue_palette(column: str, levels: list):
    """Okabe–Ito method colours for `method`, mako for every other hue."""
    if column == "method":
        return {m: METHOD_COLOR[m] for m in levels}
    return sns.color_palette("mako", len(levels))


def hue_markers(column: str, levels: list):
    if column == "method":
        return {m: METHOD_MARKER[m] for m in levels}
    return True


def display_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Map method labels to display names; unknown labels raise."""
    unknown = set(df["method"]) - set(METHOD_DISPLAY)
    if unknown:
        raise ValueError(f"method labels without a display name: {sorted(unknown)}")
    return df.assign(method=df["method"].map(METHOD_DISPLAY))


def method_order(levels) -> list:
    return [m for m in METHOD_ORDER if m in set(levels)]


def format_axis(axis, column: str, values=None) -> None:
    """Log-axis ticks: `GRID_TICKS` columns at the grid values, anything else at decades only."""
    if column in GRID_TICKS:
        if values is None:
            raise ValueError(f"{column} axis needs the grid values")
        axis.set_major_locator(FixedLocator(sorted(values)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
        return
    # Decades from the data, not the padded view, so no empty decade is added.
    lo, hi = axis.get_data_interval()
    if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0):
        lo, hi = axis.get_view_interval()
    lo_dec, hi_dec = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    if hi_dec - lo_dec < 1:
        hi_dec = lo_dec + 1
    # Widen only outward from the view, keeping the autoscale margin.
    vlo, vhi = axis.get_view_interval()
    set_lim = axis.axes.set_xlim if axis.axis_name == "x" else axis.axes.set_ylim
    set_lim(min(vlo, 10.0**lo_dec), max(vhi, 10.0**hi_dec))
    axis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
    axis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    axis.set_minor_formatter(NullFormatter())


# Median across seeds with a bootstrap CI band.
LINE_KWARGS = dict(dashes=True, estimator="median", errorbar="ci", linewidth=2.0, markersize=8)


def style_axes(ax, x, y, *, values=None, xlabel=True, ylabel=True) -> None:
    """Scales, ticks, [0, 1] limits and labels of one panel. `values` is the x grid (needed
    for `GRID_TICKS`)."""
    if x == "samp_size" or x in GRID_TICKS:
        ax.set_xscale("log")
        format_axis(ax.xaxis, x, values=values)
    if y in LOG_Y:
        ax.set_yscale("log")
        format_axis(ax.yaxis, y)
    if y in UNIT_RANGE:
        ax.set_ylim(0, 1)
    ax.set_xlabel(label(x) if xlabel else "")
    ax.set_ylabel(label(y) if ylabel else "")


def place_legend(target, title: str | None) -> None:
    """Inside a single Axes (emptiest corner); in the right margin of a FacetGrid."""
    if isinstance(target, sns.axisgrid.Grid):
        sns.move_legend(target, "center left", bbox_to_anchor=(1.0, 0.5), title=title, frameon=False)
    else:
        target.legend(title=title, loc="best", frameon=True, framealpha=0.6, facecolor="white")


def finish(fig, path) -> None:
    """Save (tight bbox) and close."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def line_panel(sub, *, x, y, hue, out_path, hue_order=None, vline=None) -> None:
    """One styled line panel saved to `out_path`; `vline` draws a dashed reference at that x."""
    levels = sub[hue].unique()
    if hue_order is not None:
        order = list(hue_order)
    elif hue == "method":
        order = method_order(levels)
    else:
        order = sorted(levels)
    outside = set(levels) - set(order)
    if outside:
        raise ValueError(f"{hue} values outside hue_order={order}: {sorted(outside)}")
    sub = sub.copy()
    sub[hue] = pd.Categorical(sub[hue], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.lineplot(
        data=sub, x=x, y=y, hue=hue, style=hue,
        hue_order=order, style_order=order,
        palette=hue_palette(hue, order), markers=hue_markers(hue, order),
        ax=ax, **LINE_KWARGS,
    )
    if vline is not None:
        ax.axvline(vline, ls="--", color="0.5", lw=1.2, zorder=0)
    style_axes(ax, x, y, values=sorted(sub[x].unique()))
    place_legend(ax, label(hue))
    finish(fig, out_path)
