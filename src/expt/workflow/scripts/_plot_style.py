"""Shared figure style for every synthetic-experiment plot script.

One palette anchored on the thesis colours (KU red #901A1E, teal #0A5963), one
typography, one label vocabulary and one axis-formatting rule, so figures from
different rule files look like one suite. Import-only module (leading underscore):
tests/test_plot_style.py imports it directly.
"""

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

# --- colours ---------------------------------------------------------------------------
KU_RED = "#901A1E"  # \definecolor{KUrod}{RGB}{144,26,30}
TEAL = "#0A5963"  # \definecolor{teal}{HTML}{0A5963}
GOLD = "#D6A43C"
SLATE = "#3F6E9A"
OLIVE = "#6B8F3A"
GREY = "#7F7F7F"
CATEGORICAL = [KU_RED, TEAL, GOLD, SLATE, OLIVE, GREY]

# --- methods ---------------------------------------------------------------------------
# evaluate.py labels every oracle run "<method>-oracle"; figures show the thesis names and
# the caption states the oracle-partition setting. COARSE is KU red in every figure.
METHOD_DISPLAY = {
    "COARSE": "COARSE",
    "COARSE-CV": "COARSE-CV",
    "COARSE-1PC": "COARSE-1PC",
    "COARSE-oracle": "COARSE",
    "kPC-k1-oracle": "COARSE-1PC",
    "kPC-k3-oracle": "COARSE-3PC",
    "RePaRe-oracle": "RePaRe",
}
METHOD_ORDER = ["COARSE", "COARSE-CV", "COARSE-1PC", "COARSE-3PC", "RePaRe"]
METHOD_COLOR = dict(zip(METHOD_ORDER, [KU_RED, TEAL, GOLD, SLATE, OLIVE]))
METHOD_MARKER = dict(zip(METHOD_ORDER, ["o", "D", "s", "^", "v"]))

# --- columns ---------------------------------------------------------------------------
COLUMN_LABEL = {
    "samp_size": "sample size (n)",
    "num_nodes": "nodes",
    "density": "density",
    "noise": "noise",
    "method": "method",
    "lambda_pen": "λ",
    "targets_per_interv": "targets per intervention",
    "ari": "ARI ↑",
    "fscore": "F-score ↑",
    "precision": "precision ↑",
    "recall": "recall ↑",
    "runtime_sec": "run time (s)",
}
UNIT_RANGE = {"ari", "fscore", "precision", "recall"}  # y in [0, 1]
LOG_Y = {"runtime_sec"}
ORDINAL_COLUMNS = {"density", "num_nodes", "samp_size"}  # hue drawn from the ramp


def label(column: str) -> str:
    return COLUMN_LABEL.get(column, column)


def apply_style() -> None:
    """Seaborn paper context at the suite's font scale, leaving matplotlib's default
    sans-serif text and default mathtext for the 10^k / 2^k ticks. No usetex: must render
    without TeX."""
    sns.set_theme(style="ticks", context="paper", font_scale=2.3, palette=CATEGORICAL)


def ordinal_palette(n: int) -> list:
    """`n` colours sampled from the teal → gold → KU red ramp (ordered hue levels)."""
    return list(sns.blend_palette([TEAL, GOLD, KU_RED], n_colors=n))


def hue_palette(column: str, levels: list):
    """Palette for a hue column: fixed method colours, the ramp for ordinal columns,
    the categorical list otherwise."""
    if column == "method":
        return {m: METHOD_COLOR[m] for m in levels}
    if column in ORDINAL_COLUMNS:
        return ordinal_palette(len(levels))
    return CATEGORICAL[: len(levels)]


def hue_markers(column: str, levels: list):
    if column == "method":
        return {m: METHOD_MARKER[m] for m in levels}
    return True


def display_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Map evaluate.py method labels to display names; unknown labels are an error so a
    new method cannot silently vanish from a figure."""
    unknown = set(df["method"]) - set(METHOD_DISPLAY)
    if unknown:
        raise ValueError(f"method labels without a display name: {sorted(unknown)}")
    return df.assign(method=df["method"].map(METHOD_DISPLAY))


def method_order(levels) -> list:
    return [m for m in METHOD_ORDER if m in set(levels)]


def format_axis(axis, column: str, values=None) -> None:
    """Tick policy per column on a log axis.
    samp_size / runtime_sec: decades only (10^k), minor ticks unlabeled, limits widened
      to the enclosing decades when fewer than two decade ticks would show.
    num_nodes: ticks at the grid values, plain integers, no minor ticks.
    lambda_pen: base-2 decades (2^k).
    """
    if column == "num_nodes":
        if values is None:
            raise ValueError("num_nodes axis needs the grid values")
        axis.set_major_locator(FixedLocator(sorted(values)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
        return
    base = 2 if column == "lambda_pen" else 10
    if base == 10:
        # Decades come from the data, not the padded view: the few percent of margin past
        # the last point would otherwise pull in a whole empty decade (n=1000 -> 10^4).
        lo, hi = axis.get_data_interval()
        if not (np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > 0):
            lo, hi = axis.get_view_interval()
        lo_dec, hi_dec = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        if hi_dec - lo_dec < 1:
            hi_dec = lo_dec + 1
        # Widen only outward, and against the *view* interval, not the data: a grid whose
        # endpoint is a power of ten (n=1000) would otherwise lose its autoscale margin and
        # leave the last marker half-clipped on the spine.
        vlo, vhi = axis.get_view_interval()
        set_lim = axis.axes.set_xlim if axis.axis_name == "x" else axis.axes.set_ylim
        set_lim(min(vlo, 10.0**lo_dec), max(vhi, 10.0**hi_dec))
    axis.set_major_locator(LogLocator(base=base, subs=(1.0,), numticks=20))
    axis.set_major_formatter(LogFormatterSciNotation(base=base, labelOnlyBase=True))
    if base == 10:
        axis.set_minor_locator(LogLocator(base=base, subs=np.arange(2, base), numticks=20))
    else:
        axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


# Every line figure in the suite draws the median across seeds with a bootstrap CI band.
LINE_KWARGS = dict(dashes=True, estimator="median", errorbar="ci", linewidth=2.0, markersize=8)


def style_axes(ax, x, y, *, values=None, xlabel=True, ylabel=True) -> None:
    """Scales, tick policy, [0, 1] limits and axis labels for one panel, keyed by column
    name. `values` is the x grid, required when x is num_nodes. Pass xlabel/ylabel False on
    a facet whose label belongs to the edge of the grid rather than to the panel."""
    if x in ("samp_size", "num_nodes"):
        ax.set_xscale("log")
        format_axis(ax.xaxis, x, values=values)
    elif x == "lambda_pen":
        ax.set_xscale("log", base=2)
        format_axis(ax.xaxis, x)
    if y in LOG_Y:
        ax.set_yscale("log")
        format_axis(ax.yaxis, y)
    if y in UNIT_RANGE:
        ax.set_ylim(0, 1)
    ax.set_xlabel(label(x) if xlabel else "")
    ax.set_ylabel(label(y) if ylabel else "")


def place_legend(target, title: str | None) -> None:
    """Single panel (an Axes): inside the axes, in whichever corner is emptiest. Grid (a
    seaborn FacetGrid): in the right margin, where no panel can be covered. The Axes branch
    rebuilds the legend from the axes' labelled artists, discarding any legend already
    built with custom handles."""
    if isinstance(target, sns.axisgrid.Grid):
        sns.move_legend(target, "center left", bbox_to_anchor=(1.0, 0.5), title=title, frameon=False)
    else:
        # loc="best" picks the emptiest corner; the translucent box keeps curves visible.
        target.legend(title=title, loc="best", frameon=True, framealpha=0.6, facecolor="white")


def finish(fig, path) -> None:
    """Save and close. The bbox settings are arguments rather than rcParams so they hold
    even if the caller never ran apply_style(); every figure in the suite saves here, so
    this is the one place that decides them. `tight` is what keeps a legend in a grid's
    right margin inside the page."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def line_panel(sub, *, x, y, hue, out_path, hue_order=None) -> None:
    """One 6.4×4.8 in line panel: median over seeds with CI band, styled by column name
    (scales, tick policy, [0,1] limits, labels, palette, markers, inset legend), saved to
    out_path. Every single-panel figure in the suite goes through here, so the panels are
    identical by construction rather than by three copies of the same block."""
    levels = sub[hue].unique()
    if hue_order is not None:
        order = list(hue_order)
    elif hue == "method":
        order = method_order(levels)
    else:
        order = sorted(levels)
    # Checked before the cast, not after: pandas 4 will raise on values outside `categories`
    # instead of silently turning them into NaN, so the membership test must come first.
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
    style_axes(ax, x, y, values=sorted(sub[x].unique()))
    place_legend(ax, label(hue))
    finish(fig, out_path)
