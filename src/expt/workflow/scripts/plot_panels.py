"""Generic line-panel renderer: one seaborn lineplot per entry of `snakemake.params.panels`.

Replaces plot.py / scalability_plot.py / plot_exp2.py / plot_exp3.py. Each panel is a dict:
    out        key of the output file in `snakemake.output`
    x, y, hue  column names; the hue column is cast to an ordered categorical so a numeric
               hue lists its real levels in the legend instead of sampled round ticks
    filter     optional {column: value} equality filter applied before plotting
    hue_order  optional explicit level order (default: sorted unique values)
    col        optional column to facet on (one row of panels sharing the y axis)
    logx/logy  axis scales (default log x, linear y)
    ylim       optional (lo, hi)
    xlabel, ylabel, legend   axis labels and legend title
Every panel draws the median across seeds with a bootstrap CI band.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.3)

df = pd.read_csv(snakemake.input[0], dtype={"targets_per_interv": str})


def _subset(panel: dict) -> pd.DataFrame:
    sub = df
    for column, value in panel.get("filter", {}).items():
        sub = sub[sub[column] == value]
    if sub.empty:
        raise ValueError(f"panel {panel['out']!r}: no rows after filter {panel.get('filter')}")
    sub = sub.copy()
    hue = panel["hue"]
    order = panel.get("hue_order") or sorted(sub[hue].unique())
    sub[hue] = pd.Categorical(sub[hue], categories=order, ordered=True)
    if sub[hue].isna().any():
        raise ValueError(f"panel {panel['out']!r}: hue values outside hue_order={order}")
    return sub


def _style(ax, panel: dict) -> None:
    if panel.get("logx", True):
        ax.set_xscale("log")
    if panel.get("logy", False):
        ax.set_yscale("log")
    if panel.get("ylim") is not None:
        ax.set_ylim(*panel["ylim"])
    ax.set_xlabel(panel.get("xlabel", panel["x"]))
    ax.set_ylabel(panel.get("ylabel", panel["y"]))


for panel in snakemake.params.panels:
    sub = _subset(panel)
    line_kwargs = dict(
        data=sub,
        x=panel["x"],
        y=panel["y"],
        hue=panel["hue"],
        style=panel["hue"],
        markers=True,
        dashes=True,
        estimator="median",
        errorbar="ci",
        linewidth=2.0,
        markersize=7,
    )
    if panel.get("col") is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        sns.lineplot(ax=ax, **line_kwargs)
        _style(ax, panel)
        # loc="best" picks the emptiest corner; the translucent box keeps curves visible.
        ax.legend(title=panel.get("legend"), loc="best", frameon=True, framealpha=0.6, facecolor="white")
    else:
        g = sns.relplot(kind="line", col=panel["col"], height=5, aspect=1.0, facet_kws={"sharey": True}, **line_kwargs)
        for ax in g.axes.flat:
            _style(ax, panel)
        g.set_titles(panel["col"] + " = {col_name}")
        sns.move_legend(
            g, "lower center", bbox_to_anchor=(0.5, -0.08),
            ncol=len(sub[panel["hue"]].cat.categories), title=panel.get("legend"), frameon=False,
        )
        fig = g.figure
    fig.savefig(snakemake.output[panel["out"]], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
