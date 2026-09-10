"""Generic line-panel renderer: one seaborn lineplot per entry of `snakemake.params.panels`.

Each panel is a dict:
    out        key of the output file in `snakemake.output`
    x, y, hue  column names; the hue column is cast to an ordered categorical so a numeric
               hue lists its real levels in the legend instead of sampled round ticks
    filter     optional {column: value} equality filter applied before plotting
    hue_order  optional explicit level order (default: METHOD_ORDER for `method`, else sorted)
    col        optional column to facet on (one row of panels sharing the y axis)
Scales, limits, labels, palette, markers and legend placement come from _plot_style, keyed
by column name, so every rule file gets identical styling for free.
Every panel draws the median across seeds with a bootstrap CI band.
"""

import pandas as pd
import seaborn as sns

import _plot_style as ps

ps.apply_style()

df = pd.read_csv(snakemake.input[0], dtype={"targets_per_interv": str})
if "method" in df.columns:
    df = ps.display_methods(df)


def _subset(panel: dict) -> pd.DataFrame:
    sub = df
    for column, value in panel.get("filter", {}).items():
        sub = sub[sub[column] == value]
    if sub.empty:
        raise ValueError(f"panel {panel['out']!r}: no rows after filter {panel.get('filter')}")
    sub = sub.copy()
    hue = panel["hue"]
    if panel.get("hue_order"):
        order = list(panel["hue_order"])
    elif hue == "method":
        order = ps.method_order(sub[hue].unique())
    else:
        order = sorted(sub[hue].unique())
    outside = set(sub[hue].unique()) - set(order)
    if outside:
        raise ValueError(f"panel {panel['out']!r}: hue values outside hue_order={order}")
    sub[hue] = pd.Categorical(sub[hue], categories=order, ordered=True)
    return sub


def _style(ax, panel: dict, sub: pd.DataFrame, *, xlabel: bool, ylabel: bool) -> None:
    """Facet-branch styling only; the single-panel branch gets the same policy from
    _plot_style.line_panel."""
    x, y = panel["x"], panel["y"]
    if x in ("samp_size", "num_nodes"):
        ax.set_xscale("log")
        ps.format_axis(ax.xaxis, x, values=sorted(sub[x].unique()))
    elif x == "lambda_pen":
        ax.set_xscale("log", base=2)
        ps.format_axis(ax.xaxis, x)
    if y in ps.LOG_Y:
        ax.set_yscale("log")
        ps.format_axis(ax.yaxis, y)
    if y in ps.UNIT_RANGE:
        ax.set_ylim(0, 1)
    ax.set_xlabel(ps.label(x) if xlabel else "")
    ax.set_ylabel(ps.label(y) if ylabel else "")


for panel in snakemake.params.panels:
    sub = _subset(panel)
    hue = panel["hue"]
    levels = list(sub[hue].cat.categories)
    out_path = snakemake.output[panel["out"]]
    if panel.get("col") is None:
        ps.line_panel(sub, x=panel["x"], y=panel["y"], hue=hue, out_path=out_path, hue_order=levels)
        continue
    g = sns.relplot(
        data=sub,
        x=panel["x"],
        y=panel["y"],
        hue=hue,
        style=hue,
        hue_order=levels,
        style_order=levels,
        palette=ps.hue_palette(hue, levels),
        markers=ps.hue_markers(hue, levels),
        dashes=True,
        estimator="median",
        errorbar="ci",
        linewidth=2.0,
        markersize=8,
        kind="line", col=panel["col"], height=4.8, aspect=1.33,
        facet_kws={"sharey": True, "sharex": True, "legend_out": False},
    )
    axes = g.axes
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            _style(axes[i, j], panel, sub, xlabel=(i == axes.shape[0] - 1), ylabel=(j == 0))
    g.set_titles(ps.label(panel["col"]) + " = {col_name}")
    ps.place_legend(g, ps.label(hue))
    ps.finish(g.figure, out_path)
