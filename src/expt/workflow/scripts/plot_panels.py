"""One line plot per `params.panels` entry: out, x, y, hue; optional filter, hue_order, col (facet),
vline (dashed x reference)."""

import pandas as pd
import seaborn as sns

import _plot_style as ps

ps.apply_style()

df = pd.read_csv(snakemake.input[0], dtype={"targets_per_interv": str})


def _subset(panel: dict) -> pd.DataFrame:
    """Filter on evaluate.py labels, then map them to display names."""
    sub = df
    for column, value in panel.get("filter", {}).items():
        sub = sub[sub[column] == value]
    if sub.empty:
        raise ValueError(f"panel {panel['out']!r}: no rows after filter {panel.get('filter')}")
    if "method" in sub.columns:
        sub = ps.display_methods(sub)
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


for panel in snakemake.params.panels:
    sub = _subset(panel)
    hue = panel["hue"]
    levels = list(sub[hue].cat.categories)
    out_path = snakemake.output[panel["out"]]
    if panel.get("col") is None:
        ps.line_panel(
            sub, x=panel["x"], y=panel["y"], hue=hue, out_path=out_path,
            hue_order=levels, vline=panel.get("vline"),
        )
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
        kind="line", col=panel["col"], height=4.8, aspect=1.33,
        **ps.LINE_KWARGS,
        facet_kws={"sharey": True, "sharex": True, "legend_out": False},
    )
    axes = g.axes
    values = sorted(sub[panel["x"]].unique())
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ps.style_axes(
                axes[i, j], panel["x"], panel["y"], values=values,
                xlabel=(i == axes.shape[0] - 1), ylabel=(j == 0),
            )
    g.set_titles(ps.label(panel["col"]) + " = {col_name}")
    ps.place_legend(g, ps.label(hue))
    ps.finish(g.figure, out_path)
