"""λ ablation: F-score vs λ (log2 x-axis), one panel per value of `params.facet`.

`facet_values` come from the rule so the smoke grid renders too. Styling comes from
_plot_style: display names, fixed method colours/markers, 2^k ticks, outside legend.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import _plot_style as ps

ps.apply_style()

df = ps.display_methods(pd.read_csv(snakemake.input[0]))
facet = str(snakemake.params.facet)
values = list(snakemake.params.facet_values)
ncols = int(snakemake.params.ncols)
title = str(snakemake.params.title)
nrows = math.ceil(len(values) / ncols)
order = ps.method_order(df["method"].unique())

fig, axes = plt.subplots(
    nrows, ncols, figsize=(6.4 * ncols, 4.8 * nrows), sharex=True, sharey=True, squeeze=False
)
axes_flat = axes.flatten()
for idx, value in enumerate(values):
    ax = axes_flat[idx]
    sub = df[df[facet] == value]
    if sub.empty:
        raise ValueError(f"no rows with {facet} == {value!r}")
    sns.lineplot(
        data=sub, x="lambda_pen", y="fscore", hue="method", style="method",
        hue_order=order, style_order=order,
        palette=ps.hue_palette("method", order), markers=ps.hue_markers("method", order),
        ax=ax, legend=(idx == 0), **ps.LINE_KWARGS,
    )
    ps.style_axes(
        ax, "lambda_pen", "fscore",
        xlabel=(idx >= len(values) - ncols), ylabel=(idx % ncols == 0),
    )
    ax.set_title(title.format(value))
for ax in axes_flat[len(values):]:
    ax.set_visible(False)

# One shared legend in the right margin (frameless, outside every panel).
handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[0].get_legend().remove()
fig.legend(handles, labels, title=ps.label("method"), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout()
ps.finish(fig, snakemake.output.pdf)
