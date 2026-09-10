"""λ ablation: F-score vs λ (log2 x-axis), one panel per value of `params.facet`.

Replaces plot_lambda_samp.py (facet=samp_size, 2x2) and plot_lambda_density.py
(facet=density, 1x3). `facet_values` come from the rule so the smoke grid renders too.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.4)

METHOD_ORDER = ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle"]

df = pd.read_csv(snakemake.input[0])
facet = str(snakemake.params.facet)
values = list(snakemake.params.facet_values)
ncols = int(snakemake.params.ncols)
title = str(snakemake.params.title)
nrows = math.ceil(len(values) / ncols)

fig, axes = plt.subplots(
    nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), sharex=True, sharey=True, squeeze=False
)
axes_flat = axes.flatten()
for idx, value in enumerate(values):
    ax = axes_flat[idx]
    sub = df[df[facet] == value]
    if sub.empty:
        raise ValueError(f"no rows with {facet} == {value!r}")
    unknown = set(sub["method"]) - set(METHOD_ORDER)
    if unknown:
        raise ValueError(f"method values outside METHOD_ORDER={METHOD_ORDER}: {unknown}")
    sns.lineplot(
        data=sub, x="lambda_pen", y="fscore", hue="method", style="method",
        hue_order=METHOD_ORDER, style_order=METHOD_ORDER, markers=True, dashes=True,
        estimator="median", errorbar="ci", linewidth=1.5, markersize=6, ax=ax,
        legend="brief" if idx == 0 else False,
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_title(title.format(value))
    ax.set_xlabel("λ" if idx >= len(values) - ncols else "")  # bottom row only
    ax.set_ylabel("F-score ↑" if idx % ncols == 0 else "")  # left column only
for ax in axes_flat[len(values):]:
    ax.set_visible(False)

# One shared legend in the figure margin so it never overlaps the curves.
handles, labels = axes_flat[0].get_legend_handles_labels()
axes_flat[0].get_legend().remove()
fig.legend(
    handles, labels, title="method", loc="center left", bbox_to_anchor=(1.0, 0.5),
    frameon=False, fontsize="small", title_fontsize="small",
)
plt.tight_layout()
plt.savefig(snakemake.output.pdf, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
