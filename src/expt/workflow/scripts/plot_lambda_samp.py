"""λ ablation — F-score vs λ at fixed density=0.5, faceted by samp_size.

2×2 grid of panels, one per samp_size in {100, 500, 1000, 5000}. A 1×4 row left
each panel too narrow to read at a thesis text column; a square grid gives each
panel roughly four times the area. Each panel: x = lambda_pen (log2),
y = fscore, hue/style = method (COARSE-oracle / kPC-k1-oracle /
kPC-k3-oracle). The lambda_pen column is populated by evaluate.py via the
optional `lambda_pen` param exposed in lambda.smk.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.4)

df = pd.read_csv(snakemake.input[0])

SAMP_SIZES = [100, 500, 1000, 5000]
METHOD_ORDER = ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle"]

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
axes_flat = axes.flatten()
for idx, n in enumerate(SAMP_SIZES):
    ax = axes_flat[idx]
    sub = df[df["samp_size"] == n]
    sns.lineplot(
        data=sub,
        x="lambda_pen",
        y="fscore",
        hue="method",
        style="method",
        hue_order=METHOD_ORDER,
        style_order=METHOD_ORDER,
        markers=True,
        dashes=True,
        estimator="median",
        errorbar="ci",
        linewidth=1.5,
        markersize=6,
        ax=ax,
        legend="brief" if idx == 0 else False,
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"n = {n}")
    ax.set_xlabel("λ" if idx >= 2 else "")          # bottom row only
    ax.set_ylabel("F-score ↑" if idx % 2 == 0 else "")  # left column only

# Single shared legend (method is identical across panels): lift the one drawn
# on the first panel out to the figure margin so it never overlaps the curves.
handles, labels = axes_flat[0].get_legend_handles_labels()
for ax in axes_flat:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
fig.legend(
    handles, labels, title="method",
    loc="center left", bbox_to_anchor=(1.0, 0.5),
    frameon=False, fontsize="small", title_fontsize="small",
)
plt.tight_layout()
plt.savefig(snakemake.output.pdf, bbox_inches="tight", pad_inches=0.02)
plt.close()
