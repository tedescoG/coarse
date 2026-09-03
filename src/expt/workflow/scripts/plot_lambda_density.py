"""λ ablation — F-score vs λ at fixed samp_size=1000, faceted by density.

Same layout as plot_lambda_samp.py but 1×3 panels, one per density in
{0.2, 0.5, 0.8}.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.4)

df = pd.read_csv(snakemake.input[0])

DENSITIES = [0.2, 0.5, 0.8]
METHOD_ORDER = ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle"]

fig, axes = plt.subplots(
    1, len(DENSITIES),
    figsize=(5 * len(DENSITIES), 5),
    sharey=True,
)
for col_idx, d in enumerate(DENSITIES):
    ax = axes[col_idx]
    sub = df[df["density"] == d]
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
        legend="brief",
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_title(f"density = {d}")
    ax.set_xlabel("λ")
    ax.set_ylabel("F-score ↑" if col_idx == 0 else "")
plt.tight_layout()
plt.savefig(snakemake.output.pdf, bbox_inches="tight", pad_inches=0.02)
plt.close()
