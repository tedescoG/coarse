import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import _plot_style as ps

ps.apply_style()

df = pd.read_csv(snakemake.input[0])
ROWS = {"ari_E": r"ARI($\hat\Pi$, $\Pi^\mathcal{E}$)", "ari_I": r"ARI($\hat\Pi$, $\Pi^\mathcal{I}$)"}
D, N = sorted(df.density.unique()), sorted(df.samp_size.unique())
palette = sns.color_palette("mako", len(N))

fig, axes = plt.subplots(len(ROWS), len(D), figsize=(6.4 * len(D), 4.8 * len(ROWS)),
                         sharex=True, sharey=True, squeeze=False)
for i, y in enumerate(ROWS):
    for j, d in enumerate(D):
        ax, sub = axes[i, j], df[df.density == d]
        sns.lineplot(sub, x="k", y=y, hue="samp_size", hue_order=N, palette=palette, marker="o",
                     estimator="mean", errorbar=("ci", 95), linewidth=2.0, markersize=8,
                     legend=False, ax=ax)
        if y == "ari_I":
            sns.lineplot(sub, x="k", y="ari_pop", errorbar=None, ax=ax, **ps.REF_LINE)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks(sorted(df.k.unique()))
        ax.set_xlabel("k" if i == len(ROWS) - 1 else "")
        ax.set_ylabel(ROWS[y] if j == 0 else "")
        if i == 0:
            ax.set_title(f"{ps.label('density')} = {d}")
handles = [Line2D([], [], color=c, marker="o", lw=2.0, ms=8, label=f"{n:,}") for n, c in zip(N, palette)]
handles.append(Line2D([], [], **ps.REF_LINE, label="population"))
fig.legend(handles=handles, title=ps.label("samp_size"), loc="center left",
           bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout()
ps.finish(fig, snakemake.output[0])
