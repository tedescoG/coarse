import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

import _plot_style as ps

ps.apply_style()

df = pd.read_csv(snakemake.input[0])
df["m"] = df["m"].astype(int)
ROWS = {"ari": ps.label("ari"), "ratio": r"$|\Pi^\mathcal{E}|\,/\,|\Pi^\mathcal{I}|$",
        "hidden": "hidden-edge fraction"}
D = sorted(df.density.unique())
single = df[(df.design == "disjoint") & (df.k == 1)]


def grid(sub, x, hue, path, reference=False):
    levels = sorted(sub[hue].unique()) if hue else [None]
    palette = sns.color_palette("mako", len(levels) + (hue is None))
    fig, axes = plt.subplots(len(ROWS), len(D), figsize=(6.4 * len(D), 4.8 * len(ROWS)),
                             sharex=True, sharey=True, squeeze=False)
    for i, y in enumerate(ROWS):
        for j, d in enumerate(D):
            ax = axes[i, j]
            sns.lineplot(sub[sub.density == d], x=x, y=y, hue=hue, hue_order=levels if hue else None,
                         palette=palette[: len(levels)] if hue else None,
                         color=None if hue else palette[0], marker="o", estimator="mean",
                         errorbar=("ci", 95), linewidth=2.0, markersize=8, legend=False, ax=ax)
            if reference:
                ax.axhline(single[single.density == d][y].mean(), **ps.REF_LINE)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(x if i == len(ROWS) - 1 else "")
            ax.set_ylabel(ROWS[y] if j == 0 else "")
            if i == 0:
                ax.set_title(f"{ps.label('density')} = {d}")
    handles = [Line2D([], [], color=c, marker="o", lw=2.0, ms=8, label=str(v))
               for v, c in zip(levels, palette)] if hue else []
    if reference:
        handles.append(Line2D([], [], **ps.REF_LINE, label="single target"))
    if handles:
        fig.legend(handles=handles, title=hue, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   frameon=False)
    fig.tight_layout()
    ps.finish(fig, path)


grid(df[df.design == "native"], "k", "m", snakemake.output.native)
grid(df[df.design == "disjoint"], "k", None, snakemake.output.disjoint)
grid(df[df.design == "overlap"], "m", "k", snakemake.output.overlap, reference=True)
