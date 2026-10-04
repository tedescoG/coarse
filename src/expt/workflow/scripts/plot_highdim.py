import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import _plot_style as ps

ps.apply_style()

graph = ps.display_methods(pd.read_csv(snakemake.input.graph))
blocks = ps.display_methods(pd.read_csv(snakemake.input.blocks))
ORDER = ps.method_order(graph["method"].unique())
P = int(graph["num_nodes"].iloc[0])
LABELS = {
    "precision": ps.label("precision"),
    "recall": ps.label("recall"),
    "fscore": ps.label("fscore"),
    "exact": "exact parent set ↑",
    "feasible": r"blocks with $n > F_j$",
}
CELL = ["num_intervs", "seed", "samp_size", "method"]


def grid(df, x, ys, xlabel, vline, estimator, path):
    ivns = sorted(df["num_intervs"].unique())
    fig, axes = plt.subplots(len(ys), len(ivns), figsize=(6.4 * len(ivns), 4.8 * len(ys)),
                             sharex=True, sharey=True, squeeze=False)
    for i, y in enumerate(ys):
        for j, m in enumerate(ivns):
            ax = axes[i, j]
            sns.lineplot(df[df["num_intervs"] == m], x=x, y=y, hue="method", style="method",
                         hue_order=ORDER, style_order=ORDER, palette=ps.hue_palette("method", ORDER),
                         markers=ps.hue_markers("method", ORDER), dashes=True, estimator=estimator,
                         errorbar="ci", linewidth=2.0, markersize=8, legend=(i == 0 and j == 0),
                         ax=ax)
            ax.axvline(vline, **ps.REF_LINE)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel(xlabel if i == len(ys) - 1 else "")
            ax.set_ylabel(LABELS[y] if j == 0 else "")
            if i == 0:
                ax.set_title(f"interventions = {m}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].get_legend().remove()
    fig.legend(handles, labels, title=ps.label("method"), loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    ps.finish(fig, path)


N_LABEL = "sample size per env (n)"
grid(graph, "samp_size", ["precision", "recall", "fscore"], N_LABEL, P, "median",
     snakemake.output.graph)

edges = 2.0 ** np.arange(-4, 11)
blocks["ratio"] = pd.cut(blocks["samp_size"] / blocks["F"], edges,
                         labels=np.sqrt(edges[:-1] * edges[1:])).astype(float)
binned = blocks.groupby([*CELL, "ratio"], as_index=False)[["exact", "recall"]].mean()
grid(binned, "ratio", ["exact", "recall"], r"$n\,/\,F_j$ (variables in the true family)", 1, "mean",
     snakemake.output.blocks)

blocks["feasible"] = (blocks["samp_size"] > blocks["F_method"]).astype(float)
feasible = blocks.groupby(CELL, as_index=False)["feasible"].mean()
grid(feasible, "samp_size", ["feasible"], N_LABEL, P, "mean", snakemake.output.feasible)
