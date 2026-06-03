"""Experiment 3 — Scalability per (density, pca) under the oracle partition.

For each (density, pca) cell produces a 3-row × 3-col grid PDF.
Rows: F-score-vs-samp_size, runtime-vs-samp_size, runtime-vs-num_nodes.
Cols: COARSE-oracle, kPC-k1-oracle, kPC-k3-oracle.
The COARSE-oracle baseline (pca_pooled == "na") appears in both pca subsets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.4)

df = pd.read_csv(snakemake.input[0])
coarse = df[df["method"] == "COARSE-oracle"]
kpc = df[df["method"].str.startswith("kPC")]

METHODS = ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle"]


def _draw_subplot(ax, sub, x, y, hue, log_x: bool, log_y: bool, ylim_unit: bool):
    # `num_nodes` / `samp_size` are numeric, so a bare `hue=` makes seaborn treat
    # them as a *continuous* scale: the colour legend then samples round tick
    # values (40/80/120/...) that match no actual curve, while `style=` emits a
    # second, categorical legend with the true values. Casting to an ordered
    # categorical forces a discrete qualitative palette and collapses hue+style
    # into a single legend carrying the real levels.
    sub = sub.copy()
    sub[hue] = pd.Categorical(sub[hue], categories=sorted(sub[hue].unique()),
                              ordered=True)
    sns.lineplot(
        data=sub, x=x, y=y, hue=hue, style=hue,
        markers=True, dashes=True,
        estimator="median", errorbar="ci",
        linewidth=1.5, markersize=5, ax=ax, legend="brief",
    )
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if ylim_unit:
        ax.set_ylim(0, 1)


# Hue legend title per row (rows 0,1 colour by node count; row 2 by sample size).
_ROW_LEGEND_TITLE = ["nodes (d)", "nodes (d)", "sample size (n)"]


def _plot_grid(cell: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex="row", sharey="row")
    for col_idx, method in enumerate(METHODS):
        sub = cell[cell["method"] == method]
        # Row 0: F-score vs samp_size (hue=num_nodes)
        _draw_subplot(axes[0, col_idx], sub, "samp_size", "fscore", "num_nodes",
                      log_x=True, log_y=False, ylim_unit=True)
        axes[0, col_idx].set_title(method)
        axes[0, col_idx].set_xlabel("sample size (n)")
        axes[0, col_idx].set_ylabel("F-score ↑" if col_idx == 0 else "")
        # Row 1: runtime vs samp_size (hue=num_nodes)
        _draw_subplot(axes[1, col_idx], sub, "samp_size", "runtime_sec", "num_nodes",
                      log_x=True, log_y=True, ylim_unit=False)
        axes[1, col_idx].set_xlabel("sample size (n)")
        axes[1, col_idx].set_ylabel("runtime (s)" if col_idx == 0 else "")
        # Row 2: runtime vs num_nodes (hue=samp_size)
        _draw_subplot(axes[2, col_idx], sub, "num_nodes", "runtime_sec", "samp_size",
                      log_x=True, log_y=True, ylim_unit=False)
        axes[2, col_idx].set_xlabel("nodes (d)")
        axes[2, col_idx].set_ylabel("runtime (s)" if col_idx == 0 else "")

    # One legend per row in the right margin: drop the nine per-panel legends
    # (which otherwise overlap the curves) and keep a single de-duplicated key.
    for row_idx in range(3):
        handles, labels = axes[row_idx, 2].get_legend_handles_labels()
        for col_idx in range(3):
            leg = axes[row_idx, col_idx].get_legend()
            if leg is not None:
                leg.remove()
        axes[row_idx, 2].legend(
            handles, labels, title=_ROW_LEGEND_TITLE[row_idx],
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            frameon=False, fontsize="small", title_fontsize="small",
        )
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


for density, pca, out_key in [
    (0.2, "obs",    "d02_obs"),
    (0.2, "pooled", "d02_pooled"),
    (0.5, "obs",    "d05_obs"),
    (0.5, "pooled", "d05_pooled"),
    (0.8, "obs",    "d08_obs"),
    (0.8, "pooled", "d08_pooled"),
]:
    cell = pd.concat(
        [coarse[coarse["density"] == density],
         kpc[(kpc["density"] == density) & (kpc["pca_pooled"] == pca)]],
        ignore_index=True,
    )
    _plot_grid(cell, snakemake.output[out_key])
