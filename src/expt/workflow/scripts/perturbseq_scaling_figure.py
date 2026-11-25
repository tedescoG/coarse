# ruff: noqa: F821
"""Scaling figure for the Perturb-seq ladder: held-out density (left) and runtime
(right) against the panel dimension.

Density: median held-out log-likelihood per gene over the four batch splits, MAD bars.
Runtime: median per fit on a log scale, stacked as the final fit (solid boundary, lower)
plus inner-fold tuning (shaded band above it), so the tuning overhead is visible without
hiding the scaling slope. The COARSE score-phase compressions (1PC, 3PC, avg) share the
COARSE colour in lighter tints; the empty model is the no-structure reference."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _plot_style as ps

ps.apply_style()

METHODS = [
    "coarse",
    "coarse_kpc1",
    "coarse_kpc3",
    "coarse_avg",
    "coarse_scored",
    "repare",
    "repare_welch",
    "utigsp",
    "singleton_empty",
    "joint",
]
DISPLAY = {
    "coarse": "COARSE",
    "coarse_kpc1": "1PC-COARSE",
    "coarse_kpc3": "3PC-COARSE",
    "coarse_avg": "avg-COARSE",
    "coarse_scored": "COARSE (scored)",
    "repare": "RePaRe",
    "repare_welch": "RePaRe (Welch)",
    "utigsp": "UT-IGSP",
    "singleton_empty": "Empty",
    "joint": "One block",
}
COLORS = {
    "coarse": "#0072B2",
    "coarse_kpc1": "#7FB3D9",
    "coarse_kpc3": "#4A97C7",
    "coarse_avg": "#A8CBE4",
    "coarse_scored": "#D55E00",
    "repare": "#E69F00",
    "repare_welch": "#F0B441",
    "utigsp": "#009E73",
    "singleton_empty": "#777777",
    "joint": "#CC79A7",
}
MARKERS = {
    "coarse": "o",
    "coarse_kpc1": "o",
    "coarse_kpc3": "o",
    "coarse_avg": "o",
    "coarse_scored": "X",
    "repare": "s",
    "repare_welch": "s",
    "utigsp": "^",
    "singleton_empty": "v",
    "joint": "D",
}
LINES = {
    "coarse": "-",
    "coarse_kpc1": "--",
    "coarse_kpc3": "-.",
    "coarse_avg": ":",
    "coarse_scored": "-.",
    "repare": "-",
    "repare_welch": "--",
    "utigsp": "-",
    "singleton_empty": "--",
    "joint": "--",
}


def median_mad(values):
    med = float(np.median(values))
    return med, float(np.median(np.abs(values - med)))


def ladder_frame(summary_path, panel):
    df = pd.read_csv(summary_path)
    df = df[(df["status"] == "ok") & (df["method"].isin(METHODS))].copy()
    df["panel"] = int(panel)
    return df


def main():
    data = pd.concat(
        [
            ladder_frame(path, panel)
            for path, panel in zip(snakemake.input.summaries, snakemake.params.panels)
        ],
        ignore_index=True,
    )
    fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(9.0, 4.6))

    for method in METHODS:
        sub = data[data["method"] == method]
        sizes = sorted(sub["panel"].unique())
        density = [sub[sub["panel"] == p]["heldout_loglik_per_gene"] for p in sizes]
        med = [median_mad(v)[0] for v in density]
        mad = [median_mad(v)[1] for v in density]
        ax_d.errorbar(
            sizes,
            med,
            yerr=mad,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6 if method == "coarse" else 4.5,
            linewidth=2.2 if method == "coarse" else 1.4,
            linestyle=LINES[method],
            capsize=3,
            label=DISPLAY[method],
        )
        fit = [median_mad(sub[sub["panel"] == p]["fit_seconds"])[0] for p in sizes]
        total = [median_mad(sub[sub["panel"] == p]["total_seconds"])[0] for p in sizes]
        ax_t.fill_between(
            sizes, fit, total, color=COLORS[method], alpha=0.15, linewidth=0
        )
        ax_t.plot(
            sizes,
            total,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6 if method == "coarse" else 4.5,
            linewidth=2.2 if method == "coarse" else 1.4,
            linestyle=LINES[method],
            label=DISPLAY[method],
        )
        ax_t.plot(
            sizes,
            fit,
            color=COLORS[method],
            linewidth=1.2,
            linestyle=LINES[method] if method != "coarse" else ":",
        )

    for ax in (ax_d, ax_t):
        ax.set_xscale("log")
        ax.set_xticks([int(p) for p in snakemake.params.panels])
        ax.set_xticklabels([str(int(p)) for p in snakemake.params.panels])
        ax.minorticks_off()
        ax.set_xlabel("genes ($d$)")
    ax_d.set_title("held-out log-likelihood per gene", pad=10)
    ax_t.set_yscale("log")
    ax_t.set_title("seconds per fit", pad=10)

    handles, labels = ax_d.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.4,
        handletextpad=0.4,
    )
    # Reserve the legend band below the axes; tight_layout alone ignores fig.legend.
    fig.tight_layout(rect=(0, 0.26, 0.93, 1))
    ps.finish(fig, snakemake.output.figure)


if __name__ == "__main__":
    main()
