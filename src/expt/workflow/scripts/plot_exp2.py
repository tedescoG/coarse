"""Experiment 2 — kPC vs plain COARSE under the oracle partition.

Produces one F-score and one runtime PDF. X = samp_size (log), hue = method
label (COARSE-oracle / kPC-k1-oracle / kPC-k3-oracle).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.0)

df = pd.read_csv(snakemake.input[0])
coarse = df[df["method"] == "COARSE-oracle"]
kpc = df[df["method"].str.startswith("kPC")]


def _plot(sub: pd.DataFrame, metric: str, ylabel: str, log_y: bool, path: str) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=sub,
        x="samp_size",
        y=metric,
        hue="method",
        style="method",
        markers=True,
        dashes=True,
        estimator="median",
        errorbar="ci",
        linewidth=2.0,
        markersize=7,
    )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel(ylabel)
    ax.legend(title="method", loc="best", fontsize="x-small")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


sub = pd.concat([coarse, kpc], ignore_index=True)
_plot(sub, "fscore", "F-score ↑", False, snakemake.output["fscore"])
_plot(sub, "runtime_sec", "runtime (s)", True, snakemake.output["runtime"])
