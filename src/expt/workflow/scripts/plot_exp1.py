"""Experiment 1 — Pooled vs Observational PCA (oracle partition).

One PDF per (density, metric) — 4 PDFs total. Each plot shows 4 lines:
k ∈ {1, 3} × pca ∈ {obs, pooled}. X = samp_size (log), hue = k,
style = pca_pooled.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.0)

df = pd.read_csv(snakemake.input[0])
df["k"] = df["method"].str.extract(r"kPC-k(\d+)-oracle").astype(int)


def _plot(sub: pd.DataFrame, metric: str, ylabel: str, log_y: bool, path: str) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=sub,
        x="samp_size",
        y=metric,
        hue="k",
        style="pca_pooled",
        markers=True,
        dashes=True,
        estimator="median",
        errorbar="ci",
        linewidth=2.0,
        markersize=8,
    )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, 1)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel(ylabel)
    ax.legend(title="k / pca", loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


for density, fkey, rkey in [
    (0.2, "d02_fscore", "d02_runtime"),
    (0.6, "d06_fscore", "d06_runtime"),
]:
    sub = df[df["density"] == density]
    _plot(sub, "fscore", "F-score ↑", False, snakemake.output[fkey])
    _plot(sub, "runtime_sec", "runtime (s)", True, snakemake.output[rkey])