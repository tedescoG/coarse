"""One PDF per (graph, num_nodes, density, metric). Each panel shows 3
method-lines (COARSE, COARSE-1PC, RePaRe) on a common samp_size x-axis.

Output filenames follow the template
  results/methods_compare/{graph}_p={p}_dens={d}_{metric}.pdf
declared in rules/methods_compare.smk — we parse each path to recover the
filter keys, so the script depends on the smk template by convention rather
than wiring.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
# Match the COARSE experiment suite (plot.py): one uniform font size for axis
# labels, tick labels, title AND legend.
sns.set_context("paper", font_scale=2.3)

METHOD_ORDER = ["COARSE", "COARSE-1PC", "RePaRe"]
PALETTE = {"COARSE": "C0", "COARSE-1PC": "C1", "RePaRe": "C2"}
MARKERS = {"COARSE": "o", "COARSE-1PC": "s", "RePaRe": "^"}

METRIC_LABEL = {
    "fscore": "F-score ↑",
    "precision": "precision ↑",
    "recall": "recall ↑",
    "runtime_sec": "run time (s)",
}
LOG_Y_METRICS = {"runtime_sec"}
UNIT_RANGE_METRICS = {"fscore", "precision", "recall"}

PATH_RE = re.compile(
    r"^(?P<graph>er|sf)_p=(?P<p>\d+)_dens=(?P<d>[\d.]+)_(?P<metric>[a-z_]+)\.pdf$"
)


def parse_output_path(path: str) -> dict:
    name = Path(path).name
    match = PATH_RE.match(name)
    if match is None:
        raise ValueError(f"Cannot parse plot output path: {path}")
    return {
        "graph": match["graph"],
        "p": int(match["p"]),
        "d": float(match["d"]),
        "metric": match["metric"],
    }


df = pd.read_csv(snakemake.input[0])

for out_path in snakemake.output:
    keys = parse_output_path(out_path)
    sub = df[
        (df["graph_family"] == keys["graph"])
        & (df["num_nodes"] == keys["p"])
        & (df["density"] == keys["d"])
        & (df["method"].isin(METHOD_ORDER))
    ]
    metric = keys["metric"]

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    sns.lineplot(
        data=sub,
        x="samp_size",
        y=metric,
        hue="method",
        style="method",
        hue_order=METHOD_ORDER,
        style_order=METHOD_ORDER,
        palette=PALETTE,
        markers=MARKERS,
        dashes=False,
        estimator="median",
        errorbar="ci",
        linewidth=2.0,
        markersize=8,
        ax=ax,
    )
    ax.set_xscale("log")
    if metric in LOG_Y_METRICS:
        ax.set_yscale("log")
    if metric in UNIT_RANGE_METRICS:
        ax.set_ylim(0, 1)
    ax.set_xlabel("sample size (n)")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    # no per-panel title: the graph/p/density are stated in the figure caption.
    # loc="best" picks the emptiest corner; a semi-transparent white box means
    # any curve passing under the legend still shows through.
    ax.legend(title=None, loc="best", frameon=True, framealpha=0.6,
              facecolor="white")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
