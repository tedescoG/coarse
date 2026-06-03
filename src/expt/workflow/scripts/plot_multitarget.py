"""Multi-target experiment plots — COARSE vs COARSE-CV vs COARSE-1PC.

Two figure families share one snakemake rule (`plot_multitarget`):

  1. **Per-(graph, p, density, tpi) line plot**
     Filename:  ``{graph}_p={p}_dens={d}_tpi={tpi}_{metric}.pdf``
     x: sample size (log)        y: metric        hue: method
     One panel per output PDF. Mirrors `plot_methods_compare.py`'s template;
     the only structural difference is the added `tpi=` segment.

  2. **Headline summary**
     Filename:  ``summary_{metric}.pdf``
     x: targets_per_interv (ordered categorical)     y: metric
     hue: method            facets: (p × density)
     This is the figure that answers "does the method degrade gracefully as
     the intervention support grows?".

Path parsing rather than wildcards-on-the-script avoids defining new
snakemake parameters per output. The script keys off filenames directly,
exactly as `plot_methods_compare.py` does for its own outputs.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.5)

METHOD_ORDER = ["COARSE", "COARSE-CV", "COARSE-1PC"]
PALETTE = {"COARSE": "C0", "COARSE-CV": "C1", "COARSE-1PC": "C2"}
MARKERS = {"COARSE": "o", "COARSE-CV": "D", "COARSE-1PC": "s"}

# Ordered categorical for the summary plot — "1to5" goes last so the reader
# sees fixed sizes 2, 3, 5 progress monotonically, then the heterogeneous
# regime as the right-most contrast.
TPI_ORDER = ["2", "3", "5", "1to5"]
TPI_LABEL = {"2": "2", "3": "3", "5": "5", "1to5": "1–5 (mix)"}

METRIC_LABEL = {
    "fscore": "F-score ↑",
    "ari": "Adjusted Rand Index ↑",
    "precision": "precision ↑",
    "recall": "recall ↑",
    "runtime_sec": "run time (s)",
}
LOG_Y_METRICS = {"runtime_sec"}
UNIT_RANGE_METRICS = {"fscore", "ari", "precision", "recall"}

LINE_RE = re.compile(
    r"^(?P<graph>er|sf)_p=(?P<p>\d+)_dens=(?P<d>[\d.]+)"
    r"_tpi=(?P<tpi>\d+(?:to\d+)?)_(?P<metric>[a-z_]+)\.pdf$"
)
SUMMARY_RE = re.compile(r"^summary_(?P<metric>[a-z_]+)\.pdf$")


def _save(fig, out_path):
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _render_line(df: pd.DataFrame, out_path: str) -> None:
    """Per-(p, density, tpi) line plot: x=samp_size, hue=method."""
    keys = LINE_RE.match(Path(out_path).name).groupdict()
    sub = df[
        (df["graph_family"] == keys["graph"])
        & (df["num_nodes"] == int(keys["p"]))
        & (df["density"] == float(keys["d"]))
        & (df["targets_per_interv"] == keys["tpi"])
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
    ax.set_title(
        f"{keys['graph']}  p={keys['p']}  density={keys['d']}  "
        f"targets/intervention = {TPI_LABEL[keys['tpi']]}"
    )
    ax.legend(title=None, loc="best", frameon=False)
    fig.tight_layout()
    _save(fig, out_path)


def _render_summary(df: pd.DataFrame, out_path: str) -> None:
    """Headline figure: x=tpi (categorical, ordered), hue=method,
    facets = (p × density). One PDF per metric.

    Uses `sns.catplot(kind='point')` so we get medians + bootstrap CIs over
    seeds + samp_sizes within each (p, density, tpi, method) cell. This
    answers the question 'does the median performance drop as targets grow?'
    rather than 'how does each individual sample size behave?' (which the
    line plots already cover)."""
    metric = SUMMARY_RE.match(Path(out_path).name).group("metric")
    sub = df[df["method"].isin(METHOD_ORDER)].copy()
    sub["tpi_label"] = sub["targets_per_interv"].map(TPI_LABEL)

    g = sns.catplot(
        data=sub,
        x="tpi_label",
        order=[TPI_LABEL[t] for t in TPI_ORDER],
        y=metric,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=PALETTE,
        markers=[MARKERS[m] for m in METHOD_ORDER],
        kind="point",
        estimator="median",
        errorbar="ci",
        dodge=0.3,
        linestyles="-",
        col="num_nodes",
        row="density",
        sharex=True,
        sharey=True,
        height=3.0,
        aspect=1.2,
        legend_out=True,
    )
    g.set_axis_labels("targets per intervention", METRIC_LABEL.get(metric, metric))
    g.set_titles(col_template="p = {col_name}", row_template="density = {row_name}")
    if metric in UNIT_RANGE_METRICS:
        for ax in g.axes.flat:
            ax.set_ylim(0, 1)
    if metric in LOG_Y_METRICS:
        for ax in g.axes.flat:
            ax.set_yscale("log")
    g.fig.suptitle(
        f"Multi-target intervention recovery — {METRIC_LABEL.get(metric, metric)}",
        y=1.02,
    )
    _save(g.fig, out_path)


df = pd.read_csv(snakemake.input[0])
# `targets_per_interv` may not exist in the schema yet if the CSV was created
# by an older evaluate.py. The dataset path already carries it as a wildcard,
# so it's available on the snakemake.wildcards object during evaluate.py —
# but evaluate.py wasn't modified, so it's the wildcard handling at *this*
# stage that owns this column. Recover it from the file path stored in the
# input by re-reading the CSV through collect.py — easiest path: re-derive
# tpi from the metrics CSV's wildcard-bearing path. But because collect.py
# concatenates without preserving paths, we instead require evaluate.py to
# pick the tpi off snakemake.wildcards. If absent here, fall back to a
# parse: the dataframe's index alone cannot reconstruct it.
if "targets_per_interv" not in df.columns:
    raise RuntimeError(
        "results.csv is missing `targets_per_interv` column. "
        "evaluate.py must be updated to log this wildcard, or the rule "
        "file must inject it via a params block."
    )

for out_path in snakemake.output:
    name = Path(out_path).name
    if LINE_RE.match(name):
        _render_line(df, out_path)
    elif SUMMARY_RE.match(name):
        _render_summary(df, out_path)
    else:
        raise ValueError(f"Cannot route output path: {out_path}")
