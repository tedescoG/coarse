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

import pandas as pd
import seaborn as sns

import _plot_style as ps

ps.apply_style()

# Ordered categorical for the summary plot — "1to5" goes last so the reader sees fixed
# sizes 2, 3, 5 progress monotonically, then the heterogeneous regime as the contrast.
TPI_ORDER = ["2", "3", "5", "1to5"]
TPI_LABEL = {"2": "2", "3": "3", "5": "5", "1to5": "1–5 (mix)"}

LINE_RE = re.compile(
    r"^(?P<graph>er|sf)_p=(?P<p>\d+)_dens=(?P<d>[\d.]+)"
    r"_tpi=(?P<tpi>\d+(?:to\d+)?)_(?P<metric>[a-z_]+)\.pdf$"
)
SUMMARY_RE = re.compile(r"^summary_(?P<metric>[a-z_]+)\.pdf$")


def _render_line(df: pd.DataFrame, out_path: str) -> None:
    """Per-(p, density, tpi) line plot: x=samp_size, hue=method."""
    keys = LINE_RE.match(Path(out_path).name).groupdict()
    sub = df[
        (df["graph_family"] == keys["graph"])
        & (df["num_nodes"] == int(keys["p"]))
        & (df["density"] == float(keys["d"]))
        & (df["targets_per_interv"] == keys["tpi"])
    ]
    ps.line_panel(sub, x="samp_size", y=keys["metric"], hue="method", out_path=out_path)


def _render_summary(df: pd.DataFrame, out_path: str) -> None:
    """Headline figure: x=tpi (categorical, ordered), hue=method,
    facets = (p × density). One PDF per metric.

    Uses `sns.catplot(kind='point')` so we get medians + bootstrap CIs over
    seeds + samp_sizes within each (p, density, tpi, method) cell. This
    answers the question 'does the median performance drop as targets grow?'
    rather than 'how does each individual sample size behave?' (which the
    line plots already cover)."""
    metric = SUMMARY_RE.match(Path(out_path).name).group("metric")
    sub = df.copy()
    sub["tpi_label"] = sub["targets_per_interv"].map(TPI_LABEL)

    order = ps.method_order(sub["method"].unique())
    g = sns.catplot(
        data=sub,
        x="tpi_label",
        order=[TPI_LABEL[t] for t in TPI_ORDER],
        y=metric,
        hue="method",
        hue_order=order,
        palette=ps.hue_palette("method", order),
        markers=[ps.METHOD_MARKER[m] for m in order],
        kind="point",
        estimator="median",
        errorbar="ci",
        dodge=0.3,
        linestyles="-",
        col="num_nodes",
        row="density",
        sharex=True,
        sharey=True,
        height=4.0,
        aspect=1.2,
        legend_out=False,
    )
    g.set_axis_labels(ps.label("targets_per_interv"), ps.label(metric))
    g.set_titles(col_template="p = {col_name}", row_template="density = {row_name}")
    if metric in ps.UNIT_RANGE:
        for ax in g.axes.flat:
            ax.set_ylim(0, 1)
    if metric in ps.LOG_Y:
        for ax in g.axes.flat:
            ax.set_yscale("log")
            ps.format_axis(ax.yaxis, metric)
    ps.place_legend(g, ps.label("method"))
    ps.finish(g.figure, out_path)


# `targets_per_interv` is a string column ("2", "1to5"); without the dtype a numeric-only
# CSV would parse it as int and the string equality filter below would match nothing.
df = pd.read_csv(snakemake.input[0], dtype={"targets_per_interv": str})
df = ps.display_methods(df)

for out_path in snakemake.output:
    name = Path(out_path).name
    if LINE_RE.match(name):
        _render_line(df, out_path)
    elif SUMMARY_RE.match(name):
        _render_summary(df, out_path)
    else:
        raise ValueError(f"Cannot route output path: {out_path}")
