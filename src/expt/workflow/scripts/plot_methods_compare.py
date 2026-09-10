"""One PDF per (graph, num_nodes, density, metric). Each panel shows 3
method-lines (COARSE-oracle / kPC-k1-oracle / RePaRe-oracle) on a common
samp_size x-axis; display names come from `_plot_style.METHOD_DISPLAY`.

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

import _plot_style as ps

ps.apply_style()

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


df = ps.display_methods(pd.read_csv(snakemake.input[0]))

for out_path in snakemake.output:
    keys = parse_output_path(out_path)
    sub = df[
        (df["graph_family"] == keys["graph"])
        & (df["num_nodes"] == keys["p"])
        & (df["density"] == keys["d"])
    ]
    metric = keys["metric"]

    order = ps.method_order(sub["method"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.lineplot(
        data=sub, x="samp_size", y=metric, hue="method", style="method",
        hue_order=order, style_order=order,
        palette=ps.hue_palette("method", order), markers=ps.hue_markers("method", order),
        dashes=True, estimator="median", errorbar="ci", linewidth=2.0, markersize=8, ax=ax,
    )
    ax.set_xscale("log")
    ps.format_axis(ax.xaxis, "samp_size")
    if metric in ps.LOG_Y:
        ax.set_yscale("log")
        ps.format_axis(ax.yaxis, metric)
    if metric in ps.UNIT_RANGE:
        ax.set_ylim(0, 1)
    ax.set_xlabel(ps.label("samp_size"))
    ax.set_ylabel(ps.label(metric))
    ps.place_legend(ax, ps.label("method"))
    ps.finish(fig, out_path)
