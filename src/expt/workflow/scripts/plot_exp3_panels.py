"""Per-panel renders of the Experiment-3 scalability grid.

`plot_exp3.py` packs nine panels (3 metrics x 3 methods) into one 3x3 PDF per
(density, pca) cell. That is fine for a quick look but unreadable at thesis
scale. This script emits each panel as its own standalone, full-size PDF for a
single (density, pca) cell, into

    results/kpc/exp3_density=<d>_pca=<pca>/

It reads the already-collected CSV (results/kpc/exp3_results.csv) and refits
nothing. Hue is cast to an ordered categorical for the same reason as in
plot_exp3.py: a bare numeric `hue=` makes seaborn build a continuous colour
legend with fabricated round tick values instead of the true discrete levels.

Run directly:
    uv run python workflow/scripts/plot_exp3_panels.py [density] [pca]
    (defaults: density=0.2, pca=obs)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
# Match the COARSE experiment suite (plot.py): one uniform font size for axis
# labels, tick labels, title AND legend. Do NOT shrink the legend afterwards.
sns.set_context("paper", font_scale=2.3)

CSV = "results/kpc/exp3_results.csv"
METHODS = ["COARSE-oracle", "kPC-k1-oracle", "kPC-k3-oracle"]
TAG = {"COARSE-oracle": "coarse", "kPC-k1-oracle": "kpc-k1", "kPC-k3-oracle": "kpc-k3"}
AXLABEL = {"samp_size": "sample size (n)", "num_nodes": "nodes (d)"}

# (stem, x, y, hue, log_x, log_y, ylim_unit, ylabel)
PANELS = [
    ("fscore_vs_n",  "samp_size", "fscore",      "num_nodes", True, False, True,  "F-score ↑"),
    ("runtime_vs_n", "samp_size", "runtime_sec", "num_nodes", True, True,  False, "runtime (s)"),
    ("runtime_vs_d", "num_nodes", "runtime_sec", "samp_size", True, True,  False, "runtime (s)"),
]


def _render_cell(cell: pd.DataFrame, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem, x, y, hue, log_x, log_y, ylim_unit, ylabel in PANELS:
        for method in METHODS:
            sub = cell[cell["method"] == method].copy()
            sub[hue] = pd.Categorical(
                sub[hue], categories=sorted(sub[hue].unique()), ordered=True
            )
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            sns.lineplot(
                data=sub, x=x, y=y, hue=hue, style=hue,
                markers=True, dashes=True,
                estimator="median", errorbar="ci",
                linewidth=1.6, markersize=6, ax=ax, legend="brief",
            )
            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")
            if ylim_unit:
                ax.set_ylim(0, 1)
            # no per-panel title: the method/metric is stated in the figure caption.
            ax.set_xlabel(AXLABEL[x])
            ax.set_ylabel(ylabel)
            # Uniform font (inherits set_context, no per-element override);
            # loc="best" picks the emptiest corner; a semi-transparent white box
            # means any curve passing under the legend still shows through.
            ax.legend(
                title=AXLABEL[hue], loc="best", ncol=2,
                frameon=True, framealpha=0.6, facecolor="white",
            )
            fig.savefig(out_dir / f"{stem}_{TAG[method]}.pdf",
                        bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)
            n += 1
    return n


def main() -> None:
    density = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    pca = sys.argv[2] if len(sys.argv) > 2 else "obs"

    df = pd.read_csv(CSV)
    coarse = df[df["method"] == "COARSE-oracle"]
    kpc = df[df["method"].str.startswith("kPC")]
    cell = pd.concat(
        [coarse[coarse["density"] == density],
         kpc[(kpc["density"] == density) & (kpc["pca_pooled"] == pca)]],
        ignore_index=True,
    )

    dtag = "%g" % density  # 0.2 -> "0.2", matching the grid's filename convention
    out_dir = Path(f"results/kpc/exp3_density={dtag}_pca={pca}")
    n = _render_cell(cell, out_dir)
    print(f"wrote {n} panels to {out_dir}/")


if __name__ == "__main__":
    main()
