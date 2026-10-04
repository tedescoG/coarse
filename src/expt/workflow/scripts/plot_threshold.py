"""threshold_correction figures: ARI against the per-test cutoff with each rule at its median
cutoff (rows n, columns density), and each rule's paired ARI gap to the oracle α."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

import _plot_style as ps

ps.apply_style()

df = pd.read_csv(snakemake.input[0])
KEY = ["density", "samp_size", "seed"]
RULES = ["bonferroni", "by", "bh", "cv", "oracle"]
DISPLAY = dict(zip(RULES, ["Bonferroni", "BY", "BH", "CV", "oracle"]))
COLOR = dict(zip(RULES, ps.hue_palette("rule", RULES)))
MARKER = dict(zip(RULES, ["s", "D", "o", "*", "P"]))
D = sorted(df.density.unique())
N = sorted(df.samp_size.unique())
curve = df[df.rule == "uncorrected"]


def n_label(n):
    k = math.log10(n)
    return f"$n = 10^{{{int(k)}}}$" if k.is_integer() else f"n = {n:,}"


def iqr(s):
    return s.quantile(0.25), s.median(), s.quantile(0.75)


fig, axes = plt.subplots(len(N), len(D), figsize=(6.4 * len(D), 4.8 * len(N)),
                         sharex=True, sharey=True, squeeze=False)
for i, n in enumerate(N):
    for j, d in enumerate(D):
        ax = axes[i, j]
        cell = df[(df.samp_size == n) & (df.density == d)]
        g = cell[cell.rule == "uncorrected"].groupby("cutoff").ari
        ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color="0.88", lw=0)
        ax.plot(g.median().index, g.median(), color="0.3", lw=2.0)
        for rule in RULES:
            r = cell[cell.rule == rule]
            (xlo, x, xhi), (ylo, y, yhi) = iqr(r.cutoff), iqr(r.ari)
            ax.errorbar(x, y, xerr=[[x - xlo], [xhi - x]], yerr=[[y - ylo], [yhi - y]],
                        fmt=MARKER[rule], color=COLOR[rule], ms=16 if rule == "cv" else 10,
                        capsize=3, zorder=3)
        ax.set_xscale("log")
        ps.format_axis(ax.xaxis, "cutoff")
        # Every other decade labelled: nine decades do not fit at this font size.
        ax.xaxis.set_major_locator(LogLocator(base=100, numticks=10))
        ax.set_ylim(0, 1)
        ax.set_xlabel(ps.label("cutoff") if i == len(N) - 1 else "")
        ax.set_ylabel(ps.label("ari") if j == 0 else "")
        if i == 0:
            ax.set_title(f"{ps.label('density')} = {d}")
    axes[i, -1].text(1.04, 0.5, n_label(n), transform=axes[i, -1].transAxes,
                     rotation=270, va="center", ha="left")
handles = [Line2D([], [], color="0.3", lw=2.0, label="uncorrected α")] + [
    Line2D([], [], ls="", marker=MARKER[r], color=COLOR[r], ms=16 if r == "cv" else 10,
           label=DISPLAY[r]) for r in RULES]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
fig.tight_layout()
ps.finish(fig, snakemake.output.curves)

# Paired gap: each rule's ARI minus the oracle ARI on the same dataset.
oracle = df[df.rule == "oracle"][KEY + ["ari"]].rename(columns={"ari": "oracle_ari"})
gap = df[df.rule.isin(RULES[:-1])].merge(oracle, on=KEY)
gap["gap"] = gap.ari - gap.oracle_ari
gap["setting"] = [f"d={d}\nn={n // 1000}k" for d, n in zip(gap.density, gap.samp_size)]
order = [f"d={d}\nn={n // 1000}k" for n in N for d in D]
fig, ax = plt.subplots(figsize=(3.2 * len(order), 6.4))
kw = dict(x="setting", y="gap", hue="rule", order=order, hue_order=RULES[:-1], palette=COLOR, ax=ax)
sns.boxplot(gap, fliersize=0, linewidth=1.0, **kw)
sns.stripplot(gap, dodge=True, size=4, edgecolor="k", linewidth=0.3, legend=False, **kw)
ax.axhline(0, ls="--", color="0.3", lw=1.2, zorder=0)
ax.set_xlabel("")
ax.set_ylabel(r"$\widehat{\mathrm{ARI}} - \mathrm{ARI}$")
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, [DISPLAY[r] for r in RULES[:-1]], ncol=len(RULES) - 1, loc="upper center",
          bbox_to_anchor=(0.5, -0.2), frameon=False)
ps.finish(fig, snakemake.output.gap)
