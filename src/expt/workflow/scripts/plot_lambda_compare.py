import pandas as pd
import seaborn as sns

sns.set_palette("colorblind")
sns.set_context("paper", font_scale=2.3)

df = pd.read_csv(snakemake.input[0])

METHOD_ORDER = ["COARSE", "COARSE-1PC", "COARSE-L2", "COARSE-1PC-L2"]
# Color encodes 1PC vs full-rank; dash encodes lambda; marker encodes 1PC vs full-rank.
palette = {
    "COARSE": "C0",
    "COARSE-L2": "C0",
    "COARSE-1PC": "C1",
    "COARSE-1PC-L2": "C1",
}
dashes = {
    "COARSE": "",
    "COARSE-1PC": "",
    "COARSE-L2": (4, 2),
    "COARSE-1PC-L2": (4, 2),
}
markers = {
    "COARSE": "o",
    "COARSE-L2": "o",
    "COARSE-1PC": "s",
    "COARSE-1PC-L2": "s",
}

g = sns.relplot(
    data=df,
    kind="line",
    x="samp_size",
    y="fscore",
    hue="method",
    style="method",
    hue_order=METHOD_ORDER,
    style_order=METHOD_ORDER,
    palette=palette,
    dashes=dashes,
    markers=markers,
    col="density",
    col_order=[0.2, 0.5, 0.8],
    estimator="median",
    errorbar="ci",
    height=5,
    aspect=1.0,
    facet_kws={"sharey": True},
)
g.set(xscale="log", ylim=(0, 1))
g.set_axis_labels("sample size (n)", "F-score ↑")
g.set_titles("density = {col_name}")
sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    title=None,
    frameon=False,
)
g.savefig(snakemake.output["fscore"], bbox_inches="tight", pad_inches=0.05)
