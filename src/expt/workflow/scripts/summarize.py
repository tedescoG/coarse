import pandas as pd


def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


df = pd.read_csv(snakemake.input[0])
out = df.groupby(list(snakemake.params.by))[list(snakemake.params.metrics)].agg(
    ["mean", "std", "median", q25, q75, "count"]
)
out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
out.reset_index().to_csv(snakemake.output[0], index=False)
