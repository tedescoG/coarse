# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
import pandas as pd

dfs = [pd.read_csv(f) for f in snakemake.input]
summary = pd.concat(dfs, ignore_index=True)
summary.to_csv(snakemake.output[0], index=False)
