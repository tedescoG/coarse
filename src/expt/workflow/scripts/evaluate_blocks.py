"""Per-block parent recovery; `F` / `F_method`: family size in variables / in the fit's units."""

import pickle

import numpy as np
import pandas as pd

from _common import block_parents, build_oracle_partition, family_size

num_nodes = int(snakemake.wildcards.num_nodes)

with open(snakemake.input.model, "rb") as f:
    model = pickle.load(f)
data = np.load(snakemake.input.data, allow_pickle=True)
true_dag, _, _, _ = build_oracle_partition(data["weights"], data["targets"], num_nodes)

partition = model.partition
index = {block: j for j, block in enumerate(partition)}
parents = block_parents(true_dag, partition)
k = 1 if model.fit_metadata["avg"] else model.fit_metadata["k"]

cell = {
    "density": float(snakemake.wildcards.density),
    "samp_size": int(snakemake.wildcards.samp_size),
    "seed": int(snakemake.wildcards.seed),
    "num_nodes": num_nodes,
    "num_intervs": int(snakemake.wildcards.num_intervs),
    "method": str(snakemake.params.method_label),
}
columns = [*cell, "block", "F", "F_method", "recall", "exact"]
rows = []
for j, block in enumerate(partition):
    if not parents[j]:
        continue
    est = {index[b] for b in model.parent_sets[block]}
    rows.append({
        **cell,
        "block": j,
        "F": family_size(partition, parents, j),
        "F_method": family_size(partition, parents, j, k),
        "recall": len(est & parents[j]) / len(parents[j]),
        "exact": float(est == parents[j]),
    })
# Header only if no block has a parent, so collect.py still reads it.
pd.DataFrame(rows, columns=columns).to_csv(snakemake.output[0], index=False)
