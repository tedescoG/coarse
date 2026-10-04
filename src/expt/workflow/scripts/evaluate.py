# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""ARI and block-edge P/R/F1 against the truth; shared by every method."""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import build_oracle_partition, partition_edge_metrics, partition_labels

density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
num_nodes = int(snakemake.wildcards["num_nodes"])
num_intervs = int(snakemake.wildcards["num_intervs"])
graph_family = snakemake.wildcards["graph"]
# Absent on oracle paths (single-target; noise only in noise.smk).
noise = getattr(snakemake.wildcards, "noise", "gaussian")
targets_per_interv = getattr(snakemake.wildcards, "targets_per_interv", "1")
method_label = str(snakemake.params.method_label)
lambda_pen = float(snakemake.params.lambda_pen)

with open(snakemake.input.model, "rb") as f:
    model = pickle.load(f)
data = np.load(snakemake.input.data, allow_pickle=True)
weights = data["weights"]
targets = data["targets"]

true_dag, _, _, true_partition = build_oracle_partition(weights, targets, num_nodes)
true_labels = partition_labels(true_partition, num_nodes)
est_labels = partition_labels(model.dag.nodes, num_nodes)
ar_index = adjusted_rand_score(true_labels, est_labels)
edges = partition_edge_metrics(model.dag, true_dag)

results = {
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "num_nodes": num_nodes,
    "num_intervs": num_intervs,
    "graph_family": graph_family,
    "method": method_label,
    "precision": edges["precision"],
    "recall": edges["recall"],
    "fscore": edges["f1"],
    "ari": ar_index,
    "runtime_sec": float(getattr(model, "fit_runtime_sec", np.nan)),
    "score": float(getattr(model, "score", np.nan)),
    "lambda_pen": lambda_pen,
    "targets_per_interv": targets_per_interv,
    "noise": noise,
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
