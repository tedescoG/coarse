"""Evaluate a fitted model against the ground-truth DAG.

The ground-truth partition is built by stacking the per-intervention descendant masks
column-wise into an M_true matrix and passing it to `infer_partition` — the same partition
machinery the model uses, so any partition mismatch explains itself as an M-row mismatch
rather than as a different algorithm.

The output CSV schema is a cross-rule contract: `collect.py` and every plot script read it,
and every method is evaluated through this one script, distinguished only by the required
`method_label` param. `lambda_pen` is likewise required so the column is never NaN.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import block_dag_prf, build_oracle_partition, partition_labels

density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
num_nodes = int(snakemake.wildcards["num_nodes"])
num_intervs = int(snakemake.wildcards["num_intervs"])
graph_family = snakemake.wildcards["graph"]
# Both wildcards are absent on the oracle paths, which are Gaussian and single-target by
# construction; the defaults are the true semantic values.
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
precision, recall, f_score = block_dag_prf(model.dag, true_dag)

results = {
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "num_nodes": num_nodes,
    "num_intervs": num_intervs,
    "graph_family": graph_family,
    "method": method_label,
    "precision": precision,
    "recall": recall,
    "fscore": f_score,
    "ari": ar_index,
    "runtime_sec": float(getattr(model, "fit_runtime_sec", np.nan)),
    "score": float(getattr(model, "score", np.nan)),
    "lambda_pen": lambda_pen,
    "targets_per_interv": targets_per_interv,
    "noise": noise,
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
