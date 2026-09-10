"""Evaluate a fitted COARSE model against the ground-truth DAG.

The ground-truth partition is built by stacking the per-intervention
descendant masks column-wise into an M_true matrix and passing it to
`infer_partition` — the same partition machinery the model uses, so any
partition mismatch explains itself as an M-row mismatch rather than as a
different algorithm.

The output CSV schema is a cross-rule contract: `collect.py` and every plot
script read it, and every method in the comparison is evaluated
through this one script, distinguished only by `method_label`.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import block_dag_prf, build_oracle_partition

density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
seed = int(snakemake.wildcards.seed)
num_nodes = int(snakemake.wildcards["num_nodes"])
num_intervs = int(snakemake.wildcards["num_intervs"])
graph_family = snakemake.wildcards["graph"]
model = pickle.load(open(snakemake.input.model, "rb"))
data = np.load(
    snakemake.input.data,
    allow_pickle=True,
)
weights = data["weights"]
targets = data["targets"]

true_dag, _, _, true_partition = build_oracle_partition(weights, targets, num_nodes)

true_labels = np.zeros(num_nodes, dtype=int)
for label, part in enumerate(true_partition):
    true_labels[list(part)] = label
est_labels = np.zeros(len(true_dag))
for label, part in enumerate(model.dag.nodes):
    est_labels[list(part)] = label
ar_index = adjusted_rand_score(true_labels, est_labels)
precision, recall, f_score = block_dag_prf(model.dag, true_dag)

method_label = getattr(snakemake.params, "method_label", "COARSE")
metric_type = getattr(snakemake.params, "metric_type", "partition")
lambda_pen = getattr(snakemake.params, "lambda_pen", np.nan)
# `targets_per_interv` is only present on the multitarget.smk paths; for every
# other rule file the wildcard is absent (those experiments are all
# single-target by construction), so we default to "1" — the true semantic
# value, which makes downstream filtering uniform across rule files.
targets_per_interv = getattr(snakemake.wildcards, "targets_per_interv", "1")


results = {
    "density": density,
    "samp_size": samp_size,
    "seed": seed,
    "num_nodes": num_nodes,
    "num_intervs": num_intervs,
    "graph_family": graph_family,
    "method": method_label,
    "metric_type": metric_type,
    "precision": precision,
    "recall": recall,
    "fscore": f_score,
    "ari": ar_index,
    "runtime_sec": float(getattr(model, "fit_runtime_sec", np.nan)),
    "score": float(getattr(model, "score", np.nan)),
    "lambda_pen": lambda_pen,
    "targets_per_interv": targets_per_interv,
}
pd.DataFrame([results]).to_csv(snakemake.output[0], index=False)
