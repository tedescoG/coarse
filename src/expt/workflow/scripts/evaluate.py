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

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import build_oracle_partition

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


def _is_adj(pa, ch):
    for atom in pa:
        for chatom in ch:
            if true_dag.has_edge(atom, chatom):
                return True
    return False


true_edge_est_partition = nx.create_empty_copy(model.dag)
node_list = list(true_edge_est_partition.nodes)
for idx, pa in enumerate(node_list[:-1]):
    for ch in node_list[idx + 1 :]:
        if _is_adj(pa, ch):
            true_edge_est_partition.add_edge(pa, ch)

true_positive = sum(
    (1 for edge in model.dag.edges if edge in true_edge_est_partition.edges)
)
try:
    precision = true_positive / len(model.dag.edges)
except ZeroDivisionError:
    precision = 1
try:
    recall = true_positive / len(true_edge_est_partition.edges)
except ZeroDivisionError:
    recall = 1
try:
    f_score = 2 * (precision * recall) / (precision + recall)
except ZeroDivisionError:
    f_score = 0

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
