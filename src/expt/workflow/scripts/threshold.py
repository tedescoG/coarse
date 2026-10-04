"""Descendant-test cutoff rules on one synthetic dataset: uncorrected α over a grid,
Bonferroni / BH / BY at level q, COARSECV, and the oracle α (max ARI over the grid, ties to the
smaller α). All rules share one Welch p-value matrix; ARI is that of the induced partition.
`nominal` is the level fixed in advance (NaN for the data-chosen CV and oracle α), `cutoff` the
per-test cutoff applied."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import (
    build_data_dict,
    build_oracle_partition,
    correct,
    normalize_intervention_type,
    partition_labels,
    welch_pvalues,
)
from coarse.cv import COARSECV
from coarse.partition import infer_partition

w = snakemake.wildcards
num_nodes = int(w.num_nodes)
seed = int(w.seed)
alphas = sorted(float(a) for a in snakemake.params.alphas)
q = float(snakemake.params.q)
intervention_type = normalize_intervention_type(snakemake.params.intervention_type)

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)
_, _, _, true_partition = build_oracle_partition(data["weights"], data["targets"], num_nodes)
true_labels = partition_labels(true_partition, num_nodes)


def ari(blocks):
    return adjusted_rand_score(true_labels, partition_labels(blocks, num_nodes))


P = welch_pvalues(data_dict)
rows = [dict(rule="uncorrected", nominal=a, cutoff=a, ari=ari(infer_partition(P < a))) for a in alphas]
oracle = max(rows, key=lambda r: r["ari"])  # first maximum = smallest α
rows.append(dict(oracle, rule="oracle", nominal=np.nan))
for rule in ("bonferroni", "bh", "by"):
    M = correct(P, rule, q)
    rows.append(dict(rule=rule, nominal=q, cutoff=P[M].max() if M.any() else 0.0,
                     ari=ari(infer_partition(M))))
cv = COARSECV(rng=np.random.default_rng(seed)).fit(
    data_dict,
    lambda_pen=float(snakemake.params.lambda_pen),
    refine_test=str(snakemake.params.refine_test),
)
rows.append(dict(rule="cv", nominal=np.nan, cutoff=cv.best_alpha, ari=ari(cv.partition),
                 runtime_sec=cv.cv_runtime_sec))

cell = {
    "density": float(w.density),
    "samp_size": int(w.samp_size),
    "seed": seed,
    "num_nodes": num_nodes,
    "num_intervs": int(w.num_intervs),
    "graph_family": w["graph"],
    "noise": w.noise,
    "targets_per_interv": w.targets_per_interv,
}
out = pd.DataFrame([cell | {"runtime_sec": np.nan} | r for r in rows])
out.to_csv(snakemake.output[0], index=False)
