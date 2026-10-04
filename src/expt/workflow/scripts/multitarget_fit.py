"""COARSECV partition ARI against Π^E (`ari_E`) and Π^I (`ari_I`); `ari_pop` = ARI(Π^E, Π^I)."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from _common import (
    build_data_dict,
    build_oracle_partition,
    interventional_partition,
    normalize_intervention_type,
    partition_labels,
)
from coarse.cv import COARSECV

w = snakemake.wildcards
num_nodes = int(w.num_nodes)
seed = int(w.seed)
intervention_type = normalize_intervention_type(snakemake.params.intervention_type)

data = np.load(snakemake.input.data, allow_pickle=True)
W, targets = data["weights"], data["targets"]
piE = build_oracle_partition(W, targets, num_nodes)[3]
piI = interventional_partition(W, targets, num_nodes)
cv = COARSECV(rng=np.random.default_rng(seed)).fit(
    build_data_dict(data, targets, intervention_type),
    lambda_pen=float(snakemake.params.lambda_pen),
    refine_test=str(snakemake.params.refine_test),
)
lE, lI, lH = (partition_labels(b, num_nodes) for b in (piE, piI, cv.partition))

row = {
    "density": float(w.density),
    "samp_size": int(w.samp_size),
    "seed": seed,
    "num_nodes": num_nodes,
    "num_intervs": int(w.num_intervs),
    "graph_family": w["graph"],
    "noise": w.noise,
    "k": int(w.targets_per_interv),
    "ari_E": adjusted_rand_score(lE, lH),
    "ari_I": adjusted_rand_score(lI, lH),
    "ari_pop": adjusted_rand_score(lI, lE),
    "nhat": len(cv.partition),
    "nE": len(piE),
    "nI": len(piI),
    "alpha": cv.best_alpha,
    "runtime_sec": cv.cv_runtime_sec,
}
pd.DataFrame([row]).to_csv(snakemake.output[0], index=False)
