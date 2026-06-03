"""Fit COARSE under the oracle partition. Handles both the no-PCA baseline
and the kPC variant; the k/pca wildcards are optional, so this single script
serves `rule fit_oracle` (kpc.smk, lambda.smk) and `rule fit_kpc_oracle` /
`rule fit_lambda_kpc_oracle` (which carry k and pca in their wildcard path).
"""

import pickle
import time

import numpy as np

from _common import (
    build_data_dict,
    build_oracle_partition,
    normalize_intervention_type,
)
from coarse.coarse import COARSEOracle

lambda_pen = float(snakemake.params.lambda_pen)
intervention_type = normalize_intervention_type(
    getattr(snakemake.params, "intervention_type", "soft")
)
num_nodes = int(snakemake.wildcards.num_nodes)

# Optional kPC wildcards — present for rules under kpc_oracle_path /
# lambda_kpc_oracle_path, absent for the non-PCA oracle path. snakemake.wildcards
# raises AttributeError on missing keys, hence the getattr default.
k_raw = getattr(snakemake.wildcards, "k", None)
pca_raw = getattr(snakemake.wildcards, "pca", None)
k = int(k_raw) if k_raw is not None else None
pca_pooled = (str(pca_raw).lower() == "pooled") if pca_raw is not None else False

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)
_, M_true, env_order, partition = build_oracle_partition(
    data["weights"], data["targets"], num_nodes,
)

model = COARSEOracle()
start = time.perf_counter()
model.fit(
    partition, M_true, env_order, data_dict,
    lambda_pen=lambda_pen, k=k, pca_pooled=pca_pooled,
)
model.fit_runtime_sec = time.perf_counter() - start

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
