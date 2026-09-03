"""Fit COARSE under the oracle partition. Handles both the no-PCA baseline
and the kPC variant; the k wildcard is optional, so this single script
serves `rule fit_oracle` (kpc.smk, lambda.smk) and `rule fit_kpc_oracle` /
`rule fit_lambda_kpc_oracle` (which carry k in their wildcard path).
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

# Optional kPC wildcard — present for rules under kpc_oracle_path /
# lambda_kpc_oracle_path, absent for the non-PCA oracle path. snakemake.wildcards
# raises AttributeError on missing keys, hence the getattr default.
k_raw = getattr(snakemake.wildcards, "k", None)
k = int(k_raw) if k_raw is not None else None

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)
_, M_true, env_order, partition = build_oracle_partition(
    data["weights"], data["targets"], num_nodes,
)

model = COARSEOracle()
start = time.perf_counter()
model.fit(
    partition, M_true, env_order, data_dict,
    lambda_pen=lambda_pen, k=k,
)
model.fit_runtime_sec = time.perf_counter() - start

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
