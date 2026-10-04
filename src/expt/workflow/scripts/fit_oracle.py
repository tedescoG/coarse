"""Fit COARSEOracle (plain, or kPC when `k` is set) under the true partition."""

import pickle

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

k_param = getattr(snakemake.params, "k", None)
k = int(k_param) if k_param is not None else None

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)
_, M_true, env_order, partition = build_oracle_partition(
    data["weights"], data["targets"], num_nodes,
)

model = COARSEOracle()
model.fit(
    partition, M_true, env_order, data_dict,
    lambda_pen=lambda_pen, k=k,
)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
