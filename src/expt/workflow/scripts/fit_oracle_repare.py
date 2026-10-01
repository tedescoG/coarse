# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""Fit RePaRe under the true partition: replays `PartitionDagModelIvn.fit` with the partition
step replaced by the noise-free masks of M_true."""

import pickle
import time
from collections import deque
from types import SimpleNamespace

import networkx as nx
import numpy as np

from repare.repare import PartitionDagModelIvn, _get_totally_ordered_partition

from _common import (
    build_data_dict,
    build_oracle_partition,
    normalize_intervention_type,
)

intervention_type = normalize_intervention_type(
    getattr(snakemake.params, "intervention_type", "soft")
)
beta = float(getattr(snakemake.params, "beta", 1e-4))
num_nodes = int(snakemake.wildcards.num_nodes)

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)
_, M_true, env_order, _ = build_oracle_partition(
    data["weights"], data["targets"], num_nodes,
)

partition_masks = {
    env_key: M_true[:, env_idx].astype(bool)
    for env_idx, env_key in enumerate(env_order)
}
oracle_partition = _get_totally_ordered_partition(partition_masks)

model = PartitionDagModelIvn()
model.assume = "gaussian"
model.refine_test = "ttest"  # unused under assume="gaussian"
model.beta = beta

obs_array = data_dict["obs"][0]
model.obs = obs_array
model.obs_type = "obs"
model.data_dict = {k: v[0] for k, v in data_dict.items() if k != "obs"}
model.env_targets = {
    k: set(int(t) for t in v[1]) for k, v in data_dict.items() if k != "obs"
}
model.env_types = {k: v[2] for k, v in data_dict.items() if k != "obs"}
model.partition = oracle_partition

model.dag = nx.DiGraph()
model.dag.add_node(tuple(range(obs_array.shape[1])))
model.refinable = deque([set(range(obs_array.shape[1]))])
model.edge_tests = []
model.edge_score = 0.0

start = time.perf_counter()
while len(model.refinable) > 0:
    model._recurse()
fit_runtime_sec = time.perf_counter() - start

# Only what evaluate.py reads.
result = SimpleNamespace(
    dag=model.dag,
    fit_runtime_sec=fit_runtime_sec,
    score=float("nan"),
)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(result, f)
