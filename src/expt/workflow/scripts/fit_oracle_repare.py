"""Fit RePaRe under the oracle partition.

Sibling of `fit_oracle.py` (which handles COARSE / kPC oracle fits). Bypasses
RePaRe's Phase 1 (per-feature intervention test + `_get_totally_ordered_partition`
over noisy p-values) by feeding it the noise-free partition masks read directly
from M_true. Phase 2 (the `while self.refinable: self._recurse()` loop with
Wilks' Lambda CCA via `_is_adj`) then runs unchanged — that's the half of
RePaRe we're benchmarking.

Output: pickled `PartitionDagModelIvn` whose `.dag`, `.fit_runtime_sec`, and
`.score` are populated, matching the shape `evaluate.py` consumes.
"""

import pickle
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np

# RePaRe is a sibling subproject inside the thesis bundle (…/THESIS/coarse/),
# not part of the COARSE Python package. parents[5] is the bundle root.
THESIS_BUNDLE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(THESIS_BUNDLE / "repare-0.2.0" / "src"))

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

# Construct the oracle partition the way RePaRe would, using M_true columns as
# noise-free intervention masks. `_get_totally_ordered_partition` returns a
# `deque[set[int]]` in topological order — exactly what `_refine` consumes.
partition_masks = {
    env_key: M_true[:, env_idx].astype(bool)
    for env_idx, env_key in enumerate(env_order)
}
oracle_partition = _get_totally_ordered_partition(partition_masks)

# Replay PartitionDagModelIvn.fit() up to L218 of repare.py, but inject the
# oracle partition deque directly instead of running the per-feature p_val loop.
model = PartitionDagModelIvn()
model.assume = "gaussian"
model.refine_test = "ttest"  # unused under assume='gaussian'; set for safety
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

# Initialise the refinement loop (repare.py L218-222).
model.dag = nx.DiGraph()
model.dag.add_node(tuple(range(obs_array.shape[1])))
model.refinable = deque([set(range(obs_array.shape[1]))])
model.edge_tests = []
model.edge_score = 0.0

start = time.perf_counter()
while len(model.refinable) > 0:
    model._recurse()
fit_runtime_sec = time.perf_counter() - start

# Pickle a minimal namespace, not the PartitionDagModelIvn itself: evaluate.py
# only reads .dag / .fit_runtime_sec / .score, and pickling the full RePaRe
# instance would force evaluate.py to have `repare` on sys.path to unpickle.
# SimpleNamespace + nx.DiGraph both live in stdlib / networkx, so the pickle is
# decoupled from RePaRe's class hierarchy.
result = SimpleNamespace(
    dag=model.dag,
    fit_runtime_sec=fit_runtime_sec,
    score=float("nan"),
)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(result, f)
