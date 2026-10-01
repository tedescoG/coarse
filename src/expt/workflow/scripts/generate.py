"""Synthetic iSCM dataset for one sweep cell."""
import warnings

warnings.filterwarnings(
    "ignore",
    message="No module named 'rpy2'.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="covariance is not symmetric positive-semidefinite",
    category=RuntimeWarning,
)

import numpy as np

from _common import make_dataset, normalize_intervention_type, parse_targets_per_interv

seed = int(snakemake.wildcards.seed)
density = float(snakemake.wildcards.density)
samp_size = int(snakemake.wildcards.samp_size)
num_nodes = int(snakemake.wildcards.num_nodes)
num_intervs = int(snakemake.wildcards.num_intervs)
graph = snakemake.wildcards["graph"]
targets_size = parse_targets_per_interv(getattr(snakemake.wildcards, "targets_per_interv", None))
noise = getattr(snakemake.wildcards, "noise", "gaussian")
normalize_intervention_type(getattr(snakemake.params, "intervention_type", "soft"))

weights, targets, data_dict = make_dataset(
    graph, num_nodes, num_intervs, density * (num_nodes - 1), seed, samp_size, targets_size,
    noise=noise,
)

# Object array: target sets can be ragged.
np.savez(
    snakemake.output[0],
    weights=weights,
    targets=np.array(targets, dtype=object),
    **data_dict,
)
