"""Synthetic dataset for one sweep cell: an iSCM (Ormaniec et al. 2024) sampled through
sempler. All of the DGP lives in ``_common.make_dataset`` so the diagnostics sample exactly
the same model; this script only maps Snakemake wildcards onto it and writes the ``.npz``.
"""
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
# Every DATA path carries a noise segment; the default only guards ad-hoc invocations.
noise = getattr(snakemake.wildcards, "noise", "gaussian")
# COARSE only supports soft (shift) interventions; anything else is a misconfiguration.
normalize_intervention_type(getattr(snakemake.params, "intervention_type", "soft"))

weights, targets, data_dict = make_dataset(
    graph, num_nodes, num_intervs, density * (num_nodes - 1), seed, samp_size, targets_size,
    noise=noise,
)

# Heterogeneous target-size sweeps (e.g. ``targets_per_interv="1to5"``) produce
# a ragged ``targets`` list — np.savez would error trying to stack them into
# a homogeneous 2D array. Wrap in an explicit object array so each entry stays
# its own variable-length sequence; downstream code only ever iterates, so the
# change is invisible to consumers (and a no-op for the fixed-size case).
np.savez(
    snakemake.output[0],
    weights=weights,
    targets=np.array(targets, dtype=object),
    **data_dict,
)
