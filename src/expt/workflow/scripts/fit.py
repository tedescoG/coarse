"""Fit COARSE on a synthetic dataset produced by `generate.py`.

Mirrors RePaRe's `fit.py` API surface so the rules / wildcards stay aligned;
the only differences are dropping `beta` and `assume` (which COARSE doesn't
use) and adding `lambda_pen` (the BIC penalty coefficient).
"""

import pickle
import time

import numpy as np

from _common import build_data_dict, normalize_intervention_type
from coarse.coarse import COARSE

alpha = float(snakemake.params.alpha)
lambda_pen = float(snakemake.params.lambda_pen)
refine_param = getattr(snakemake.params, "refine_test", "welch")
refine_test = ("welch" if refine_param is None else str(refine_param)).lower()
intervention_type = normalize_intervention_type(
    getattr(snakemake.params, "intervention_type", "soft")
)
# Optional kPC params — present for the 1PC suite, absent for normal COARSE.
# Same getattr-with-default shape as fit_oracle.py:28-31.
k_param = getattr(snakemake.params, "k", None)
k = int(k_param) if k_param is not None else None
pca_pooled = bool(getattr(snakemake.params, "pca_pooled", False))

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)

model = COARSE()
start = time.perf_counter()
model.fit(
    data_dict,
    alpha=alpha,
    lambda_pen=lambda_pen,
    refine_test=refine_test,
    k=k,
    pca_pooled=pca_pooled,
)
model.fit_runtime_sec = time.perf_counter() - start

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
