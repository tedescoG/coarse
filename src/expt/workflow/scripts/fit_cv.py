"""Fit COARSECV (k-fold α-selection wrapper around COARSE) on a synthetic dataset.

Mirrors `fit.py` but swaps `COARSE().fit(...)` for `cv_coarse(...)`. The CV
defaults from `coarse.cv` (alpha_grid=DEFAULT_ALPHA_GRID, n_folds=5) are used —
the outer Snakemake sweep varies data-generation parameters only.

After the CV run we overwrite `fit_runtime_sec` with `cv_runtime_sec` so the
unmodified `evaluate.py` reports total CV wall time rather than just the
post-CV refit (otherwise the runtime column would silently undercount the
work and skew the scalability plot against the fixed-α baseline).
"""

import pickle

import numpy as np

from _common import build_data_dict, normalize_intervention_type
from coarse.cv import cv_coarse

lambda_pen = float(snakemake.params.lambda_pen)
refine_param = getattr(snakemake.params, "refine_test", "welch")
refine_test = ("welch" if refine_param is None else str(refine_param)).lower()
intervention_type = normalize_intervention_type(
    getattr(snakemake.params, "intervention_type", "soft")
)

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)

model = cv_coarse(data_dict, lambda_pen=lambda_pen, refine_test=refine_test)
model.fit_runtime_sec = model.cv_runtime_sec

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
