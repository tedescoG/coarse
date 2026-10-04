"""Fit COARSE on one synthetic dataset and pickle the model."""

import pickle

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
k_param = getattr(snakemake.params, "k", None)
k = int(k_param) if k_param is not None else None

data = np.load(snakemake.input.data, allow_pickle=True)
data_dict = build_data_dict(data, data["targets"], intervention_type)

model = COARSE()
model.fit(
    data_dict,
    alpha=alpha,
    lambda_pen=lambda_pen,
    refine_test=refine_test,
    k=k,
)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(model, f)
