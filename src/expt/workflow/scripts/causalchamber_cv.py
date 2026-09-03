#!/usr/bin/env python3
"""Run COARSE-CV (5-fold α-selection wrapper) on the CausalChamber light tunnel.

Mirrors `causalchamber_coarse.py` but swaps the α × λ grid for a λ-only outer
loop. For each λ in `snakemake.params.lambdas`, a fresh `COARSECV` is fitted
which internally cross-validates α over `alpha_grid` and refits at α̂ on the
full data. The forwarded `model.dag` / `model.score` come from the final refit
(see `coarse/cv.py:353-366`), so ARI / edge-metric computation is identical to
the plain-COARSE path.

`fit_time` is set to `model.cv_runtime_sec` (the total CV wall time including
all K × |A| inner fits + the final refit), matching the convention in
`fit_cv.py:32` — otherwise the runtime column would silently undercount the
work and skew any future cross-method timing plot.

Score / oracle selection across the 4 λ cells uses the same comparators as
`causalchamber_coarse.py:118-121`. Because α is CV-selected inside each cell,
the resulting (α̂, λ) grid is a 1-D sweep over λ rather than a 2-D α × λ grid.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from coarse.cv import COARSECV
from _causalchamber_common import (
    build_data_dict,
    ground_truth_partition,
    labeled_summary,
    partition_edge_metrics,
    partition_labels_from_dag,
    save_dag_plot,
)


def select_targets(mode, group_targets, single_env_labels, name_to_idx):
    if mode == "grouped":
        return {label: set(t) for label, t in group_targets.items()}
    if mode == "ungrouped":
        return {label: {name_to_idx[label]} for label in single_env_labels}
    raise ValueError(f"Unknown mode: {mode!r}")


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.partition, "rb") as f:
        partition_parts = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.truelabels, "rb") as f:
        true_labels = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    lambdas = [float(lam) for lam in snakemake.params.lambdas]
    alpha_grid = tuple(float(a) for a in snakemake.params.alpha_grid)
    n_folds = int(snakemake.params.n_folds)
    mode = snakemake.params.mode

    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    data_dict = build_data_dict(blocks, targets)
    num_atoms = len(partition_parts)

    # Ungrouped mode intervenes on single variables, inducing a finer
    # ground-truth partition than the GROUPED `true_labels`. Recompute so ARI
    # is comparable to the COARSE / RePaRe ungrouped cells.
    if mode == "ungrouped":
        with open(snakemake.input.truedagfull, "rb") as f:
            true_dag_full = pickle.load(f)
        _, true_labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    records = []
    dags = {}
    for lambda_pen in lambdas:
        start = time.perf_counter()
        model = COARSECV().fit(
            data_dict,
            alpha_grid=alpha_grid,
            n_folds=n_folds,
            lambda_pen=lambda_pen,
            refine_test="welch",
        )
        # Outer wall-clock includes the full CV loop (inner fits + refit).
        # `cv_runtime_sec` is the canonical "total work" figure — see
        # `coarse/cv.py:351` and `fit_cv.py:32`.
        fit_time = max(time.perf_counter() - start, float(model.cv_runtime_sec))

        est_labels = partition_labels_from_dag(model.dag, num_atoms)
        ari = adjusted_rand_score(true_labels, est_labels)
        edge_stats = partition_edge_metrics(model.dag, true_graph)

        records.append({
            "alpha": float(model.best_alpha),
            "lambda": float(lambda_pen),
            "ari": float(ari),
            "score": float(model.score),
            "fit_time": float(fit_time),
            "num_parts": int(model.dag.number_of_nodes()),
            "num_edges": int(model.dag.number_of_edges()),
            "n_folds": int(n_folds),
            "cv_log_lik_at_best": float(model.cv_log_lik[model.best_alpha]),
            **edge_stats,
        })
        dags[(float(model.best_alpha), float(lambda_pen))] = model.dag

    df = pd.DataFrame(records)
    df.to_csv(snakemake.output.metrics_csv, index=False)

    # Score-selected: max model.score (Eq. 21 partition BIC, higher is better);
    # tie-break by ARI. Oracle-selected: max ARI, then F1, then precision, then
    # score. Mirrors `causalchamber_coarse.py:118-121`.
    score_row = max(records, key=lambda r: (r["score"], r["ari"]))
    oracle_row = max(
        records, key=lambda r: (r["ari"], r["f1"], r["precision"], r["score"])
    )

    score_dag = dags[(score_row["alpha"], score_row["lambda"])]
    oracle_dag = dags[(oracle_row["alpha"], oracle_row["lambda"])]

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    score_parts, score_edges = labeled_summary(score_dag, feature_cols)
    oracle_parts, oracle_edges = labeled_summary(oracle_dag, feature_cols)

    def _params_payload(row, parts, edges):
        return {
            "alpha": row["alpha"],
            "lambda": row["lambda"],
            "ari": row["ari"],
            "score": row["score"],
            "fit_time": row["fit_time"],
            "num_parts": row["num_parts"],
            "num_edges": row["num_edges"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "k": None,
            "n_folds": row["n_folds"],
            "alpha_grid": list(alpha_grid),
            "cv_log_lik_at_best": row["cv_log_lik_at_best"],
            "parts": parts,
            "edges": edges,
        }

    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            _params_payload(score_row, score_parts, score_edges),
            f, indent=2, default=float,
        )
    with open(snakemake.output.oracle_params, "w") as f:
        json.dump(
            _params_payload(oracle_row, oracle_parts, oracle_edges),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
