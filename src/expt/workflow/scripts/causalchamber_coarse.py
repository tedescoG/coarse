#!/usr/bin/env python3
"""Run COARSE on the CausalChamber light tunnel: an α grid at fixed λ.

Two cells per intervention regime, mirroring RePaRe's score / oracle rows:

    oracle → plain `COARSE().fit` at every α in `snakemake.params.alphas`
             (λ = `snakemake.params.lambda_pen`, never swept); the row is the
             grid cell that scores best against ground truth
             (`select_oracle_row`: max ARI, F1, precision, then BIC).
    score  → one `COARSECV` fit at the same λ: α̂ is chosen by K-fold held-out
             log-likelihood over the same α grid and the model refit at α̂.
             Ground truth never enters. Reported `fit_time` is the **final
             refit only** (`model.fit_runtime_sec`) — RePaRe reports the
             selected grid cell's single fit, not its search — and the whole
             CV loop goes to `search_runtime_sec`.

Stage-1 test is `snakemake.params.refine_test` (the rule passes
`"gaussian_lrt"`, the mean + variance LRT that RePaRe's `assume="gaussian"`
path runs).

Score convention: COARSE's `model.score` is the Eq. 21 partition BIC summed
over all environments; **higher is better**. Not comparable across methods —
cross-method comparison uses ARI / block-edge precision / recall / F1.
"""

import json
import pickle
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from coarse import COARSE
from coarse.cv import COARSECV
from _common import partition_labels
from _causalchamber_common import (
    all_edge_metrics,
    build_block_data_dict,
    ground_truth_partition,
    labeled_summary,
    params_payload,
    save_dag_plot,
    select_oracle_row,
    select_targets,
)


def _row_from_model(model, alpha, lambda_pen, fit_time, true_labels, true_graph, num_atoms):
    est_labels = partition_labels(model.dag.nodes, num_atoms)
    return {
        "alpha": float(alpha),
        "lambda": float(lambda_pen),
        "second_hp": float(lambda_pen),
        "second_hp_name": "lambda",
        "ari": float(adjusted_rand_score(true_labels, est_labels)),
        "score": float(model.score),
        "fit_time": float(fit_time),
        "num_parts": int(model.dag.number_of_nodes()),
        "num_edges": int(model.dag.number_of_edges()),
        **all_edge_metrics(model.dag, true_graph, native="partition"),
    }


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

    alphas = [float(a) for a in snakemake.params.alphas]
    lambda_pen = float(snakemake.params.lambda_pen)
    n_folds = int(snakemake.params.n_folds)
    refine_test = str(snakemake.params.refine_test)
    mode = snakemake.params.mode

    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    data_dict = build_block_data_dict(blocks, targets)
    num_atoms = len(partition_parts)

    # `true_labels` was computed under the GROUPED ground-truth partition.
    # The ungrouped run intervenes on single variables, which generally induces
    # a finer ground-truth partition — recompute it here so ARI is comparable
    # to RePaRe's ungrouped numbers.
    if mode == "ungrouped":
        with open(snakemake.input.truedagfull, "rb") as f:
            true_dag_full = pickle.load(f)
        _, true_labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    # --- α grid at fixed λ → metrics.csv + oracle row ---------------------------
    records = []
    dags = {}
    for alpha in alphas:
        start = time.perf_counter()
        model = COARSE().fit(
            data_dict, alpha=alpha, lambda_pen=lambda_pen, refine_test=refine_test
        )
        fit_time = time.perf_counter() - start
        records.append(
            _row_from_model(model, alpha, lambda_pen, fit_time, true_labels, true_graph, num_atoms)
        )
        dags[float(alpha)] = model.dag

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    pd.DataFrame(records).to_csv(snakemake.output.metrics_csv, index=False)

    oracle_row = select_oracle_row(records)
    oracle_dag = dags[oracle_row["alpha"]]

    # --- COARSECV at the same λ over the same α grid → score row ----------------
    cv_model = COARSECV().fit(
        data_dict,
        alpha_grid=tuple(alphas),
        n_folds=n_folds,
        lambda_pen=lambda_pen,
        refine_test=refine_test,
    )
    score_row = _row_from_model(
        cv_model, cv_model.best_alpha, lambda_pen, cv_model.fit_runtime_sec,
        true_labels, true_graph, num_atoms,
    )
    score_row["search_runtime_sec"] = float(cv_model.cv_runtime_sec)
    score_dag = cv_model.dag

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    common = {"refine_test": refine_test, "alpha_grid": alphas, "n_folds": n_folds}
    score_parts, score_edges = labeled_summary(score_dag, feature_cols)
    oracle_parts, oracle_edges = labeled_summary(oracle_dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(
                score_row, score_parts, score_edges,
                score_selection="cv",
                cv_log_lik_at_best=float(cv_model.cv_log_lik[cv_model.best_alpha]),
                **common,
            ),
            f, indent=2, default=float,
        )
    with open(snakemake.output.oracle_params, "w") as f:
        json.dump(
            params_payload(oracle_row, oracle_parts, oracle_edges, score_selection="oracle", **common),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
