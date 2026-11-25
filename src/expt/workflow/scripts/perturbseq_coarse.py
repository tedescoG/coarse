#!/usr/bin/env python3
"""COARSE α grid and COARSECV on Perturb-seq pseudobulk; the CV search time is
`search_runtime_sec`. Grid rows carry silver (CollecTRI) metrics, reported but never
used for selection: the score row is CV-only."""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from coarse import COARSE
from coarse.cv import COARSECV

from _common import partition_edge_metrics, partition_labels
from _causalchamber_common import (
    build_block_data_dict,
    ground_truth_partition,
    labeled_summary,
    params_payload,
    save_dag_plot,
)
from _perturbseq_common import ps_targets


def _row_from_model(model, alpha, lambda_pen, fit_time, silver_labels, silver_graph):
    est_labels = partition_labels(model.dag.nodes, len(silver_labels))
    return {
        "metric_type": "partition",
        "alpha": float(alpha),
        "lambda": float(lambda_pen),
        "ari": float(adjusted_rand_score(silver_labels, est_labels)),
        "score": float(model.score),
        "fit_time": float(fit_time),
        "num_parts": int(model.dag.number_of_nodes()),
        "num_edges": int(model.dag.number_of_edges()),
        **partition_edge_metrics(model.dag, silver_graph),
    }


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.partition, "rb") as f:
        partition_parts = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        silver_graph = pickle.load(f)
    with open(snakemake.input.truedagfull, "rb") as f:
        true_dag_full = pickle.load(f)
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

    targets = ps_targets(mode, group_targets, single_env_labels, name_to_idx)
    data_dict = build_block_data_dict(blocks, targets)
    # Silver reference partition (CollecTRI regulons): diagnostic only.
    _, silver_labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    records = []
    for alpha in alphas:
        model = COARSE().fit(
            data_dict, alpha=alpha, lambda_pen=lambda_pen, refine_test=refine_test
        )
        records.append(
            _row_from_model(
                model, alpha, lambda_pen, model.fit_runtime_sec, silver_labels, silver_graph
            )
        )

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    pd.DataFrame(records).to_csv(snakemake.output.metrics_csv, index=False)

    cv_model = COARSECV().fit(
        data_dict,
        alpha_grid=tuple(alphas),
        n_folds=n_folds,
        lambda_pen=lambda_pen,
        refine_test=refine_test,
    )
    score_row = _row_from_model(
        cv_model,
        cv_model.best_alpha,
        lambda_pen,
        cv_model.fit_runtime_sec,
        silver_labels,
        silver_graph,
    )
    score_row["search_runtime_sec"] = float(cv_model.cv_runtime_sec)
    score_dag = cv_model.dag

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))

    score_parts, score_edges = labeled_summary(score_dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(
                score_row,
                score_parts,
                score_edges,
                score_selection="cv",
                refine_test=refine_test,
                alpha_grid=alphas,
                n_folds=n_folds,
                env_labels=sorted(targets),
                cv_log_lik_at_best=float(cv_model.cv_log_lik[cv_model.best_alpha]),
            ),
            f,
            indent=2,
            default=float,
        )


if __name__ == "__main__":
    main()
