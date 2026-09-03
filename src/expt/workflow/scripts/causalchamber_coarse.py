#!/usr/bin/env python3
"""Run a COARSE (or COARSE-1PC) α × λ grid on the CausalChamber light tunnel.

Score convention: COARSE's `model.score` is the Eq. 21 partition BIC summed
over all environments; **higher is better** (README "BIC is maximized"). The
data-driven "score" selection therefore uses `max(...)` — opposite of
RePaRe's `min(...)` convention, which negates GnIES BIC. The two scores are
NOT comparable across methods; cross-method comparison happens via
ARI / partition-edge precision / recall / F1, never via the `score` column.

COARSE vs COARSE-1PC is controlled entirely by `snakemake.params.k`:
    k = None  → plain COARSE (no PCA on parent blocks)
    k = 1     → COARSE-1PC (1-component PCA per parent block)
"""

import json
import pickle
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from coarse import COARSE
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

    alphas = [float(a) for a in snakemake.params.alphas]
    lambdas = [float(lam) for lam in snakemake.params.lambdas]
    mode = snakemake.params.mode
    k_param = snakemake.params.k
    k = None if k_param in (None, "None", "none", "") else int(k_param)

    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    data_dict = build_data_dict(blocks, targets)
    num_atoms = len(partition_parts)

    # `true_labels` was computed under the GROUPED ground-truth partition.
    # The ungrouped run intervenes on single variables, which generally induces
    # a finer ground-truth partition — recompute it here so ARI is comparable
    # to RePaRe's ungrouped numbers.
    if mode == "ungrouped":
        with open(snakemake.input.truedagfull, "rb") as f:
            true_dag_full = pickle.load(f)
        _, true_labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    records = []
    dags = {}
    for alpha, lambda_pen in product(alphas, lambdas):
        start = time.perf_counter()
        model = COARSE().fit(
            data_dict,
            alpha=alpha,
            lambda_pen=lambda_pen,
            k=k,
            refine_test="welch",
        )
        fit_time = time.perf_counter() - start

        est_labels = partition_labels_from_dag(model.dag, num_atoms)
        ari = adjusted_rand_score(true_labels, est_labels)
        edge_stats = partition_edge_metrics(model.dag, true_graph)

        records.append({
            "alpha": float(alpha),
            "lambda": float(lambda_pen),
            "ari": float(ari),
            "score": float(model.score),
            "fit_time": float(fit_time),
            "num_parts": int(model.dag.number_of_nodes()),
            "num_edges": int(model.dag.number_of_edges()),
            **edge_stats,
        })
        dags[(float(alpha), float(lambda_pen))] = model.dag

    df = pd.DataFrame(records)
    df.to_csv(snakemake.output.metrics_csv, index=False)

    # Score-selected: max model.score (higher is better); tie-break by ARI.
    # Oracle-selected: max ARI, then F1, then precision, then score.
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
            "k": k,
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
