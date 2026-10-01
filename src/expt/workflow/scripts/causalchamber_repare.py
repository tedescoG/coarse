#!/usr/bin/env python3
# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""RePaRe α × β grid on the Causal Chamber. Score row = min GnIES score of the expanded DAG,
oracle row = max ARI, then F1."""

import json
import pickle
import time
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from gnies.scores.gnies_score import GnIESScore
from repare.repare import PartitionDagModelIvn

from _common import partition_edge_metrics, partition_labels
from _causalchamber_common import (
    build_block_data_dict,
    ground_truth_partition,
    labeled_summary,
    params_payload,
    save_dag_plot,
    select_oracle_row,
    select_score_row,
    select_targets,
)


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.partition, "rb") as f:
        partition_parts = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.truedagfull, "rb") as f:
        true_dag_full = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    alphas = [float(a) for a in snakemake.params.alphas]
    betas = [float(b) for b in snakemake.params.betas]
    mode = snakemake.params.mode

    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    _, true_labels = ground_truth_partition(targets, partition_parts, true_dag_full)

    env_labels = list(targets)
    gnies_data = [blocks["obs"], *(blocks[label] for label in env_labels)]
    intervention_union = set().union(*targets.values()) if targets else set()
    gnies_score = GnIESScore(gnies_data, intervention_union, lmbda=0.0, centered=True)

    num_atoms = len(partition_parts)

    records = []
    dags = {}
    for alpha, beta in product(alphas, betas):
        start = time.perf_counter()
        model = PartitionDagModelIvn().fit(
            build_block_data_dict(blocks, targets),
            alpha=float(alpha),
            beta=float(beta),
            assume="gaussian",
            refine_test="ks",
        )
        fit_time = time.perf_counter() - start

        est_labels = partition_labels(model.dag.nodes, num_atoms)
        ari = adjusted_rand_score(true_labels, est_labels)

        expanded_adj = model.expand_coarsened_dag(fully_connected=True)
        score_value = -float(gnies_score.full_score(expanded_adj))

        records.append({
            "metric_type": "partition",
            "alpha": float(alpha),
            "beta": float(beta),
            "ari": float(ari),
            "score": score_value,
            "fit_time": float(fit_time),
            "num_parts": int(model.dag.number_of_nodes()),
            "num_edges": int(model.dag.number_of_edges()),
            **partition_edge_metrics(model.dag, true_graph),
        })
        dags[(float(alpha), float(beta))] = model.dag

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    df = pd.DataFrame(records)
    df.to_csv(snakemake.output.metrics_csv, index=False)

    score_row = select_score_row(records)
    oracle_row = select_oracle_row(records)

    score_dag = dags[(score_row["alpha"], score_row["beta"])]
    oracle_dag = dags[(oracle_row["alpha"], oracle_row["beta"])]

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    score_parts, score_edges = labeled_summary(score_dag, feature_cols)
    oracle_parts, oracle_edges = labeled_summary(oracle_dag, feature_cols)

    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(score_row, score_parts, score_edges),
            f, indent=2, default=float,
        )
    with open(snakemake.output.oracle_params, "w") as f:
        json.dump(
            params_payload(oracle_row, oracle_parts, oracle_edges),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
