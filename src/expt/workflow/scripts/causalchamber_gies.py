#!/usr/bin/env python3
# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""GIES (known targets) on the Causal Chamber, ungrouped regime."""

import json
import pickle
import time
from pathlib import Path

import gies
import pandas as pd

from _causalchamber_common import (
    atomic_graph,
    baseline_env_lists,
    graph_edge_metrics,
    labeled_summary,
    params_payload,
    save_dag_plot,
    singleton_dag,
)


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    data_list, target_lists, env_labels = baseline_env_lists(
        blocks, single_env_labels, name_to_idx
    )

    start = time.perf_counter()
    adj, score = gies.fit_bic(data_list, target_lists)
    fit_time = time.perf_counter() - start

    graph = atomic_graph(adj)
    row = {
        "metric_type": "edges",
        "ari": float("nan"),
        "score": float(score),
        "fit_time": float(fit_time),
        "search_runtime_sec": float(fit_time),
        "num_parts": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        **graph_edge_metrics(graph, true_graph),
    }
    pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)

    dag = singleton_dag(graph)
    save_dag_plot(dag, feature_cols, Path(snakemake.output.score_dag))
    parts, edges = labeled_summary(dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(row, parts, edges, env_labels=env_labels, targets=target_lists),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
