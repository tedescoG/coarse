#!/usr/bin/env python3
# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""GnIES (unknown targets) on the Causal Chamber, ungrouped regime, all rows."""

import json
import pickle
import time
from pathlib import Path

import gnies
import pandas as pd

from _causalchamber_common import (
    atomic_graph,
    baseline_env_lists,
    labeled_summary,
    params_payload,
    save_dag_plot,
    singleton_dag,
    skeleton_metrics,
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

    data_list, _, env_labels = baseline_env_lists(blocks, single_env_labels, name_to_idx)

    start = time.perf_counter()
    score, adj, est_targets = gnies.fit(
        data=data_list,
        known_targets=set(),
        approach="greedy",
        center=True,
        ges_iterate=False,
        phases=["forward"],
        ges_phases=["forward"],
    )
    fit_time = time.perf_counter() - start

    graph = atomic_graph(adj)
    row = {
        "metric_type": "skeleton",
        "ari": float("nan"),
        "score": float(score),
        "fit_time": float(fit_time),
        "search_runtime_sec": float(fit_time),
        "num_parts": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        **skeleton_metrics(graph, true_graph),
    }
    pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)

    dag = singleton_dag(graph)
    save_dag_plot(dag, feature_cols, Path(snakemake.output.score_dag))
    parts, edges = labeled_summary(dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(
                row, parts, edges,
                env_labels=env_labels,
                estimated_targets=sorted(int(t) for t in est_targets),
            ),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
