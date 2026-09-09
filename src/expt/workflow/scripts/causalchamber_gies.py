#!/usr/bin/env python3
"""Run GIES (known targets) on the CausalChamber light tunnel.

Mirrors `repare-0.2.0/src/expt/workflow/scripts/causalchamber_gies.py`:
`gies.fit_bic(data, I)` with all defaults (forward / backward / turning,
iterated), Gaussian BIC, intercept fitted. Unlike RePaRe's version the env
regime is a parameter: "grouped" uses obs + rgb + pol with the multi-target
sets, "ungrouped" uses obs + the single-variable experiments.

GIES returns the I-essential graph; undirected edges arrive as both `A[i, j]`
and `A[j, i]` and are kept as two directed edges (RePaRe's convention), so
the native metric is directed edge precision / recall (`metric_type="edges"`)
and the skeleton view lives in `skel_*`.

There is no hyperparameter grid, so only the score-selected artifacts are
written; `fit_time == search_runtime_sec` (one run).
"""

import json
import pickle
import time
from pathlib import Path

import gies
import pandas as pd

from _causalchamber_common import (
    adjacency_to_singleton_dag,
    all_edge_metrics,
    baseline_env_lists,
    labeled_summary,
    params_payload,
    save_dag_plot,
)


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    mode = snakemake.params.mode
    data_list, target_lists, env_labels = baseline_env_lists(
        mode, blocks, group_targets, single_env_labels, name_to_idx
    )

    start = time.perf_counter()
    adj, score = gies.fit_bic(data_list, target_lists)
    fit_time = time.perf_counter() - start

    dag = adjacency_to_singleton_dag(adj)
    row = {
        "alpha": None,
        "second_hp": None,
        "second_hp_name": "",
        "ari": float("nan"),
        "score": float(score),
        "fit_time": float(fit_time),
        "search_runtime_sec": float(fit_time),
        "num_parts": int(dag.number_of_nodes()),
        "num_edges": int(dag.number_of_edges()),
        **all_edge_metrics(dag, true_graph, native="edges"),
    }
    pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)

    save_dag_plot(dag, feature_cols, Path(snakemake.output.score_dag))
    parts, edges = labeled_summary(dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(row, parts, edges, env_labels=env_labels, targets=target_lists),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
