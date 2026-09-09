#!/usr/bin/env python3
"""Run GnIES (unknown targets) on the CausalChamber light tunnel.

Mirrors `repare-0.2.0/src/expt/workflow/scripts/causalchamber_gnies.py`,
including its two speed concessions, which are kept so the numbers stay
comparable to RePaRe's Table 1:

- every environment is subsampled to `snakemake.params.max_rows` rows
  (seed = position in the env list; RePaRe used 2000);
- forward-only outer and inner phases, `ges_iterate=False`.

GnIES returns the I-CPDAG; its native metric is the **skeleton**
precision / recall (`metric_type="skeleton"`, as in RePaRe) — the directed
view, with undirected edges counted as two directed ones, lives in `dir_*`.
No hyperparameter grid (BIC penalty `0.5·log N` by default), so only the
score-selected artifacts are written; `fit_time == search_runtime_sec`.
"""

import json
import pickle
import time
from pathlib import Path

import gnies
import pandas as pd

from _causalchamber_common import (
    adjacency_to_singleton_dag,
    all_edge_metrics,
    baseline_env_lists,
    labeled_summary,
    params_payload,
    save_dag_plot,
    subsample_rows,
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
    max_rows = int(snakemake.params.max_rows)
    data_list, _, env_labels = baseline_env_lists(
        mode, blocks, group_targets, single_env_labels, name_to_idx
    )
    data_list = [subsample_rows(arr, max_rows, seed=i) for i, arr in enumerate(data_list)]

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
        **all_edge_metrics(dag, true_graph, native="skeleton"),
    }
    pd.DataFrame([row]).to_csv(snakemake.output.metrics_csv, index=False)

    save_dag_plot(dag, feature_cols, Path(snakemake.output.score_dag))
    parts, edges = labeled_summary(dag, feature_cols)
    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(
                row, parts, edges,
                env_labels=env_labels,
                max_rows=max_rows,
                estimated_targets=sorted(int(t) for t in est_targets),
            ),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
