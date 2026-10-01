#!/usr/bin/env python3
# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""UT-IGSP `alpha_ci` × `alpha_inv` grid on the Causal Chamber, ungrouped regime, each cell
seeded. Score row = min observational BIC, oracle row = max `f1 · precision`."""

import json
import pickle
import random
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import ut_igsp

from _causalchamber_common import (
    atomic_graph,
    baseline_env_lists,
    gaussian_bic_score,
    graph_edge_metrics,
    labeled_summary,
    params_payload,
    save_dag_plot,
    select_utigsp_oracle_row,
    select_utigsp_score_row,
    singleton_dag,
)


def _record_targets(found):
    # ut_igsp.fit returns only the union; keep the per-env list for the I-CPDAG.
    inner = ut_igsp.unknown_target_igsp

    def spy(*args, **kwargs):
        dag, targets = inner(*args, **kwargs)
        found[:] = [sorted(int(t) for t in tgt) for tgt in targets]
        return dag, targets

    ut_igsp.unknown_target_igsp = spy


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

    alpha_cis = [float(a) for a in snakemake.params.alpha_ci]
    alpha_invs = [float(a) for a in snakemake.params.alpha_inv]
    data_list, _, env_labels = baseline_env_lists(blocks, single_env_labels, name_to_idx)

    records = []
    dags = {}
    env_targets = {}
    found = []
    _record_targets(found)
    for alpha_ci, alpha_inv in product(alpha_cis, alpha_invs):
        # Its restarts draw from `random`.
        random.seed(0)
        np.random.seed(0)
        start = time.perf_counter()
        adj, _ = ut_igsp.fit(
            data_list,
            alpha_ci=alpha_ci,
            alpha_inv=alpha_inv,
            test="gauss",
            obs_idx=0,
        )
        fit_time = time.perf_counter() - start

        graph = atomic_graph(adj)
        bic = gaussian_bic_score(blocks["obs"], graph)

        records.append({
            "metric_type": "edges",
            "alpha_ci": alpha_ci,
            "alpha_inv": alpha_inv,
            "ari": float("nan"),
            "bic": float(bic),
            "score": float(bic),
            "fit_time": float(fit_time),
            "num_parts": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            **graph_edge_metrics(graph, true_graph),
        })
        dags[(alpha_ci, alpha_inv)] = singleton_dag(graph)
        env_targets[(alpha_ci, alpha_inv)] = list(found)

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    pd.DataFrame(records).to_csv(snakemake.output.metrics_csv, index=False)

    score_row = select_utigsp_score_row(records)
    oracle_row = select_utigsp_oracle_row(records)
    score_key = (score_row["alpha_ci"], score_row["alpha_inv"])
    oracle_key = (oracle_row["alpha_ci"], oracle_row["alpha_inv"])
    score_dag, oracle_dag = dags[score_key], dags[oracle_key]

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    for row, dag, key, path in (
        (score_row, score_dag, score_key, snakemake.output.score_params),
        (oracle_row, oracle_dag, oracle_key, snakemake.output.oracle_params),
    ):
        parts, edges = labeled_summary(dag, feature_cols)
        with open(path, "w") as f:
            json.dump(
                params_payload(
                    row, parts, edges, env_labels=env_labels,
                    estimated_targets=env_targets[key],
                ),
                f, indent=2, default=float,
            )


if __name__ == "__main__":
    main()
