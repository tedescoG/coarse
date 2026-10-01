#!/usr/bin/env python3
"""Run a UT-IGSP `alpha_ci` × `alpha_inv` grid on the CausalChamber light tunnel.

Mirrors `repare-0.2.0/src/expt/workflow/scripts/causalchamber_utigsp.py`
(partial-correlation CI test, Gaussian invariance test, unknown targets), with
the grid moved from the script into the rule params. The wrapper
`causal_chambers.ut_igsp` is the BSD-licensed Gamella code vendored in the
RePaRe bundle; it is reached via the same sys.path insert as
`causalchamber_repare.py`.

Selection (RePaRe's convention):
    score  → min Gaussian BIC of the estimated DAG on the observational
             sample (`gaussian_bic_score`, lower is better; ground truth never
             enters), tie → smaller alphas;
    oracle → max `f1 · precision`.
The DAG is scored after projection onto the true coarsening. `fit_time` is the selected cell's
single fit; the grid total goes to `search_runtime_sec`.
"""

import json
import pickle
import random
import sys
import time
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

THESIS_BUNDLE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(THESIS_BUNDLE / "repare-0.2.0" / "src"))

from causal_chambers import ut_igsp

from _causalchamber_common import (
    all_edge_metrics,
    baseline_env_lists,
    ground_truth_partition,
    gaussian_bic_score,
    labeled_summary,
    params_payload,
    quotient_graph,
    save_dag_plot,
    select_targets,
    select_utigsp_oracle_row,
    select_utigsp_score_row,
)


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.grouptargets, "rb") as f:
        group_targets = pickle.load(f)
    with open(snakemake.input.truegraph, "rb") as f:
        true_graph = pickle.load(f)
    with open(snakemake.input.partition, "rb") as f:
        partition_parts = pickle.load(f)
    with open(snakemake.input.truedagfull, "rb") as f:
        true_dag_full = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.singleenvlabels, "rb") as f:
        single_env_labels = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]

    mode = snakemake.params.mode
    targets = select_targets(mode, group_targets, single_env_labels, name_to_idx)
    true_blocks, _ = ground_truth_partition(targets, partition_parts, true_dag_full)
    alpha_cis = [float(a) for a in snakemake.params.alpha_ci]
    alpha_invs = [float(a) for a in snakemake.params.alpha_inv]
    data_list, _, env_labels = baseline_env_lists(
        mode, blocks, group_targets, single_env_labels, name_to_idx
    )

    records = []
    dags = {}
    for alpha_ci, alpha_inv in product(alpha_cis, alpha_invs):
        # `unknown_target_igsp` starts its restarts from `random`-module draws
        # (unseeded in RePaRe's script, so its recorded numbers are one draw);
        # seed per cell so this grid is reproducible.
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

        dag = quotient_graph(zip(*np.nonzero(adj)), true_blocks)
        bic = gaussian_bic_score(
            blocks["obs"], nx.from_numpy_array(adj != 0, create_using=nx.DiGraph)
        )

        records.append({
            "alpha": alpha_ci,
            "alpha_ci": alpha_ci,
            "alpha_inv": alpha_inv,
            "second_hp": alpha_inv,
            "second_hp_name": "alpha_inv",
            "ari": float("nan"),
            "bic": float(bic),
            "score": float(bic),
            "fit_time": float(fit_time),
            "num_parts": int(dag.number_of_nodes()),
            "num_edges": int(dag.number_of_edges()),
            **all_edge_metrics(dag, true_graph, true_blocks),
        })
        dags[(alpha_ci, alpha_inv)] = dag

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    pd.DataFrame(records).to_csv(snakemake.output.metrics_csv, index=False)

    score_row = select_utigsp_score_row(records)
    oracle_row = select_utigsp_oracle_row(records)
    score_dag = dags[(score_row["alpha_ci"], score_row["alpha_inv"])]
    oracle_dag = dags[(oracle_row["alpha_ci"], oracle_row["alpha_inv"])]

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    for row, dag, path in (
        (score_row, score_dag, snakemake.output.score_params),
        (oracle_row, oracle_dag, snakemake.output.oracle_params),
    ):
        parts, edges = labeled_summary(dag, feature_cols)
        with open(path, "w") as f:
            json.dump(
                params_payload(
                    row, parts, edges,
                    alpha_ci=row["alpha_ci"], alpha_inv=row["alpha_inv"],
                    env_labels=env_labels,
                ),
                f, indent=2, default=float,
            )


if __name__ == "__main__":
    main()
