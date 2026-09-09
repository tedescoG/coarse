#!/usr/bin/env python3
"""Run a RePaRe α × β grid on the CausalChamber light tunnel.

Lifted from `repare-0.2.0/src/expt/workflow/scripts/causalchamber_repare.py`.
Two structural changes:

1. RePaRe lives outside this Python package; we reach it via the same
   `sys.path` trick as `fit_oracle_repare.py`.
2. **No model pickles** — the partition + edges are serialized into
   `score_params.json` / `oracle_params.json`, and the DAGs are written as
   PNGs at fit time. The aggregator reconstructs everything from JSON, so it
   doesn't need `repare` on its sys.path and no data sneaks out via pickled
   `PartitionDagModelIvn` instances.

Score convention: RePaRe selects HPs by minimizing `score`, which is
`-gnies.full_score(expanded_adj)` (the negation flips GnIES BIC into
lower-is-better). NOT comparable across methods — cross-method comparison
uses ARI / block-edge precision / recall / F1.

Runtime convention (RePaRe's): `fit_time` is the selected cell's single fit;
the grid total goes to `search_runtime_sec`.
"""

import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

# RePaRe is a sibling subproject inside the thesis bundle, not part of the
# COARSE Python package. parents[5] is the bundle root (…/THESIS/).
THESIS_BUNDLE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(THESIS_BUNDLE / "repare-0.2.0" / "src"))

from gnies.scores.gnies_score import GnIESScore
from repare.repare import PartitionDagModelIvn

from _causalchamber_common import (
    all_edge_metrics,
    build_data_dict,
    ground_truth_partition,
    labeled_summary,
    params_payload,
    partition_labels_from_dag,
    save_dag_plot,
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
    with open(snakemake.input.truelabels, "rb") as f:
        true_labels = pickle.load(f)
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
    if mode == "ungrouped":
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
            build_data_dict(blocks, targets),
            alpha=float(alpha),
            beta=float(beta),
            assume="gaussian",
            refine_test="ks",
        )
        fit_time = time.perf_counter() - start

        est_labels = partition_labels_from_dag(model.dag, num_atoms)
        ari = adjusted_rand_score(true_labels, est_labels)

        expanded_adj = model.expand_coarsened_dag(fully_connected=True)
        score_value = -float(gnies_score.full_score(expanded_adj))

        records.append({
            "alpha": float(alpha),
            "beta": float(beta),
            "second_hp": float(beta),
            "second_hp_name": "beta",
            "ari": float(ari),
            "score": score_value,
            "fit_time": float(fit_time),
            "num_parts": int(model.dag.number_of_nodes()),
            "num_edges": int(model.dag.number_of_edges()),
            **all_edge_metrics(model.dag, true_graph, native="partition"),
        })
        dags[(float(alpha), float(beta))] = model.dag

    grid_runtime = float(sum(r["fit_time"] for r in records))
    for r in records:
        r["search_runtime_sec"] = grid_runtime

    df = pd.DataFrame(records)
    df.to_csv(snakemake.output.metrics_csv, index=False)

    # RePaRe's native selection: min score (lower is better after negation);
    # tie-break: smaller α (ground truth never enters the score-selected row).
    # Oracle: max ARI / F1 / precision, then min score.
    score_row = min(records, key=lambda r: (r["score"], r["alpha"]))
    oracle_row = max(
        records, key=lambda r: (r["ari"], r["f1"], r["precision"], -r["score"])
    )

    score_dag = dags[(score_row["alpha"], score_row["beta"])]
    oracle_dag = dags[(oracle_row["alpha"], oracle_row["beta"])]

    save_dag_plot(score_dag, feature_cols, Path(snakemake.output.score_dag))
    save_dag_plot(oracle_dag, feature_cols, Path(snakemake.output.oracle_dag))

    score_parts, score_edges = labeled_summary(score_dag, feature_cols)
    oracle_parts, oracle_edges = labeled_summary(oracle_dag, feature_cols)

    with open(snakemake.output.score_params, "w") as f:
        json.dump(
            params_payload(score_row, score_parts, score_edges, beta=score_row["beta"]),
            f, indent=2, default=float,
        )
    with open(snakemake.output.oracle_params, "w") as f:
        json.dump(
            params_payload(oracle_row, oracle_parts, oracle_edges, beta=oracle_row["beta"]),
            f, indent=2, default=float,
        )


if __name__ == "__main__":
    main()
