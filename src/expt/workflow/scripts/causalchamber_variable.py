#!/usr/bin/env python3
"""Ungrouped regime, every method on one footing: variable level (block methods expanded) and
block level (collapsed onto the true partition), with exact and simulated negative controls."""

import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd

from _causalchamber_common import (
    LEVELS, METHOD_LABEL, VARIABLE_METRICS, collapse, dag_from_labeled_parts, exact_nc,
    expand_blocks, ground_truth_partition, icpdag, random_expanded, random_icpdag,
    select_targets, sim_nc, split_edges, variable_metrics,
)

BLOCK_METHODS = ("coarse", "repare")


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    true_graph = _load(snakemake.input.truegraph)
    name_to_idx = _load(snakemake.input.nametoidx)
    targets = select_targets("ungrouped", {}, _load(snakemake.input.singleenvlabels), name_to_idx)
    partition, _ = ground_truth_partition(
        targets, _load(snakemake.input.partition), _load(snakemake.input.truedagfull)
    )
    block_of = {v: tuple(sorted(b)) for b in partition for v in b}
    p = len(block_of)
    truths = {"variable": set(true_graph.edges)}
    true_blocks = collapse((truths["variable"], set()), block_of)
    assert not true_blocks[1] and nx.is_directed_acyclic_graph(nx.DiGraph(list(true_blocks[0])))
    truths["block"] = true_blocks[0]
    m_max = {"variable": p * (p - 1) // 2, "block": len(partition) * (len(partition) - 1) // 2}

    rows = []
    for method, selections in snakemake.params.method_selections.items():
        for sel in selections:
            with open(getattr(snakemake.input, f"{method}_{sel}_params")) as f:
                params = json.load(f)
            dag = dag_from_labeled_parts(params["parts"], params["edges"], name_to_idx)
            if method in BLOCK_METHODS:
                graph = expand_blocks(dag.edges)
                blocks, k = list(dag.nodes), dag.number_of_edges()
                draw = lambda rng, blocks=blocks, k=k: random_expanded(rng, blocks, k)
            else:
                arcs = {(u[0], v[0]) for u, v in dag.edges}
                if method == "gies":
                    family = params["targets"][1:]
                elif method == "gnies":  # GnIES orients every edge at a target
                    family = [[t] for t in params["estimated_targets"]]
                else:
                    family = params["estimated_targets"]
                graph = split_edges(arcs) if method != "utigsp" else icpdag(arcs, family, p)
                m_adj = len(graph[0]) + len(graph[1])
                draw = lambda rng, m=m_adj, fam=family: random_icpdag(rng, p, m, fam)

            observed = {
                lv: variable_metrics(g, truths[lv])
                for lv, g in zip(LEVELS, (graph, collapse(graph, block_of)))
            }
            rng = np.random.default_rng(snakemake.params.seed)
            sim = sim_nc(draw, snakemake.params.draws, rng, truths, block_of, observed)
            name = METHOD_LABEL[method] + ("_ungrouped" if method in BLOCK_METHODS else "")
            for lv in LEVELS:
                o = observed[lv]
                exact = exact_nc(m_max[lv], len(truths[lv]), o["m_est"], o["tp_adj"])
                for k in VARIABLE_METRICS:
                    base = {"method": name, "selection": sel, "level": lv, "metric": k,
                            "value": o[k], "m_est": o["m_est"]}
                    for nc, res in (("exact", exact.get(k)), ("sim", sim[(lv, k)])):
                        if res is not None:
                            rows.append({**base, "nc": nc,
                                         **dict(zip(("nc_mean", "nc_lo", "nc_hi", "p"), res))})

    cols = ["method", "selection", "level", "metric", "value", "nc", "nc_mean", "nc_lo",
            "nc_hi", "p", "m_est"]
    pd.DataFrame(rows, columns=cols).to_csv(snakemake.output.summary, index=False)


if __name__ == "__main__":
    main()
