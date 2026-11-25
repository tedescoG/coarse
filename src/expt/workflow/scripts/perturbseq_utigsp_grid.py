# ruff: noqa: F821
"""UT-IGSP tuning-grid check for the Perturb-seq ladder: does the sink placement of the
perturbed genes (the finding behind the intervention-prediction metric) depend on the
tuning point?

For every split, refits UT-IGSP at all four grid points (alpha_ci, alpha_inv) on the
same transformed training batches as the comparison pipeline, and records, per
environment: the summed out-degree of the target genes and the asserted shift set
(targets plus descendants). The fit at the candidate that tuning selected is compared
against the stored final DAG, so the grid run doubles as a determinism check of the
pipeline. Writes utigsp_grid.csv with rows (split, candidate, env)."""

import json
import pickle
import random

import networkx as nx
import numpy as np
import pandas as pd
import ut_igsp
from _perturbseq_common import PS_GRIDS, outer_train_batches
from perturbseq_cells_fit import subset

from coarse.normalize import apply_inverse_normal, inverse_normal_reference


def _gene_dag(blocks, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(g for blk in blocks for g in blk)
    graph.add_edges_from((e[0][0], e[1][0]) for e in edges)
    return graph


def _target_metrics(graph, targets):
    """Summed out-degree over the target genes and the asserted shift set
    (targets plus all their descendants) for one environment."""
    present = [t for t in targets if t in graph]
    out_degree = sum(graph.out_degree(t) for t in present)
    asserted = set(targets)
    for t in present:
        asserted |= nx.descendants(graph, t)
    asserted &= set(graph.nodes)
    return out_degree, len(asserted)


def _stored_edges(rec):
    return frozenset((tuple(sorted(e[0])), tuple(sorted(e[1]))) for e in rec["edges"])


def main() -> None:
    with open(snakemake.input.cells, "rb") as f:
        data = pickle.load(f)
    with open(snakemake.input.metadata) as f:
        metadata = json.load(f)
    features = metadata["features"]
    envs = sorted(k for k in data if k != "obs")
    targets = {
        env: {g for g in env.split("_") if g} for env in metadata["environments"]
    }
    # Paths arrive in PS_CELL_SPLITS order; the JSONs do not carry the split.
    selections = {
        i: json.load(open(path)) for i, path in enumerate(snakemake.input.selections)
    }
    stored = {}
    for i, path in enumerate(snakemake.input.finals):
        with open(path) as f:
            stored[i] = json.load(f)

    candidates = PS_GRIDS["utigsp"]
    rows = []
    for split in range(len(snakemake.input.selections)):
        train = subset(data, outer_train_batches(split))
        ref = inverse_normal_reference(train["obs"])
        train = {k: apply_inverse_normal(ref, v) for k, v in train.items()}
        stored_edges = _stored_edges(stored[split])
        for candidate, (alpha_ci, alpha_inv) in enumerate(candidates):
            random.seed(0)
            np.random.seed(0)
            adj, _ = ut_igsp.fit(
                [train["obs"], *(train[e] for e in envs)],
                alpha_ci=alpha_ci,
                alpha_inv=alpha_inv,
                test="gauss",
                obs_idx=0,
            )
            blocks = [[features[i]] for i in range(adj.shape[0])]
            edges = [[[features[i]], [features[j]]] for i, j in zip(*np.nonzero(adj))]
            graph = _gene_dag(blocks, edges)
            selected = candidate == selections[split]["selected_index"]
            # The selected candidate must reproduce the pipeline's stored DAG exactly;
            # other candidates legitimately differ.
            matches = (
                _stored_edges({"edges": edges}) == stored_edges
                and len(blocks) == len(stored[split]["blocks"])
                if selected
                else None
            )
            for env, t in targets.items():
                out_degree, asserted = _target_metrics(graph, t)
                rows.append(
                    {
                        "split": split,
                        "candidate": candidate,
                        "alpha_ci": alpha_ci,
                        "alpha_inv": alpha_inv,
                        "selected": selected,
                        "matches_stored": matches,
                        "n_edges": graph.number_of_edges(),
                        "env": env,
                        "target_out_degree": out_degree,
                        "asserted_size": asserted,
                    }
                )

    frame = pd.DataFrame(rows)
    checked = frame[frame["selected"]]
    if not checked["matches_stored"].all():
        mismatches = checked[~checked["matches_stored"]][["split", "candidate"]]
        raise ValueError(
            f"selected-candidate refits disagree with stored DAGs: {mismatches}"
        )
    frame.to_csv(snakemake.output.csv, index=False)


if __name__ == "__main__":
    main()
