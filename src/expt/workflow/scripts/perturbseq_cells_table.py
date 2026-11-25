# ruff: noqa: F821
"""Comprehensive results table for the Perturb-seq comparison.

Post-hoc: reads summary.csv plus the final-fit JSONs, rebuilds each serialized graph,
recomputes held-out density and the no-edge ablation uniformly for every method on the
copula-Gaussian scale, adds CollecTRI directed interaction support against a
gene-relabeling null, and emits per-method medians and MADs across splits as a CSV and
a booktabs fragment.
Splits overlap, so the spread describes across-split variation on this dataset,
not sampling uncertainty about the procedure.
"""

import itertools
import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from _perturbseq_common import outer_train_batches
from perturbseq_cells_compare import DIAGNOSTIC_ALPHA, int_apply
from perturbseq_cells_fit import (
    graph_density_score,
    heldout_agreement,
    subset,
)

DISPLAY_NAMES = {
    "coarse": "COARSE (Welch partition)",
    "coarse_kpc1": "1PC-COARSE",
    "coarse_kpc3": "3PC-COARSE",
    "coarse_avg": "avg-COARSE",
    "coarse_scored": "COARSE (scored)",
    "gnies": "GnIES",
    "repare": "RePaRe",
    "repare_welch": "RePaRe (Welch partition)",
    "utigsp": "UT-IGSP",
    "singleton_empty": "Empty ({p} singletons)",
    "joint": "One joint block",
}


def rebuild_graph(blocks, edges, features):
    """Serialized gene-name blocks/edges back to a DAG over gene-index tuples."""
    index = {g: i for i, g in enumerate(features)}
    nodes = [tuple(sorted(index[g] for g in b)) for b in blocks]
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for parent, child in edges:
        parent_node = tuple(sorted(index[g] for g in parent))
        child_node = tuple(sorted(index[g] for g in child))
        graph.add_edge(parent_node, child_node)
    return graph


def collectri_support(blocks, edges, collectri_pairs, features, rng, num_draws=200):
    """Two-sided qualitative CollecTRI reference: the fraction of blocks containing at
    least one documented directed interaction (regulon-level support) and the fraction
    of selected parent->child block pairs with a documented interaction across the two
    blocks. Both are compared against a null that relabels genes over the same block
    structure, preserving block sizes and the edge topology."""
    index = {g: i for i, g in enumerate(features)}

    def block_support(labels):
        hits = sum(
            1
            for b in blocks
            if any(
                (labels[index[a]], labels[index[b]]) in collectri_pairs
                for a, b in itertools.combinations(b, 2)
            )
        )
        return hits / len(blocks)

    def edge_support(labels):
        if not edges:
            return None
        hits = sum(
            1
            for parent, child in edges
            if any(
                (labels[index[a]], labels[index[b]]) in collectri_pairs
                for a in parent
                for b in child
            )
        )
        return hits / len(edges)

    block_draws, edge_draws = [], []
    for _ in range(num_draws):
        labels = list(rng.permutation(features))
        block_draws.append(block_support(labels))
        edge_draw = edge_support(labels)
        if edge_draw is not None:
            edge_draws.append(edge_draw)
    return {
        "blocks": block_support(features),
        "blocks_random": float(np.mean(block_draws)),
        "edges": edge_support(features),
        "edges_random": (float(np.mean(edge_draws)) if edge_draws else None),
    }


def collect_records(summary, finals, cells, metadata, methods=None, seed=0):
    features = metadata["features"]
    collectri_pairs = {tuple(pair) for pair in metadata["collectri_edges"]}
    rng = np.random.default_rng(seed)
    records = []
    for path in finals:
        with open(path) as f:
            final = json.load(f)
        if final.get("status") != "ok":
            continue
        split, method = final["split"], final["method"]
        graph = rebuild_graph(final["blocks"], final["edges"], features)
        train = subset(cells, outer_train_batches(split))
        test = subset(cells, [int(b) for b in final["test_batches"]])
        train, test = int_apply(train, test)
        density = graph_density_score(graph, train, test)["mean_per_gene"]
        if graph.number_of_edges():
            no_edges = graph.copy()
            no_edges.remove_edges_from(list(no_edges.edges))
            ablated = graph_density_score(no_edges, train, test)["mean_per_gene"]
            edge_contribution = density - ablated
        else:
            edge_contribution = 0.0
        agreement = heldout_agreement(graph, train, test, DIAGNOSTIC_ALPHA)
        if agreement["within"] is None:
            codet_delta = None
        else:
            codet_delta = (
                agreement["within"]["both_detected"]
                - agreement["matched_random_expectation"]["both_detected"]
            )
        collectri = collectri_support(
            final["blocks"], final["edges"], collectri_pairs, features, rng
        )
        row = summary[(summary.method == method) & (summary.split == split)].iloc[0]
        if abs(row.heldout_loglik_per_gene - density) > 1e-6:
            print(
                f"warning: density mismatch {method} split {split}: "
                f"{row.heldout_loglik_per_gene} vs {density}"
            )
        records.append(
            {
                "method": method,
                "split": split,
                "density": density,
                "edge_contribution": edge_contribution,
                "codet_delta": codet_delta,
                "collectri_blocks": collectri["blocks"],
                "collectri_blocks_random": collectri["blocks_random"],
                "collectri_edges": collectri["edges"],
                "collectri_edges_random": collectri["edges_random"],
                "blocks": len(final["blocks"]),
                "edges": len(final["edges"]),
                "seconds": row.total_seconds,
            }
        )
    frame = pd.DataFrame(records)
    if methods:
        frame = frame[frame.method.isin(methods)]
    return frame


def summarize(records):
    rows = []
    for method, group in records.groupby("method"):
        row = {"method": method}
        for metric in (
            "density",
            "edge_contribution",
            "codet_delta",
            "collectri_blocks",
            "collectri_edges",
            "seconds",
            "blocks",
            "edges",
        ):
            values = group[metric].astype(float).dropna()
            if values.empty:
                continue
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_mad"] = float(np.median(np.abs(values - values.median())))
        sup = group.dropna(subset=["collectri_blocks_random"])
        if len(sup):
            delta = sup.collectri_blocks - sup.collectri_blocks_random
            row["collectri_blocks_delta_median"] = delta.median()
            row["collectri_blocks_delta_mad"] = float(
                np.median(np.abs(delta - delta.median()))
            )
        ed = group.dropna(subset=["collectri_edges_random"])
        if len(ed):
            delta = ed.collectri_edges - ed.collectri_edges_random
            row["collectri_edges_delta_median"] = delta.median()
            row["collectri_edges_delta_mad"] = float(
                np.median(np.abs(delta - delta.median()))
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("density_median", ascending=False)


def _tex_cell(median, mad, fmt):
    if median is None or (isinstance(median, float) and np.isnan(median)):
        return "---"
    if mad is None or mad == 0:
        return f"${fmt.format(median)}$"
    mad_fmt = "{:.1f}" if fmt == "{:.0f}" and mad % 1 else fmt
    return f"${fmt.format(median)}$ (${mad_fmt.format(mad)}$)"


def format_tex(table, method_order, num_features):
    """Booktabs fragment: one row per method, median (MAD) cells; the paper wraps it
    in a table environment with its own caption."""
    columns = [
        ("Density", "density_median", "density_mad", "{:.3f}"),
        (
            "Edge contribution",
            "edge_contribution_median",
            "edge_contribution_mad",
            "{:.3f}",
        ),
        ("Co-detection $\\Delta$", "codet_delta_median", "codet_delta_mad", "{:.2f}"),
        (
            "CollecTRI blocks $\\Delta$",
            "collectri_blocks_delta_median",
            None,
            "{:+.2f}",
        ),
        ("Blocks", "blocks_median", "blocks_mad", "{:.0f}"),
        ("Edges", "edges_median", "edges_mad", "{:.0f}"),
        ("Time (s)", "seconds_median", "seconds_mad", "{:.1f}"),
    ]
    lines = [
        "% auto-generated by perturbseq_cells_table.py; do not edit",
        "\\begin{tabular}{l" + "c" * len(columns) + "}",
        "\\toprule",
        "Method & " + " & ".join(h for h, _, _, _ in columns) + " \\\\",
        "\\midrule",
    ]
    present = [m for m in method_order if m in table.method.values]
    for method in present:
        row = table[table.method == method].iloc[0]
        cells = [
            _tex_cell(row.get(med), row.get(mad) if mad else None, fmt)
            for _, med, mad, fmt in columns
        ]
        lines.append(
            DISPLAY_NAMES.get(method, method).format(p=num_features)
            + " & "
            + " & ".join(cells)
            + " \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    with open(snakemake.input.metadata) as f:
        metadata = json.load(f)
    with open(snakemake.input.cells, "rb") as f:
        cells = pickle.load(f)
    summary = pd.read_csv(snakemake.input.summary)
    records = collect_records(
        summary,
        list(snakemake.input.finals),
        cells,
        metadata,
        methods=list(snakemake.params.methods),
    )
    table = summarize(records)
    table.to_csv(snakemake.output.csv, index=False)
    Path(snakemake.output.tex).write_text(
        format_tex(table, list(snakemake.params.methods), len(metadata["features"]))
    )


if __name__ == "__main__":
    main()


main()
