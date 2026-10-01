#!/usr/bin/env python3
# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
"""Collect every method's selected rows into the summary CSV, the DAG dump and figure."""

import json
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from _causalchamber_common import METHOD_LABEL, dag_from_labeled_parts, draw_dag

SUMMARY_COLUMNS = [
    "method", "selection", "metric_type", "ari", "precision", "recall", "f1",
    "runtime_sec", "score", "alpha", "beta", "alpha_ci", "alpha_inv",
    "lambda", "search_runtime_sec",
]
HP_KEYS = ("alpha", "beta", "alpha_ci", "alpha_inv", "lambda")
BLOCK_METHODS = ("coarse", "repare")


def _method_name(method: str, mode: str) -> str:
    return f"{METHOD_LABEL[method]}_{mode}" if method in BLOCK_METHODS else METHOD_LABEL[method]


def _hp_string(p: dict) -> str:
    parts = [f"{key}={p[key]:.0e}" for key in HP_KEYS if p.get(key) is not None]
    return ", ".join(parts) if parts else "no hyperparameters"


def main():
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)

    method_selections = dict(snakemake.params.method_selections)
    method_modes = dict(snakemake.params.method_modes)
    cells = [
        (method, mode, sel)
        for method, selections in method_selections.items()
        for mode in method_modes[method]
        for sel in selections
    ]

    params_by_cell = {}
    for method, mode, sel in cells:
        with open(getattr(snakemake.input, f"{method}_{mode}_{sel}_params"), "r") as f:
            params_by_cell[(method, mode, sel)] = json.load(f)

    summary_rows = []
    for method, mode, sel in cells:
        p = params_by_cell[(method, mode, sel)]
        summary_rows.append({
            "method": _method_name(method, mode),
            "selection": sel,
            **{key: p[key] for key in SUMMARY_COLUMNS[2:] if key in p},
            "runtime_sec": p["fit_time"],
        })
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        snakemake.output.summary, index=False
    )

    fig, axes = plt.subplots(1, len(BLOCK_METHODS), figsize=(8 * len(BLOCK_METHODS), 7))
    for ax, method in zip(axes, BLOCK_METHODS):
        p = params_by_cell[(method, "grouped", "score")]
        dag = dag_from_labeled_parts(p["parts"], p["edges"], name_to_idx)
        title = (
            f"{METHOD_LABEL[method]} (grouped, score-selected)\n"
            f"{_hp_string(p)}, ARI={p['ari']:.3f}, F1={p['f1']:.3f}, "
            f"runtime={p['fit_time']:.2f}s"
        )
        draw_dag(dag, feature_cols, ax, title=title)
    fig.tight_layout()
    fig.savefig(snakemake.output.dag, bbox_inches="tight")
    plt.close(fig)

    lines = []
    for method, mode, sel in cells:
        p = params_by_cell[(method, mode, sel)]
        lines.append(f"=== {_method_name(method, mode)} ({sel}-selected) ===")
        lines.append(
            f"  {_hp_string(p)}, ari={p['ari']:.4f}, "
            f"{p['metric_type']} precision/recall/f1="
            f"{p['precision']:.4f}/{p['recall']:.4f}/{p['f1']:.4f}, "
            f"runtime_sec={p['fit_time']:.3f}, "
            f"search_runtime_sec={p['search_runtime_sec']:.3f}, "
            f"num_parts={p['num_parts']}, num_edges={p['num_edges']}"
        )
        lines.append("  Partition nodes:")
        for idx, labels in p["parts"]:
            lines.append(f"    Node {idx}: {tuple(labels)}")
        lines.append("  Edges (u -> v):")
        for u, v in p["edges"]:
            lines.append(f"    {u} -> {v}")
        lines.append("")

    with open(snakemake.output.dag_summary, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
