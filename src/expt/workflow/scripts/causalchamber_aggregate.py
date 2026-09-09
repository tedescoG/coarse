#!/usr/bin/env python3
"""Aggregate the CausalChamber results of every method into the headline
artifacts:

  - results/causalchamber/grid_metrics.csv   (COARSE-grouped α grid at λ=1)
  - results/causalchamber/dag.png            (one panel per method: grouped, score-selected)
  - results/causalchamber/grid_runs/         (empty directory placeholder)
  - results/causalchamber_summary.csv        (one row per method × mode × selection)
  - results/causalchamber_dags.txt           (plain-text DAG dump, same rows)

Reads `score_params.json` / `oracle_params.json` per (method, mode) cell and
reconstructs DAGs via `dag_from_labeled_parts` — no model pickles, no
`repare` dependency. Which (method, selection) cells exist comes from
`snakemake.params.method_selections` (GIES / GnIES have no grid, hence no
oracle row).

Column semantics (see `_causalchamber_common.all_edge_metrics`):
  - `metric_type` + `precision/recall/f1`: the method's native metric as in
    RePaRe's Table 1 (partition-level for COARSE/RePaRe, directed atomic
    edges for GIES/UT-IGSP, skeleton for GnIES).
  - `dir_*` / `skel_*`: the same directed / skeleton block-collapsed metrics
    for every method — the cross-method comparison columns.
  - `runtime_sec`: the selected single fit (COARSE: the CV refit only);
    `search_runtime_sec`: the whole grid / CV loop.
  - `score_native` is NOT comparable across methods (different objectives
    and sign conventions).
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from _causalchamber_common import dag_from_labeled_parts, draw_dag


METHOD_LABEL = {
    "coarse": "COARSE",
    "repare": "RePaRe",
    "gies": "GIES",
    "gnies": "GnIES",
    "utigsp": "UT-IGSP",
}
SECOND_HP_SYMBOL = {"lambda": "λ", "beta": "β", "alpha_inv": "α_inv", "": ""}

SUMMARY_COLUMNS = [
    "method", "mode", "selection", "metric_type", "ari",
    "precision", "recall", "f1",
    "dir_precision", "dir_recall", "dir_f1",
    "skel_precision", "skel_recall", "skel_f1",
    "runtime_sec", "search_runtime_sec", "score_native",
    "alpha", "second_hp", "second_hp_name", "num_parts", "num_edges",
]


def _params_path_attr(method: str, mode: str, selection: str) -> str:
    return f"{method}_{mode}_{selection}_params"


def _metrics_path_attr(method: str, mode: str) -> str:
    return f"{method}_{mode}_metrics"


def _fmt(value, spec=".0e"):
    return "–" if value is None else format(value, spec)


def _hp_string(p: dict, method: str, selection: str) -> str:
    """`α=1e-03, λ=1e+00` style string; α̂ marks a CV-selected threshold."""
    parts = []
    if p.get("alpha") is not None:
        alpha_sym = "α̂" if p.get("score_selection") == "cv" and selection == "score" else "α"
        if p.get("second_hp_name") == "alpha_inv":
            alpha_sym = "α_ci"
        parts.append(f"{alpha_sym}={_fmt(p['alpha'])}")
    if p.get("second_hp") is not None:
        parts.append(f"{SECOND_HP_SYMBOL[p['second_hp_name']]}={_fmt(p['second_hp'])}")
    return ", ".join(parts) if parts else "no hyperparameters"


def main():
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)

    method_selections: dict[str, list[str]] = dict(snakemake.params.method_selections)
    modes: list[str] = list(snakemake.params.modes)
    cells = [
        (method, mode, sel)
        for method, selections in method_selections.items()
        for mode in modes
        for sel in selections
    ]

    params_by_cell: dict[tuple[str, str, str], dict] = {}
    for method, mode, sel in cells:
        path = getattr(snakemake.input, _params_path_attr(method, mode, sel))
        with open(path, "r") as f:
            params_by_cell[(method, mode, sel)] = json.load(f)

    # --- results/causalchamber_summary.csv --------------------------------------
    summary_rows = []
    for method, mode, sel in cells:
        p = params_by_cell[(method, mode, sel)]
        summary_rows.append({
            "method": METHOD_LABEL[method],
            "mode": mode,
            "selection": sel,
            "metric_type": p["metric_type"],
            "ari": p["ari"],
            "precision": p["precision"],
            "recall": p["recall"],
            "f1": p["f1"],
            "dir_precision": p["dir_precision"],
            "dir_recall": p["dir_recall"],
            "dir_f1": p["dir_f1"],
            "skel_precision": p["skel_precision"],
            "skel_recall": p["skel_recall"],
            "skel_f1": p["skel_f1"],
            "runtime_sec": p["fit_time"],
            "search_runtime_sec": p["search_runtime_sec"],
            "score_native": p["score"],
            "alpha": p["alpha"],
            "second_hp": p["second_hp"],
            "second_hp_name": p["second_hp_name"],
            "num_parts": p["num_parts"],
            "num_edges": p["num_edges"],
        })
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_df.to_csv(snakemake.output.summary, index=False)

    # --- results/causalchamber/grid_metrics.csv ---------------------------------
    grouped_grid = pd.read_csv(
        getattr(snakemake.input, _metrics_path_attr("coarse", "grouped"))
    )
    grouped_grid.to_csv(snakemake.output.grid_metrics, index=False)

    # --- results/causalchamber/dag.png: one panel per method (grouped, score) ---
    methods = list(method_selections)
    fig, axes = plt.subplots(1, len(methods), figsize=(8 * len(methods), 7))
    for ax, method in zip(axes, methods):
        p = params_by_cell[(method, "grouped", "score")]
        dag = dag_from_labeled_parts(p["parts"], p["edges"], name_to_idx)
        ari_str = "" if p["ari"] is None or p["ari"] != p["ari"] else f"ARI={p['ari']:.3f}, "
        title = (
            f"{METHOD_LABEL[method]} (grouped, score-selected)\n"
            f"{_hp_string(p, method, 'score')}, "
            f"{ari_str}{p['metric_type']} F1={p['f1']:.3f}, "
            f"runtime={p['fit_time']:.2f}s"
        )
        draw_dag(dag, feature_cols, ax, title=title)
    fig.tight_layout()
    fig.savefig(snakemake.output.dag, bbox_inches="tight")
    plt.close(fig)

    # --- results/causalchamber/grid_runs/ (placeholder) -------------------------
    Path(snakemake.output.grid_dir).mkdir(parents=True, exist_ok=True)

    # --- results/causalchamber_dags.txt: one block per cell ---------------------
    lines = []
    for method, mode, sel in cells:
        p = params_by_cell[(method, mode, sel)]
        lines.append(f"=== {METHOD_LABEL[method]} ({mode}, {sel}-selected) ===")
        lines.append(
            f"  {_hp_string(p, method, sel)}, metric_type={p['metric_type']}, "
            f"ari={_fmt(p['ari'], '.4f')}, precision={p['precision']:.4f}, "
            f"recall={p['recall']:.4f}, f1={p['f1']:.4f}, "
            f"dir_f1={p['dir_f1']:.4f}, skel_f1={p['skel_f1']:.4f}, "
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
