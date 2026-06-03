#!/usr/bin/env python3
"""Aggregate the COARSE / COARSE-1PC / RePaRe CausalChamber results into the
four headline artifacts:

  - results/causalchamber/grid_metrics.csv   (COARSE-grouped full grid)
  - results/causalchamber/dag.png            (3-panel grouped score DAGs)
  - results/causalchamber/grid_runs/         (empty directory placeholder)
  - results/causalchamber_summary.csv        (12-row method × mode × selection)
  - results/causalchamber_dags.txt           (12-block plain-text DAG dump)

Reads `score_params.json` / `oracle_params.json` per (method, mode) cell and
reconstructs partition DAGs via `dag_from_labeled_parts` — no model pickles,
no `repare` dependency, no risk of data leaking through serialized model state.

The `score` columns are NOT comparable across methods (COARSE / 1PC use
Eq. 21 partition BIC, max-better; RePaRe uses negated GnIES-on-expanded BIC,
min-better). Cross-method comparison is via ARI / partition-edge F1.
"""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from _causalchamber_common import dag_from_labeled_parts, draw_dag


METHOD_ORDER = ["coarse", "onepc", "cv", "repare"]
METHOD_LABEL = {
    "coarse": "COARSE",
    "onepc": "COARSE-1PC",
    "cv": "COARSE-CV",
    "repare": "RePaRe",
}
MODES = ["grouped", "ungrouped"]
SELECTIONS = ["score", "oracle"]


def _params_path_attr(method: str, mode: str, selection: str) -> str:
    return f"{method}_{mode}_{selection}_params"


def _metrics_path_attr(method: str, mode: str) -> str:
    return f"{method}_{mode}_metrics"


def main():
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)

    params_by_cell: dict[tuple[str, str, str], dict] = {}
    for method in METHOD_ORDER:
        for mode in MODES:
            for sel in SELECTIONS:
                path = getattr(snakemake.input, _params_path_attr(method, mode, sel))
                with open(path, "r") as f:
                    params_by_cell[(method, mode, sel)] = json.load(f)

    # --- results/causalchamber_summary.csv: 12 rows -----------------------------
    summary_rows = []
    for method in METHOD_ORDER:
        for mode in MODES:
            for sel in SELECTIONS:
                p = params_by_cell[(method, mode, sel)]
                second_knob_value = p.get("lambda", p.get("beta"))
                summary_rows.append({
                    "method": METHOD_LABEL[method],
                    "mode": mode,
                    "selection": sel,
                    "ari": p["ari"],
                    "precision": p["precision"],
                    "recall": p["recall"],
                    "f1": p["f1"],
                    "runtime_sec": p["fit_time"],
                    "score_native": p["score"],
                    "alpha": p["alpha"],
                    "lambda_or_beta": second_knob_value,
                    "num_parts": p["num_parts"],
                    "num_edges": p["num_edges"],
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(snakemake.output.summary, index=False)

    # --- results/causalchamber/grid_metrics.csv ---------------------------------
    grouped_grid = pd.read_csv(
        getattr(snakemake.input, _metrics_path_attr("coarse", "grouped"))
    )
    grouped_grid.to_csv(snakemake.output.grid_metrics, index=False)

    # --- results/causalchamber/dag.png: 4-panel grouped score DAGs --------------
    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(8 * len(METHOD_ORDER), 7))
    for ax, method in zip(axes, METHOD_ORDER):
        p = params_by_cell[(method, "grouped", "score")]
        dag = dag_from_labeled_parts(p["parts"], p["edges"], name_to_idx)
        second_knob = "β" if method == "repare" else "λ"
        second_val = p.get("lambda", p.get("beta"))
        # For CV, α was selected internally via 5-fold cv_log_lik argmax —
        # mark it with α̂ rather than α to avoid implying it was a free knob.
        alpha_sym = "α̂" if method == "cv" else "α"
        title = (
            f"{METHOD_LABEL[method]} (grouped, score-selected)\n"
            f"{alpha_sym}={p['alpha']:.0e}, {second_knob}={second_val:.0e}, "
            f"ARI={p['ari']:.3f}, F1={p['f1']:.3f}, "
            f"runtime={p['fit_time']:.2f}s"
        )
        draw_dag(dag, feature_cols, ax, title=title)
    fig.tight_layout()
    fig.savefig(snakemake.output.dag, bbox_inches="tight")
    plt.close(fig)

    # --- results/causalchamber/grid_runs/ (placeholder) -------------------------
    Path(snakemake.output.grid_dir).mkdir(parents=True, exist_ok=True)

    # --- results/causalchamber_dags.txt: 12-block text dump ---------------------
    lines = []
    for method in METHOD_ORDER:
        for mode in MODES:
            for sel in SELECTIONS:
                p = params_by_cell[(method, mode, sel)]
                second_knob = "beta" if method == "repare" else "lambda"
                second_val = p.get("lambda", p.get("beta"))
                lines.append(
                    f"=== {METHOD_LABEL[method]} ({mode}, {sel}-selected) ==="
                )
                lines.append(
                    f"  alpha={p['alpha']:.0e}, {second_knob}={second_val:.0e}, "
                    f"ari={p['ari']:.4f}, precision={p['precision']:.4f}, "
                    f"recall={p['recall']:.4f}, f1={p['f1']:.4f}, "
                    f"runtime_sec={p['fit_time']:.3f}, "
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
