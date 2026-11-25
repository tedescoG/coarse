# ruff: noqa: F821
"""Held-out intervention prediction for the Perturb-seq comparison.

For every (panel, split, environment): take the environment's target genes, predict the
shift set the learned model asserts -- the target genes' blocks plus, for the `dag`
variant, every descendant block (interventional soundness is exactly this claim) -- and
score precision/recall/F1 against Welch shift calls on the held-out batches at the
co-detection diagnostic's level (alpha = 0.01, copula-Gaussian scale, transform
anchored on training controls). The `partition` variant (no edges) isolates what the
learned edges add at the signature level. Trivial brackets: `target_only` (the
perturbed gene(s) alone) and `all_genes`. UT-IGSP propagates through its fine DAG over
singletons. Reads stored final fits and the preprocessed cells; recomputes only the
signature calls and cross-checks them against the stored held-out shift matrices.
Splits overlap and the calls ignore within-batch dependence, so numbers are
exploratory. Outputs: one shiftpred.csv per panel plus the results table fragment
shiftpred-table.tex (median precision / recall / F1 per asserted set and panel).
"""

import json
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from _perturbseq_common import BATCH_PAIRS, outer_train_batches
from perturbseq_cells_fit import subset

from coarse.hypothesis_tests import compute_M
from coarse.normalize import apply_inverse_normal, inverse_normal_reference

ALPHA = 0.01  # heldout_agreement's diagnostic level
PANELS = ("50", "100", "200", "500")
TABLE_ROWS = (
    ("coarse", "partition", "COARSE: blocks"),
    ("coarse", "dag", "COARSE: blocks + edges"),
    ("coarse_kpc1", "dag", "1PC-COARSE: blocks + edges"),
    ("coarse_kpc3", "dag", "3PC-COARSE: blocks + edges"),
    ("coarse_avg", "dag", "avg-COARSE: blocks + edges"),
    ("coarse_scored", "partition", "COARSE (scored): blocks"),
    ("coarse_scored", "dag", "COARSE (scored): blocks + edges"),
    ("repare", "dag", "RePaRe"),
    ("repare_welch", "dag", "RePaRe (Welch): blocks + edges"),
    ("utigsp", "dag", "UT-IGSP (fine DAG)"),
    ("target_only", "-", "Target gene(s) only"),
    ("all_genes", "-", "All genes"),
)


def _transform(train, test):
    ref = inverse_normal_reference(train["obs"])
    return (
        {k: apply_inverse_normal(ref, v) for k, v in train.items()},
        {k: apply_inverse_normal(ref, v) for k, v in test.items()},
    )


def _seed_blocks(blocks, targets):
    return {i for i, blk in enumerate(blocks) if targets & set(blk)}


def _descendant_genes(edges, blocks, seeds):
    """Genes of the seed blocks plus every block-level descendant of a seed block."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(blocks)))
    index = {tuple(sorted(b)): i for i, b in enumerate(blocks)}
    for parent, child in edges:
        graph.add_edge(index[tuple(sorted(parent))], index[tuple(sorted(child))])
    genes = set()
    for seed in seeds:
        genes |= set(blocks[seed])
        for d in nx.descendants(graph, seed):
            genes |= set(blocks[d])
    return genes


def _prf(predicted, observed):
    hit = len(predicted & observed)
    precision = hit / len(predicted) if predicted else np.nan
    recall = hit / len(observed) if observed else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
        else np.nan
    )
    return precision, recall, f1


def _panel_rows(cells_path, metadata_path, details_path, panel: int):
    """Asserted-vs-observed scoring rows for one panel."""
    with open(cells_path, "rb") as f:
        data = pickle.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)
    with open(details_path) as f:
        finals = json.load(f)
    features = np.asarray(metadata["features"])
    targets = {
        env: {g for g in env.split("_") if g} for env in metadata["environments"]
    }
    graphs = {
        (rec["split"], rec["method"]): rec
        for rec in finals
        if rec.get("status") == "ok"
    }

    rows = []
    for split in range(len(BATCH_PAIRS)):
        train = subset(data, outer_train_batches(split))
        test = subset(data, BATCH_PAIRS[split])
        train, test = _transform(train, test)
        M, env_order = compute_M(test, alpha=ALPHA)

        # Protocol check: our calls must equal the stored co-detection diagnostic's.
        for (s, method), rec in graphs.items():
            if s == split and "shift_agreement" in rec:
                stored = np.asarray(
                    rec["shift_agreement"]["heldout_shift_matrix"], dtype=bool
                )
                if stored.shape != M.shape or not np.array_equal(stored, M):
                    raise ValueError(
                        f"held-out shift calls disagree with stored matrix "
                        f"(split {split}, method {method})"
                    )

        observed = {
            env: set(features[np.flatnonzero(M[:, j])].tolist())
            for j, env in enumerate(env_order)
        }
        for env in env_order:
            t = targets.get(env, set())
            predicted_by = {
                ("target_only", "-"): set(t),
                ("all_genes", "-"): set(features),
            }
            for (s, method), rec in graphs.items():
                if s != split:
                    continue
                blocks, edges = rec["blocks"], rec["edges"]
                seeds = _seed_blocks(blocks, t)
                if method in (
                    "coarse_scored",
                    "coarse",
                    "coarse_kpc1",
                    "coarse_kpc3",
                    "coarse_avg",
                    "coarse_scored",
                    "repare",
                    "repare_welch",
                ):
                    predicted_by[(method, "partition")] = {
                        g for i in seeds for g in blocks[i]
                    }
                if method in (
                    "coarse_scored",
                    "coarse",
                    "coarse_kpc1",
                    "coarse_kpc3",
                    "coarse_avg",
                    "coarse_scored",
                    "repare",
                    "repare_welch",
                    "utigsp",
                ):
                    predicted_by[(method, "dag")] = _descendant_genes(
                        edges, blocks, seeds
                    )
            for (method, variant), predicted in predicted_by.items():
                precision, recall, f1 = _prf(predicted, observed[env])
                rows.append(
                    {
                        "panel": panel,
                        "split": split,
                        "env": env,
                        "method": method,
                        "variant": variant,
                        "n_observed": len(observed[env]),
                        "n_predicted": len(predicted),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
    return rows


def _results_table(frame: pd.DataFrame) -> str:
    """Booktabs fragment: median precision / recall / F1 per asserted set and panel."""
    lines = [
        "% auto-generated by perturbseq_cells_shiftpred.py; do not edit",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Asserted shift set & $d{=}50$ & $d{=}100$ & $d{=}200$ & $d{=}500$ \\\\",
        "\\midrule",
    ]
    for method, variant, label in TABLE_ROWS:
        cells = []
        for panel in PANELS:
            sub = frame[
                (frame["panel"] == int(panel))
                & (frame["method"] == method)
                & (frame["variant"] == variant)
            ]
            if len(sub):
                cells.append(
                    " / ".join(
                        f"{sub[column].median():.2f}"
                        for column in ("precision", "recall", "f1")
                    )
                )
            else:
                cells.append("---")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    rows = []
    for cells, metadata, details, csv, panel in zip(
        snakemake.input.cells,
        snakemake.input.metadata,
        snakemake.input.details,
        snakemake.output.csvs,
        PANELS,
        strict=True,
    ):
        panel_rows = _panel_rows(cells, metadata, details, int(panel))
        pd.DataFrame(panel_rows).to_csv(csv, index=False)
        rows.extend(panel_rows)
    frame = pd.DataFrame(rows)
    with open(snakemake.output.tex, "w") as f:
        f.write(_results_table(frame))


if __name__ == "__main__":
    main()
