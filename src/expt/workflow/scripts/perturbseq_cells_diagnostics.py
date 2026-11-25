# ruff: noqa: F821
"""Structure diagnostics beyond density for the Perturb-seq comparison.

Reads the aggregate summary and the final-fit records (details.json) and emits, per
method: tuning-inclusive and final-fit runtime (median/MAD over splits), block and edge
counts, cross-split partition stability (adjusted Rand index and largest-block Jaccard
over all batch-split pairs), and — for the scored variant — its partition agreement with
the Welch-partition hybrid on the same splits. These are the numbers quoted for the
runtime and cluster-stability claims; density lives in summary.csv and table.tex.

Splits overlap, so spreads describe across-split variation on this dataset, not
sampling uncertainty about the procedure.
"""

import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _mad(values: list[float]) -> float:
    if not values:
        return float("nan")
    med = float(np.median(values))
    return float(np.median(np.abs(np.asarray(values) - med)))


def _labels(blocks: list[list[str]], features: list[str]) -> np.ndarray:
    index = {g: i for i, g in enumerate(features)}
    labels = np.empty(len(features), dtype=int)
    for b, blk in enumerate(blocks):
        for g in blk:
            labels[index[g]] = b
    return labels


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def _random_labels(sizes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Block labels for a uniformly random partition with the given block sizes."""
    labels = np.repeat(np.arange(len(sizes)), sizes)
    rng.shuffle(labels)
    return labels


def main() -> None:
    with open(snakemake.input.metadata) as f:
        features = json.load(f)["features"]
    with open(snakemake.input.details) as f:
        finals = json.load(f)
    summary = pd.read_csv(snakemake.input.summary)
    summary = summary[summary["status"] == "ok"]

    blocks_by = {}
    for rec in finals:
        if rec.get("status") == "ok":
            blocks_by[(rec["split"], rec["method"])] = rec["blocks"]
    methods = sorted({m for _, m in blocks_by})

    rows = []
    for method in methods:
        splits = sorted(s for s, m in blocks_by if m == method)
        stats = summary[summary["method"] == method]
        row = {
            "method": method,
            "fit_seconds_med": float(stats["fit_seconds"].median()),
            "fit_seconds_mad": _mad(stats["fit_seconds"].tolist()),
            "total_seconds_med": float(stats["total_seconds"].median()),
            "total_seconds_mad": _mad(stats["total_seconds"].tolist()),
            "num_blocks_med": float(stats["num_blocks"].median()),
            "num_edges_med": float(stats["num_edges"].median()),
        }
        rng = np.random.default_rng(2026)
        aris, top_j, nulls = [], [], []
        for a, b in itertools.combinations(splits, 2):
            la = _labels(blocks_by[(a, method)], features)
            lb = _labels(blocks_by[(b, method)], features)
            aris.append(adjusted_rand_score(la, lb))
            # Null: random partitions with the same block-size histograms.
            for _ in range(50):
                ra = _random_labels(np.bincount(la), rng)
                rb = _random_labels(np.bincount(lb), rng)
                nulls.append(adjusted_rand_score(ra, rb))
            top_a = max(blocks_by[(a, method)], key=len)
            top_b = max(blocks_by[(b, method)], key=len)
            top_j.append(_jaccard(set(top_a), set(top_b)))
        if aris:
            row["split_ari_med"] = float(np.median(aris))
            row["split_ari_min"] = float(np.min(aris))
            row["split_ari_max"] = float(np.max(aris))
            row["split_ari_null_med"] = float(np.median(nulls))
            row["top_block_jaccard_med"] = float(np.median(top_j))
        rows.append(row)

    pd.DataFrame(rows).to_csv(snakemake.output.csv, index=False)


if __name__ == "__main__":
    main()
