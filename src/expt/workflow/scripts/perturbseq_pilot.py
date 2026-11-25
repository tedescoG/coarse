# ruff: noqa: F821
"""Exploratory batch-split reproducibility of COARSE's marginal shift partition.

Splits share batch IDs across environments.  The random baseline is the exact expected
within-block agreement under a uniformly random reassignment with the same block sizes.
Overlapping splits are dependent; their spread is descriptive, not a confidence interval.
"""

import itertools
import json
import pickle

import numpy as np
import pandas as pd

from coarse import COARSE
from coarse.hypothesis_tests import compute_M
from coarse.partition import infer_partition


def align_batches(blocks, batch_ids):
    """Align rows by common batch ID and reject duplicates or malformed inputs."""
    if set(blocks) != set(batch_ids) or "obs" not in blocks:
        raise ValueError(
            "blocks and batch IDs must cover the same environments, including obs"
        )
    common = None
    for env, X in blocks.items():
        ids = batch_ids[env]
        if len(ids) != len(X) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid batch IDs for {env}")
        common = set(ids) if common is None else common & set(ids)
    ids = sorted(common)
    if len(ids) < 8:
        raise ValueError(f"need at least 8 common batches; found {len(ids)}")
    aligned = {}
    for env, X in blocks.items():
        positions = {batch: i for i, batch in enumerate(batch_ids[env])}
        aligned[env] = np.asarray(X)[[positions[batch] for batch in ids]]
    return aligned, ids


def agreement(partition, heldout):
    """Within-block match and exact matched-size random-partition expectation.

    Singleton-only partitions have no within-block pairs. One-block partitions give
    no contrast with the random baseline. Return None for their non-informative contrast.
    """
    p = heldout.shape[0]
    pairs = list(itertools.combinations(range(p), 2))
    scores = {(i, j): float(np.mean(heldout[i] == heldout[j])) for i, j in pairs}
    within = [
        scores[tuple(sorted((i, j)))]
        for block in partition
        for i, j in itertools.combinations(sorted(block), 2)
    ]
    if not within or len(within) == len(pairs):
        return None, None, None
    observed = float(np.mean(within))
    baseline = float(np.mean(list(scores.values())))
    return observed, baseline, observed - baseline


def run_pilot(blocks, batch_ids, features, alphas):
    aligned, ids = align_batches(blocks, batch_ids)
    if len(ids) != 8:
        raise ValueError(
            f"pilot requires exactly eight common batches, found {len(ids)}"
        )
    if any(
        x.ndim != 2 or x.shape[1] != len(features) or not np.isfinite(x).all()
        for x in aligned.values()
    ):
        raise ValueError("non-finite or misaligned expression matrix")
    rows = []
    for alpha in alphas:
        for train_idx in itertools.combinations(range(8), 4):
            train_set = set(train_idx)
            test_idx = [i for i in range(8) if i not in train_set]
            train = {env: X[list(train_idx)] for env, X in aligned.items()}
            test = {env: X[test_idx] for env, X in aligned.items()}
            M_train, env_order = compute_M(train, alpha=alpha, test_name="welch")
            M_test, test_order = compute_M(test, alpha=alpha, test_name="welch")
            if env_order != test_order:
                raise ValueError("environment order differs between splits")
            parts = infer_partition(M_train)
            identical_test = bool(np.all(M_test == M_test[0]))
            obs, null, delta = agreement(parts, M_test)
            if identical_test:
                # An identical held-out signature for every gene cannot discriminate partitions.
                obs, null, delta = None, None, None
            rows.append(
                {
                    "alpha": float(alpha),
                    "train_batches": ",".join(str(ids[i]) for i in train_idx),
                    "test_batches": ",".join(str(ids[i]) for i in test_idx),
                    "blocks": len(parts),
                    "heldout_shift_fraction": float(M_test.mean()),
                    "heldout_all_zero": bool(not M_test.any()),
                    "heldout_all_identical": identical_test,
                    "agreement": obs,
                    "random_agreement": null,
                    "difference": delta,
                }
            )
    full = {}
    for alpha in alphas:
        model = COARSE().fit(aligned, alpha=alpha, refine_test="welch")
        full[str(alpha)] = {
            "environment_order": list(model.env_order),
            "shift_matrix": model.M.astype(int).tolist(),
            "blocks": [[features[i] for i in node] for node in model.dag.nodes],
            "edges": [
                [[features[i] for i in u], [features[i] for i in v]]
                for u, v in model.dag.edges
            ],
            "score_finite": bool(np.isfinite(model.score)),
        }
    summary = {
        "scope": "exploratory; overlapping splits are dependent; not a confidence interval",
        "common_batches": ids,
        "features": features,
        "full_data_models": full,
        "per_alpha": {},
    }
    for alpha in alphas:
        group = [r for r in rows if r["alpha"] == float(alpha)]
        valid = [r["difference"] for r in group if r["difference"] is not None]
        summary["per_alpha"][str(alpha)] = {
            "splits": len(group),
            "informative_partitions": len(valid),
            "all_zero_test_signatures": sum(r["heldout_all_zero"] for r in group),
            "all_identical_test_signatures": sum(
                r["heldout_all_identical"] for r in group
            ),
            "mean_difference": float(np.mean(valid)) if valid else None,
        }
    return rows, summary


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.batch_ids) as f:
        batch_ids = json.load(f)
    with open(snakemake.input.features) as f:
        features = json.load(f)["feature_cols"]
    rows, summary = run_pilot(blocks, batch_ids, features, snakemake.params.alphas)
    with open(snakemake.input.reference_edges) as f:
        summary["collectri_edges_unmodified"] = json.load(f)
    pd.DataFrame(rows).to_csv(snakemake.output.splits, index=False)
    with open(snakemake.output.summary, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
