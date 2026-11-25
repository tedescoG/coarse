# ruff: noqa: F821
"""Select the tuning candidate for one (split, method) from its inner-fold records.

The selected candidate index is consumed by perturbseq_cells_compare.py's "final"
stage. A method whose candidates all failed is recorded as failed, not dropped.
"""

import json
import math

from _perturbseq_common import PS_FIXED_METHODS, PS_GRIDS


def select_candidates(fold_records, method):
    """`{status, selected_index, candidates, ...}` from one split/method's fold records."""
    if method not in PS_GRIDS:
        raise ValueError(f"unknown method {method}")
    by_candidate = {c: [] for c in range(len(PS_GRIDS[method]))}
    for record in fold_records:
        if record.get("status") != "ok" and "error" not in record:
            raise ValueError(f"malformed fold record: {record}")
        by_candidate[record["candidate"]].append(record)

    candidates = []
    for index, records in sorted(by_candidate.items()):
        records = sorted(records, key=lambda r: r["fold"])
        params = list(PS_GRIDS[method][index])
        if method in PS_FIXED_METHODS:
            candidates.append(
                {
                    "index": index,
                    "parameters": params,
                    "fold_scores": [],
                    "mean_score": None,
                }
            )
            continue
        scores = [r.get("score") for r in records]
        complete = bool(scores) and all(
            s is not None and math.isfinite(s) for s in scores
        )
        candidates.append(
            {
                "index": index,
                "parameters": params,
                "fold_scores": scores,
                "mean_score": float(sum(scores) / len(scores)) if complete else None,
                "errors": [r["error"] for r in records if "error" in r],
            }
        )

    if method in PS_FIXED_METHODS:
        selected = 0
        status = "ok"
    else:
        valid = [c for c in candidates if c["mean_score"] is not None]
        if not valid:
            selected = None
            status = "failed"
        else:
            # Highest mean inner score; grid order is the deterministic tie-break.
            selected = max(valid, key=lambda c: (c["mean_score"], -c["index"]))["index"]
            status = "ok"
    return {
        "status": status,
        "selected_index": selected,
        "selected_parameters": list(PS_GRIDS[method][selected])
        if selected is not None
        else None,
        "candidates": candidates,
    }


def main():
    method = snakemake.wildcards.method
    records = []
    for path in snakemake.input.folds:
        with open(path) as f:
            records.append(json.load(f))
    selection = select_candidates(records, method)
    with open(snakemake.output.selection, "w") as f:
        json.dump(selection, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
