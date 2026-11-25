# ruff: noqa: F821
"""Collect per-(split, method) final results without hiding failed fits."""

import json

import pandas as pd


def summarize(finals, fold_records):
    rows = []
    for r in sorted(finals, key=lambda x: (x["split"], x["method"])):
        score = r.get("heldout_loglik", {})
        agreement = r.get("shift_agreement", {})
        within = agreement.get("within") or {}
        random = agreement.get("matched_random_expectation") or {}
        selection = r.get("selection", {})
        candidates = selection.get("candidates", [])
        selected = next(
            (
                c
                for c in candidates
                if c.get("index") == selection.get("selected_index")
            ),
            {},
        )
        tuning_seconds = sum(
            f.get("seconds", 0.0)
            for f in fold_records
            if f["split"] == r["split"] and f["method"] == r["method"]
        )
        rows.append(
            {
                "split": r["split"],
                "method": r["method"],
                "status": r["status"],
                "error": str((r.get("error") or {}).get("message", "")),
                "selected_parameters": json.dumps(selection.get("selected_parameters")),
                "candidates_failed": sum(
                    1 for c in candidates if c.get("mean_score") is None
                ),
                "validation_mean": selected.get("mean_score"),
                "num_blocks": len(r["blocks"]) if "blocks" in r else None,
                "num_edges": len(r["edges"]) if "edges" in r else None,
                "heldout_loglik_per_gene": score.get("mean_per_gene"),
                "same_blocks_no_edges_loglik": r.get(
                    "same_blocks_no_edges_loglik", {}
                ).get("mean_per_gene"),
                "both_detected": within.get("both_detected"),
                "both_detected_random": random.get("both_detected"),
                "both_undetected": within.get("both_undetected"),
                "both_undetected_random": random.get("both_undetected"),
                "fit_seconds": r.get("fit_seconds"),
                "tuning_seconds": tuning_seconds,
                "total_seconds": r.get("fit_seconds", 0.0) + tuning_seconds,
            }
        )
    return pd.DataFrame(rows)


def main():
    finals = []
    for path in snakemake.input.finals:
        with open(path) as f:
            finals.append(json.load(f))
    fold_records = []
    for path in snakemake.input.folds:
        with open(path) as f:
            fold_records.append(json.load(f))
    summarize(finals, fold_records).to_csv(snakemake.output.table, index=False)
    with open(snakemake.output.details, "w") as f:
        json.dump(finals, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
