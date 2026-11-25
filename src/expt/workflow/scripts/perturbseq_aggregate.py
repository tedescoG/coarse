#!/usr/bin/env python3
"""Collect per-method Perturb-seq metrics (selection grids + held-out-environment eval)
into one summary CSV."""

import pandas as pd


def main():
    frames = []
    for key, path in snakemake.input.items():
        kind = "grid" if key.endswith("_metrics") else "heldout"
        stem = key.removesuffix("_metrics").removesuffix("_heldout")
        method, mode = stem.split("_", 1)
        df = pd.read_csv(path)
        df["method"] = method
        df["mode"] = mode
        df["metric_kind"] = kind
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(snakemake.output.summary, index=False)


if __name__ == "__main__":
    main()
