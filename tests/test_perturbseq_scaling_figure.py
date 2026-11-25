"""Tests for the scaling-figure script: ladder frame building and PDF rendering."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

SCRIPTS = str(Path(__file__).resolve().parents[1] / "src/expt/workflow/scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

figure = importlib.import_module("perturbseq_scaling_figure")


def write_summary(path, panel):
    rng = np.random.default_rng(panel)
    rows = []
    for method, base in (("coarse", -1.40), ("coarse_kpc3", -1.38), ("utigsp", -1.35)):
        for split in range(4):
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "status": "ok",
                    "heldout_loglik_per_gene": base + rng.normal(0.0, 0.01),
                    "fit_seconds": 5.0 + panel / 100.0,
                    "tuning_seconds": 10.0 + panel / 50.0,
                    "total_seconds": 15.0 + panel * 3.0 / 100.0,
                }
            )
    rows.append(
        {
            "method": "coarse_kpc3",
            "split": 9,
            "status": "failed",
            "heldout_loglik_per_gene": -99.0,
            "fit_seconds": 1.0,
            "tuning_seconds": 1.0,
            "total_seconds": 2.0,
        }
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_figure_renders_from_summaries_and_drops_failed(tmp_path):
    panels = ["50", "200"]
    summaries = []
    for panel in panels:
        path = tmp_path / f"summary-{panel}.csv"
        write_summary(path, int(panel))
        summaries.append(str(path))
    out = tmp_path / "figure.pdf"
    figure.snakemake = SimpleNamespace(
        input=SimpleNamespace(summaries=summaries),
        params=SimpleNamespace(panels=panels),
        output=SimpleNamespace(figure=str(out)),
    )
    figure.main()
    assert out.exists() and out.stat().st_size > 0


def test_ladder_frame_filters_status_and_restricts_methods(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary(path, 50)
    frame = figure.ladder_frame(path, "50")
    assert set(frame["status"]) == {"ok"}
    assert set(frame["method"]) <= set(figure.METHODS)
    assert (frame["panel"] == 50).all()
