# Perturb-seq real-data benchmark: cell-level comparison of the constraint-based
# COARSE (Welch-partition + block-BIC) and its score-phase compressions (1PC, 3PC,
# avg), RePaRe, UT-IGSP and fixed controls on Norman et al. (2019) CRISPRa, with
# held-out density, signature co-detection, CollecTRI support, cross-split stability,
# intervention prediction and runtime diagnostics, and the panel-size scaling ladder.
# Deliberately NOT in `rule all`: prepare downloads external data, fits are CPU-heavy.
# Ladder + figure: snakemake results/perturbseq-scaling-figure.pdf
# (builds every missing ladder leg); brace targets address individual panels,
# e.g. `snakemake results/perturbseq-cells-compare-{200,500}/summary.csv`.

import os
import sys


# One source of truth for grids/batches: the shared helper module, imported at parse time.
sys.path.insert(0, os.path.join(workflow.basedir, "scripts"))
from _perturbseq_common import BATCH_PAIRS, PS_FIXED_METHODS, PS_GRIDS  # noqa: E402

ALPHAS = [1e-4, 1e-3, 1e-2, 0.1]
BETAS = [1e-4, 1e-3, 1e-2, 0.1]
FIXED_LAMBDA = 1.0
# The smoke dataset has eight gemgroup rows per environment.
PS_CV_N_FOLDS = 8 if bool(int(config.get("smoke", 0))) else 10
# scRNA counts violate the Gaussian-LRT model: mean-shift (Welch) only.
COARSE_REFINE_TEST = "welch"
# Held-out-environment folds for the ground-truth-free eval.
N_ENV_FOLDS = 5

DATASET = "norman2019"

# `--config smoke=1`: Norman 2019 smoke subset (4 single-gene envs x 10 genes); outputs
# under results/perturbseq-smoke/.
SMOKE = bool(int(config.get("smoke", 0)))
BASE = "results/perturbseq-smoke/" if SMOKE else "results/perturbseq/"

PREP_FILES = {
    "blocks": BASE + "preprocessed/blocks.pkl",
    "batch_ids": BASE + "preprocessed/batch_ids.json",
    "reference_edges": BASE + "preprocessed/reference_edges.json",
    "cells": BASE + "preprocessed/cells.pkl",
    "features": BASE + "preprocessed/features.json",
    "partition": BASE + "preprocessed/partition_parts.pkl",
    "grouptargets": BASE + "preprocessed/grouptargets.pkl",
    "truegraph": BASE + "preprocessed/truegraph.pkl",
    "truedagfull": BASE + "preprocessed/truedagfull.pkl",
    "nametoidx": BASE + "preprocessed/nametoidx.pkl",
    "singleenvlabels": BASE + "preprocessed/singleenvlabels.pkl",
}

PS_COMMON = [
    "workflow/scripts/_perturbseq_common.py",
    "workflow/scripts/_causalchamber_common.py",
    "workflow/scripts/_common.py",
]

PS_MODES = ["grouped", "ungrouped"]
PS_METHOD_MODES = {
    "coarse": PS_MODES,
    "repare": PS_MODES,
    "utigsp": ["ungrouped"],
}
# Silver (CollecTRI) metrics go into every metrics.csv row but never pick a row:
# each method emits its ground-truth-free score row only.
PS_METHOD_SELECTIONS = ["score"]


def ps_outputs_for(method, mode="{mode}"):
    out = {"metrics_csv": BASE + method + f"_{mode}/metrics.csv"}
    for sel in PS_METHOD_SELECTIONS:
        out[f"{sel}_dag"] = BASE + method + f"_{mode}/" + sel + "_dag.png"
        out[f"{sel}_params"] = BASE + method + f"_{mode}/" + sel + "_params.json"
    return out


rule perturbseq_prepare:
    output:
        **PREP_FILES,
    params:
        dataset=DATASET,
        root="data/perturbseq",
        smoke=SMOKE,
    script:
        "../scripts/perturbseq_prepare.py"


rule perturbseq_coarse:
    input:
        **PREP_FILES,
        common=PS_COMMON,
    output:
        **ps_outputs_for("coarse"),
    params:
        alphas=ALPHAS,
        lambda_pen=FIXED_LAMBDA,
        n_folds=PS_CV_N_FOLDS,
        refine_test=COARSE_REFINE_TEST,
        mode="{mode}",
    script:
        "../scripts/perturbseq_coarse.py"


rule perturbseq_repare:
    input:
        **PREP_FILES,
        common=PS_COMMON,
    output:
        **ps_outputs_for("repare"),
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="{mode}",
    script:
        "../scripts/perturbseq_repare.py"


rule perturbseq_utigsp:
    input:
        **PREP_FILES,
        common=PS_COMMON,
        ut_igsp="workflow/scripts/ut_igsp.py",
    output:
        **ps_outputs_for("utigsp", "ungrouped"),
    params:
        alpha_ci=ALPHAS,
        alpha_inv=ALPHAS,
    script:
        "../scripts/perturbseq_utigsp.py"


rule perturbseq_eval:
    input:
        **PREP_FILES,
        params=BASE + "{method}_{mode}/score_params.json",
        common=PS_COMMON,
    output:
        metrics=BASE + "{method}_{mode}/heldout_metrics.csv",
    wildcard_constraints:
        method="coarse|repare|utigsp",
        mode="grouped|ungrouped",
    params:
        n_env_folds=N_ENV_FOLDS,
    script:
        "../scripts/perturbseq_eval.py"


def ps_aggregate_inputs():
    inputs = {}
    for method, modes in PS_METHOD_MODES.items():
        for mode in modes:
            inputs[f"{method}_{mode}_metrics"] = BASE + f"{method}_{mode}/metrics.csv"
            inputs[f"{method}_{mode}_heldout"] = (
                BASE + f"{method}_{mode}/heldout_metrics.csv"
            )
    return inputs


rule perturbseq_aggregate:
    input:
        **ps_aggregate_inputs(),
        common=PS_COMMON,
    output:
        summary=f"results/perturbseq{'-smoke' if SMOKE else ''}_summary.csv",
    script:
        "../scripts/perturbseq_aggregate.py"


rule perturbseq_pilot:
    input:
        blocks=PREP_FILES["blocks"],
        batch_ids=PREP_FILES["batch_ids"],
        reference_edges=PREP_FILES["reference_edges"],
        features=PREP_FILES["features"],
    output:
        splits=BASE + "pilot/splits.csv",
        summary=BASE + "pilot/summary.json",
    params:
        alphas=ALPHAS,
    script:
        "../scripts/perturbseq_pilot.py"


rule perturbseq:
    input:
        rules.perturbseq_pilot.output if SMOKE else rules.perturbseq_aggregate.output.summary,


# Separate cell-level analysis; never part of the default experiment targets. The
# panel size is a wildcard, so the scaling ladder is one DAG: brace targets build any
# set of panels, and the scaling figure consumes every ladder leg.
PS_PANELS = "22|40|50|100|200|500|1000"
PS_LADDER_PANELS = ["50", "100", "200", "500"]
PS_CELL_SPLITS = ["0", "1", "2", "3"]
PS_CELL_METHODS = [
    "coarse",
    "coarse_kpc1",
    "coarse_kpc3",
    "coarse_avg",
    "coarse_scored",
    "repare",
    "repare_welch",
    "utigsp",
    "singleton_empty",
    "joint",
]
# gnies ships only on the 40-gene CollecTRI-guided panel.
PS_GNIES_PANEL = "40"
# Presentation row order for the results table; any method without fits is dropped by
# the table script itself.
PS_TABLE_METHODS = [
    "coarse",
    "coarse_kpc1",
    "coarse_kpc3",
    "coarse_avg",
    "coarse_scored",
    "repare",
    "repare_welch",
    "utigsp",
    "gnies",
    "singleton_empty",
    "joint",
]
PS_N_INNER_FOLDS = len(BATCH_PAIRS) - 1
PS_COMPARE_HELPERS = [
    "workflow/scripts/perturbseq_cells_fit.py",
    "workflow/scripts/_perturbseq_common.py",
    "workflow/scripts/ut_igsp.py",
]


def ps_compare_methods(panel):
    """Methods requested for one panel; `compare_methods` filters, gnies is 40-only."""
    methods = list(PS_CELL_METHODS)
    if panel == PS_GNIES_PANEL:
        methods.append("gnies")
    requested = str(config.get("compare_methods", "")).split()
    return [m for m in methods if not requested or m in requested]


def ps_fold_inputs(wildcards):
    """Inner-fold records feeding selection; fixed baselines have none."""
    if wildcards.method in PS_FIXED_METHODS:
        return []
    return expand(
        "results/perturbseq-cells-compare-{panel}/split_{split}/{method}_c{candidate}_v{fold}.json",
        panel=[wildcards.panel],
        split=[wildcards.split],
        method=[wildcards.method],
        candidate=range(len(PS_GRIDS[wildcards.method])),
        fold=range(PS_N_INNER_FOLDS),
    )


def ps_all_fold_inputs(wildcards):
    """Every fold record for the requested methods; grids vary per method."""
    files = []
    for split in PS_CELL_SPLITS:
        for method in ps_compare_methods(wildcards.panel):
            if method in PS_FIXED_METHODS:
                continue
            files += expand(
                "results/perturbseq-cells-compare-{panel}/split_{split}/{method}_c{candidate}_v{fold}.json",
                panel=[wildcards.panel],
                split=[split],
                method=[method],
                candidate=range(len(PS_GRIDS[method])),
                fold=range(PS_N_INNER_FOLDS),
            )
    return files


rule perturbseq_cells_prepare:
    input:
        common=[
            "workflow/scripts/_perturbseq_common.py",
            "workflow/scripts/perturbseq_prepare.py",
        ],
    output:
        cells="results/perturbseq-cells-{panel}/preprocessed/cells.pkl",
        metadata="results/perturbseq-cells-{panel}/preprocessed/metadata.json",
    wildcard_constraints:
        panel=PS_PANELS,
    params:
        root="data/perturbseq",
        panel_size=lambda wildcards: int(wildcards.panel),
    script:
        "../scripts/perturbseq_cells_prepare.py"


rule perturbseq_cells_fold:
    input:
        cells=rules.perturbseq_cells_prepare.output.cells,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
        helpers=[
            "workflow/scripts/perturbseq_cells_fit.py",
            "workflow/scripts/_perturbseq_common.py",
            "workflow/scripts/ut_igsp.py",
        ],
    output:
        result="results/perturbseq-cells-compare-{panel}/split_{split}/{method}_c{candidate}_v{fold}.json",
    wildcard_constraints:
        panel=PS_PANELS,
        split="0|1|2|3",
        candidate=r"\d+",
        fold=r"\d+",
        method="coarse|coarse_kpc1|coarse_kpc3|coarse_avg|coarse_scored|repare|repare_welch|utigsp|gnies",
    params:
        stage="fold",
    script:
        "../scripts/perturbseq_cells_compare.py"


rule perturbseq_cells_select:
    input:
        cells=rules.perturbseq_cells_prepare.output.cells,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
        folds=ps_fold_inputs,
        helpers=PS_COMPARE_HELPERS,
    output:
        selection="results/perturbseq-cells-compare-{panel}/split_{split}/{method}_selection.json",
    wildcard_constraints:
        panel=PS_PANELS,
        split="0|1|2|3",
        method="coarse|coarse_kpc1|coarse_kpc3|coarse_avg|coarse_scored|repare|repare_welch|utigsp|gnies|singleton_empty|joint",
    script:
        "../scripts/perturbseq_cells_select.py"


rule perturbseq_cells_final:
    input:
        cells=rules.perturbseq_cells_prepare.output.cells,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
        selection="results/perturbseq-cells-compare-{panel}/split_{split}/{method}_selection.json",
        helpers=PS_COMPARE_HELPERS,
    output:
        result="results/perturbseq-cells-compare-{panel}/split_{split}/{method}_final.json",
    wildcard_constraints:
        panel=PS_PANELS,
        split="0|1|2|3",
        method="coarse|coarse_kpc1|coarse_kpc3|coarse_avg|coarse_scored|repare|repare_welch|utigsp|gnies|singleton_empty|joint",
    params:
        stage="final",
    script:
        "../scripts/perturbseq_cells_compare.py"


rule perturbseq_cells_compare_aggregate:
    input:
        finals=lambda wildcards: expand(
            "results/perturbseq-cells-compare-{panel}/split_{split}/{method}_final.json",
            panel=[wildcards.panel],
            split=PS_CELL_SPLITS,
            method=ps_compare_methods(wildcards.panel),
        ),
        folds=ps_all_fold_inputs,
    output:
        table="results/perturbseq-cells-compare-{panel}/summary.csv",
        details="results/perturbseq-cells-compare-{panel}/details.json",
    wildcard_constraints:
        panel=PS_PANELS,
    script:
        "../scripts/perturbseq_cells_compare_aggregate.py"


rule perturbseq_cells_table:
    input:
        summary=rules.perturbseq_cells_compare_aggregate.output.table,
        finals=lambda wildcards: expand(
            "results/perturbseq-cells-compare-{panel}/split_{split}/{method}_final.json",
            panel=[wildcards.panel],
            split=PS_CELL_SPLITS,
            method=ps_compare_methods(wildcards.panel),
        ),
        cells=rules.perturbseq_cells_prepare.output.cells,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
        helpers=PS_COMPARE_HELPERS,
    output:
        csv="results/perturbseq-cells-compare-{panel}/table.csv",
        tex="results/perturbseq-cells-compare-{panel}/table.tex",
    wildcard_constraints:
        panel=PS_PANELS,
    params:
        methods=lambda wildcards: [
            m for m in PS_TABLE_METHODS if m in ps_compare_methods(wildcards.panel)
        ],
    script:
        "../scripts/perturbseq_cells_table.py"


rule perturbseq_cells_diagnostics:
    input:
        summary=rules.perturbseq_cells_compare_aggregate.output.table,
        details=rules.perturbseq_cells_compare_aggregate.output.details,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
    output:
        csv="results/perturbseq-cells-compare-{panel}/diagnostics.csv",
    wildcard_constraints:
        panel=PS_PANELS,
    script:
        "../scripts/perturbseq_cells_diagnostics.py"


rule perturbseq_cells_shiftpred:
    input:
        cells=[
            f"results/perturbseq-cells-{p}/preprocessed/cells.pkl"
            for p in PS_LADDER_PANELS
        ],
        metadata=[
            f"results/perturbseq-cells-{p}/preprocessed/metadata.json"
            for p in PS_LADDER_PANELS
        ],
        details=expand(
            "results/perturbseq-cells-compare-{panel}/details.json",
            panel=PS_LADDER_PANELS,
        ),
        helpers=PS_COMPARE_HELPERS,
    output:
        csvs=[
            f"results/perturbseq-cells-compare-{p}/shiftpred.csv"
            for p in PS_LADDER_PANELS
        ],
        tex="results/perturbseq-shiftpred-table.tex",
    script:
        "../scripts/perturbseq_cells_shiftpred.py"


rule perturbseq_utigsp_grid:
    input:
        cells=rules.perturbseq_cells_prepare.output.cells,
        metadata=rules.perturbseq_cells_prepare.output.metadata,
        selections=expand(
            "results/perturbseq-cells-compare-{panel}/split_{split}/utigsp_selection.json",
            panel=["{panel}"],
            split=PS_CELL_SPLITS,
        ),
        finals=expand(
            "results/perturbseq-cells-compare-{panel}/split_{split}/utigsp_final.json",
            panel=["{panel}"],
            split=PS_CELL_SPLITS,
        ),
        helpers=PS_COMPARE_HELPERS,
    output:
        csv="results/perturbseq-cells-compare-{panel}/utigsp_grid.csv",
    wildcard_constraints:
        panel=PS_PANELS,
    script:
        "../scripts/perturbseq_utigsp_grid.py"


rule perturbseq_utigsp_grid_figure:
    input:
        grids=expand(
            "results/perturbseq-cells-compare-{panel}/utigsp_grid.csv",
            panel=PS_LADDER_PANELS,
        ),
        shiftpred=expand(
            "results/perturbseq-cells-compare-{panel}/shiftpred.csv",
            panel=PS_LADDER_PANELS,
        ),
        script="workflow/scripts/perturbseq_utigsp_grid_figure.py",
    output:
        figure="results/perturbseq-utigsp-grid-figure.pdf",
    params:
        panels=PS_LADDER_PANELS,
    script:
        "../scripts/perturbseq_utigsp_grid_figure.py"


rule perturbseq_cells_compare:
    input:
        rules.perturbseq_cells_compare_aggregate.output,
        rules.perturbseq_cells_table.output,
        rules.perturbseq_cells_diagnostics.output,
        rules.perturbseq_cells_shiftpred.output,
        rules.perturbseq_utigsp_grid.output,

# Density and tuning-inclusive runtime as the panel dimension grows: one figure from
# every ladder leg's aggregate. Building the figure builds any missing ladder legs.
rule perturbseq_scaling_figure:
    input:
        summaries=expand(
            "results/perturbseq-cells-compare-{panel}/summary.csv",
            panel=PS_LADDER_PANELS,
        ),
        script="workflow/scripts/perturbseq_scaling_figure.py",
    output:
        figure="results/perturbseq-scaling-figure.pdf",
    params:
        panels=PS_LADDER_PANELS,
    script:
        "../scripts/perturbseq_scaling_figure.py"
