"""CausalChamber experiment — COARSE / RePaRe / GIES / GnIES / UT-IGSP on the
light tunnel.

Superset of `repare-0.2.0/src/expt/workflow/rules/causalchamber.smk`: same
dataset, ground truth and baseline calls, plus COARSE, and every method run on
both intervention regimes (grouped: obs + rgb + pol; ungrouped: obs +
single-variable experiments). One α grid (`ALPHAS`) is shared by every
α-tuned method: COARSE's CV / oracle, RePaRe (× β) and UT-IGSP (× α_inv).
No model pickles are persisted — per-fit outputs are metrics.csv /
*_params.json / *_dag.png, and the aggregator works from JSON without
importing `repare`.

Per-method directory contract (read by causalchamber_aggregate.py):
    results/causalchamber/<method>_<mode>/{metrics.csv, score_params.json, score_dag.png}
    + {oracle_params.json, oracle_dag.png} for methods with a hyperparameter grid.
"""

from snakemake.io import directory

ALPHAS = [1e-4, 1e-3, 1e-2, 0.1]
BETAS = [1e-4, 1e-3, 1e-2, 0.1]
# λ is never swept: BIC is not comparable across λ, so COARSE runs at the
# standard BIC penalty and only α is tuned (score: 10-fold CV; oracle: best
# grid cell against ground truth) — the same one-threshold design as RePaRe.
FIXED_LAMBDA = 1.0
CV_N_FOLDS = 10
# Stage-1 test: Gaussian mean + variance LRT, the test RePaRe's `assume="gaussian"`
# path runs (its `refine_test="ks"` argument is inert under that assumption).
COARSE_REFINE_TEST = "gaussian_lrt"
# GnIES: per-env row cap inherited from RePaRe's experiment. Override for a
# quick wiring check with `--config gnies_max_rows=200` (the rule re-runs when
# the param changes, so the real run is not contaminated).
GNIES_MAX_ROWS = int(config.get("gnies_max_rows", 2000))

DATASET = "lt_interventions_standard_v1"
CHAMBER = "lt"
CONFIGURATION = "standard"

BASE = "results/causalchamber/"

PREP_FILES = {
    "blocks": BASE + "preprocessed/blocks.pkl",
    "features": BASE + "preprocessed/features.json",
    "partition": BASE + "preprocessed/partition_parts.pkl",
    "grouptargets": BASE + "preprocessed/grouptargets.pkl",
    "truegraph": BASE + "preprocessed/truegraph.pkl",
    "truelabels": BASE + "preprocessed/truelabels.pkl",
    "truedagfull": BASE + "preprocessed/truedagfull.pkl",
    "nametoidx": BASE + "preprocessed/nametoidx.pkl",
    "singleenvlabels": BASE + "preprocessed/singleenvlabels.pkl",
    "singleenvtargets": BASE + "preprocessed/singleenvtargets.pkl",
    "singleenvdata": BASE + "preprocessed/singleenvdata.pkl",
}

CC_MODES = ["grouped", "ungrouped"]
# Methods with a hyperparameter grid emit an oracle row; GIES / GnIES do not.
CC_METHOD_SELECTIONS = {
    "coarse": ["score", "oracle"],
    "repare": ["score", "oracle"],
    "gies": ["score"],
    "gnies": ["score"],
    "utigsp": ["score", "oracle"],
}


def cc_outputs_for(method, selections):
    out = {"metrics_csv": BASE + method + "_{mode}/metrics.csv"}
    for sel in selections:
        out[f"{sel}_dag"] = BASE + method + "_{mode}/" + sel + "_dag.png"
        out[f"{sel}_params"] = BASE + method + "_{mode}/" + sel + "_params.json"
    return out


rule causalchamber_prepare:
    output:
        **PREP_FILES,
    params:
        dataset=DATASET,
        chamber=CHAMBER,
        configuration=CONFIGURATION,
        root="data/causalchamber",
    script:
        "../scripts/causalchamber_prepare.py"


# ---------------------------------------------------------------------------
# COARSE — α grid at λ=1: oracle row = best cell vs ground truth; score row =
# one COARSECV fit (α̂ by K-fold held-out likelihood over the same grid).
# ---------------------------------------------------------------------------


rule causalchamber_coarse:
    input:
        **PREP_FILES,
    output:
        **cc_outputs_for("coarse", CC_METHOD_SELECTIONS["coarse"]),
    params:
        alphas=ALPHAS,
        lambda_pen=FIXED_LAMBDA,
        n_folds=CV_N_FOLDS,
        refine_test=COARSE_REFINE_TEST,
        mode="{mode}",
    script:
        "../scripts/causalchamber_coarse.py"


# ---------------------------------------------------------------------------
# RePaRe — α × β grid, native GnIES-on-expanded score selection.
# ---------------------------------------------------------------------------


rule causalchamber_repare:
    input:
        **PREP_FILES,
    output:
        **cc_outputs_for("repare", CC_METHOD_SELECTIONS["repare"]),
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="{mode}",
    script:
        "../scripts/causalchamber_repare.py"


# ---------------------------------------------------------------------------
# Baselines from RePaRe's experiment — GIES (known targets), GnIES and UT-IGSP
# (unknown targets). Calls are identical to repare-0.2.0's scripts.
# ---------------------------------------------------------------------------


rule causalchamber_gies:
    input:
        **PREP_FILES,
    output:
        **cc_outputs_for("gies", CC_METHOD_SELECTIONS["gies"]),
    params:
        mode="{mode}",
    script:
        "../scripts/causalchamber_gies.py"


rule causalchamber_gnies:
    input:
        **PREP_FILES,
    output:
        **cc_outputs_for("gnies", CC_METHOD_SELECTIONS["gnies"]),
    params:
        mode="{mode}",
        max_rows=GNIES_MAX_ROWS,
    script:
        "../scripts/causalchamber_gnies.py"


rule causalchamber_utigsp:
    input:
        **PREP_FILES,
    output:
        **cc_outputs_for("utigsp", CC_METHOD_SELECTIONS["utigsp"]),
    params:
        mode="{mode}",
        alpha_ci=ALPHAS,
        alpha_inv=ALPHAS,
    script:
        "../scripts/causalchamber_utigsp.py"


# ---------------------------------------------------------------------------
# Aggregator — produces the headline artifacts. Input keys follow the
# `{method}_{mode}_metrics` / `{method}_{mode}_{selection}_params` naming the
# aggregator resolves via getattr(snakemake.input, ...).
# ---------------------------------------------------------------------------


def cc_aggregate_inputs():
    inputs = {
        "features": PREP_FILES["features"],
        "nametoidx": PREP_FILES["nametoidx"],
    }
    for method, selections in CC_METHOD_SELECTIONS.items():
        for mode in CC_MODES:
            inputs[f"{method}_{mode}_metrics"] = BASE + f"{method}_{mode}/metrics.csv"
            for sel in selections:
                inputs[f"{method}_{mode}_{sel}_params"] = (
                    BASE + f"{method}_{mode}/{sel}_params.json"
                )
    return inputs


rule causalchamber_aggregate:
    input:
        **cc_aggregate_inputs(),
    output:
        grid_metrics=BASE + "grid_metrics.csv",
        dag=BASE + "dag.png",
        grid_dir=directory(BASE + "grid_runs"),
        summary="results/causalchamber_summary.csv",
        dag_summary="results/causalchamber_dags.txt",
    params:
        method_selections=CC_METHOD_SELECTIONS,
        modes=CC_MODES,
    script:
        "../scripts/causalchamber_aggregate.py"


# ---------------------------------------------------------------------------
# Top-level entry point — mirrors the `lambda_all` / `l2_all` convention.
# ---------------------------------------------------------------------------

cc_outputs = [
    "results/causalchamber/dag.png",
    "results/causalchamber/grid_metrics.csv",
    "results/causalchamber_summary.csv",
    "results/causalchamber_dags.txt",
]


rule causalchamber:
    input:
        cc_outputs,
