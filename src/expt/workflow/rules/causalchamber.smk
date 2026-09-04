"""CausalChamber experiment — COARSE / COARSE-1PC / RePaRe on the light tunnel.

Mirrors `repare-0.2.0/src/expt/workflow/rules/causalchamber.smk` but swaps the
method set (drops GIES/GnIES/UT-IGSP, adds COARSE and COARSE-1PC) and persists
no model pickles. The preprocessed dataset still uses .pkl, but per-fit outputs
are only metrics.csv / params.json / dag.png — the aggregator works from JSON
and never imports `repare`.
"""

from snakemake.io import directory

ALPHAS = [1e-4, 1e-3, 1e-2, 0.1]
LAMBDAS = [0.5, 1.0, 2.0, 4.0]
BETAS = [1e-4, 1e-3, 1e-2, 0.1]

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


rule causalchamber_prepare:
    output:
        blocks=PREP_FILES["blocks"],
        features=PREP_FILES["features"],
        partition=PREP_FILES["partition"],
        grouptargets=PREP_FILES["grouptargets"],
        truegraph=PREP_FILES["truegraph"],
        truelabels=PREP_FILES["truelabels"],
        truedagfull=PREP_FILES["truedagfull"],
        nametoidx=PREP_FILES["nametoidx"],
        singleenvlabels=PREP_FILES["singleenvlabels"],
        singleenvtargets=PREP_FILES["singleenvtargets"],
        singleenvdata=PREP_FILES["singleenvdata"],
    params:
        dataset=DATASET,
        chamber=CHAMBER,
        configuration=CONFIGURATION,
        root="data/causalchamber",
    script:
        "../scripts/causalchamber_prepare.py"


# ---------------------------------------------------------------------------
# COARSE — α × λ grid, soft interventions, no PCA on parent blocks.
# ---------------------------------------------------------------------------


rule causalchamber_coarse_grouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "coarse_grouped/metrics.csv",
        score_dag=BASE + "coarse_grouped/score_dag.png",
        oracle_dag=BASE + "coarse_grouped/oracle_dag.png",
        score_params=BASE + "coarse_grouped/score_params.json",
        oracle_params=BASE + "coarse_grouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        lambdas=LAMBDAS,
        mode="grouped",
        k="None",
    script:
        "../scripts/causalchamber_coarse.py"


rule causalchamber_coarse_ungrouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "coarse_ungrouped/metrics.csv",
        score_dag=BASE + "coarse_ungrouped/score_dag.png",
        oracle_dag=BASE + "coarse_ungrouped/oracle_dag.png",
        score_params=BASE + "coarse_ungrouped/score_params.json",
        oracle_params=BASE + "coarse_ungrouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        lambdas=LAMBDAS,
        mode="ungrouped",
        k="None",
    script:
        "../scripts/causalchamber_coarse.py"


# ---------------------------------------------------------------------------
# COARSE-1PC — same α × λ grid, k=1.
# ---------------------------------------------------------------------------


rule causalchamber_onepc_grouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "onepc_grouped/metrics.csv",
        score_dag=BASE + "onepc_grouped/score_dag.png",
        oracle_dag=BASE + "onepc_grouped/oracle_dag.png",
        score_params=BASE + "onepc_grouped/score_params.json",
        oracle_params=BASE + "onepc_grouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        lambdas=LAMBDAS,
        mode="grouped",
        k=1,
    script:
        "../scripts/causalchamber_coarse.py"


rule causalchamber_onepc_ungrouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "onepc_ungrouped/metrics.csv",
        score_dag=BASE + "onepc_ungrouped/score_dag.png",
        oracle_dag=BASE + "onepc_ungrouped/oracle_dag.png",
        score_params=BASE + "onepc_ungrouped/score_params.json",
        oracle_params=BASE + "onepc_ungrouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        lambdas=LAMBDAS,
        mode="ungrouped",
        k=1,
    script:
        "../scripts/causalchamber_coarse.py"


# ---------------------------------------------------------------------------
# COARSE-CV — λ-only outer sweep; α selected internally by 10-fold CV from
# DEFAULT_ALPHA_GRID (coarse.cv.DEFAULT_ALPHA_GRID). k=None.
# ---------------------------------------------------------------------------

CV_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 0.05, 0.1]
CV_N_FOLDS = 10


rule causalchamber_cv_grouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "cv_grouped/metrics.csv",
        score_dag=BASE + "cv_grouped/score_dag.png",
        oracle_dag=BASE + "cv_grouped/oracle_dag.png",
        score_params=BASE + "cv_grouped/score_params.json",
        oracle_params=BASE + "cv_grouped/oracle_params.json",
    params:
        lambdas=LAMBDAS,
        alpha_grid=CV_ALPHA_GRID,
        n_folds=CV_N_FOLDS,
        mode="grouped",
    script:
        "../scripts/causalchamber_cv.py"


rule causalchamber_cv_ungrouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "cv_ungrouped/metrics.csv",
        score_dag=BASE + "cv_ungrouped/score_dag.png",
        oracle_dag=BASE + "cv_ungrouped/oracle_dag.png",
        score_params=BASE + "cv_ungrouped/score_params.json",
        oracle_params=BASE + "cv_ungrouped/oracle_params.json",
    params:
        lambdas=LAMBDAS,
        alpha_grid=CV_ALPHA_GRID,
        n_folds=CV_N_FOLDS,
        mode="ungrouped",
    script:
        "../scripts/causalchamber_cv.py"


# ---------------------------------------------------------------------------
# RePaRe — α × β grid, native GnIES-on-expanded score selection.
# ---------------------------------------------------------------------------


rule causalchamber_repare_grouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "repare_grouped/metrics.csv",
        score_dag=BASE + "repare_grouped/score_dag.png",
        oracle_dag=BASE + "repare_grouped/oracle_dag.png",
        score_params=BASE + "repare_grouped/score_params.json",
        oracle_params=BASE + "repare_grouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="grouped",
    script:
        "../scripts/causalchamber_repare.py"


rule causalchamber_repare_ungrouped:
    input:
        **PREP_FILES,
    output:
        metrics_csv=BASE + "repare_ungrouped/metrics.csv",
        score_dag=BASE + "repare_ungrouped/score_dag.png",
        oracle_dag=BASE + "repare_ungrouped/oracle_dag.png",
        score_params=BASE + "repare_ungrouped/score_params.json",
        oracle_params=BASE + "repare_ungrouped/oracle_params.json",
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="ungrouped",
    script:
        "../scripts/causalchamber_repare.py"


# ---------------------------------------------------------------------------
# Aggregator — produces the 4 headline artifacts.
# ---------------------------------------------------------------------------


rule causalchamber_aggregate:
    input:
        features=PREP_FILES["features"],
        nametoidx=PREP_FILES["nametoidx"],
        coarse_grouped_metrics=BASE + "coarse_grouped/metrics.csv",
        coarse_ungrouped_metrics=BASE + "coarse_ungrouped/metrics.csv",
        onepc_grouped_metrics=BASE + "onepc_grouped/metrics.csv",
        onepc_ungrouped_metrics=BASE + "onepc_ungrouped/metrics.csv",
        cv_grouped_metrics=BASE + "cv_grouped/metrics.csv",
        cv_ungrouped_metrics=BASE + "cv_ungrouped/metrics.csv",
        repare_grouped_metrics=BASE + "repare_grouped/metrics.csv",
        repare_ungrouped_metrics=BASE + "repare_ungrouped/metrics.csv",
        coarse_grouped_score_params=BASE + "coarse_grouped/score_params.json",
        coarse_grouped_oracle_params=BASE + "coarse_grouped/oracle_params.json",
        coarse_ungrouped_score_params=BASE + "coarse_ungrouped/score_params.json",
        coarse_ungrouped_oracle_params=BASE + "coarse_ungrouped/oracle_params.json",
        onepc_grouped_score_params=BASE + "onepc_grouped/score_params.json",
        onepc_grouped_oracle_params=BASE + "onepc_grouped/oracle_params.json",
        onepc_ungrouped_score_params=BASE + "onepc_ungrouped/score_params.json",
        onepc_ungrouped_oracle_params=BASE + "onepc_ungrouped/oracle_params.json",
        cv_grouped_score_params=BASE + "cv_grouped/score_params.json",
        cv_grouped_oracle_params=BASE + "cv_grouped/oracle_params.json",
        cv_ungrouped_score_params=BASE + "cv_ungrouped/score_params.json",
        cv_ungrouped_oracle_params=BASE + "cv_ungrouped/oracle_params.json",
        repare_grouped_score_params=BASE + "repare_grouped/score_params.json",
        repare_grouped_oracle_params=BASE + "repare_grouped/oracle_params.json",
        repare_ungrouped_score_params=BASE + "repare_ungrouped/score_params.json",
        repare_ungrouped_oracle_params=BASE + "repare_ungrouped/oracle_params.json",
    output:
        grid_metrics=BASE + "grid_metrics.csv",
        dag=BASE + "dag.png",
        grid_dir=directory(BASE + "grid_runs"),
        summary="results/causalchamber_summary.csv",
        dag_summary="results/causalchamber_dags.txt",
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
