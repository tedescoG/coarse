# Adapted from RePaRe (https://github.com/Alex-Markham/repare, v0.2.0), GPL-3.0-or-later.
# Causal Chamber light tunnel: COARSE and RePaRe on the grouped and ungrouped regimes, GIES,
# GnIES and UT-IGSP on the ungrouped one. Score rows never use ground truth; oracle rows take
# the best grid cell. Per method: results/causalchamber/<method>_<mode>/.

ALPHAS = [1e-4, 1e-3, 1e-2, 0.1]
BETAS = [1e-4, 1e-3, 1e-2, 0.1]
# λ is not tuned: BIC is not comparable across λ.
FIXED_LAMBDA = 1.0
CV_N_FOLDS = 10
# Gaussian mean + variance LRT.
COARSE_REFINE_TEST = "gaussian_lrt"

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
    "truedagfull": BASE + "preprocessed/truedagfull.pkl",
    "nametoidx": BASE + "preprocessed/nametoidx.pkl",
    "singleenvlabels": BASE + "preprocessed/singleenvlabels.pkl",
}

# Snakemake hashes only `script:`, so imported helpers are explicit inputs.
CC_COMMON = [
    "workflow/scripts/_causalchamber_common.py",
    "workflow/scripts/_common.py",
]

CC_MODES = ["grouped", "ungrouped"]
CC_METHOD_MODES = {
    "coarse": CC_MODES,
    "repare": CC_MODES,
    "gies": ["ungrouped"],
    "gnies": ["ungrouped"],
    "utigsp": ["ungrouped"],
}
# Methods with a hyperparameter grid emit an oracle row; GIES / GnIES do not.
CC_METHOD_SELECTIONS = {
    "coarse": ["score", "oracle"],
    "repare": ["score", "oracle"],
    "gies": ["score"],
    "gnies": ["score"],
    "utigsp": ["score", "oracle"],
}


def cc_outputs_for(method, selections, mode="{mode}"):
    out = {"metrics_csv": BASE + method + f"_{mode}/metrics.csv"}
    for sel in selections:
        out[f"{sel}_dag"] = BASE + method + f"_{mode}/" + sel + "_dag.png"
        out[f"{sel}_params"] = BASE + method + f"_{mode}/" + sel + "_params.json"
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


rule causalchamber_coarse:
    input:
        **PREP_FILES,
        common=CC_COMMON,
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


rule causalchamber_repare:
    input:
        **PREP_FILES,
        common=CC_COMMON,
    output:
        **cc_outputs_for("repare", CC_METHOD_SELECTIONS["repare"]),
    params:
        alphas=ALPHAS,
        betas=BETAS,
        mode="{mode}",
    script:
        "../scripts/causalchamber_repare.py"


rule causalchamber_gies:
    input:
        **PREP_FILES,
        common=CC_COMMON,
    output:
        **cc_outputs_for("gies", CC_METHOD_SELECTIONS["gies"], "ungrouped"),
    script:
        "../scripts/causalchamber_gies.py"


rule causalchamber_gnies:
    input:
        **PREP_FILES,
        common=CC_COMMON,
    output:
        **cc_outputs_for("gnies", CC_METHOD_SELECTIONS["gnies"], "ungrouped"),
    script:
        "../scripts/causalchamber_gnies.py"


rule causalchamber_utigsp:
    input:
        **PREP_FILES,
        common=CC_COMMON,
        ut_igsp="workflow/scripts/ut_igsp.py",
    output:
        **cc_outputs_for("utigsp", CC_METHOD_SELECTIONS["utigsp"], "ungrouped"),
    params:
        alpha_ci=ALPHAS,
        alpha_inv=ALPHAS,
    script:
        "../scripts/causalchamber_utigsp.py"


def cc_aggregate_inputs():
    inputs = {
        "features": PREP_FILES["features"],
        "nametoidx": PREP_FILES["nametoidx"],
    }
    for method, selections in CC_METHOD_SELECTIONS.items():
        for mode in CC_METHOD_MODES[method]:
            for sel in selections:
                inputs[f"{method}_{mode}_{sel}_params"] = (
                    BASE + f"{method}_{mode}/{sel}_params.json"
                )
    return inputs


rule causalchamber_aggregate:
    input:
        **cc_aggregate_inputs(),
        common=CC_COMMON,
    output:
        dag=BASE + "dag.png",
        summary="results/causalchamber_summary.csv",
        dag_summary="results/causalchamber_dags.txt",
    params:
        method_selections=CC_METHOD_SELECTIONS,
        method_modes=CC_METHOD_MODES,
    script:
        "../scripts/causalchamber_aggregate.py"


rule causalchamber_variable:
    input:
        **{
            f"{method}_{sel}_params": BASE + f"{method}_ungrouped/{sel}_params.json"
            for method, selections in CC_METHOD_SELECTIONS.items()
            for sel in selections
        },
        truegraph=PREP_FILES["truegraph"],
        nametoidx=PREP_FILES["nametoidx"],
        partition=PREP_FILES["partition"],
        truedagfull=PREP_FILES["truedagfull"],
        singleenvlabels=PREP_FILES["singleenvlabels"],
        common=CC_COMMON,
    output:
        summary="results/causalchamber_variable_summary.csv",
    params:
        method_selections=CC_METHOD_SELECTIONS,
        draws=1000,
        seed=0,
    script:
        "../scripts/causalchamber_variable.py"


cc_outputs = [
    "results/causalchamber/dag.png",
    "results/causalchamber_summary.csv",
    "results/causalchamber_dags.txt",
    "results/causalchamber_variable_summary.csv",
]


rule causalchamber:
    input:
        cc_outputs,
