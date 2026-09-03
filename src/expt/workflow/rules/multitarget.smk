# 3-method head-to-head on MULTI-TARGET interventions: COARSE vs COARSE-CV vs
# COARSE-1PC. Unlike methods_compare.smk, this rule file drives the FULL
# learning pipelines (fit.py / fit_cv.py), not the oracle-partition variant —
# the question we answer here is "does partition discovery itself survive
# multi-target interventions?", which fit_oracle.py would short-circuit.
#
# Data lives under a separate namespace `results/_data_multitarget/...` keyed
# on a new `targets_per_interv` wildcard. The existing `results/_data/...`
# cache is untouched, so every existing rule and figure continues to build
# without regeneration.
#
# Wildcard `targets_per_interv` (alias `tpi`):
#   - "k"     → fixed-size: every env intervenes on exactly k nodes.
#   - "AtoB"  → heterogeneous: env j intervenes on a random subset
#               whose size is sampled uniformly in [A, B].
# The parser lives in scripts/_common.py:parse_targets_per_interv (also
# exercised from tests/test_multitarget.py).

data_path = (
    "results/_data_multitarget/graph={graph}/num_nodes={num_nodes}/"
    "num_intervs={num_intervs}/targets_per_interv={targets_per_interv}/density={density}/"
    "samp_size={samp_size}/seed={seed}/"
)
coarse_path = (
    "results/multitarget/coarse/graph={graph}/num_nodes={num_nodes}/"
    "num_intervs={num_intervs}/targets_per_interv={targets_per_interv}/density={density}/"
    "samp_size={samp_size}/seed={seed}/"
)
cv_path = (
    "results/multitarget/cv/graph={graph}/num_nodes={num_nodes}/"
    "num_intervs={num_intervs}/targets_per_interv={targets_per_interv}/density={density}/"
    "samp_size={samp_size}/seed={seed}/"
)
onepc_path = (
    "results/multitarget/onepc/graph={graph}/num_nodes={num_nodes}/"
    "num_intervs={num_intervs}/targets_per_interv={targets_per_interv}/density={density}/"
    "samp_size={samp_size}/seed={seed}/"
)


wildcard_constraints:
    graph="er|sf",
    num_intervs=r"\d+",
    targets_per_interv=r"\d+(to\d+)?",


MT_GRAPHS = ["er"]
MT_NUM_NODES = [10, 20, 50]
MT_DENSITY = [0.2, 0.5]
MT_NUM_INTERVS = [5]
MT_TPI = ["2", "3", "5", "1to5"]
MT_SAMP_SIZE = [500, 2000, 10000, 50000]
MT_SEEDS = list(range(10))
MT_METRICS = ["fscore", "ari", "runtime_sec"]
MT_SUMMARY_METRICS = ["fscore", "ari"]


# ---------------------------------------------------------------------------
# Dataset generation — reuses scripts/generate.py. The script reads the
# `targets_per_interv` wildcard via getattr(snakemake.wildcards, ..., None);
# absent → size=1 (single-target, existing behavior). Present here.
# ---------------------------------------------------------------------------

rule generate_multitarget:
    output:
        data_path + "dataset.npz",
    params:
        intervention_type="soft",
    script:
        "../scripts/generate.py"


# ---------------------------------------------------------------------------
# Fit rules — three methods over the same multi-target dataset.
#   - COARSE: synth.smk's production defaults (alpha=1e-4, lambda=1.0, welch).
#     This is the same configuration the existing main figures use, so
#     multi-target results stay comparable.
#   - COARSE-CV: cv.smk's defaults (cv_coarse internal alpha grid).
#   - COARSE-1PC: onepc.smk's defaults (k=1, pca_pooled=False).
# ---------------------------------------------------------------------------

rule fit_multitarget_coarse:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(coarse_path + "model.pkl"),
    params:
        alpha=0.0001,
        lambda_pen=1.0,
        refine_test="welch",
        intervention_type="soft",
    script:
        "../scripts/fit.py"


rule fit_multitarget_cv:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(cv_path + "model.pkl"),
    params:
        lambda_pen=1.0,
        refine_test="welch",
        intervention_type="soft",
    script:
        "../scripts/fit_cv.py"


rule fit_multitarget_onepc:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(onepc_path + "model.pkl"),
    params:
        alpha=0.0001,
        lambda_pen=1.0,
        refine_test="welch",
        intervention_type="soft",
        k=1,
        pca_pooled=False,
    script:
        "../scripts/fit.py"


# ---------------------------------------------------------------------------
# Evaluate rules — same evaluate.py for all three, distinguished only by
# method_label. metrics.csv is NOT temp(): the collect rule reads it.
# ---------------------------------------------------------------------------

rule evaluate_multitarget_coarse:
    input:
        data=data_path + "dataset.npz",
        model=coarse_path + "model.pkl",
    output:
        coarse_path + "metrics.csv",
    params:
        method_label="COARSE",
        pca_pooled="na",
        lambda_pen=1.0,
    script:
        "../scripts/evaluate.py"


rule evaluate_multitarget_cv:
    input:
        data=data_path + "dataset.npz",
        model=cv_path + "model.pkl",
    output:
        cv_path + "metrics.csv",
    params:
        method_label="COARSE-CV",
        pca_pooled="na",
        lambda_pen=1.0,
    script:
        "../scripts/evaluate.py"


rule evaluate_multitarget_onepc:
    input:
        data=data_path + "dataset.npz",
        model=onepc_path + "model.pkl",
    output:
        onepc_path + "metrics.csv",
    params:
        method_label="COARSE-1PC",
        pca_pooled="obs",
        lambda_pen=1.0,
    script:
        "../scripts/evaluate.py"


# ---------------------------------------------------------------------------
# Collect — concatenate all three methods' per-cell CSVs.
# ---------------------------------------------------------------------------

rule collect_multitarget:
    input:
        expand(
            coarse_path + "metrics.csv",
            graph=MT_GRAPHS,
            num_nodes=MT_NUM_NODES,
            num_intervs=MT_NUM_INTERVS,
            targets_per_interv=MT_TPI,
            density=MT_DENSITY,
            samp_size=MT_SAMP_SIZE,
            seed=MT_SEEDS,
        ),
        expand(
            cv_path + "metrics.csv",
            graph=MT_GRAPHS,
            num_nodes=MT_NUM_NODES,
            num_intervs=MT_NUM_INTERVS,
            targets_per_interv=MT_TPI,
            density=MT_DENSITY,
            samp_size=MT_SAMP_SIZE,
            seed=MT_SEEDS,
        ),
        expand(
            onepc_path + "metrics.csv",
            graph=MT_GRAPHS,
            num_nodes=MT_NUM_NODES,
            num_intervs=MT_NUM_INTERVS,
            targets_per_interv=MT_TPI,
            density=MT_DENSITY,
            samp_size=MT_SAMP_SIZE,
            seed=MT_SEEDS,
        ),
    output:
        "results/multitarget/results.csv",
    script:
        "../scripts/collect.py"


# ---------------------------------------------------------------------------
# Plot — two figure families:
#   (a) per-(graph, p, density, tpi) line plot, x=samp_size, hue=method.
#   (b) headline summary, x=tpi (ordered), hue=method, faceted by (p, dens).
# Filenames are parsed by plot_multitarget.py to recover filter keys.
# ---------------------------------------------------------------------------

rule plot_multitarget:
    input:
        rules.collect_multitarget.output[0],
    output:
        expand(
            "results/multitarget/{graph}_p={p}_dens={d}_tpi={targets_per_interv}_{metric}.pdf",
            graph=MT_GRAPHS,
            p=MT_NUM_NODES,
            d=MT_DENSITY,
            targets_per_interv=MT_TPI,
            metric=MT_METRICS,
        )
        + expand(
            "results/multitarget/summary_{metric}.pdf",
            metric=MT_SUMMARY_METRICS,
        ),
    script:
        "../scripts/plot_multitarget.py"


# ---------------------------------------------------------------------------
# Visualization — one PDF per method, multi-page with 3 panels per page:
# (1) atomic ground-truth DAG, (2) true partition DAG, (3) learned partition.
#
# VIZ_CONFIGS picks 4 small-p configs that exercise different multi-target
# regimes (fixed k=2, k=3, larger p, heterogeneous 1–5). The script fits each
# method on each dataset inline. Datasets are declared as inputs so snakemake
# generates them on demand from rule generate_multitarget — no inline call to
# sempler in the visualization script keeps the data path through the
# canonical generator.
# ---------------------------------------------------------------------------

VIZ_CONFIGS = [
    dict(graph="er", num_nodes=10, num_intervs=5, targets_per_interv="2",    density=0.2, samp_size=10000, seed=0),
    dict(graph="er", num_nodes=10, num_intervs=5, targets_per_interv="3",    density=0.2, samp_size=10000, seed=0),
    dict(graph="er", num_nodes=20, num_intervs=5, targets_per_interv="3",    density=0.2, samp_size=10000, seed=0),
    dict(graph="er", num_nodes=10, num_intervs=5, targets_per_interv="1to5", density=0.5, samp_size=10000, seed=0),
]


rule visualize_multitarget:
    input:
        datasets=[data_path.format(**cfg) + "dataset.npz" for cfg in VIZ_CONFIGS],
    output:
        coarse="results/multitarget/viz/coarse_multitarget.pdf",
        cv="results/multitarget/viz/cv_multitarget.pdf",
        onepc="results/multitarget/viz/onepc_multitarget.pdf",
    params:
        configs=VIZ_CONFIGS,
    script:
        "../scripts/visualize_methods.py"


# ---------------------------------------------------------------------------
# Convenience target — `snakemake multitarget_all` builds everything.
# ---------------------------------------------------------------------------

rule multitarget_all:
    input:
        rules.plot_multitarget.output,
        rules.visualize_multitarget.output,
