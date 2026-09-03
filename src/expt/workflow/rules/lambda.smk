# λ ablation experiments. Two sub-experiments, both under the oracle
# partition (so ARI = 1 by construction and only the BIC penalty
# multiplier is varied):
#   `lambda_samp`     — λ × samp_size at fixed density.
#   `lambda_density`  — λ × density   at fixed samp_size.
#
# Reuses the same fit/evaluate scripts as kpc.smk (scripts/fit_oracle.py
# handles both no-PCA and kPC variants via the optional k wildcard;
# scripts/evaluate.py), but writes to its own output namespace
# `results/lambda/...` so the kpc/ theme is not invalidated.
# Both sub-experiments share `oracle/` and `kpc_oracle/` model artifacts
# within `results/lambda/` wherever their (density, samp_size, λ) tuples
# overlap.
#
# λ grid is log2-spaced on the *unit* interval boundary: 2^k for k ∈ [-3, 3].
# Values are declared as floats so str(λ) always emits the dotted form,
# matching the `lambda_pen=r"\d+\.\d+"` wildcard constraint below.

data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
lambda_oracle_path = "results/lambda/oracle/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/lambda={lambda_pen}/seed={seed}/"
lambda_kpc_oracle_path = "results/lambda/kpc_oracle/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/k={k}/lambda={lambda_pen}/seed={seed}/"


wildcard_constraints:
    k=r"\d+",
    lambda_pen=r"\d+\.\d+",


# ---------------------------------------------------------------------------
# Fit + evaluate rules — scripts reused from kpc.smk unchanged.
# ---------------------------------------------------------------------------


rule fit_lambda_oracle:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(lambda_oracle_path + "model.pkl"),
    params:
        lambda_pen=lambda w: float(w.lambda_pen),
        intervention_type="soft",
    script:
        "../scripts/fit_oracle.py"


rule fit_lambda_kpc_oracle:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(lambda_kpc_oracle_path + "model.pkl"),
    params:
        lambda_pen=lambda w: float(w.lambda_pen),
        intervention_type="soft",
    script:
        "../scripts/fit_oracle.py"


rule evaluate_lambda_oracle:
    input:
        data=data_path + "dataset.npz",
        model=lambda_oracle_path + "model.pkl",
    output:
        lambda_oracle_path + "metrics.csv",
    params:
        method_label="COARSE-oracle",
        lambda_pen=lambda w: float(w.lambda_pen),
    script:
        "../scripts/evaluate.py"


rule evaluate_lambda_kpc_oracle:
    input:
        data=data_path + "dataset.npz",
        model=lambda_kpc_oracle_path + "model.pkl",
    output:
        lambda_kpc_oracle_path + "metrics.csv",
    params:
        method_label=lambda w: f"kPC-k{w.k}-oracle",
        lambda_pen=lambda w: float(w.lambda_pen),
    script:
        "../scripts/evaluate.py"


# ---------------------------------------------------------------------------
# Sub-experiment 1 — λ × samp_size at fixed density.
# ---------------------------------------------------------------------------

LAMBDA_SAMP_GRAPH = ["er"]
LAMBDA_SAMP_NODES = [30]
LAMBDA_SAMP_INTERVS = [5]
LAMBDA_SAMP_DENSITY = [0.5]
LAMBDA_SAMP_SAMP = [100, 500, 1000, 5000]
LAMBDA_SAMP_K = [1, 3]
LAMBDA_SAMP_LAMBDA = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
LAMBDA_SAMP_SEEDS = list(range(20))


rule collect_lambda_samp:
    input:
        expand(
            lambda_oracle_path + "metrics.csv",
            graph=LAMBDA_SAMP_GRAPH,
            num_nodes=LAMBDA_SAMP_NODES,
            num_intervs=LAMBDA_SAMP_INTERVS,
            density=LAMBDA_SAMP_DENSITY,
            samp_size=LAMBDA_SAMP_SAMP,
            lambda_pen=LAMBDA_SAMP_LAMBDA,
            seed=LAMBDA_SAMP_SEEDS,
        ),
        expand(
            lambda_kpc_oracle_path + "metrics.csv",
            graph=LAMBDA_SAMP_GRAPH,
            num_nodes=LAMBDA_SAMP_NODES,
            num_intervs=LAMBDA_SAMP_INTERVS,
            density=LAMBDA_SAMP_DENSITY,
            samp_size=LAMBDA_SAMP_SAMP,
            k=LAMBDA_SAMP_K,
            lambda_pen=LAMBDA_SAMP_LAMBDA,
            seed=LAMBDA_SAMP_SEEDS,
        ),
    output:
        "results/lambda/lambda_samp_results.csv",
    script:
        "../scripts/collect.py"


rule plot_lambda_samp:
    input:
        rules.collect_lambda_samp.output[0],
    output:
        pdf="results/lambda/lambda_samp.pdf",
    script:
        "../scripts/plot_lambda_samp.py"


rule lambda_samp:
    input:
        "results/lambda/lambda_samp.pdf",


# ---------------------------------------------------------------------------
# Sub-experiment 2 — λ × density at fixed samp_size.
# ---------------------------------------------------------------------------

LAMBDA_DENSITY_GRAPH = ["er"]
LAMBDA_DENSITY_NODES = [30]
LAMBDA_DENSITY_INTERVS = [5]
LAMBDA_DENSITY_DENSITY = [0.2, 0.5, 0.8]
LAMBDA_DENSITY_SAMP = [1000]
LAMBDA_DENSITY_K = [1, 3]
LAMBDA_DENSITY_LAMBDA = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
LAMBDA_DENSITY_SEEDS = list(range(20))


rule collect_lambda_density:
    input:
        expand(
            lambda_oracle_path + "metrics.csv",
            graph=LAMBDA_DENSITY_GRAPH,
            num_nodes=LAMBDA_DENSITY_NODES,
            num_intervs=LAMBDA_DENSITY_INTERVS,
            density=LAMBDA_DENSITY_DENSITY,
            samp_size=LAMBDA_DENSITY_SAMP,
            lambda_pen=LAMBDA_DENSITY_LAMBDA,
            seed=LAMBDA_DENSITY_SEEDS,
        ),
        expand(
            lambda_kpc_oracle_path + "metrics.csv",
            graph=LAMBDA_DENSITY_GRAPH,
            num_nodes=LAMBDA_DENSITY_NODES,
            num_intervs=LAMBDA_DENSITY_INTERVS,
            density=LAMBDA_DENSITY_DENSITY,
            samp_size=LAMBDA_DENSITY_SAMP,
            k=LAMBDA_DENSITY_K,
            lambda_pen=LAMBDA_DENSITY_LAMBDA,
            seed=LAMBDA_DENSITY_SEEDS,
        ),
    output:
        "results/lambda/lambda_density_results.csv",
    script:
        "../scripts/collect.py"


rule plot_lambda_density:
    input:
        rules.collect_lambda_density.output[0],
    output:
        pdf="results/lambda/lambda_density.pdf",
    script:
        "../scripts/plot_lambda_density.py"


rule lambda_density:
    input:
        "results/lambda/lambda_density.pdf",


# ---------------------------------------------------------------------------
# Top-level aggregator.
# ---------------------------------------------------------------------------


rule lambda_all:
    input:
        "results/lambda/lambda_samp.pdf",
        "results/lambda/lambda_density.pdf",
