# λ ablation under the oracle partition (ARI = 1 by construction; only the BIC penalty
# multiplier varies). λ grid is log2-spaced, 2^k for k ∈ [-3, 3].
#   lambda_samp     — λ × samp_size at density 0.5 (documented low-n exception).
#   lambda_density  — λ × density at n = 1000.
# Shared fits: lambda_samp's λ=1 cells are kpc.smk exp2's; lambda_density's density-0.5
# column is lambda_samp's n=1000 column (full grid only; the smoke LAMBDA_SAMP has no
# n=1000).
LAMBDA_NODES = [30]
LAMBDA_DENSITY_SAMP = [1000]


rule collect_lambda_samp:
    input:
        cells(
            ORACLE + "metrics.csv",
            method=ORACLE_METHODS,
            lambda_pen=LAMBDAS,
            graph=["er"],
            num_nodes=LAMBDA_NODES,
            num_intervs=[5],
            density=[0.5],
            samp_size=LAMBDA_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/lambda/lambda_samp_results.csv",
    script:
        "../scripts/collect.py"


rule plot_lambda_samp:
    input:
        rules.collect_lambda_samp.output[0],
        style=PLOT_STYLE,
    output:
        pdf="results/lambda/lambda_samp.pdf",
    params:
        facet="samp_size",
        facet_values=LAMBDA_SAMP,
        ncols=2,
        title="n = {}",
    script:
        "../scripts/plot_lambda.py"


rule collect_lambda_density:
    input:
        cells(
            ORACLE + "metrics.csv",
            method=ORACLE_METHODS,
            lambda_pen=LAMBDAS,
            graph=["er"],
            num_nodes=LAMBDA_NODES,
            num_intervs=[5],
            density=DENSITIES,
            samp_size=LAMBDA_DENSITY_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/lambda/lambda_density_results.csv",
    script:
        "../scripts/collect.py"


rule plot_lambda_density:
    input:
        rules.collect_lambda_density.output[0],
        style=PLOT_STYLE,
    output:
        pdf="results/lambda/lambda_density.pdf",
    params:
        facet="density",
        facet_values=DENSITIES,
        ncols=3,
        title="density = {}",
    script:
        "../scripts/plot_lambda.py"


rule lambda_samp:
    input:
        rules.plot_lambda_samp.output,


rule lambda_density:
    input:
        rules.plot_lambda_density.output,


rule lambda_all:
    input:
        rules.plot_lambda_samp.output,
        rules.plot_lambda_density.output,
