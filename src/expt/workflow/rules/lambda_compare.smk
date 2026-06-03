data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
coarse_path = "results/coarse/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
onepc_path = "results/coarse_1pc/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
coarseL2_path = "results/coarseL2/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
onepcL2_path = "results/1pcL2/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


wildcard_constraints:
    graph="er|sf",
    num_intervs=r"\d+",


# Concat the metrics.csv files produced by the four sibling pipelines
# (synth.smk, onepc.smk, coarseL2.smk, onepcL2.smk) for the shared grid.
# No fitting happens here — metrics.csv files are not temp(), only model.pkl is.
rule collect_lambda_compare:
    input:
        expand(
            coarse_path + "metrics.csv",
            num_nodes=[10],
            seed=range(10),
            density=[0.2, 0.5, 0.8],
            samp_size=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
            allow_missing=True,
        ),
        expand(
            onepc_path + "metrics.csv",
            num_nodes=[10],
            seed=range(10),
            density=[0.2, 0.5, 0.8],
            samp_size=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
            allow_missing=True,
        ),
        expand(
            coarseL2_path + "metrics.csv",
            num_nodes=[10],
            seed=range(10),
            density=[0.2, 0.5, 0.8],
            samp_size=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
            allow_missing=True,
        ),
        expand(
            onepcL2_path + "metrics.csv",
            num_nodes=[10],
            seed=range(10),
            density=[0.2, 0.5, 0.8],
            samp_size=[100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000],
            allow_missing=True,
        ),
    output:
        results="results/lambda_compare/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot_lambda_compare:
    input:
        rules.collect_lambda_compare.output["results"],
    output:
        fscore="results/lambda_compare/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot_lambda_compare.py"


rule lambda_compare_all:
    input:
        expand(
            "results/lambda_compare/{graph}_fscore_ivn={num_intervs}.pdf",
            graph=["er", "sf"],
            num_intervs=[2, 5, 8],
        ),
