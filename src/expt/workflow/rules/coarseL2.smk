data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
coarseL2_path = "results/coarseL2/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule fit_l2:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(coarseL2_path + "model.pkl"),
    params:
        alpha=0.0001,
        lambda_pen=2.0,
        refine_test="welch",
        intervention_type="soft",
        scale=True,
    script:
        "../scripts/fit.py"


rule evaluate_l2:
    input:
        data=data_path + "dataset.npz",
        model=coarseL2_path + "model.pkl",
    output:
        coarseL2_path + "metrics.csv",
    params:
        method_label="COARSE-L2",
        lambda_pen=2.0,
    script:
        "../scripts/evaluate.py"


rule collect_l2:
    input:
        expand(
            coarseL2_path + "metrics.csv",
            num_nodes=[10],
            seed=range(10),
            density=[0.2, 0.5, 0.8],
            samp_size=[
                100,
                200,
                500,
                1000,
                2000,
                5000,
                10000,
                20000,
                50000,
                100000,
            ],
            allow_missing=True,
        ),
    output:
        results="results/coarseL2/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot_l2:
    input:
        rules.collect_l2.output["results"],
    output:
        ari="results/coarseL2/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/coarseL2/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot.py"


# Cap num_nodes at 100: same rationale as the lambda=1 variant in synth.smk —
# at p=200 the synthetic LGANM data has deep multiplicative chains producing
# near-exact linear dependencies inside large partition blocks. Cholesky of
# the block-Σ̂ fails, BIC short-circuits to -inf, COARSE saves a degenerate
# model. Doubling the penalty does not change this; the failure is numerical.
rule collect_scalability_l2:
    input:
        expand(
            coarseL2_path + "metrics.csv",
            graph=["er"],
            num_nodes=[10, 20, 50, 100],
            seed=range(10),
            density=[0.2],
            num_intervs=[5],
            samp_size=[
                100,
                1000,
                10000,
                100000,
            ],
            allow_missing=True,
        ),
    output:
        results="results/coarseL2/scalability.csv",
    script:
        "../scripts/collect.py"


rule plot_scalability_l2:
    input:
        rules.collect_scalability_l2.output["results"],
    output:
        ari_samp="results/coarseL2/scalability_ari_samp.pdf",
        time_samp="results/coarseL2/scalability_time_samp.pdf",
        time_nodes="results/coarseL2/scalability_time_nodes.pdf",
    script:
        "../scripts/scalability_plot.py"


rule coarseL2_all:
    input:
        expand(
            "results/coarseL2/{graph}_{metric}_ivn={num_intervs}.pdf",
            graph=["er", "sf"],
            metric=["ari", "fscore"],
            num_intervs=[2, 5, 8],
        ),
        expand(
            "results/coarseL2/scalability_{compare}.pdf",
            compare=["ari_samp", "time_samp", "time_nodes"],
        ),
