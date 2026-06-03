data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
coarse_path = "results/coarse/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule generate:
    output:
        data_path + "dataset.npz",
    params:
        intervention_type="soft",
    script:
        "../scripts/generate.py"


rule fit:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(coarse_path + "model.pkl"),
    params:
        alpha=0.0001,
        lambda_pen=1.0,
        refine_test="welch",
        intervention_type="soft",
        scale=True,
    script:
        "../scripts/fit.py"


rule evaluate:
    input:
        data=data_path + "dataset.npz",
        model=coarse_path + "model.pkl",
    output:
        coarse_path + "metrics.csv",
    script:
        "../scripts/evaluate.py"


rule collect:
    input:
        expand(
            coarse_path + "metrics.csv",
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
        results="results/coarse/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot:
    input:
        rules.collect.output["results"],
    output:
        ari="results/coarse/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/coarse/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot.py"


# Cap num_nodes at 100: at p=200 the synthetic LGANM data has deep multiplicative
# chains producing near-exact linear dependencies inside large partition blocks
# (122-155 features). Even after the per-env z-score (scale=True in rule fit),
# Cholesky of the block-Σ̂ fails and the BIC scorer short-circuits to -inf, which
# COARSE silently saves as a degenerate (0-edge, score=-inf) model. The kPC
# scalability sweeps stay at p=200 because SVD projects each block to k_j dims.
rule collect_scalability:
    input:
        expand(
            coarse_path + "metrics.csv",
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
        results="results/coarse/scalability.csv",
    script:
        "../scripts/collect.py"


rule plot_scalability:
    input:
        rules.collect_scalability.output["results"],
    output:
        ari_samp="results/coarse/scalability_ari_samp.pdf",
        time_samp="results/coarse/scalability_time_samp.pdf",
        time_nodes="results/coarse/scalability_time_nodes.pdf",
    script:
        "../scripts/scalability_plot.py"
