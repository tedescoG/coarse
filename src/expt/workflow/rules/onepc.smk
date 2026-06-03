data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
onepc_path = "results/coarse_1pc/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule fit_1pc:
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


rule evaluate_1pc:
    input:
        data=data_path + "dataset.npz",
        model=onepc_path + "model.pkl",
    output:
        onepc_path + "metrics.csv",
    params:
        method_label="COARSE-1PC",
        pca_pooled="obs",
    script:
        "../scripts/evaluate.py"


rule collect_1pc:
    input:
        expand(
            onepc_path + "metrics.csv",
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
        results="results/coarse_1pc/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot_1pc:
    input:
        rules.collect_1pc.output["results"],
    output:
        ari="results/coarse_1pc/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/coarse_1pc/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot.py"


rule collect_scalability_1pc:
    input:
        expand(
            onepc_path + "metrics.csv",
            graph=["er"],
            num_nodes=[10, 20, 50, 100, 200],
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
        results="results/coarse_1pc/scalability.csv",
    script:
        "../scripts/collect.py"


rule plot_scalability_1pc:
    input:
        rules.collect_scalability_1pc.output["results"],
    output:
        ari_samp="results/coarse_1pc/scalability_ari_samp.pdf",
        time_samp="results/coarse_1pc/scalability_time_samp.pdf",
        time_nodes="results/coarse_1pc/scalability_time_nodes.pdf",
    script:
        "../scripts/scalability_plot.py"
