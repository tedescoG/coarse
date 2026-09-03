data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
cv_path = "results/coarse_cv/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


rule fit_cv:
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


rule evaluate_cv:
    input:
        data=data_path + "dataset.npz",
        model=cv_path + "model.pkl",
    output:
        cv_path + "metrics.csv",
    params:
        method_label="COARSE-CV",
    script:
        "../scripts/evaluate.py"


rule collect_cv:
    input:
        expand(
            cv_path + "metrics.csv",
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
        results="results/coarse_cv/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot_cv:
    input:
        rules.collect_cv.output["results"],
    output:
        ari="results/coarse_cv/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/coarse_cv/{graph}_fscore_ivn={num_intervs}.pdf",
    script:
        "../scripts/plot.py"


# Drop cells where samp_size <= num_nodes: with 5-fold CV the per-env train
# fold is 0.8*samp_size, so the per-env Σ̂ is rank-deficient and every (α,
# fold) cell returns -inf, tripping COARSECV's "all pairs failed" RuntimeError.
# Not an identifiable regime; no point burning compute on it.
#
# Cap num_nodes at 100: at p=200 the synthetic LGANM data has deep multiplicative
# chains producing near-exact linear dependencies inside large partition blocks
# (122-155 features). The train block-Σ is rank-deficient and Cholesky fails; 0/7 on-disk p=200 cells
# admit a fully-finite all-folds α. Not an identifiable regime; matches the
# spirit of the s > n guard.
rule collect_scalability_cv:
    input:
        [
            cv_path.format(
                graph="er",
                num_nodes=n,
                num_intervs=5,
                density=0.2,
                samp_size=s,
                seed=seed,
            )
            + "metrics.csv"
            for n in [10, 20, 50, 100]
            for s in [100, 1000, 10000, 100000]
            for seed in range(10)
            if s > n
        ],
    output:
        results="results/coarse_cv/scalability.csv",
    script:
        "../scripts/collect.py"


rule plot_scalability_cv:
    input:
        rules.collect_scalability_cv.output["results"],
    output:
        ari_samp="results/coarse_cv/scalability_ari_samp.pdf",
        time_samp="results/coarse_cv/scalability_time_samp.pdf",
        time_nodes="results/coarse_cv/scalability_time_nodes.pdf",
    script:
        "../scripts/scalability_plot.py"
