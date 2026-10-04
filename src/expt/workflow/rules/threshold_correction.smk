# Descendant-test cutoff: uncorrected α, Bonferroni / BH / BY, COARSECV and oracle α; er, ivn 5.
TC_NODES = [20] if SMOKE else [250]
TC_SAMP = [1000, 2000] if SMOKE else [10000, 100000]
TC_DENSITIES = [0.2, 0.5, 0.8]
TC_DIR = RESULTS + "threshold_correction/"


rule collect_threshold_correction:
    input:
        cells(
            THRESHOLD + "metrics.csv",
            **DEFAULT,
            graph=["er"],
            num_nodes=TC_NODES,
            num_intervs=[5],
            density=TC_DENSITIES,
            samp_size=TC_SAMP,
            seed=SEEDS,
        ),
    output:
        TC_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_threshold_correction:
    input:
        rules.collect_threshold_correction.output[0],
    output:
        TC_DIR + "summary.csv",
    params:
        by=["num_nodes", "density", "samp_size", "rule", "nominal"],
        metrics=["ari", "cutoff"],
    script:
        "../scripts/summarize.py"


rule plot_threshold_correction:
    input:
        rules.collect_threshold_correction.output[0],
        style=PLOT_STYLE,
    output:
        curves=TC_DIR + "curves.pdf",
        gap=TC_DIR + "gap.pdf",
    script:
        "../scripts/plot_threshold.py"


rule threshold_correction_all:
    input:
        rules.plot_threshold_correction.output,
        rules.summarize_threshold_correction.output,
