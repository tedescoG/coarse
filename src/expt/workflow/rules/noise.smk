# Noise-family robustness: COARSE on Gaussian / uniform / Laplace iSCM noise at p=10,
# same (density × n) grid as synth.smk. The Gaussian cells *are* synth.smk's COARSE cells
# (same PIPELINE paths), so nothing is fit twice.
NOISE_IVNS = [5]


rule collect_noise:
    input:
        cells(
            PIPELINE + "metrics.csv",
            method=["coarse"],
            noise=NOISES,
            targets_per_interv=["1"],
            num_nodes=[10],
            density=DENSITIES,
            samp_size=SAMP_SIZES,
            seed=SEEDS,
        ),
    output:
        "results/noise/{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule plot_noise:
    input:
        rules.collect_noise.output[0],
    output:
        ari="results/noise/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/noise/{graph}_fscore_ivn={num_intervs}.pdf",
    params:
        panels=[
            dict(
                out="ari",
                x="samp_size",
                y="ari",
                hue="noise",
                hue_order=NOISES,
                col="density",
                ylim=(0, 1),
                xlabel="sample size (n)",
                ylabel="ARI ↑",
                legend="noise",
            ),
            dict(
                out="fscore",
                x="samp_size",
                y="fscore",
                hue="noise",
                hue_order=NOISES,
                col="density",
                ylim=(0, 1),
                xlabel="sample size (n)",
                ylabel="F-score ↑",
                legend="noise",
            ),
        ],
    script:
        "../scripts/plot_panels.py"


rule noise_all:
    input:
        expand(
            "results/noise/{graph}_{metric}_ivn={num_intervs}.pdf",
            graph=["er", "sf"],
            metric=["ari", "fscore"],
            num_intervs=NOISE_IVNS,
        ),
