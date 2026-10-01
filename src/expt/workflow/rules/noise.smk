# Noise families under the true partition: COARSEOracle, er, ivn 5, density 0.2, n > p.
NOISE_SAMP = sorted({100, 500, *SCAL_SAMP})
NOISE_DIR = RESULTS + "noise/"


rule collect_noise:
    input:
        cells(
            ORACLE + "metrics.csv",
            guard=identifiable,
            method=["coarse"],
            noise=NOISES,
            lambda_pen=[LAMBDA_PEN],
            graph=["er"],
            num_nodes=NOISE_NODES,
            num_intervs=[5],
            density=[0.2],
            samp_size=NOISE_SAMP,
            seed=SEEDS,
        ),
    output:
        NOISE_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_noise:
    input:
        rules.collect_noise.output[0],
    output:
        NOISE_DIR + "summary.csv",
    params:
        by=["noise", "num_nodes", "samp_size"],
        metrics=["fscore"],
    script:
        "../scripts/summarize.py"


rule plot_noise:
    input:
        rules.collect_noise.output[0],
        style=PLOT_STYLE,
    output:
        **{f"p{p}": NOISE_DIR + f"fscore_p={p}.pdf" for p in NOISE_NODES},
    params:
        panels=[
            dict(
                out=f"p{p}",
                filter={"num_nodes": p},
                x="samp_size",
                y="fscore",
                hue="noise",
                hue_order=NOISES,
            )
            for p in NOISE_NODES
        ],
    script:
        "../scripts/plot_panels.py"


rule noise_all:
    input:
        rules.plot_noise.output,
        rules.summarize_noise.output,
