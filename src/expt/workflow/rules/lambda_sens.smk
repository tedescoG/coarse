# BIC penalty multiplier λ, true partition (er, p=50, 5 envs). Per density: x=λ, y=P | R | F1,
# hue=n, dashed λ=1.
LS_LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
LS_NODES = [10] if SMOKE else [50]
LS_DENSITIES = [0.2, 0.5]
LS_SAMP = [100, 500] if SMOKE else [100, 500, 1000, 5000]
LS_METRICS = ["precision", "recall", "fscore"]
LS_DIR = RESULTS + "lambda_sens/"
LS_PANELS = [
    dict(
        out=f"lambda_sens_{m}_d{d}",
        filter={"density": d},
        x="lambda_pen",
        y=m,
        hue="samp_size",
        vline=LAMBDA_PEN,
    )
    for m in LS_METRICS
    for d in LS_DENSITIES
]


rule collect_lambda_sens:
    input:
        cells(
            ORACLE + "metrics.csv",
            method=["coarse"],
            noise=["gaussian"],
            lambda_pen=LS_LAMBDAS,
            graph=["er"],
            num_nodes=LS_NODES,
            num_intervs=[5],
            density=LS_DENSITIES,
            samp_size=LS_SAMP,
            seed=SEEDS,
        ),
    output:
        LS_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_lambda_sens:
    input:
        rules.collect_lambda_sens.output[0],
    output:
        LS_DIR + "summary.csv",
    params:
        by=["density", "samp_size", "lambda_pen"],
        metrics=LS_METRICS,
    script:
        "../scripts/summarize.py"


rule plot_lambda_sens:
    input:
        rules.collect_lambda_sens.output[0],
        style=PLOT_STYLE,
    output:
        **{panel["out"]: LS_DIR + panel["out"] + ".pdf" for panel in LS_PANELS},
    params:
        panels=LS_PANELS,
    script:
        "../scripts/plot_panels.py"


rule lambda_sens_all:
    input:
        rules.plot_lambda_sens.output,
        rules.summarize_lambda_sens.output,
