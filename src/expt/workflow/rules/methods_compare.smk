# Methods under the true partition (edge recovery only).
MC_METHODS = ["coarse", "avg", "repare"]
MC_NODES = [10, 20] if SMOKE else [10, 20, 50, 100, 200, 300, 500]
MC_METRICS = ["fscore", "runtime_sec"]
MC_DIR = RESULTS + "methods_compare/"
MC_PANELS = [
    dict(
        out=f"er_p={p}_dens={d}_{m}",
        filter={"num_nodes": p, "density": d},
        x="samp_size",
        y=m,
        hue="method",
    )
    for p in MC_NODES
    for d in DENSITIES
    for m in MC_METRICS
]


rule collect_methods_compare:
    input:
        cells(
            ORACLE + "metrics.csv",
            noise=["gaussian"],
            method=MC_METHODS,
            lambda_pen=[LAMBDA_PEN],
            graph=["er"],
            num_nodes=MC_NODES,
            num_intervs=[5],
            density=DENSITIES,
            samp_size=SAMP_SIZES,
            seed=SEEDS,
        ),
    output:
        MC_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_methods_compare:
    input:
        rules.collect_methods_compare.output[0],
    output:
        MC_DIR + "summary.csv",
    params:
        by=["method", "num_nodes", "density", "samp_size"],
        metrics=["fscore", "precision", "recall", "runtime_sec"],
    script:
        "../scripts/summarize.py"


rule plot_methods_compare:
    input:
        rules.collect_methods_compare.output[0],
        style=PLOT_STYLE,
    output:
        **{panel["out"]: MC_DIR + panel["out"] + ".pdf" for panel in MC_PANELS},
    params:
        panels=MC_PANELS,
    script:
        "../scripts/plot_panels.py"


rule methods_compare_all:
    input:
        rules.plot_methods_compare.output,
        rules.summarize_methods_compare.output,
