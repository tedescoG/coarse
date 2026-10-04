# High-dimensional regime (n ≤ p) under the true partition: COARSEOracle plain, 1PC, 3PC and avg;
# er, p=100, density 0.1, graph-level and per-block parent recovery.
HD_METHODS = ["coarse", "kpc1", "kpc3", "avg"]
HD_IVNS = [5, 10] if SMOKE else [5, 10, 20]
HD_SAMP = (
    [10, 50, 200] if SMOKE else [10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
)
HD_DIR = RESULTS + "highdim/"
HD_GRID = dict(
    noise=["gaussian"],
    method=HD_METHODS,
    lambda_pen=[LAMBDA_PEN],
    graph=["er"],
    num_nodes=[100],
    num_intervs=HD_IVNS,
    density=[0.1],
    samp_size=HD_SAMP,
    seed=SEEDS,
)


rule collect_highdim:
    input:
        cells(ORACLE + "metrics.csv", **HD_GRID),
    output:
        HD_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule collect_highdim_blocks:
    input:
        cells(ORACLE + "blocks.csv", **HD_GRID),
    output:
        HD_DIR + "blocks.csv",
    script:
        "../scripts/collect.py"


rule summarize_highdim:
    input:
        rules.collect_highdim.output[0],
    output:
        HD_DIR + "summary.csv",
    params:
        by=["method", "num_intervs", "samp_size"],
        metrics=["precision", "recall", "fscore"],
    script:
        "../scripts/summarize.py"


rule plot_highdim:
    input:
        graph=rules.collect_highdim.output[0],
        blocks=rules.collect_highdim_blocks.output[0],
        style=PLOT_STYLE,
    output:
        graph=HD_DIR + "graph.pdf",
        blocks=HD_DIR + "blocks.pdf",
        feasible=HD_DIR + "feasible.pdf",
    script:
        "../scripts/plot_highdim.py"


rule highdim_all:
    input:
        rules.plot_highdim.output,
        rules.summarize_highdim.output,
