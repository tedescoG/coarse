# Population gap between the env and interventional partitions with multi-target envs; er, p=50.
MT_NODES = [50]
MT_DENSITIES = [0.2, 0.5, 0.8]
MT_SEEDS = list(range(2 if SMOKE else 200))
MT_DIR = RESULTS + "multitarget/"


rule collect_multitarget:
    input:
        cells(
            MULTITARGET + "metrics.csv",
            graph=["er"],
            num_nodes=MT_NODES,
            density=MT_DENSITIES,
            seed=MT_SEEDS,
        ),
    output:
        MT_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_multitarget:
    input:
        rules.collect_multitarget.output[0],
    output:
        MT_DIR + "summary.csv",
    params:
        by=["design", "density", "k", "m"],
        metrics=["ari", "ratio", "hidden", "nE", "nI"],
    script:
        "../scripts/summarize.py"


rule plot_multitarget:
    input:
        rules.collect_multitarget.output[0],
        style=PLOT_STYLE,
    output:
        native=MT_DIR + "native.pdf",
        disjoint=MT_DIR + "disjoint.pdf",
        overlap=MT_DIR + "overlap.pdf",
    script:
        "../scripts/plot_multitarget.py"


rule multitarget_all:
    input:
        rules.plot_multitarget.output,
        rules.summarize_multitarget.output,
