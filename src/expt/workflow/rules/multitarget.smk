# Population gap between the env and interventional partitions with multi-target envs; er, p=50.
# Estimation: COARSECV on data with k targets per env, m=10 (in no other grid).
MT_NODES = [50]
MT_DENSITIES = [0.2, 0.5, 0.8]
MT_SEEDS = list(range(2 if SMOKE else 200))
MT_DIR = RESULTS + "multitarget/"
MF_NODES = [20] if SMOKE else [50]
MF_SAMP = [1000, 2000] if SMOKE else [1000, 10000, 100000]
MF_DIR = MT_DIR + "estimation/"


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


rule collect_multitarget_fit:
    input:
        cells(
            MTFIT + "metrics.csv",
            noise=["gaussian"],
            targets_per_interv=["1", "2", "3", "4", "5"],
            graph=["er"],
            num_nodes=MF_NODES,
            num_intervs=[10],
            density=MT_DENSITIES,
            samp_size=MF_SAMP,
            seed=SEEDS,
        ),
    output:
        MF_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_multitarget_fit:
    input:
        rules.collect_multitarget_fit.output[0],
    output:
        MF_DIR + "summary.csv",
    params:
        by=["density", "k", "samp_size"],
        metrics=["ari_E", "ari_I", "ari_pop", "nhat", "alpha"],
    script:
        "../scripts/summarize.py"


rule plot_multitarget_fit:
    input:
        rules.collect_multitarget_fit.output[0],
        style=PLOT_STYLE,
    output:
        MF_DIR + "curves.pdf",
    script:
        "../scripts/plot_multitarget_fit.py"


rule multitarget_all:
    input:
        rules.plot_multitarget.output,
        rules.summarize_multitarget.output,
        rules.plot_multitarget_fit.output,
        rules.summarize_multitarget_fit.output,
