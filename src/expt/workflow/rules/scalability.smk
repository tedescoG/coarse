# COARSE as p and n grow (er, 5 envs, density 0.2). x=n, y=ARI | F1 | runtime, hue=p;
# time_nodes: x=p, hue=n.
SCAL_STEMS = ["ari_samp", "fscore_samp", "time_samp", "time_nodes"]
SCAL_DIR = RESULTS + "scalability/coarse/"


rule collect_scalability:
    input:
        cells(
            PIPELINE + "metrics.csv",
            **DEFAULT,
            method=["coarse"],
            graph=["er"],
            num_intervs=[5],
            density=[0.2],
            num_nodes=SCAL_NODES,
            samp_size=SCAL_SAMP,
            seed=SEEDS,
        ),
    output:
        SCAL_DIR + "results.csv",
    script:
        "../scripts/collect.py"


rule summarize_scalability:
    input:
        rules.collect_scalability.output[0],
    output:
        SCAL_DIR + "summary.csv",
    params:
        by=["num_nodes", "samp_size"],
        metrics=["ari", "fscore", "runtime_sec"],
    script:
        "../scripts/summarize.py"


rule plot_scalability:
    input:
        rules.collect_scalability.output[0],
        style=PLOT_STYLE,
    output:
        **{stem: SCAL_DIR + stem + ".pdf" for stem in SCAL_STEMS},
    params:
        panels=[
            dict(out="ari_samp", x="samp_size", y="ari", hue="num_nodes"),
            dict(out="fscore_samp", x="samp_size", y="fscore", hue="num_nodes"),
            dict(out="time_samp", x="samp_size", y="runtime_sec", hue="num_nodes"),
            dict(out="time_nodes", x="num_nodes", y="runtime_sec", hue="samp_size"),
        ],
    script:
        "../scripts/plot_panels.py"


rule scalability_all:
    input:
        rules.plot_scalability.output,
        rules.summarize_scalability.output,
