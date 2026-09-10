# Main synthetic sweeps for the full-pipeline methods (COARSE, COARSE-CV, COARSE-1PC) on
# Gaussian single-target data: (density × n) at p=10 per graph family and ivn, plus
# scalability (p × n). Fits come from pipeline.smk; this file only collects and plots.
SYNTH_IVNS = [2, 5, 8]
SYNTH_GRAPHS = ["er", "sf"]

AXIS_N = "sample size (n)"


rule collect_synth:
    input:
        cells(
            PIPELINE + "metrics.csv",
            **DEFAULT,
            num_nodes=[10],
            density=DENSITIES,
            samp_size=SAMP_SIZES,
            seed=SEEDS,
        ),
    output:
        "results/synth/{method}/{graph}_results_ivn={num_intervs}.csv",
    wildcard_constraints:
        method="coarse|cv|onepc",
    script:
        "../scripts/collect.py"


rule plot_synth:
    input:
        rules.collect_synth.output[0],
    output:
        ari="results/synth/{method}/{graph}_ari_ivn={num_intervs}.pdf",
        fscore="results/synth/{method}/{graph}_fscore_ivn={num_intervs}.pdf",
    params:
        panels=[
            dict(
                out="ari",
                x="samp_size",
                y="ari",
                hue="density",
                ylim=(0, 1),
                xlabel=AXIS_N,
                ylabel="ARI ↑",
                legend="density",
            ),
            dict(
                out="fscore",
                x="samp_size",
                y="fscore",
                hue="density",
                ylim=(0, 1),
                xlabel=AXIS_N,
                ylabel="F-score ↑",
                legend="density",
            ),
        ],
    script:
        "../scripts/plot_panels.py"


# The iSCM generator standardizes every variable, so nothing breaks numerically as p grows.
# Budget: a dataset at p=500, n=100000 is 6 environments × 0.4 GB = 2.4 GB on disk and in
# memory during fit, so keep --cores modest at the top of this grid.
rule collect_scalability:
    input:
        cells(
            PIPELINE + "metrics.csv",
            guard=identifiable,
            **DEFAULT,
            graph=["er"],
            num_intervs=[5],
            density=[0.2],
            num_nodes=SCAL_NODES,
            samp_size=SCAL_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/synth/{method}/scalability.csv",
    wildcard_constraints:
        method="coarse|cv|onepc",
    script:
        "../scripts/collect.py"


rule plot_scalability:
    input:
        rules.collect_scalability.output[0],
    output:
        ari_samp="results/synth/{method}/scalability_ari_samp.pdf",
        time_samp="results/synth/{method}/scalability_time_samp.pdf",
        time_nodes="results/synth/{method}/scalability_time_nodes.pdf",
    params:
        panels=[
            dict(
                out="ari_samp",
                x="samp_size",
                y="ari",
                hue="num_nodes",
                ylim=(0, 1),
                xlabel=AXIS_N,
                ylabel="ARI ↑",
                legend="nodes (d)",
            ),
            dict(
                out="time_samp",
                x="samp_size",
                y="runtime_sec",
                hue="num_nodes",
                logy=True,
                xlabel=AXIS_N,
                ylabel="run time (s)",
                legend="nodes (d)",
            ),
            dict(
                out="time_nodes",
                x="num_nodes",
                y="runtime_sec",
                hue="samp_size",
                logy=True,
                xlabel="number of nodes (d)",
                ylabel="run time (s)",
                legend=AXIS_N,
            ),
        ],
    script:
        "../scripts/plot_panels.py"


rule synth_main:
    input:
        expand(
            "results/synth/{method}/{graph}_{metric}_ivn={num_intervs}.pdf",
            method=PIPELINE_METHODS,
            graph=SYNTH_GRAPHS,
            metric=["ari", "fscore"],
            num_intervs=SYNTH_IVNS,
        ),


rule synth_scalability:
    input:
        expand(
            "results/synth/{method}/scalability_{compare}.pdf",
            method=PIPELINE_METHODS,
            compare=["ari_samp", "time_samp", "time_nodes"],
        ),


rule synth_all:
    input:
        rules.synth_main.input,
        rules.synth_scalability.input,
