# Multi-target interventions, full pipelines (COARSE, COARSE-CV, COARSE-1PC): does
# partition discovery itself survive |targets| > 1? Data and fits come from pipeline.smk
# through the `targets_per_interv` wildcard ("k" → every env intervenes on k nodes;
# "AtoB" → env sizes drawn uniformly in [A, B]; parser: _common.parse_targets_per_interv).
MT_NODES = [10, 20] if SMOKE else [10, 20, 50]
MT_DENSITY = [0.2, 0.5]
MT_TPI = ["2", "3", "5", "1to5"]
MT_SAMP = [500, 1000] if SMOKE else [500, 2000, 10000, 50000]
MT_METRICS = ["fscore", "ari", "runtime_sec"]
MT_SUMMARY_METRICS = ["fscore", "ari"]


rule collect_multitarget:
    input:
        cells(
            PIPELINE + "metrics.csv",
            method=PIPELINE_METHODS,
            noise=["gaussian"],
            targets_per_interv=MT_TPI,
            graph=["er"],
            num_nodes=MT_NODES,
            num_intervs=[5],
            density=MT_DENSITY,
            samp_size=MT_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/multitarget/results.csv",
    script:
        "../scripts/collect.py"


# (a) per-(graph, p, density, tpi) line plot, x=samp_size, hue=method;
# (b) headline summary, x=tpi (ordered), hue=method, faceted by (p, density).
# Filenames are parsed by plot_multitarget.py to recover the filter keys.
rule plot_multitarget:
    input:
        rules.collect_multitarget.output[0],
    output:
        expand(
            "results/multitarget/{graph}_p={p}_dens={d}_tpi={tpi}_{metric}.pdf",
            graph=["er"],
            p=MT_NODES,
            d=MT_DENSITY,
            tpi=MT_TPI,
            metric=MT_METRICS,
        )
        + expand("results/multitarget/summary_{metric}.pdf", metric=MT_SUMMARY_METRICS),
    script:
        "../scripts/plot_multitarget.py"


rule multitarget_all:
    input:
        rules.plot_multitarget.output,
