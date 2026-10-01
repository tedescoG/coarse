# Densities sweep for COARSE: density × n at p = 10, per graph family and ivn.
DENS_IVNS = [2, 5, 8]
DENS_GRAPHS = ["er", "sf"]
DENS_METRICS = ["ari", "fscore", "runtime_sec"]
DENS_DIR = RESULTS + "densities/coarse/"


rule collect_densities:
    input:
        cells(
            PIPELINE + "metrics.csv",
            **DEFAULT,
            method=["coarse"],
            num_nodes=DENS_NODES,
            density=DENSITIES,
            samp_size=DENS_SAMP,
            seed=SEEDS,
        ),
    output:
        DENS_DIR + "{graph}_results_ivn={num_intervs}.csv",
    script:
        "../scripts/collect.py"


rule summarize_densities:
    input:
        rules.collect_densities.output[0],
    output:
        DENS_DIR + "{graph}_summary_ivn={num_intervs}.csv",
    params:
        by=["density", "samp_size"],
        metrics=DENS_METRICS,
    script:
        "../scripts/summarize.py"


rule plot_densities:
    input:
        rules.collect_densities.output[0],
        style=PLOT_STYLE,
    output:
        **{
            m: DENS_DIR + "{graph}_" + m + "_ivn={num_intervs}.pdf"
            for m in DENS_METRICS
        },
    params:
        panels=[dict(out=m, x="samp_size", y=m, hue="density") for m in DENS_METRICS],
    script:
        "../scripts/plot_panels.py"


rule densities_all:
    input:
        expand(
            DENS_DIR + "{graph}_{stem}_ivn={num_intervs}.pdf",
            graph=DENS_GRAPHS,
            stem=DENS_METRICS,
            num_intervs=DENS_IVNS,
        ),
        expand(
            DENS_DIR + "{graph}_summary_ivn={num_intervs}.csv",
            graph=DENS_GRAPHS,
            num_intervs=DENS_IVNS,
        ),
