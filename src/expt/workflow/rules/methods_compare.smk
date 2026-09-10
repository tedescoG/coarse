# Head-to-head under the oracle partition: COARSE vs 1PC-COARSE (kpc1) vs RePaRe. Each
# receives the same oracle partition, so only the edge-recovery half is benchmarked.
# Shared fits: the coarse/kpc1 cells are kpc.smk exp3's; the RePaRe arm and the SAMP_SIZES
# values outside SCAL_SAMP (n = 2000, 5000, 20000, 50000) are new work.
MC_METHODS = ["coarse", "kpc1", "repare"]
MC_NODES = [10, 20] if SMOKE else [10, 20, 50, 100]
MC_METRICS = ["fscore", "precision", "recall", "runtime_sec"]


rule collect_methods_compare:
    input:
        cells(
            ORACLE + "metrics.csv",
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
        "results/methods_compare/results.csv",
    script:
        "../scripts/collect.py"


# Filenames are parsed by plot_methods_compare.py to recover the filter keys.
rule plot_methods_compare:
    input:
        rules.collect_methods_compare.output[0],
    output:
        expand(
            "results/methods_compare/{graph}_p={p}_dens={d}_{metric}.pdf",
            graph=["er"],
            p=MC_NODES,
            d=DENSITIES,
            metric=MC_METRICS,
        ),
    script:
        "../scripts/plot_methods_compare.py"


rule methods_compare_all:
    input:
        rules.plot_methods_compare.output,
