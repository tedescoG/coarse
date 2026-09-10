# kPC-COARSE under the oracle partition (COARSEOracle). Fits come from oracle.smk.
#   exp2 — kPC vs plain COARSE at low n (documented exception to the n >= 500 floor).
#   exp3 — scalability across (density, num_nodes) for COARSE / kPC-k1 / kPC-k3.
# Shared fits: exp2's λ=1 cells at n ∈ LAMBDA_SAMP are lambda_samp's (n=200 is exp2-only);
# exp3's coarse/kpc1 cells are methods_compare's. Nothing is fit twice.
EXP2_NODES = [30]
EXP3_NODES = [10, 20] if SMOKE else [10, 20, 50, 100, 200]
EXP3_STEMS = ["fscore_vs_n", "runtime_vs_n", "runtime_vs_d"]
EXP3_PANEL = {
    "fscore_vs_n": dict(x="samp_size", y="fscore", hue="num_nodes"),
    "runtime_vs_n": dict(x="samp_size", y="runtime_sec", hue="num_nodes"),
    "runtime_vs_d": dict(x="num_nodes", y="runtime_sec", hue="samp_size"),
}


rule collect_exp2:
    input:
        cells(
            ORACLE + "metrics.csv",
            method=ORACLE_METHODS,
            lambda_pen=[LAMBDA_PEN],
            graph=["er"],
            num_nodes=EXP2_NODES,
            num_intervs=[5],
            density=[0.5],
            samp_size=LOW_N_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/kpc/exp2_results.csv",
    script:
        "../scripts/collect.py"


rule plot_exp2:
    input:
        rules.collect_exp2.output[0],
        style=PLOT_STYLE,
    output:
        fscore="results/kpc/exp2_fscore.pdf",
        runtime="results/kpc/exp2_runtime.pdf",
    params:
        panels=[
            dict(out="fscore", x="samp_size", y="fscore", hue="method"),
            dict(out="runtime", x="samp_size", y="runtime_sec", hue="method"),
        ],
    script:
        "../scripts/plot_panels.py"


rule collect_exp3:
    input:
        cells(
            ORACLE + "metrics.csv",
            method=ORACLE_METHODS,
            lambda_pen=[LAMBDA_PEN],
            graph=["er"],
            num_nodes=EXP3_NODES,
            num_intervs=[5],
            density=DENSITIES,
            samp_size=SCAL_SAMP,
            seed=SEEDS,
        ),
    output:
        "results/kpc/exp3_results.csv",
    script:
        "../scripts/collect.py"


# One standalone PDF per (panel, method) per density: readable at thesis scale, unlike a
# 3×3 grid. Output keys are `<stem>_<method>`.
rule plot_exp3:
    input:
        rules.collect_exp3.output[0],
        style=PLOT_STYLE,
    output:
        **{
            f"{stem}_{m}": f"results/kpc/exp3/density={{density}}/{stem}_{m}.pdf"
            for stem in EXP3_STEMS
            for m in ORACLE_METHODS
        },
    params:
        panels=lambda w: [
            dict(
                out=f"{stem}_{m}",
                filter={"density": float(w.density), "method": ORACLE_LABELS[m]},
                **EXP3_PANEL[stem],
            )
            for stem in EXP3_STEMS
            for m in ORACLE_METHODS
        ],
    script:
        "../scripts/plot_panels.py"


rule exp2:
    input:
        rules.plot_exp2.output,


rule exp3:
    input:
        expand(
            "results/kpc/exp3/density={density}/{stem}_{method}.pdf",
            density=DENSITIES,
            stem=EXP3_STEMS,
            method=ORACLE_METHODS,
        ),
