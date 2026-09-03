# 3-method head-to-head under the oracle partition: COARSE vs 1pc-COARSE vs
# RePaRe. Each method receives the *same* oracle partition from
# `_common.build_oracle_partition`, so partition discovery is bypassed and only
# the edge-recovery half of each algorithm is benchmarked.
#
# Output namespace: `results/methods_compare/{coarse,onepc,repare}/...`.
# Fits hit fit_oracle.py (COARSE / 1pc — kPC wildcards select 1pc) or the new
# fit_oracle_repare.py (RePaRe). All three feed the existing evaluate.py with
# distinct `method_label` params so the collected CSV carries the right
# `method` column for the plot legend.

data_path = "results/_data/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
coarse_oracle_path = "results/methods_compare/coarse/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"
onepc_oracle_path = "results/methods_compare/onepc/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/k={k}/pca={pca}/seed={seed}/"
repare_oracle_path = "results/methods_compare/repare/graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/density={density}/samp_size={samp_size}/seed={seed}/"


wildcard_constraints:
    graph="er|sf",
    num_intervs=r"\d+",
    k=r"\d+",
    pca=r"obs|pooled",


GRAPHS = ["er"]
NUM_NODES = [10, 20, 50, 100]
DENSITY = [0.2, 0.5, 0.8]
NUM_INTERVS = [5]
SAMP_SIZE = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
SEEDS = list(range(10))
METRICS = ["fscore", "precision", "recall", "runtime_sec"]


# ---------------------------------------------------------------------------
# Fit rules — COARSE and 1pc reuse scripts/fit_oracle.py (1pc selects rank-1
# PCA via the k/pca wildcards encoded in onepc_oracle_path). RePaRe uses the
# new scripts/fit_oracle_repare.py.
# ---------------------------------------------------------------------------

rule fit_methods_compare_coarse:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(coarse_oracle_path + "model.pkl"),
    params:
        lambda_pen=1.0,
        intervention_type="soft",
    script:
        "../scripts/fit_oracle.py"


rule fit_methods_compare_onepc:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(onepc_oracle_path + "model.pkl"),
    params:
        lambda_pen=1.0,
        intervention_type="soft",
    script:
        "../scripts/fit_oracle.py"


rule fit_methods_compare_repare:
    input:
        data=data_path + "dataset.npz",
    output:
        temp(repare_oracle_path + "model.pkl"),
    params:
        beta=1e-4,
        intervention_type="soft",
    script:
        "../scripts/fit_oracle_repare.py"


# ---------------------------------------------------------------------------
# Evaluate rules — same evaluate.py for all three, distinguished only by
# the `method_label` param. metrics.csv files are *not* temp(): they're the
# permanent artefacts the collect/plot stage reads.
# ---------------------------------------------------------------------------

rule evaluate_methods_compare_coarse:
    input:
        data=data_path + "dataset.npz",
        model=coarse_oracle_path + "model.pkl",
    output:
        coarse_oracle_path + "metrics.csv",
    params:
        method_label="COARSE",
        pca_pooled="na",
        lambda_pen=1.0,
    script:
        "../scripts/evaluate.py"


rule evaluate_methods_compare_onepc:
    input:
        data=data_path + "dataset.npz",
        model=onepc_oracle_path + "model.pkl",
    output:
        onepc_oracle_path + "metrics.csv",
    params:
        method_label="COARSE-1PC",
        pca_pooled=lambda w: w.pca,
        lambda_pen=1.0,
    script:
        "../scripts/evaluate.py"


rule evaluate_methods_compare_repare:
    input:
        data=data_path + "dataset.npz",
        model=repare_oracle_path + "model.pkl",
    output:
        repare_oracle_path + "metrics.csv",
    params:
        method_label="RePaRe",
        pca_pooled="na",
        lambda_pen=1.0,  # placeholder; RePaRe has no penalty, keep column non-NaN for CSV consistency
    script:
        "../scripts/evaluate.py"


# ---------------------------------------------------------------------------
# Collect + plot.
# ---------------------------------------------------------------------------

rule collect_methods_compare:
    input:
        expand(
            coarse_oracle_path + "metrics.csv",
            graph=GRAPHS,
            num_nodes=NUM_NODES,
            num_intervs=NUM_INTERVS,
            density=DENSITY,
            samp_size=SAMP_SIZE,
            seed=SEEDS,
        ),
        expand(
            onepc_oracle_path + "metrics.csv",
            graph=GRAPHS,
            num_nodes=NUM_NODES,
            num_intervs=NUM_INTERVS,
            density=DENSITY,
            samp_size=SAMP_SIZE,
            k=[1],
            pca=["obs"],
            seed=SEEDS,
        ),
        expand(
            repare_oracle_path + "metrics.csv",
            graph=GRAPHS,
            num_nodes=NUM_NODES,
            num_intervs=NUM_INTERVS,
            density=DENSITY,
            samp_size=SAMP_SIZE,
            seed=SEEDS,
        ),
    output:
        "results/methods_compare/results.csv",
    script:
        "../scripts/collect.py"


rule plot_methods_compare:
    input:
        rules.collect_methods_compare.output[0],
    output:
        expand(
            "results/methods_compare/{graph}_p={p}_dens={d}_{metric}.pdf",
            graph=GRAPHS,
            p=NUM_NODES,
            d=DENSITY,
            metric=METRICS,
        ),
    script:
        "../scripts/plot_methods_compare.py"


rule methods_compare_all:
    input:
        rules.plot_methods_compare.output,
