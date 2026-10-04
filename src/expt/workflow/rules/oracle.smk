# Fits under the true partition: COARSEOracle (plain / kPC / avg) and RePaRe; per-block
# parent recovery for COARSEOracle.


rule fit_oracle:
    input:
        data=ORACLE_DATA + "dataset.npz",
    output:
        temp(ORACLE + "model.pkl"),
    wildcard_constraints:
        method="coarse|kpc1|kpc3|avg",
    resources:
        mem_mb=cell_mem_mb,
    params:
        lambda_pen=lambda w: float(w.lambda_pen),
        intervention_type=INTERVENTION_TYPE,
        k=lambda w: ORACLE_K[w.method],
        avg=lambda w: w.method == "avg",
    script:
        "../scripts/fit_oracle.py"


rule fit_oracle_repare:
    input:
        data=ORACLE_DATA + "dataset.npz",
    output:
        temp(ORACLE + "model.pkl"),
    wildcard_constraints:
        method="repare",
    resources:
        mem_mb=cell_mem_mb,
    params:
        beta=1e-4,
        intervention_type=INTERVENTION_TYPE,
    script:
        "../scripts/fit_oracle_repare.py"


rule evaluate_oracle:
    input:
        data=ORACLE_DATA + "dataset.npz",
        model=ORACLE + "model.pkl",
    output:
        ORACLE + "metrics.csv",
    wildcard_constraints:
        method="coarse|kpc1|kpc3|avg|repare",
    resources:
        mem_mb=cell_mem_mb,
    params:
        method_label=lambda w: ORACLE_LABELS[w.method],
        lambda_pen=lambda w: float(w.lambda_pen),
    script:
        "../scripts/evaluate.py"


rule evaluate_oracle_blocks:
    input:
        data=ORACLE_DATA + "dataset.npz",
        model=ORACLE + "model.pkl",
    output:
        ORACLE + "blocks.csv",
    wildcard_constraints:
        method="coarse|kpc1|kpc3|avg",
    resources:
        mem_mb=cell_mem_mb,
    params:
        method_label=lambda w: ORACLE_LABELS[w.method],
    script:
        "../scripts/evaluate_blocks.py"
