# Oracle-partition fits: COARSEOracle (plain / kPC) and RePaRe with the true partition
# injected. Only ever on Gaussian single-target data (ORACLE_DATA). λ is a path wildcard
# because lambda.smk sweeps it; every other consumer expands lambda_pen=[LAMBDA_PEN].


rule fit_oracle:
    input:
        data=ORACLE_DATA + "dataset.npz",
    output:
        temp(ORACLE + "model.pkl"),
    wildcard_constraints:
        method="coarse|kpc1|kpc3",
    params:
        lambda_pen=lambda w: float(w.lambda_pen),
        intervention_type=INTERVENTION_TYPE,
        k=lambda w: ORACLE_K[w.method],
    script:
        "../scripts/fit_oracle.py"


rule fit_oracle_repare:
    input:
        data=ORACLE_DATA + "dataset.npz",
    output:
        temp(ORACLE + "model.pkl"),
    wildcard_constraints:
        method="repare",
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
        method="coarse|kpc1|kpc3|repare",
    params:
        method_label=lambda w: ORACLE_LABELS[w.method],
        # RePaRe has no penalty; the column is kept non-NaN for CSV consistency.
        lambda_pen=lambda w: float(w.lambda_pen),
    script:
        "../scripts/evaluate.py"
