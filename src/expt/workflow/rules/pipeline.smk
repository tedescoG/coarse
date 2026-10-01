# generate → fit → evaluate for full-pipeline COARSE (partition discovery + scoring).


rule generate:
    output:
        dataset(DATA + "dataset.npz"),
    resources:
        mem_mb=cell_mem_mb,
    params:
        intervention_type=INTERVENTION_TYPE,
    script:
        "../scripts/generate.py"


rule fit_coarse:
    input:
        data=DATA + "dataset.npz",
    output:
        temp(PIPELINE + "model.pkl"),
    wildcard_constraints:
        method="coarse",
    resources:
        mem_mb=cell_mem_mb,
    params:
        alpha=ALPHA,
        lambda_pen=LAMBDA_PEN,
        refine_test=REFINE_TEST,
        intervention_type=INTERVENTION_TYPE,
        # Kept so existing cells' params provenance still matches.
        k=None,
    script:
        "../scripts/fit.py"


rule evaluate:
    input:
        data=DATA + "dataset.npz",
        model=PIPELINE + "model.pkl",
    output:
        PIPELINE + "metrics.csv",
    wildcard_constraints:
        method="coarse",
    resources:
        mem_mb=cell_mem_mb,
    params:
        method_label="COARSE",
        lambda_pen=LAMBDA_PEN,
    script:
        "../scripts/evaluate.py"
