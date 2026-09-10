# generate → fit → evaluate for the full-pipeline methods (COARSE, COARSE-CV, COARSE-1PC).
# Snakemake cannot pick `script:` by wildcard, so the two fit rules dispatch on disjoint
# `method` constraints. Never add a third fit rule here without one.


rule generate:
    output:
        DATA + "dataset.npz",
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
        method="coarse|onepc",
    params:
        alpha=ALPHA,
        lambda_pen=LAMBDA_PEN,
        refine_test=REFINE_TEST,
        intervention_type=INTERVENTION_TYPE,
        k=lambda w: PIPELINE_K[w.method],
    script:
        "../scripts/fit.py"


rule fit_cv:
    input:
        data=DATA + "dataset.npz",
    output:
        temp(PIPELINE + "model.pkl"),
    wildcard_constraints:
        method="cv",
    params:
        lambda_pen=LAMBDA_PEN,
        refine_test=REFINE_TEST,
        intervention_type=INTERVENTION_TYPE,
    script:
        "../scripts/fit_cv.py"


rule evaluate:
    input:
        data=DATA + "dataset.npz",
        model=PIPELINE + "model.pkl",
    output:
        PIPELINE + "metrics.csv",
    wildcard_constraints:
        method="coarse|cv|onepc",
    params:
        method_label=lambda w: PIPELINE_LABELS[w.method],
        lambda_pen=LAMBDA_PEN,
    script:
        "../scripts/evaluate.py"
