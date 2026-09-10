# Shared path templates, constants and grids for every synthetic rule file. Included first
# by the Snakefile; defines no rule. Result paths encode the sweep as wildcards:
#   DATA      results/_data/noise=…/tpi=…/<cell>/dataset.npz
#   PIPELINE  results/pipeline/method=…/noise=…/tpi=…/<cell>/{model.pkl,metrics.csv}
#   ORACLE    results/oracle/method=…/lambda_pen=…/<cell>/{model.pkl,metrics.csv}
from itertools import product

CELL = (
    "graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/"
    "density={density}/samp_size={samp_size}/seed={seed}/"
)
DATA = "results/_data/noise={noise}/tpi={targets_per_interv}/" + CELL
PIPELINE = (
    "results/pipeline/method={method}/noise={noise}/tpi={targets_per_interv}/" + CELL
)
ORACLE = "results/oracle/method={method}/lambda_pen={lambda_pen}/" + CELL
# Oracle experiments only ever run on Gaussian single-target data.
ORACLE_DATA = DATA.replace("{noise}", "gaussian").replace("{targets_per_interv}", "1")
DEFAULT = dict(noise=["gaussian"], targets_per_interv=["1"])

ALPHA = 1e-4
LAMBDA_PEN = 1.0
REFINE_TEST = "welch"
INTERVENTION_TYPE = "soft"

# Full-pipeline methods (partition discovery + scoring) and their evaluate.py labels.
PIPELINE_METHODS = ["coarse", "cv", "onepc"]
PIPELINE_LABELS = {"coarse": "COARSE", "cv": "COARSE-CV", "onepc": "COARSE-1PC"}
PIPELINE_K = {"coarse": None, "onepc": 1}
# Oracle-partition methods (COARSEOracle / RePaRe with the true partition injected).
ORACLE_METHODS = ["coarse", "kpc1", "kpc3"]
ORACLE_LABELS = {
    "coarse": "COARSE-oracle",
    "kpc1": "kPC-k1-oracle",
    "kpc3": "kPC-k3-oracle",
    "repare": "RePaRe-oracle",
}
ORACLE_K = {"coarse": None, "kpc1": 1, "kpc3": 3}
NOISES = ["gaussian", "uniform", "laplace"]

# `--config smoke=1` shrinks every grid to p <= 20 (30 where p is fixed), n <= 1000 and
# 2 seeds. Smoke cells sit at the same paths as full-grid cells, so a full run reuses every
# smoke cell whose (p, n) is also in the full grid.
SMOKE = bool(int(config.get("smoke", 0)))
SEEDS = list(range(2 if SMOKE else 20))
DENSITIES = [0.2, 0.5, 0.8]
SAMP_SIZES = (
    [500, 1000] if SMOKE else [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
)
SCAL_NODES = [10, 20] if SMOKE else [10, 20, 50, 100, 200, 500]
SCAL_SAMP = [500, 1000] if SMOKE else [500, 1000, 10000, 100000]
# The two documented exceptions to the n >= 500 floor: exp2 and lambda_samp are about low n.
LOW_N_SAMP = [100, 500] if SMOKE else [100, 200, 500, 1000]
LAMBDA_SAMP = [100, 500] if SMOKE else [100, 500, 1000, 5000]
# Floats so the path renders "1.0", matching the lambda_pen=\d+\.\d+ constraint.
LAMBDAS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def cells(template, guard=None, **grid):
    """Expand `template` over the Cartesian product of `grid`, keeping the cells for which
    `guard(**cell)` is true. Wildcards absent from `grid` stay unexpanded."""
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
    if guard is not None:
        combos = [c for c in combos if guard(**c)]
    if not combos:
        return []
    columns = {k: [c[k] for c in combos] for k in keys}
    return expand(template, zip, allow_missing=True, **columns)


def identifiable(samp_size, num_nodes, **_):
    """Drop cells with n <= p: the per-env covariance is rank-deficient, every score is
    -inf, and COARSECV raises on the all-failed grid. Applied to every scalability grid.
    """
    return samp_size > num_nodes
