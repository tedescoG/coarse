# Shared path templates, constants and grids for every synthetic rule file. Included first
# by the Snakefile; defines no rule. Per-cell paths encode the sweep as wildcards and live
# under results/_data (datasets) and results/_run (fits + per-cell scorecards); the collected
# CSVs and figures of each experiment live directly under RESULTS (results/<experiment>/):
#   DATA      results/_data/noise=…/tpi=…/<cell>/dataset.npz
#   PIPELINE  results/_run/pipeline/method=…/noise=…/tpi=…/<cell>/{model.pkl,metrics.csv}
#   ORACLE    results/_run/oracle/method=…/noise=…/lambda_pen=…/<cell>/{model.pkl,metrics.csv}
from itertools import product

CELL = (
    "graph={graph}/num_nodes={num_nodes}/num_intervs={num_intervs}/"
    "density={density}/samp_size={samp_size}/seed={seed}/"
)
DATA = "results/_data/noise={noise}/tpi={targets_per_interv}/" + CELL
PIPELINE = (
    "results/_run/pipeline/method={method}/noise={noise}/tpi={targets_per_interv}/"
    + CELL
)
ORACLE = (
    "results/_run/oracle/method={method}/noise={noise}/lambda_pen={lambda_pen}/" + CELL
)
# Oracle experiments only ever run on single-target data; noise stays a wildcard because
# noise.smk sweeps it. Every other oracle consumer expands noise=["gaussian"].
ORACLE_DATA = DATA.replace("{targets_per_interv}", "1")
DEFAULT = dict(noise=["gaussian"], targets_per_interv=["1"])

# Datasets are temp() by default: a p=500, n=100000 cell is 2.4 GB and make_dataset is
# seed-deterministic, so a re-run regenerates it in seconds. `--config keep_data=1` keeps them.
KEEP_DATA = bool(int(config.get("keep_data", 0)))


def dataset(path):
    return path if KEEP_DATA else temp(path)


def cell_mem_mb(wildcards):
    """Memory budget of one cell: the dataset, its centered copy and covariance headroom.
    Declared as `resources: mem_mb` on every per-cell rule so `--resources mem_mb=<cap>`
    limits how many p=500, n=100000 cells run at once."""
    envs = 1 + int(wildcards.num_intervs)
    n = int(wildcards.samp_size)
    p = int(wildcards.num_nodes)
    return int(3 * envs * n * p * 8 / 2**20) + 300


ALPHA = 1e-4
LAMBDA_PEN = 1.0
REFINE_TEST = "welch"
INTERVENTION_TYPE = "soft"
# Every plot script imports this module; Snakemake hashes only the `script:` file, so it
# must be an explicit input or a style change leaves every figure stale.
PLOT_STYLE = "workflow/scripts/_plot_style.py"

# Oracle-partition methods (COARSEOracle / RePaRe with the true partition injected).
ORACLE_LABELS = {
    "coarse": "COARSE-oracle",
    "kpc1": "kPC-k1-oracle",
    "kpc3": "kPC-k3-oracle",
    "repare": "RePaRe-oracle",
}
ORACLE_K = {"coarse": None, "kpc1": 1, "kpc3": 3}
NOISES = ["gaussian", "uniform", "laplace", "t3", "t4"]

# `--config smoke=1` shrinks every grid to p <= 20 (30 where p is fixed), small n and
# 2 seeds. Smoke *cells* sit at the same per-cell paths as full-grid cells, so a full run
# reuses every smoke cell whose (p, n) is also in the full grid; the *collected* CSVs and
# figures go under results/smoke/ instead, so a smoke run can never masquerade as a finished
# experiment (a full run then still schedules its own collect + plot).
SMOKE = bool(int(config.get("smoke", 0)))
RESULTS = "results/smoke/" if SMOKE else "results/"
SEEDS = list(range(2 if SMOKE else 20))
DENSITIES = [0.2, 0.5, 0.8]
# Densities sweep (p fixed, n from 100).
DENS_NODES = [10]
DENS_SAMP = (
    [100, 500, 1000]
    if SMOKE
    else [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
)
# Head-to-head (methods_compare) n-grid.
SAMP_SIZES = (
    [500, 1000] if SMOKE else [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
)
# Scalability grid; the smoke n-grid keeps two values so a line can be drawn.
SCAL_NODES = [10, 20] if SMOKE else [10, 20, 50, 100, 200, 500]
SCAL_SAMP = [100, 1000] if SMOKE else [100, 1000, 10000, 100000]
# Noise robustness (oracle partition) shares the scalability n-grid so Gaussian datasets
# are generated once for both experiments.
NOISE_NODES = [10, 20] if SMOKE else [10, 50, 100, 200]


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
    -inf, and COARSECV raises on the all-failed grid. Applied to every grid whose smallest
    n does not exceed its largest p (scalability, noise).
    """
    return samp_size > num_nodes
