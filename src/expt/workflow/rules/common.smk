# Path templates, constants and grids of the synthetic experiments. Per-cell outputs:
#   DATA      results/_data/noise=…/tpi=…/<cell>/dataset.npz
#   PIPELINE  results/_run/pipeline/method=…/noise=…/tpi=…/<cell>/{model.pkl,metrics.csv}
#   ORACLE    results/_run/oracle/method=…/noise=…/lambda_pen=…/<cell>/{model.pkl,metrics.csv}
#   THRESHOLD results/_run/threshold/noise=…/tpi=…/<cell>/metrics.csv
#   MULTITARGET results/_run/multitarget/graph=…/num_nodes=…/density=…/seed=…/metrics.csv
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
THRESHOLD = "results/_run/threshold/noise={noise}/tpi={targets_per_interv}/" + CELL
MULTITARGET = (
    "results/_run/multitarget/graph={graph}/num_nodes={num_nodes}/"
    "density={density}/seed={seed}/"
)
# Oracle fits use single-target data; noise.smk sweeps the noise wildcard.
ORACLE_DATA = DATA.replace("{targets_per_interv}", "1")
DEFAULT = dict(noise=["gaussian"], targets_per_interv=["1"])

# Datasets are temp() (a p=1000, n=100000 cell is 4.8 GB); `--config keep_data=1` keeps them.
KEEP_DATA = bool(int(config.get("keep_data", 0)))


def dataset(path):
    return path if KEEP_DATA else temp(path)


def cell_mem_mb(wildcards):
    """Memory budget of one cell, for `--resources mem_mb=<cap>`."""
    envs = 1 + int(wildcards.num_intervs)
    n = int(wildcards.samp_size)
    p = int(wildcards.num_nodes)
    return int(3 * envs * n * p * 8 / 2**20) + 300


ALPHA = 1e-4
LAMBDA_PEN = 1.0
REFINE_TEST = "welch"
INTERVENTION_TYPE = "soft"
# threshold_correction: uncorrected α curve (10^-8 … 10^-1 in quarter decades, plus 0.2) and
# the level of Bonferroni / BH / BY.
TC_ALPHAS = [10 ** (k / 4) for k in range(-32, -3)] + [0.2]
TC_Q = 0.05
# multitarget: size of the fixed target pool of the disjoint and overlap designs.
MT_POOL = 20
# Imported by every plot script; an explicit input since Snakemake hashes only `script:`.
PLOT_STYLE = "workflow/scripts/_plot_style.py"

ORACLE_LABELS = {
    "coarse": "COARSE-oracle",
    "kpc1": "kPC-k1-oracle",
    "kpc3": "kPC-k3-oracle",
    "repare": "RePaRe-oracle",
}
ORACLE_K = {"coarse": None, "kpc1": 1, "kpc3": 3}
NOISES = ["gaussian", "uniform", "laplace", "t3", "t4"]

# `--config smoke=1`: small grids and 2 seeds; collected outputs go under results/smoke/.
SMOKE = bool(int(config.get("smoke", 0)))
RESULTS = "results/smoke/" if SMOKE else "results/"
SEEDS = list(range(2 if SMOKE else 20))
DENSITIES = [0.2, 0.5, 0.8]
DENS_NODES = [10]
DENS_SAMP = (
    [100, 500, 1000]
    if SMOKE
    else [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
)
SAMP_SIZES = [1000, 2000] if SMOKE else [1000, 2000, 5000, 10000, 20000, 50000, 100000]
SCAL_NODES = [10, 20] if SMOKE else [10, 20, 50, 100, 200, 500, 1000]
SCAL_SAMP = [1000, 5000] if SMOKE else [1000, 5000, 10000, 50000, 100000]
NOISE_NODES = [10, 20] if SMOKE else [50, 100, 200, 500]


def cells(template, **grid):
    """Expand `template` over the product of `grid`."""
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]
    if not combos:
        return []
    columns = {k: [c[k] for c in combos] for k in keys}
    return expand(template, zip, allow_missing=True, **columns)
