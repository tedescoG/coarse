# Score-Based Learning of Cluster DAGs from Interventions

**COARSE** (**C**ausal **O**rdering via **A**bstraction and **R**ecovery from **S**oft
int**E**rventions) learns a partition of the variables and a DAG over its blocks from
observational and soft-interventional data.

## Installation

With [uv](https://docs.astral.sh/uv/getting-started/):

```bash
uv sync
```

## Usage

```python
from coarse import COARSE, COARSECV

# One (n_e, p) array per environment; "obs" is the observational one.
data_dict = {"obs": X_obs, 1: X_env1, 2: X_env2}

model = COARSE().fit(data_dict, alpha=1e-4)
model.partition   # list of blocks (frozensets of column indices)
model.dag         # networkx.DiGraph over the blocks

model = COARSECV().fit(data_dict)   # alpha chosen by 10-fold cross-validation

COARSE().fit(data_dict, k=1)        # kPC-COARSE: score each block's top-k principal components
COARSE().fit(data_dict, avg=True)   # avg-COARSE: score each block's mean
COARSEOracle().fit(partition, M, env_order, data_dict)   # score step under a given partition
```

## Tests

```bash
uv run pytest tests/
```

## Reproducing the experiments

Run from `src/expt/`. Outputs go to `src/expt/results/`.

| Experiment | What it measures | Command |
|---|---|---|
| Densities | COARSE vs n across graph densities (p = 10) | `uv run snakemake densities_all --cores 12` |
| Scalability | COARSE as p and n grow | `uv run snakemake scalability_all --cores 12 --resources mem_mb=24000` |
| Noise | score step under non-Gaussian noise | `uv run snakemake noise_all --cores 12` |
| Method comparison | COARSE, avg-COARSE and RePaRe under the true partition | `uv run snakemake methods_compare_all --cores 12 --resources mem_mb=24000` |
| λ sensitivity | BIC penalty multiplier | `uv run snakemake lambda_sens_all --cores 12` |
| Threshold correction | descendant-test cutoff rules (α, Bonferroni, BH, BY, CV) | `uv run snakemake threshold_correction_all --cores 12` |
| Multi-target | partition under multi-target interventions (population and estimated) | `uv run snakemake multitarget_all --cores 12` |
| High-dimensional | score step for n ≤ p (COARSE, kPC, avg) | `uv run snakemake highdim_all --cores 12` |
| Causal Chamber | real-data benchmark against RePaRe, GIES, GnIES, UT-IGSP | `uv run snakemake causalchamber --cores 4` |
| All of the above | | `uv run snakemake all --cores 12 --resources mem_mb=24000 --keep-going` |

`coarse_all` runs every COARSE-only sweep (densities, scalability, noise, λ, threshold, high-dim).

- `mem_mb` caps the memory of concurrent cells; the largest scalability cells need about 14 GB.
- Add `--config smoke=1` (after the target) for a quick run on small grids.
- Add `--scheduler greedy` if Snakemake's ILP scheduler fails.
- The first Causal Chamber run downloads the dataset.

## Third-party code

- [RePaRe](https://github.com/Alex-Markham/repare) (GPL-3.0): a dependency. The synthetic
  pipeline, the Causal Chamber scripts and `fit_oracle_repare.py` are adapted from its experiments
  (files marked `Adapted from RePaRe`).
- UT-IGSP wrapper `src/expt/workflow/scripts/ut_igsp.py` (BSD-3-Clause, J. L. Gamella): vendored
  from [causal-chamber-paper](https://github.com/juangamella/causal-chamber-paper).
- [GIES](https://github.com/juangamella/gies), [GnIES](https://github.com/juangamella/gnies),
  [sempler](https://github.com/juangamella/sempler) and
  [causaldag](https://github.com/uhlerlab/causaldag) (BSD-3-Clause): dependencies.
- [causalchamber](https://github.com/juangamella/causal-chamber-package) (MIT) and the
  [Causal Chamber datasets](https://github.com/juangamella/causal-chamber) (CC-BY 4.0).

## License

GPL-3.0-or-later (see `COPYING`).

