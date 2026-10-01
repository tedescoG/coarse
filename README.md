# Learning Causal Abstractions by Scoring Coarse Parents

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
```

## Tests

```bash
uv run pytest tests/
```

## Reproducing the experiments

Run from `src/expt/`. Outputs go to `src/expt/results/`.

| Experiment | Command |
|---|---|
| Densities | `uv run snakemake densities_all --cores 12` |
| Scalability | `uv run snakemake scalability_all --cores 12 --resources mem_mb=24000` |
| Noise | `uv run snakemake noise_all --cores 12` |
| Method comparison | `uv run snakemake methods_compare_all --cores 12 --resources mem_mb=24000` |
| Causal Chamber | `uv run snakemake causalchamber --cores 4` |
| All of the above | `uv run snakemake all --cores 12 --resources mem_mb=24000 --keep-going` |

- `mem_mb` caps the memory of concurrent cells; the largest scalability cells need about 14 GB.
- Add `--config smoke=1` (after the target) for a quick run on small grids.
- Add `--scheduler greedy` if Snakemake's ILP scheduler fails.
- The first Causal Chamber run downloads the dataset.

## Third-party code

- [RePaRe](https://github.com/Alex-Markham/repare) (GPL-3.0): a dependency. The Causal
  Chamber scripts and `fit_oracle_repare.py` are adapted from its experiments.
- UT-IGSP wrapper `src/expt/workflow/scripts/ut_igsp.py` (BSD-3-Clause, J. L. Gamella): vendored
  from [causal-chamber-paper](https://github.com/juangamella/causal-chamber-paper).
- [GIES](https://github.com/juangamella/gies), [GnIES](https://github.com/juangamella/gnies),
  [sempler](https://github.com/juangamella/sempler) and
  [causaldag](https://github.com/uhlerlab/causaldag) (BSD-3-Clause): dependencies.
- [causalchamber](https://github.com/juangamella/causal-chamber-package) (MIT) and the
  [Causal Chamber datasets](https://github.com/juangamella/causal-chamber) (CC-BY 4.0).

## License

GPL-3.0-or-later (see `COPYING`).

## Citing

```bibtex
@misc{alma99128284276405763,
author = {Tedesco, Gaetano},
copyright = {Unrestricted online access},
keywords = {Causal Abstraction ; Coarsening ; Heterogeneous Data},
language = {eng},
publisher = {Department of Mathematical Sciences, Faculty of Science, University of Copenhagen},
title = {Learning Causal Abstractions by Scoring Coarse Parents},
year = {2026},
}
```
