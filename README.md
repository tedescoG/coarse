# Learning Causal Abstractions by Scoring Coarse Parents

This repository contains the pure-Python implementation of **COARSE** (**C**ausal
**O**rdering via **A**bstraction and **R**ecovery from **S**oft int**E**rventions), the
method developed for my MSc thesis.

The implementation is provided by the `COARSE` class in
[`src/coarse/coarse.py`](src/coarse/coarse.py); cross-validated threshold selection is
provided by `cv_coarse` / `COARSECV` in [`src/coarse/cv.py`](src/coarse/cv.py).

Experiments are organized into a Snakemake workflow, with the
[`src/expt/workflow/Snakefile`](src/expt/workflow/Snakefile) entry point.

## Installation

Dependencies, versioning, and installation are handled by
[`uv`](https://docs.astral.sh/uv/getting-started/) via the included `pyproject.toml` and
`uv.lock`. The first time `uv run <...>` is called, it downloads and installs everything.

```bash
uv sync
```

## Usage

```python
import numpy as np
from coarse import COARSE

# data_dict maps each environment to an (n_e, p) array of samples.
# "obs" is the observational/baseline environment; the other keys are
# soft-intervention environments. COARSE models *soft* interventions only.
data_dict = {
    "obs": X_obs,    # ndarray (n, p)
    1:     X_env1,
    2:     X_env2,
}

model = COARSE().fit(data_dict)
print(model.dag)        # networkx.DiGraph over variable blocks
print(model.partition)  # the inferred partition Π_E
```

## Reproducing the experiments

- `uv run pytest tests/` — quick check from the project root.
- From `src/expt/`:
  - `uv run snakemake results/coarse/er_ari_ivn=2.pdf --cores all` — reproduce one figure.
  - `uv run snakemake all --cores all` — reproduce all experiments; the first call
    downloads the third-party `causalchamber` dataset.

All Snakemake outputs are saved in `src/expt/results/`. Decrease `--cores` (e.g. `10`) as needed.

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
