"""Helpers shared by the Perturb-seq scripts."""

from __future__ import annotations

import itertools

# Shared capture batches; each outer split holds out one pair.
BATCH_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8))

# One candidate tuple per tuning parameter set; fixed baselines take no parameters.
# COARSE's score-phase compressions are separate entries with fixed compression:
# coarse_kpc1/kpc3 project each block to its first k principal directions of the
# observational environment, coarse_avg to the standardized mean; each tunes alpha
# like unconstrained coarse. (The retired scored-partition entry tuned lambda/c knobs.)
PS_GRIDS = {
    "coarse": ((1e-4,), (1e-3,), (1e-2,), (0.1,)),
    "coarse_kpc1": ((1e-4,), (1e-3,), (1e-2,), (0.1,)),
    "coarse_kpc3": ((1e-4,), (1e-3,), (1e-2,), (0.1,)),
    "coarse_avg": ((1e-4,), (1e-3,), (1e-2,), (0.1,)),
    # The retired scored-partition partition replacement: baseline-null-calibrated
    # mean-only evidence with two tuned knobs, declaration cost (lambda) and merge
    # tolerance (c); score-phase BIC penalty at its default.
    "coarse_scored": tuple(itertools.product((0.5, 1.0, 2.0, 4.0), (0.03, 1.0))),
    # repare's cell-level grid started at alpha=1e-3 while every coarse variant could
    # tune to 1e-4; the strict levels are appended AFTER the original four so existing
    # fold files keep their candidate indices (the recorded parameters stay valid).
    "repare": tuple(itertools.product((1e-3, 0.1), repeat=2))
    + tuple(itertools.product((1e-4, 1e-2), (1e-3, 0.1))),
    # Same grid and edge phase as repare, but the partition test is the mean-only
    # Welch t-test (vendored module), matching the constraint-based procedure's
    # partition instrument exactly.
    "repare_welch": tuple(itertools.product((1e-3, 0.1), repeat=2))
    + tuple(itertools.product((1e-4, 1e-2), (1e-3, 0.1))),
    "utigsp": tuple(itertools.product((1e-3, 0.1), repeat=2)),
    "gnies": ((0.5,), (1.0,), (2.0,), (4.0,)),
    "singleton_empty": ((),),
    "joint": ((),),
}
PS_FIXED_METHODS = frozenset({"singleton_empty", "joint"})
PS_METHODS = tuple(PS_GRIDS)


def outer_train_batches(split: int) -> tuple[int, ...]:
    """The three batch pairs kept for training when pair `split` is held out."""
    if not 0 <= split < len(BATCH_PAIRS):
        raise ValueError(f"invalid split {split}")
    test_pair = BATCH_PAIRS[split]
    return tuple(b for pair in BATCH_PAIRS if pair != test_pair for b in pair)


def inner_validation_pairs(split: int) -> tuple[tuple[int, int], ...]:
    """The three batch pairs used as inner validation folds for outer split `split`."""
    if not 0 <= split < len(BATCH_PAIRS):
        raise ValueError(f"invalid split {split}")
    test_pair = BATCH_PAIRS[split]
    return tuple(pair for pair in BATCH_PAIRS if pair != test_pair)


def canonical_condition(condition: str) -> str:
    """Canonicalize a GEARS condition label to the GEO environment convention."""
    genes = sorted(g for g in condition.split("+") if g != "ctrl")
    return "_".join(genes) if genes else "obs"


def ps_targets(
    mode: str,
    group_targets: dict[str, set[int]],
    single_env_labels: list[str],
    name_to_idx: dict[str, int],
) -> dict[str, set[int]]:
    """`{env: targets}`: "grouped" keeps every perturbation env (pairs multi-target,
    singles single-target); "ungrouped" keeps single-gene envs only."""
    if mode == "grouped":
        targets = {label: set(t) for label, t in group_targets.items()}
        for label in single_env_labels:
            targets[label] = {name_to_idx[label]}
        return targets
    if mode == "ungrouped":
        return {label: {name_to_idx[label]} for label in single_env_labels}
    raise ValueError(f"Unknown mode: {mode!r}")
