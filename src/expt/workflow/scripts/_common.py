"""Shared helpers for snakemake scripts. Not part of the coarse public API.

Snakemake's `script:` directive adds the script's directory to sys.path before
execution, so sibling imports like `from _common import ...` work without any
extra wiring. The leading underscore signals "module-local; do not import from
elsewhere in the codebase."
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from coarse.partition import infer_partition


def normalize_intervention_type(value: str) -> str:
    """COARSE supports only soft (shift) interventions; reject anything
    else explicitly.

    Replaces the silent hard/do fallthrough from the previous four `_map_type`
    copies — COARSE has no hard-intervention pathway (Algorithm 1's two-sample
    test detects shift, not structural cuts), so a 'hard' string would silently
    mislead the algorithm. A raise surfaces the misconfiguration at fit time.
    """
    lowered = str(value).lower()
    if lowered == "soft":
        return "soft"
    raise ValueError(f"COARSE supports only soft interventions; got {value!r}")


def parse_targets_per_interv(token):
    """Parse the optional ``targets_per_interv`` snakemake wildcard.

    Returns the ``size`` argument for ``sempler.generators.intervention_targets``:
    an ``int`` for fixed-size targets, or a ``(min, max)`` tuple for
    heterogeneous (random-size) targets.

    Token grammar:
      - ``None`` (wildcard absent)  → ``1``  (single-target, backward-compatible default)
      - ``"k"`` (integer string)    → ``int(k)``
      - ``"AtoB"`` (e.g. ``"1to5"``) → ``(A, B)``

    Lives in _common.py rather than inline in generate.py so the parser is
    importable from the test suite (generate.py is a snakemake script and
    therefore not importable as a plain Python module).
    """
    if token is None:
        return 1
    s = str(token)
    if "to" in s:
        lo, hi = s.split("to", 1)
        return (int(lo), int(hi))
    return int(s)


def build_data_dict(data, targets, intervention_type: str) -> dict:
    """Build the (data, targets, type) dict from a generate.py .npz archive.

    Used by fit.py and fit_oracle.py — keeping the construction in one place
    means a future change to the env-tuple shape only needs one edit.
    """
    data_dict = {"obs": (data["obs"], set(), "obs")}
    for idx, target in enumerate(targets):
        tgt = set(np.atleast_1d(target).astype(int))
        data_dict[str(idx)] = (data[str(idx)], tgt, intervention_type)
    return data_dict


def build_oracle_partition(weights: np.ndarray, targets, num_nodes: int):
    """Reconstruct (true_dag, M_true, env_order, partition) from a ground-truth
    weight matrix and intervention targets.

    Each entry of ``targets`` is an iterable of node indices — typically a
    single int (the canonical single-target case) but possibly a set/tuple of
    several ints (multi-target interventions). The affected mask for an
    environment is the union of each target node together with all of its
    descendants in the true DAG.

    For singleton targets this reduces to ``{t} ∪ descendants(t)`` — identical
    to the historical single-target behavior — so cached results from existing
    single-target sweeps are unchanged.
    """
    true_dag = nx.DiGraph(weights.astype(bool))
    masks = []
    for target in targets:
        target_nodes = [int(n) for n in np.atleast_1d(target)]
        affected: set[int] = set(target_nodes)
        for node in target_nodes:
            affected.update(nx.descendants(true_dag, node))
        mask = np.zeros(num_nodes, dtype=bool)
        mask[list(affected)] = True
        masks.append(mask)
    M_true = np.column_stack(masks).astype(bool)
    env_order = [str(idx) for idx in range(len(targets))]
    partition = infer_partition(M_true)
    return true_dag, M_true, env_order, partition
