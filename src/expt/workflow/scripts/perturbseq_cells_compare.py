# ruff: noqa: F821
"""One Snakemake job = one fit.

Stage "fold" (wildcards split/method/candidate/fold): fit one candidate on the outer
training batches minus one validation pair, score on that pair. Stage "final": fit the
candidate selected by perturbseq_cells_select.py on the full outer training set and
score on the held-out pair. Splits overlap and are exploratory; within-batch cell
dependence is ignored.
"""

import json
import pickle
import random
import time
import traceback

import networkx as nx
import numpy as np
from _perturbseq_common import (
    BATCH_PAIRS,
    PS_GRIDS,
    inner_validation_pairs,
    outer_train_batches,
)
from perturbseq_cells_fit import (
    graph_density_score,
    heldout_agreement,
    serialize_graph,
    singleton_graph,
    subset,
)

from coarse import COARSE
from coarse.normalize import apply_inverse_normal, inverse_normal_reference

DIAGNOSTIC_ALPHA = 0.01


def int_apply(fit_data, eval_data):
    """Per-gene inverse-normal transform anchored on the training control cells.

    Applied uniformly to every method so densities stay comparable; anchors come from
    `fit_data['obs']` only, so no evaluation cells inform the reference."""
    ref = inverse_normal_reference(fit_data["obs"])
    return (
        {k: apply_inverse_normal(ref, v) for k, v in fit_data.items()},
        {k: apply_inverse_normal(ref, v) for k, v in eval_data.items()},
    )


def fit_graph(method, data, params, seed=0):
    """Fit on given environments; targets are not supplied to any discovery method."""
    p = data["obs"].shape[1]
    if method == "coarse":
        model = COARSE(rng=np.random.default_rng(seed)).fit(
            data, alpha=params[0], refine_test="welch"
        )
        if not np.isfinite(model.score):
            raise ValueError("COARSE returned non-finite BIC")
        return model.dag
    if method == "coarse_scored":
        from coarse.coarse import COARSEOracle
        from coarse.scored_partition import null_calibrated_evidence, scored_partition

        order = sorted(k for k in data if k != "obs")
        z, _ = null_calibrated_evidence(
            data, mean_only=True, rng=np.random.default_rng(seed)
        )
        n_e = [float(len(data[k])) for k in order]
        sp = scored_partition(
            z,
            n_e,
            float(len(data["obs"])),
            lam=params[0],
            c=params[1],
            # Blocks stay within the score phase's estimability region (the per-env
            # BIC needs n_e > r_j + s_j), so evidence-null genes split into several
            # same-signature blocks instead of one unscorable mega-block; sp.partition
            # -- not the M-row canonicalization of sp.M -- preserves those splits.
            size_cap=max(int(min(n_e)) - 1, 1),
        )
        model = COARSEOracle(rng=np.random.default_rng(seed)).fit(
            sp.partition, sp.M, order, data
        )
        if not np.isfinite(model.score):
            raise ValueError("COARSEOracle returned non-finite BIC")
        return model.dag
    if method in ("coarse_kpc1", "coarse_kpc3", "coarse_avg"):
        # Score-phase compressions of the same constraint-based partition: blocks are
        # projected onto the observational environment's leading directions (kPC) or
        # standardized mean (avg) before the block-BIC search; the partition, supports
        # and candidate pools are identical to unconstrained coarse.
        kwargs = {"avg": True} if method == "coarse_avg" else {"k": int(method[-1])}
        model = COARSE(rng=np.random.default_rng(seed)).fit(
            data, alpha=params[0], refine_test="welch", **kwargs
        )
        if not np.isfinite(model.score):
            raise ValueError(f"{method} returned non-finite BIC")
        return model.dag
    if method == "repare":
        from repare.repare import PartitionDagModelIvn

        # Explicitly soft: this version uses target metadata only to skip hard interventions.
        inputs = {"obs": (data["obs"], set(), "obs")}
        inputs.update({k: (v, set(), "soft") for k, v in data.items() if k != "obs"})
        model = PartitionDagModelIvn(rng=np.random.default_rng(seed)).fit(
            inputs, alpha=params[0], beta=params[1], assume="gaussian"
        )
        return model.dag
    if method == "repare_welch":
        import repare_welch

        # Same procedure and grid as repare with ONLY the partition test swapped to
        # the mean-only Welch t-test; the edge phase keeps RePaRe's Gaussian
        # canonical-correlation CI tests (edge_assume, see the vendored header).
        inputs = {"obs": (data["obs"], set(), "obs")}
        inputs.update({k: (v, set(), "soft") for k, v in data.items() if k != "obs"})
        model = repare_welch.PartitionDagModelIvn(rng=np.random.default_rng(seed))
        model.edge_assume = "gaussian"
        model.fit(inputs, alpha=params[0], beta=params[1], assume=None, refine_test="welch")
        return model.dag
    if method == "utigsp":
        import ut_igsp

        random.seed(seed)
        np.random.seed(seed)
        envs = sorted(k for k in data if k != "obs")
        adj, _ = ut_igsp.fit(
            [data["obs"], *(data[e] for e in envs)],
            alpha_ci=params[0],
            alpha_inv=params[1],
            test="gauss",
            obs_idx=0,
        )
        if adj.shape != (p, p):
            raise ValueError(f"UT-IGSP returned adjacency of shape {adj.shape}")
        graph = singleton_graph(p)
        graph.add_edges_from(((int(i),), (int(j),)) for i, j in zip(*np.nonzero(adj)))
        return graph
    if method == "gnies":
        import gnies
        from gnies.utils import pdag_to_dag

        arrays = [data["obs"], *(data[k] for k in sorted(data) if k != "obs")]
        penalty = params[0] * 0.5 * np.log(sum(map(len, arrays)))
        _, pdag, _ = gnies.fit(data=arrays, lmbda=penalty, known_targets=set())
        # Extend in feature order, without using held-out scores to orient edges.
        adj = pdag_to_dag(pdag)
        graph = singleton_graph(p)
        graph.add_edges_from(((int(i),), (int(j),)) for i, j in zip(*np.nonzero(adj)))
        return graph
    if method == "singleton_empty":
        return singleton_graph(p)
    if method == "joint":
        graph = nx.DiGraph()
        graph.add_node(tuple(range(p)))
        return graph
    raise ValueError(f"unknown method {method}")


def error_record(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=8),
    }


def run_fold(data, features, split, method, candidate, fold, fit=fit_graph):
    """Fit one candidate on one inner training set and score one validation pair."""
    if method not in PS_GRIDS:
        raise ValueError(f"unknown method {method}")
    params = PS_GRIDS[method][candidate]
    valid_pair = inner_validation_pairs(split)[fold]
    train_batches = tuple(b for b in outer_train_batches(split) if b not in valid_pair)
    record = {
        "split": split,
        "method": method,
        "candidate": candidate,
        "fold": fold,
        "parameters": list(params),
        "train_batches": list(train_batches),
        "validation_batches": list(valid_pair),
        "status": "failed",
    }
    start = time.perf_counter()
    try:
        fit_data = subset(data, train_batches)
        valid_data = subset(data, valid_pair)
        fit_data, valid_data = int_apply(fit_data, valid_data)
        graph = fit(method, fit_data, params)
        score = graph_density_score(graph, fit_data, valid_data)["mean_per_gene"]
        if not np.isfinite(score):
            raise ValueError("non-finite validation score")
        record["score"] = float(score)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - retain fold failure
        record["error"] = error_record(exc)
    record["seconds"] = time.perf_counter() - start
    return record


def run_final(data, features, split, method, selection, fit=fit_graph):
    """Fit the selected candidate on the outer training set; score the held-out pair."""
    result = {
        "split": split,
        "method": method,
        "test_batches": list(BATCH_PAIRS[split]),
        "diagnostic_alpha": DIAGNOSTIC_ALPHA,
        "selection": selection,
        "status": "failed",
    }
    if selection.get("status") != "ok":
        result["error"] = {
            "message": "no candidate succeeded in every inner fold",
            "candidate_errors": selection.get("candidates"),
        }
        return result
    params = tuple(selection["selected_parameters"])
    try:
        start = time.perf_counter()
        train = subset(data, outer_train_batches(split))
        test = subset(data, BATCH_PAIRS[split])
        train, test = int_apply(train, test)
        graph = fit(method, train, params)
        result["fit_seconds"] = time.perf_counter() - start
        result.update(serialize_graph(graph, features))
        result["heldout_loglik"] = graph_density_score(graph, train, test)
        if method in {
            "coarse",
            "coarse_kpc1",
            "coarse_kpc3",
            "coarse_avg",
            "coarse_scored",
            "repare",
            "repare_welch",
        }:
            result["shift_agreement"] = heldout_agreement(
                graph, train, test, DIAGNOSTIC_ALPHA
            )
        if method in {"coarse", "coarse_kpc1", "coarse_kpc3", "coarse_avg", "coarse_scored"}:
            without_edges = graph.copy()
            without_edges.remove_edges_from(list(without_edges.edges))
            result["same_blocks_no_edges_loglik"] = graph_density_score(
                without_edges, train, test
            )
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - retain final fit/scoring failure
        result["error"] = error_record(exc)
    return result


def main():
    with open(snakemake.input.cells, "rb") as f:
        data = pickle.load(f)
    with open(snakemake.input.metadata) as f:
        metadata = json.load(f)
    features = metadata["features"]
    split = int(snakemake.wildcards.split)
    method = snakemake.wildcards.method
    if snakemake.params.stage == "fold":
        result = run_fold(
            data,
            features,
            split,
            method,
            int(snakemake.wildcards.candidate),
            int(snakemake.wildcards.fold),
        )
    else:
        with open(snakemake.input.selection) as f:
            selection = json.load(f)
        result = run_final(data, features, split, method, selection)
        result["protocol"] = {
            "evaluation": (
                "Gaussian, environment-specific OLS and shrunk covariance on held-out cells"
            ),
            "int": (
                "Per-gene inverse-normal transform anchored on training control cells; "
                "all methods fit and evaluated on transformed data"
            ),
            "warning": "Within-batch cell dependence is ignored; splits overlap and are exploratory.",
            "repare": "Gaussian LRT and Gaussian CI; unknown direct targets, soft interventions.",
            "repare_welch": (
                "RePaRe with the partition test swapped to the mean-only Welch "
                "t-test (vendored repare_welch module, edge phase unchanged); "
                "otherwise as repare."
            ),
            "utigsp": "Gaussian CI and invariance tests; unknown direct targets.",
            "gnies": "Unknown targets; penalty multiplier on 0.5 log(n); fixed consistent DAG extension.",
            "coarse": "Welch shift test and block BIC; unknown direct targets.",
            "coarse_kpc1": (
                "COARSE with blocks projected to the observational environment's "
                "leading principal direction before the block-BIC search; score "
                "family uses k_j in place of r_j; unknown direct targets."
            ),
            "coarse_kpc3": (
                "COARSE with blocks projected to the first three principal "
                "directions; otherwise as coarse_kpc1."
            ),
            "coarse_avg": (
                "COARSE with each block summarized by the standardized mean "
                "(rank-one, fixed uniform direction); otherwise as coarse_kpc1."
            ),
            "coarse_scored": (
                "Baseline-null-calibrated squared-Welch (mean-only) evidence in "
                "units of null MAD; agglomerative scored partition with declaration "
                "cost lambda and merge tolerance c tuned on inner folds; block BIC."
            ),
            "data_selection": metadata["note"],
        }
    with open(snakemake.output.result, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
