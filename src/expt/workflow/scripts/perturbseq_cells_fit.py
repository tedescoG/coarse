# ruff: noqa: F821
"""Exploratory cell-level fit with capture-batch validation and held-out scoring.

Each model sees the same cells. Hyperparameters are chosen using training batches only.
The evaluation density uses environment-specific OLS and diagonal covariance shrinkage;
it is a common post-fit score, not COARSE's BIC. Cells within batches are dependent.
"""

import itertools
import json
import pickle
import random

import networkx as nx
import numpy as np

from coarse import COARSE
from coarse.hypothesis_tests import compute_M

TRAIN_BATCHES = (1, 2, 3, 4, 5, 6)
TEST_BATCHES = (7, 8)
VALIDATION_BATCHES = ((1, 2), (3, 4), (5, 6))
SHRINKAGE = 0.1
FLOOR = 1e-6


def subset(data, batches):
    """Subset every environment by batch, rejecting missing batches."""
    out = {}
    for env, record in data.items():
        X = np.asarray(record["X"], dtype=float)
        b = np.asarray(record["batch"])
        if X.ndim != 2 or len(X) != len(b) or not np.isfinite(X).all():
            raise ValueError(f"invalid data for {env}")
        if not set(batches).issubset(set(b)):
            raise ValueError(f"missing batches in {env}")
        out[env] = X[np.isin(b, batches)]
    return out


def validate_training_variation(data, features):
    """Fail early when an environment has a constant training feature."""
    folds = [TRAIN_BATCHES] + [
        tuple(b for b in TRAIN_BATCHES if b not in validation)
        for validation in VALIDATION_BATCHES
    ]
    failures = []
    for batches in folds:
        for env, X in subset(data, batches).items():
            constant = np.flatnonzero(np.ptp(X, axis=0) == 0)
            if len(constant):
                failures.append(
                    f"batches={batches}, env={env}: {[features[i] for i in constant]}"
                )
    if failures:
        raise ValueError(
            "constant training features; COARSE BIC is singular: " + "; ".join(failures)
        )


def singleton_graph(p):
    g = nx.DiGraph()
    g.add_nodes_from((i,) for i in range(p))
    return g


def graph_density_score(graph, train, test, shrinkage=SHRINKAGE):
    """Mean per-gene log density, averaging environments equally.

    Within each observed environment, fit an intercept, coefficients and residual
    covariance using only that environment's training cells. The same shrinkage and
    variance floor apply to all graphs; parent values are observed at test time.
    """
    if set(train) != set(test) or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("environment mismatch or graph is not a DAG")
    p = train["obs"].shape[1]
    nodes = list(graph.nodes)
    columns = [int(i) for node in nodes for i in node]
    if sorted(columns) != list(range(p)):
        raise ValueError("graph nodes do not partition the features")
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be in (0, 1]")
    # Floor is derived exclusively from baseline training cells.
    variance_floor = np.maximum(np.var(train["obs"], axis=0), 1e-12) * FLOOR
    per_env = {}
    for env, X in train.items():
        Y = test[env]
        if len(X) < 2 or len(Y) < 1:
            raise ValueError(f"insufficient cells for {env}")
        mean = X.mean(axis=0)
        Xt, Xv = X - mean, Y - mean
        logp = np.zeros(len(Y))
        for node in nodes:
            cols = list(node)
            parents = sorted(i for pa in graph.predecessors(node) for i in pa)
            if parents:
                coef = np.linalg.lstsq(Xt[:, parents], Xt[:, cols], rcond=None)[0]
                train_resid = Xt[:, cols] - Xt[:, parents] @ coef
                test_resid = Xv[:, cols] - Xv[:, parents] @ coef
            else:
                train_resid, test_resid = Xt[:, cols], Xv[:, cols]
            cov = train_resid.T @ train_resid / len(X)
            cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
            cov += np.diag(variance_floor[cols])
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(logdet):
                raise ValueError(f"non-positive covariance in {env}")
            solved = np.linalg.solve(cov, test_resid.T).T
            logp += -0.5 * (
                len(cols) * np.log(2 * np.pi)
                + logdet
                + np.einsum("ij,ij->i", test_resid, solved)
            )
        per_env[env] = float(logp.mean() / p)
        if not np.isfinite(per_env[env]):
            raise ValueError(f"non-finite held-out likelihood for {env}")
    return {
        "mean_per_gene": float(np.mean(list(per_env.values()))),
        "per_environment": per_env,
    }


def heldout_agreement(graph, train, test, alpha):
    """Diagnostic Welch support agreement; p-values do not account for batch clustering."""
    train_M, order = compute_M(train, alpha=alpha, test_name="welch")
    test_M, test_order = compute_M(test, alpha=alpha, test_name="welch")
    if order != test_order:
        raise ValueError("environment order mismatch")
    p = test_M.shape[0]
    pairs = list(itertools.combinations(range(p), 2))
    within = [
        tuple(sorted(pair))
        for node in graph.nodes
        for pair in itertools.combinations(node, 2)
    ]

    def rates(items):
        if not items:
            return None
        left = np.array([test_M[i] for i, _ in items])
        right = np.array([test_M[j] for _, j in items])
        return {
            "both_detected": float((left & right).mean()),
            "both_undetected": float((~left & ~right).mean()),
            "disagree": float((left != right).mean()),
        }

    identifiable = bool(
        within and len(within) != len(pairs) and np.any(test_M != test_M[0])
    )
    return {
        "environment_order": order,
        "training_shift_matrix": train_M.astype(int).tolist(),
        "heldout_shift_matrix": test_M.astype(int).tolist(),
        "informative": identifiable,
        "within": rates(within) if identifiable else None,
        "matched_random_expectation": rates(pairs) if identifiable else None,
        "warning": "Cell-level Welch calls ignore dependence within capture batches.",
    }


def serialize_graph(graph, features):
    return {
        "blocks": [[features[i] for i in n] for n in graph.nodes],
        "edges": [
            [[features[i] for i in u], [features[i] for i in v]] for u, v in graph.edges
        ],
    }


def fit_and_evaluate(data, metadata, alphas, include_utigsp=False):
    """Fixed outer holdout; three inner batch folds for COARSE alpha selection."""
    train, test = subset(data, TRAIN_BATCHES), subset(data, TEST_BATCHES)
    features = metadata["features"]
    validate_training_variation(data, features)
    validation, validation_errors = {}, {}
    for alpha in alphas:
        fold_scores = []
        for valid_batches in VALIDATION_BATCHES:
            fit_batches = tuple(b for b in TRAIN_BATCHES if b not in valid_batches)
            fit_data, valid_data = (
                subset(data, fit_batches),
                subset(data, valid_batches),
            )
            try:
                model = COARSE(rng=np.random.default_rng(0)).fit(
                    fit_data, alpha=alpha, refine_test="welch"
                )
                if not np.isfinite(model.score):
                    raise ValueError("COARSE returned non-finite training score")
                fold_scores.append(
                    graph_density_score(model.dag, fit_data, valid_data)[
                        "mean_per_gene"
                    ]
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                fold_scores.append(None)
                validation_errors[f"{alpha}:{valid_batches}"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        validation[str(alpha)] = fold_scores
    valid = [a for a in alphas if all(v is not None for v in validation[str(a)])]
    if not valid:
        raise RuntimeError(
            f"all alpha candidates failed validation: {validation_errors}"
        )
    selected = max(valid, key=lambda a: (np.mean(validation[str(a)]), -a))
    model = COARSE(rng=np.random.default_rng(0)).fit(
        train, alpha=selected, refine_test="welch"
    )
    if not np.isfinite(model.score):
        raise ValueError("COARSE returned non-finite final training score")
    baseline = singleton_graph(len(features))
    result = {
        "scope": "Exploratory, not proof of causal robustness to latent confounding.",
        "train_batches": list(TRAIN_BATCHES),
        "test_batches": list(TEST_BATCHES),
        "validation_batches": [list(x) for x in VALIDATION_BATCHES],
        "alpha_grid": list(alphas),
        "validation_loglik_per_gene": validation,
        "validation_errors": validation_errors,
        "selected_alpha": float(selected),
        "evaluation": {
            "covariance_shrinkage": SHRINKAGE,
            "variance_floor_factor": FLOOR,
            "environment_weighting": "equal",
            "parent_values": "observed",
            "note": "Gaussian score on original features; within-batch cell dependence ignored.",
        },
        "methods": {},
        "metadata": metadata,
    }
    for label, graph in [("coarse", model.dag), ("singleton_empty_dag", baseline)]:
        result["methods"][label] = {
            **serialize_graph(graph, features),
            "heldout_loglik": graph_density_score(graph, train, test),
        }
    result["methods"]["coarse"]["shift_agreement"] = heldout_agreement(
        model.dag, train, test, selected
    )
    if include_utigsp:
        # Fixed thresholds, unlike CV-tuned COARSE; failures are reported, not omitted.
        try:
            import ut_igsp

            random.seed(0)
            np.random.seed(0)
            envs = sorted(k for k in train if k != "obs")
            adj, _ = ut_igsp.fit(
                [train["obs"], *(train[e] for e in envs)],
                alpha_ci=0.01,
                alpha_inv=0.01,
                test="gauss",
                obs_idx=0,
            )
            g = singleton_graph(len(features))
            g.add_edges_from(((int(i),), (int(j),)) for i, j in zip(*np.nonzero(adj)))
            result["methods"]["utigsp_fixed_0.01"] = {
                **serialize_graph(g, features),
                "heldout_loglik": graph_density_score(g, train, test),
                "note": "Fixed CI and invariance thresholds; not CV-tuned.",
            }
        except Exception as exc:  # noqa: BLE001 - report baseline failures in the output
            result["methods"]["utigsp_fixed_0.01"] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
    return result


def main():
    with open(snakemake.input.cells, "rb") as f:
        data = pickle.load(f)
    with open(snakemake.input.metadata) as f:
        metadata = json.load(f)
    result = fit_and_evaluate(
        data,
        metadata,
        tuple(snakemake.params.alphas),
        bool(snakemake.params.include_utigsp),
    )
    with open(snakemake.output.summary, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
