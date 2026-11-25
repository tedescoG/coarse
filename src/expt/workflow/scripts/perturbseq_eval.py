#!/usr/bin/env python3
"""Ground-truth-free eval of a fitted coarse DAG on held-out environments.

SKETCH: the protocol below should be agreed on before the first run.

- Environments are split into `n_env_folds` folds (LOO when the fold count reaches the
  env count); each fold's block parameters are fit on the training envs and frozen for
  the held-out envs.
- Data is centered by the observational baseline mean, matching the centered block model
  (Def. 15): `X_j = B_j X_pa(j) + eps_j`, residuals Gaussian.
- Held-out metrics per env: RMSE and Pearson of the predicted block-mean shifts
  (through the fitted B matrices) against the observed shifts, RMSE of the null
  predictor "no shift", and mean per-gene Gaussian log-likelihood of the held-out
  pseudobulk rows under the frozen parameters.
"""

import json
import pickle

import numpy as np
import pandas as pd

from _causalchamber_common import dag_from_labeled_parts


def _fit_blocks(dag, X_train):
    """`{node: (B, Sigma)}` fit by OLS per block on pooled centered training rows."""
    params = {}
    for node in dag.nodes:
        node_cols = list(node)
        parents = [p for parent in dag.predecessors(node) for p in parent]
        Y = X_train[:, node_cols]
        if parents:
            Xp = X_train[:, parents]
            beta, *_ = np.linalg.lstsq(Xp, Y, rcond=None)
            resid = Y - Xp @ beta
        else:
            beta = None
            resid = Y - Y.mean(axis=0)
        params[node] = (beta, np.atleast_2d(np.cov(resid, rowvar=False)))
    return params


def _mvn_logpdf(rows, mean, cov):
    """Mean logpdf of `rows` under N(mean, cov); nan when `cov` is degenerate."""
    d = cov.shape[0]
    if rows.shape[0] <= d:
        return float("nan")
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    diff = rows - mean
    sol = np.linalg.solve(cov, diff.T).T
    maha = np.einsum("ij,ij->i", diff, sol)
    return float(np.mean(-0.5 * (d * np.log(2 * np.pi) + logdet + maha)))


def _predict_shifts(dag, params, mean_shift):
    """(pred, obs) component vectors of the block-mean shifts predicted through parents."""
    preds, obs = [], []
    for node in dag.nodes:
        parents = [p for parent in dag.predecessors(node) for p in parent]
        beta, _ = params[node]
        obs.append(mean_shift[list(node)])
        preds.append(beta @ mean_shift[parents] if beta is not None else np.zeros(len(node)))
    return np.concatenate(preds), np.concatenate(obs)


def evaluate(dag, blocks, env_labels, n_env_folds):
    """Environment-fold CV of the held-out metrics; returns one record per env."""
    mu0 = blocks["obs"].mean(axis=0)
    centered = {label: blocks[label] - mu0 for label in ["obs", *env_labels]}
    train_pool = np.vstack([centered["obs"], *(centered[e] for e in env_labels)])

    fold_size = max(1, len(env_labels) // n_env_folds)
    folds = (
        [env_labels[i : i + fold_size] for i in range(0, len(env_labels), fold_size)]
        if len(env_labels) > n_env_folds
        else [[e] for e in env_labels]
    )

    records = []
    for held_out in folds:
        train_envs = [e for e in env_labels if e not in held_out]
        X_train = np.vstack([centered[e] for e in train_envs])
        params = _fit_blocks(dag, X_train)

        for env in held_out:
            Xe = centered[env]
            mean_shift = Xe.mean(axis=0)
            pred, obs = _predict_shifts(dag, params, mean_shift)
            rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
            null_rmse = float(np.sqrt(np.mean(obs**2)))
            pearson = (
                float(np.corrcoef(pred, obs)[0, 1])
                if pred.size > 1 and np.std(pred) > 0 and np.std(obs) > 0
                else float("nan")
            )
            logliks = []
            for node in dag.nodes:
                parents = [p for parent in dag.predecessors(node) for p in parent]
                beta, cov = params[node]
                if beta is None:
                    rows = Xe[:, list(node)]
                    logliks.append(_mvn_logpdf(rows, np.zeros(len(node)), cov))
                else:
                    Xp = Xe[:, parents]
                    logliks.append(
                        _mvn_logpdf(
                            Xe[:, list(node)] - Xp @ beta,
                            np.zeros(len(node)),
                            cov,
                        )
                    )
            loglik_per_row = float(np.nanmean(logliks)) if logliks else float("nan")
            records.append(
                {
                    "env": env,
                    "rmse": rmse,
                    "null_rmse": null_rmse,
                    "pearson": pearson,
                    "loglik_per_row": loglik_per_row,
                }
            )
    return records


def main():
    with open(snakemake.input.blocks, "rb") as f:
        blocks = pickle.load(f)
    with open(snakemake.input.nametoidx, "rb") as f:
        name_to_idx = pickle.load(f)
    with open(snakemake.input.features, "r") as f:
        feature_cols = json.load(f)["feature_cols"]
    with open(snakemake.input.params, "r") as f:
        payload = json.load(f)

    dag = dag_from_labeled_parts(payload["parts"], payload["edges"], name_to_idx)
    env_labels = payload.get("env_labels") or [label for label in blocks if label != "obs"]
    n_env_folds = int(snakemake.params.n_env_folds)

    records = evaluate(dag, blocks, sorted(env_labels), n_env_folds)
    for r in records:
        r.update(
            method=snakemake.wildcards.method,
            mode=snakemake.wildcards.mode,
            n_env_folds=n_env_folds,
            num_parts=dag.number_of_nodes(),
            num_edges=dag.number_of_edges(),
        )
    mean_record = {
        key: float(np.nanmean([r[key] for r in records]))
        for key in ("rmse", "null_rmse", "pearson", "loglik_per_row")
    }
    mean_record.update(
        env="__mean__",
        method=snakemake.wildcards.method,
        mode=snakemake.wildcards.mode,
        n_env_folds=n_env_folds,
        num_parts=dag.number_of_nodes(),
        num_edges=dag.number_of_edges(),
    )
    pd.DataFrame([*records, mean_record]).to_csv(snakemake.output.metrics, index=False)


if __name__ == "__main__":
    main()
