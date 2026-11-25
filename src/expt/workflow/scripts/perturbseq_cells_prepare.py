# ruff: noqa: F821
"""Prepare the approved 22-gene Norman subset as individual-cell samples.

Expression is taken as supplied by GEARS; it is not aggregated by capture batch.
Batch IDs are retained for splitting, not treated as biological replicates.
"""

import hashlib
import json
import pickle
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from _perturbseq_common import canonical_condition
from perturbseq_prepare import _ensure_sources, _load_identities
from scipy import stats

ENVS = ("CEBPB", "CEBPA", "IRF1", "PRDM1")
GENES = (
    "CEBPB",
    "CEBPA",
    "IRF1",
    "PRDM1",
    "UCP2",
    "HSPA1A",
    "MYC",
    "NPM1",
    "HSP90AA1",
    "STMN1",
    "GAPDH",
    "COMT",
    "MT2A",
    "TIMP1",
    "TGFB1",
    "LGALS1",
    "ODC1",
    "BIRC5",
    "SOCS2",
    "LMO2",
    "PCNA",
    "MKI67",
)
COLLECTRI_SHA256 = "98ef1163146d2e28ee413e7c909174998cd9d742ea93d97ccfe93f3a14c160c1"


# Panel sizes routed to the blind environment-and-gene selection; other sizes fall
# back to the fixed 22-gene anchor or the 40-gene CollecTRI-guided panel.
BLIND_PANEL_SIZES = (50, 100, 200, 500, 1000)


def _select_panel(adata, ids, names, panel_size):
    """Blind selection of five perturbation environments and a gene panel.

    Environments: all guide conditions with at least 20 cells in every capture batch,
    ranked by the number of genes with a detected mean shift against controls
    (Welch, BH FDR 1%, all measured genes);
    five are picked greedily to maximize the minimum pairwise Jaccard distance of
    their significant-gene sets, skipping candidates that share a perturbed gene with
    an already-picked environment.
    Genes: the perturbed genes plus the highest-control-variance genes detected in at
    least 30% of control cells, up to panel_size.
    No regulator database enters the selection; CollecTRI is used only as an
    evaluation-side reference. Environments are chosen to have detectable marginal
    effects, which favors every method equally since all consume the same data."""
    candidates = ids[ids.env.notna() & (ids.env != "obs")]
    counts = (
        candidates.groupby(["env", "gemgroup"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 9), fill_value=0)
    )
    qualifying = counts[(counts >= 20).all(axis=1)]
    env_rows = {
        env: adata.obs_names.isin(candidates[candidates.env == env].index)
        for env in qualifying.index
    }
    obs_rows = adata.obs_names.isin(ids[ids.env == "obs"].index)
    controls = adata[obs_rows].to_memory().X
    m0 = np.asarray(controls.mean(axis=0)).ravel()
    v0 = np.maximum(np.asarray(controls.power(2).mean(axis=0)).ravel() - m0**2, 1e-12)
    n0 = controls.shape[0]
    sig_sets, counts_sig, cell_counts = {}, {}, {}
    for env in qualifying.index:
        env_cells = adata[env_rows[env]].to_memory().X
        me = np.asarray(env_cells.mean(axis=0)).ravel()
        ve = np.maximum(
            np.asarray(env_cells.power(2).mean(axis=0)).ravel() - me**2, 1e-12
        )
        ne = env_cells.shape[0]
        se0, se1 = v0 / n0, ve / ne
        t2 = (me - m0) ** 2 / (se0 + se1)
        df = (se0 + se1) ** 2 / (se0**2 / (n0 - 1) + se1**2 / (ne - 1))
        q = stats.false_discovery_control(stats.f.sf(t2, 1, df), method="bh")
        sig_sets[env] = frozenset(np.flatnonzero(q < 0.01).tolist())
        counts_sig[env] = len(sig_sets[env])
        cell_counts[env] = ne
    ranked = sorted(qualifying.index, key=lambda e: (-counts_sig[e], e))

    def jaccard(a, b):
        union = sig_sets[a] | sig_sets[b]
        inter = sig_sets[a] & sig_sets[b]
        return len(inter) / len(union) if union else 0.0

    def targets_of(env):
        return set(env.split("_"))

    picked = [ranked[0]]
    while len(picked) < 5:
        picked_targets = set().union(*map(targets_of, picked))
        rest = [
            e
            for e in ranked[:15]
            if e not in picked and not targets_of(e) & picked_targets
        ]
        picked.append(
            max(
                rest,
                key=lambda e: (
                    min(jaccard(e, p) for p in picked),
                    counts_sig[e],
                    e,
                ),
            )
        )
    envs = sorted(picked)

    picked_genes = sorted({g for e in envs for g in e.split("_")})
    detection = np.asarray((controls > 0).mean(axis=0)).ravel()
    cands = [
        i
        for i, g in enumerate(names)
        if names.count(g) == 1
        and detection[i] >= 0.3
        and v0[i] > 0
        and g not in picked_genes
    ]
    cands.sort(key=lambda i: (-v0[i], -detection[i], names[i]))
    genes = picked_genes + [names[i] for i in cands[: panel_size - len(picked_genes)]]
    selection = {
        "protocol": (
            "Environments: guide conditions with at least 20 cells in every batch, "
            "ranked by genes with a mean shift vs controls (Welch, BH FDR 1%); "
            "five picked to maximize minimum pairwise Jaccard distance of "
            "significant-gene sets, skipping shared perturbed genes. "
            "Genes: perturbed genes plus top control-variance genes detected in "
            "at least 30% of controls. Selection uses observed marginals only; "
            "no regulator database."
        ),
        "universe_size": len(qualifying),
        "candidates": {
            env: {
                "cells": int(cell_counts[env]),
                "significant_genes": int(counts_sig[env]),
            }
            for env in ranked[:15]
        },
        "picked": envs,
    }
    return envs, genes, selection


def prepare(root, panel_size=22):
    h5ad, identities, reference = _ensure_sources(Path(root))
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    if digest != COLLECTRI_SHA256:
        raise ValueError(f"CollecTRI snapshot mismatch: {digest}")
    ids = _load_identities(identities).set_index("cell_barcode", verify_integrity=True)
    adata = anndata.read_h5ad(h5ad, backed="r")
    names = list(adata.var["gene_name"])
    envs, genes = list(ENVS), list(GENES)
    selection = None
    if panel_size in BLIND_PANEL_SIZES:
        envs, genes, selection = _select_panel(adata, ids, names, panel_size)
    ids = ids[ids.env.isin(["obs", *envs])]
    rows = np.flatnonzero(adata.obs_names.isin(ids.index))
    cell_ids = adata.obs_names[rows]
    meta = ids.loc[cell_ids].copy()
    conditions = adata.obs.iloc[rows]["condition"].astype(str).map(canonical_condition)
    if not np.array_equal(meta.env.to_numpy(), conditions.to_numpy()):
        raise ValueError("GEO guide assignments and GEARS conditions disagree")
    sub = adata[rows].to_memory()
    ref = pd.read_csv(reference, sep="\t")
    if panel_size == 40:
        controls = sub.X[meta.env.to_numpy() == "obs"]
        detection = dict(zip(names, np.asarray((controls > 0).mean(axis=0)).ravel()))
        links = ref[ref.source_genesymbol.isin(ENVS)].drop_duplicates(
            ["source_genesymbol", "target_genesymbol"]
        )
        targets = links.groupby("target_genesymbol").size()
        candidates = [
            g
            for g in targets.index
            if g not in {*GENES, "FOXA1", "FOSB"}
            and detection.get(g, 0) >= 0.3
            and names.count(g) == 1
        ]
        genes.extend(
            sorted(candidates, key=lambda g: (targets[g], -detection[g], g))[:18]
        )
    if len(genes) != panel_size or any(names.count(g) != 1 for g in genes):
        raise ValueError(
            f"requested {panel_size} unique features; assembled {len(genes)}"
        )
    X = sub.X[:, [names.index(g) for g in genes]]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    if not np.isfinite(X).all():
        raise ValueError("non-finite expression")
    data = {}
    for env in ["obs", *envs]:
        mask = meta.env.to_numpy() == env
        data[env] = {"X": X[mask], "batch": meta.gemgroup.to_numpy()[mask]}
        counts = pd.Series(data[env]["batch"]).value_counts()
        if any(counts.get(batch, 0) < 20 for batch in range(1, 9)):
            raise ValueError(f"{env}: fewer than 20 cells in a required batch")
    ref = ref.loc[
        ref.source_genesymbol.isin(genes) & ref.target_genesymbol.isin(genes),
        ["source_genesymbol", "target_genesymbol"],
    ].drop_duplicates()
    return data, {
        "features": genes,
        "panel_selection": (
            "Perturbed genes plus top control-variance genes detected in at least "
            "30% of controls."
            if panel_size in BLIND_PANEL_SIZES
            else "Anchor 22 plus CollecTRI targets of the four perturbed TFs detected in "
            "at least 30% of controls; rank by fewer regulators, higher detection, "
            "then symbol."
        ),
        "environments": list(envs),
        "environments_selection": selection,
        "cells_per_batch": {
            env: {
                str(k): int(v)
                for k, v in sorted(pd.Series(d["batch"]).value_counts().items())
            }
            for env, d in data.items()
        },
        "collectri_edges": ref.values.tolist(),
        "collectri_sha256": digest,
        "excluded_from_initial_subset": {
            "features_and_environments": ["FOXA1", "FOSB"],
            "reason": "Constant expression in training environments caused singular covariances.",
        },
        "note": (
            "Exploratory; cells within capture batches need not be independent. "
            "Gene selection used control expression from all batches, including the test batches."
            + (
                " Environments were selected for detectable marginal effects, "
                "which favors every method equally."
                if panel_size in BLIND_PANEL_SIZES
                else ""
            )
        ),
    }


def main():
    data, metadata = prepare(snakemake.params.root, snakemake.params.panel_size)
    with open(snakemake.output.cells, "wb") as f:
        pickle.dump(data, f)
    with open(snakemake.output.metadata, "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
