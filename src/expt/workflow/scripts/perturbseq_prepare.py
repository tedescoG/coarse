#!/usr/bin/env python3
"""Norman et al. (2019) K562 CRISPRa Perturb-seq: gemgroup pseudobulk + CollecTRI regulons.

Data sources are pinned:

- Expression + conditions: GEARS-processed h5ad (log-normalized, 91,205 cells x 5,045
  genes; Harvard Dataverse file 6154020, md5 cdc41d6050e619c37fd9dd44d440e2b4). Its
  `cell_type` column says A549 (GEARS metadata bug); the data is Norman K562.
- Guide assignment + capture batch: GEO GSE133344 `filtered_cell_identities.csv.gz`
  (md5 5011c6e2cafc61c46a60f6ad061f90c4), joined on the 10x cell barcode (97.4% join).
- TF-target reference: CollecTRI via the OmniPath REST API (TSV snapshot, fetched
  2026-11-20; the API serves the current release, so re-runs may drift).

Smoke mode (`--config smoke=1`): 4 single-gene environments + controls over 10
CollecTRI-connected feature genes, one pseudobulk row per (environment, gemgroup).
Full mode: TODO (all 105 genes, combo environments, all-vs-regulon gene filter).
"""

import hashlib
import json
import pickle
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GEARS_ZIP_URL = "https://dataverse.harvard.edu/api/access/datafile/6154020"
GEARS_ZIP_MD5 = "cdc41d6050e619c37fd9dd44d440e2b4"
IDENTITIES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/"
    "GSE133344_filtered_cell_identities.csv.gz"
)
IDENTITIES_MD5 = "5011c6e2cafc61c46a60f6ad061f90c4"
COLLECTRI_URL = (
    "https://omnipathdb.org/interactions?datasets=collectri&organisms=9606"
    "&genesymbols=true&limit=100000"
)
COLLECTRI_DATE = "2026-11-20"

MIN_CELLS_PER_GEMGROUP = 20
# Cell-level matrices for UT-IGSP (its Gaussian CI / invariance tests need per-env
# regressions that 8 gemgroup rows cannot support); seeded caps keep it fast.
SMOKE_MAX_CELLS_OBS = 2000
SMOKE_MAX_CELLS_PER_ENV = 500
# Smoke subset (`--config smoke=1`); pairs are full-run material. Features are
# CollecTRI targets of the four perturbed TFs that are well measured in control cells
# (>= 35% nonzero), except the TFs themselves, which CRISPRa raises in their own env.
SMOKE_ENVS = ("KLF1", "CEBPE", "ETS2", "EGR1")
SMOKE_GENES = (
    "KLF1",
    "HBG2",
    "GYPB",
    "CEBPE",
    "SPI1",
    "ETS2",
    "KRT8",
    "EGR1",
    "GADD45A",
    "CCND3",
)


# The Dataverse front rejects the bare `Python-urllib` User-Agent with 403; an
# identifying UA passes, which is also what GEARS's own loader does (requests' default
# `python-requests/x.y` UA) against the same unauthenticated endpoint.
USER_AGENT = "coarse-perturbseq/0.1 (python-urllib; contact alex.markham@causal.dev)"


def _download(url: str, path: Path, md5: str | None = None) -> None:
    """Stream `url` to `path`, verify md5 when pinned; atomic via a `.part` file."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {path}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(request) as response, open(part, "wb") as f:  # noqa: S310
        shutil.copyfileobj(response, f)
    if md5 is not None:
        digest = hashlib.md5(part.read_bytes()).hexdigest()  # noqa: S324
        if digest != md5:
            raise ValueError(f"{path}: md5 {digest} != pinned {md5}")
    part.replace(path)


def _ensure_sources(root: Path) -> tuple[Path, Path, Path]:
    zip_path = root / "norman.zip"
    _download(GEARS_ZIP_URL, zip_path, GEARS_ZIP_MD5)
    h5ad_path = root / "norman" / "perturb_processed.h5ad"
    if not h5ad_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("norman/perturb_processed.h5ad", root)
    ids_path = root / "GSE133344_filtered_cell_identities.csv.gz"
    _download(IDENTITIES_URL, ids_path, IDENTITIES_MD5)
    collectri_path = root / f"collectri_omnipath_{COLLECTRI_DATE}.tsv"
    _download(COLLECTRI_URL, collectri_path)
    return h5ad_path, ids_path, collectri_path


def _env_label(identity: str) -> str | None:
    """Canonical env label of a guide-pair label ("KLF1_NegCtrl0" -> "KLF1",
    "NegCtrl0_NegCtrl1" -> "obs"); None for empty splits."""
    genes = sorted(t for t in identity.split("_") if t and not t.startswith("NegCtrl"))
    return "_".join(genes) if genes else "obs"


def _load_identities(path: Path) -> pd.DataFrame:
    ids = pd.read_csv(path)
    ids = ids[ids["good_coverage"]].copy()
    first = ids["guide_identity"].str.split("__").str[0]
    ids["env"] = first.map(_env_label)
    ids["gemgroup"] = ids["gemgroup"].astype(int)
    return ids[["cell_barcode", "env", "gemgroup"]]


def _collectri_edges(path: Path, genes: list[str]) -> pd.DataFrame:
    """TF -> target edges among `genes`, with row counts per symbol pair.

    The row count is not an established measure of evidence strength.
    """
    ref = pd.read_csv(path, sep="\t")
    keep = set(genes)
    ref = ref[
        ref["source_genesymbol"].isin(keep)
        & ref["target_genesymbol"].isin(keep)
        & (ref["source_genesymbol"] != ref["target_genesymbol"])
    ]
    edges = (
        ref.groupby(["source_genesymbol", "target_genesymbol"]).size().reset_index(name="weight")
    )
    edges.columns = ["source", "target", "weight"]
    return edges


def _acyclic(edges: pd.DataFrame) -> pd.DataFrame:
    """Greedily drop cycle-closing edges, using row counts and symbol order.

    This is a provisional debugging heuristic, not a biological evidence ranking or
    an optimal acyclic subgraph; it must not be treated as a silver-standard DAG.
    """
    kept: list[tuple[str, str]] = []
    weight = {
        (source, target): int(w)
        for source, target, w in edges[["source", "target", "weight"]].itertuples(
            index=False, name=None
        )
    }
    for u, v in sorted(weight, key=lambda e: (-weight[e], e)):
        # Adding u -> v closes a cycle iff v already reaches u via kept edges.
        children: dict[str, set[str]] = {}
        for a, b in kept:
            children.setdefault(a, set()).add(b)
        stack, seen = [v], set()
        cyclic = False
        while stack:
            node = stack.pop()
            if node == u:
                cyclic = True
                break
            if node in seen:
                continue
            seen.add(node)
            stack.extend(children.get(node, ()))
        if not cyclic:
            kept.append((u, v))
    return pd.DataFrame(kept, columns=["source", "target"])


def _regulon_parts(genes: list[str], edges: pd.DataFrame, targets: set[str]) -> list[set[int]]:
    """Provisional grouping by regulator; leftovers become singletons.

    These groups are not necessarily signature-equivalence blocks.
    """
    index = {g: i for i, g in enumerate(genes)}
    sub = edges[edges["source"].isin(targets)]
    counts = sub.groupby(["source", "target"]).size().reset_index(name="n")
    best_tf = counts.sort_values(["n", "target"], ascending=[False, True]).drop_duplicates(
        "target"
    )
    parts, assigned = [], set()
    for tf, grp in best_tf.groupby("source"):
        members = {index[g] for g in grp["target"] if g in index}
        if len(members) >= 2:
            parts.append(members)
            assigned.update(grp["target"])
    parts += [{index[g]} for g in genes if g not in assigned]
    return parts


def _silver_graphs(
    genes: list[str], parts: list[set[int]], gene_edges: pd.DataFrame
) -> tuple["nx.DiGraph", "nx.DiGraph"]:
    import networkx as nx

    index = {g: i for i, g in enumerate(genes)}
    gene_of_part = {i: {genes[a] for a in part} for i, part in enumerate(parts)}

    true_dag_full = nx.DiGraph()
    true_dag_full.add_nodes_from(range(len(genes)))
    true_dag_full.add_edges_from(
        (index[u], index[v]) for u, v in gene_edges.itertuples(index=False, name=None)
    )

    true_graph = nx.DiGraph()
    true_graph.add_nodes_from(range(len(parts)))
    for u in range(len(parts)):
        for v in range(len(parts)):
            if u != v and any(
                true_dag_full.has_edge(index[a], index[b])
                for a in gene_of_part[u]
                for b in gene_of_part[v]
            ):
                true_graph.add_edge(u, v)
    return true_graph, true_dag_full


def _pseudobulk(
    X_sub: np.ndarray, envs: list[str], env_of_cell: list[str], gemgroup_of_cell: list[int]
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Mean expression and matching batch IDs, one row per qualifying batch."""
    frame = pd.DataFrame({"env": env_of_cell, "gemgroup": gemgroup_of_cell})
    blocks, batch_ids = {}, {}
    for label in ["obs", *envs]:
        rows, ids = [], []
        mask = frame["env"] == label
        for gg, grp in frame[mask].groupby("gemgroup"):
            if len(grp) < MIN_CELLS_PER_GEMGROUP:
                continue
            rows.append(X_sub[grp.index.to_numpy()].mean(axis=0))
            ids.append(int(gg))
        blocks[label] = np.asarray(rows, dtype=float)
        batch_ids[label] = ids
        print(f"{label}: {len(rows)} gemgroup rows over {int(mask.sum())} cells")
    return blocks, batch_ids


def _prepare_smoke(root: Path) -> dict:
    h5ad_path, ids_path, collectri_path = _ensure_sources(root)

    import anndata

    ids = _load_identities(ids_path)
    keep_envs = list(SMOKE_ENVS)
    ids = ids[ids["env"].isin(["obs", *keep_envs])]
    ids = ids.set_index("cell_barcode")

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    var_genes = list(adata.var["gene_name"])
    missing = [g for g in SMOKE_GENES if g not in var_genes]
    if missing:
        raise ValueError(f"smoke genes missing from h5ad var: {missing}")
    # Column order must match SMOKE_GENES: features are labelled in that order below.
    gene_cols = [var_genes.index(g) for g in SMOKE_GENES]

    common_mask = adata.obs_names.isin(ids.index)
    keep_idx = np.where(common_mask)[0]
    common_barcodes = pd.Index(adata.obs_names[keep_idx])
    order = np.argsort(common_barcodes.argsort())
    print(f"h5ad cells: {adata.n_obs}, joined: {len(keep_idx)}")

    sub = adata[keep_idx].to_memory()
    X = sub.X
    X = X[:, gene_cols]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    barcodes = common_barcodes[order]
    X = X[order]
    joined = ids.loc[barcodes]
    env_of_cell = joined["env"].tolist()
    gemgroup_of_cell = joined["gemgroup"].tolist()

    blocks, batch_ids = _pseudobulk(X, keep_envs, env_of_cell, gemgroup_of_cell)

    rng = np.random.default_rng(0)
    cells = {}
    for label in ["obs", *keep_envs]:
        idx = np.where(np.asarray(env_of_cell) == label)[0]
        cap = SMOKE_MAX_CELLS_OBS if label == "obs" else SMOKE_MAX_CELLS_PER_ENV
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, size=cap, replace=False))
        cells[label] = X[idx].astype(np.float32)

    genes = list(SMOKE_GENES)
    edges = _collectri_edges(collectri_path, genes)
    print(f"collectri edges among smoke genes (raw): {len(edges)}")
    dag_edges = _acyclic(edges)
    print(f"collectri edges after acyclization: {len(dag_edges)}")

    parts = _regulon_parts(genes, edges, set(keep_envs))
    true_graph, true_dag_full = _silver_graphs(genes, parts, dag_edges)
    name_to_idx = {g: i for i, g in enumerate(genes)}
    return {
        "blocks": blocks,
        "batch_ids": batch_ids,
        "reference_edges": edges[["source", "target"]].values.tolist(),
        "cells": cells,
        "feature_cols": genes,
        "partition_parts": parts,
        "group_targets": {},  # smoke keeps single-gene envs only
        "true_graph": true_graph,
        "true_dag_full": true_dag_full,
        "name_to_idx": name_to_idx,
        "single_env_labels": keep_envs,
    }


def _prepare_full(root: Path) -> dict:
    """TODO(full scale): all 105 perturbed genes incl. pair environments (multi-target),
    regulons of all measured perturbed TFs, same pinned sources. Download of the full
    GEO matrix (1.13 GB) only needed if GEARS's 5,045-gene filter drops features."""
    raise NotImplementedError("full-scale prepare is not implemented yet")


def prepare(root: str, smoke: bool) -> dict:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    return _prepare_smoke(root_path) if smoke else _prepare_full(root_path)


def main():
    data = prepare(snakemake.params.root, bool(int(snakemake.params.smoke)))
    with open(snakemake.output.blocks, "wb") as f:
        pickle.dump(data["blocks"], f)
    with open(snakemake.output.cells, "wb") as f:
        pickle.dump(data["cells"], f)
    with open(snakemake.output.batch_ids, "w") as f:
        json.dump(data["batch_ids"], f)
    with open(snakemake.output.reference_edges, "w") as f:
        json.dump(data["reference_edges"], f)
    with open(snakemake.output.features, "w") as f:
        json.dump({"feature_cols": data["feature_cols"]}, f)
    with open(snakemake.output.partition, "wb") as f:
        pickle.dump(data["partition_parts"], f)
    with open(snakemake.output.grouptargets, "wb") as f:
        pickle.dump(data["group_targets"], f)
    with open(snakemake.output.truegraph, "wb") as f:
        pickle.dump(data["true_graph"], f)
    with open(snakemake.output.truedagfull, "wb") as f:
        pickle.dump(data["true_dag_full"], f)
    with open(snakemake.output.nametoidx, "wb") as f:
        pickle.dump(data["name_to_idx"], f)
    with open(snakemake.output.singleenvlabels, "wb") as f:
        pickle.dump(data["single_env_labels"], f)


if __name__ == "__main__":
    main()
