"""Population gap between the env partition Π^E and the interventional partition Π^I on one
random DAG, for three target designs: `native` (sempler targets, k per env, m envs),
`disjoint` (a fixed pool S split into |S|/k envs) and `overlap` (the disjoint cover of S plus
random k-subsets of S, m envs in total). `hidden` is the fraction of true edges between Π^I
blocks that fall inside one Π^E block."""

import networkx as nx
import numpy as np
import pandas as pd
from sempler.generators import intervention_targets
from sklearn.metrics import adjusted_rand_score

from _common import build_oracle_partition, interventional_partition, make_weights, partition_labels

w = snakemake.wildcards
p, seed, density = int(w.num_nodes), int(w.seed), float(w.density)
prm = snakemake.params
W = make_weights(w.graph, p, density * (p - 1), seed)
dag = nx.DiGraph(W.astype(bool))


def metrics(targets):
    piE = build_oracle_partition(W, targets, p)[3]
    piI = interventional_partition(W, targets, p)
    lE, lI = partition_labels(piE, p), partition_labels(piI, p)
    cross = [(u, v) for u, v in dag.edges if lI[u] != lI[v]]
    return dict(ari=adjusted_rand_score(lI, lE), nE=len(piE), nI=len(piI), ratio=len(piE) / len(piI),
                hidden=np.mean([lE[u] == lE[v] for u, v in cross]) if cross else np.nan)


rows = []
for k in prm.native_k:
    for m in prm.native_m:
        T = intervention_targets(p, m, k, replace=k > 1, random_state=seed)
        rows.append(dict(design="native", k=k, m=m, **metrics(T)))
rng = np.random.default_rng(10_000 + seed)
S = rng.choice(p, prm.pool, replace=False)
for k in prm.disjoint_k:
    perm = rng.permutation(S)
    T = [perm[i:i + k] for i in range(0, prm.pool, k)]
    rows.append(dict(design="disjoint", k=k, m=len(T), **metrics(T)))
for k in prm.overlap_k:
    perm = rng.permutation(S)
    cover = [perm[i:i + k] for i in range(0, prm.pool, k)]
    extra = [rng.choice(S, k, replace=False) for _ in range(max(prm.overlap_m))]
    for m in prm.overlap_m:
        if m >= len(cover):
            rows.append(dict(design="overlap", k=k, m=m, **metrics(cover + extra[: m - len(cover)])))

df = pd.DataFrame(rows)
df.insert(1, "density", density)
df.insert(2, "seed", seed)
df.to_csv(snakemake.output[0], index=False)
