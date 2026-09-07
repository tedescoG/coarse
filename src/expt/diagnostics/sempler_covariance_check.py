# Why the synthetic sweep cannot go past p=100.
#
# generate.py draws ER graphs with a fixed DENSITY, so the average number of parents grows
# with p (20 at p=100, 40 at p=200, 100 at p=500). Edge weights up to |2| then make variances
# snowball along the graph: the largest variance in the model reaches 1e10 / 1e22 / 1e34.
# Sempler builds the exact population covariance and hands it to numpy's multivariate
# sampler. A double holds ~16 significant digits, so beyond p=100 that matrix is rounded
# into something that is no longer a covariance matrix and the samples are wrong.
# Fixing the average degree instead (independent of p) keeps the model within double
# precision at every size. This script shows both, on several seeds, with a table and a figure.
#
#   uv run python src/expt/diagnostics/sempler_covariance_check.py

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import default_rng

warnings.filterwarnings("ignore", message="No module named 'rpy2'.*")
from sempler import LGANM
from sempler.generators import dag_avg_deg

N_OBS = 5000
GRAPHS = ["er", "sf"]  # Erdos-Renyi (sempler) and scale-free (Barabasi-Albert), as in the suite
SIZES = [10, 100, 200, 500, 1000]  # 10 is where the density sweeps actually run; everything works there
SEEDS = range(10)
DENSITY = 0.2          # what the suite uses today
AVG_DEGREES = [4, 10, 20]  # proposed parametrisation; 20 is included to show where it stops working
EPS = np.finfo(float).eps  # 2.2e-16, relative rounding error of a double
NUMPY_WARNING = "numpy 'covariance is not positive-semidefinite'"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "diagnostics"
OUT = OUT_DIR / "sempler_covariance_check"  # .pdf, _summary.csv, _runs.csv


# Identical to workflow/scripts/generate.py, except that the average degree is an argument:
# the suite passes DENSITY * (p - 1) for both graph types.
def make_weights(graph, p, avg_deg, seed):
    if graph == "er":
        W = dag_avg_deg(p, avg_deg, w_min=0.5, w_max=2, return_ordering=False, random_state=seed)
    else:
        rng = default_rng(seed)
        m = max(1, min(p - 1, int(round(max(avg_deg / 2, 1)))))
        base = nx.barabasi_albert_graph(p, m, seed=seed)
        rank = {node: i for i, node in enumerate(rng.permutation(p))}
        W = np.zeros((p, p))
        for u, v in base.edges():
            src, dst = (u, v) if rank[u] < rank[v] else (v, u)
            W[src, dst] = rng.uniform(0.5, 2.0)
    edges = np.flatnonzero(W)
    negative = edges[default_rng(seed).choice([True, False], len(edges))]
    W[np.unravel_index(negative, (p, p))] *= -1
    return W


def sample(W, seed):
    model = LGANM(W, means=(-2, 2), variances=(0.5, 2), random_state=seed)
    # Same closed form sempler uses internally before calling np.random.multivariate_normal.
    A = np.linalg.inv(np.eye(model.p) - model.W.T)
    Sigma = A @ np.diag(model.variances) @ A.T
    # generate.py silences this warning; here it is recorded because it is the symptom.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = model.sample(N_OBS)
    warned = any("positive-semidefinite" in str(w.message) for w in caught)
    return X, Sigma, warned


def diagnose(X, Sigma):
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / len(X)
    eig = np.linalg.eigvalsh(S)
    try:
        np.linalg.cholesky(S)
        cholesky_ok = True
    except np.linalg.LinAlgError:
        cholesky_ok = False
    model_var = np.diag(Sigma)
    return {
        "var_span_log10": np.log10(model_var.max() / model_var.min()),  # digits the model needs
        "min_eig": eig[0],
        # Size of the rounding error made while forming a matrix whose largest entry is eig[-1].
        # When it exceeds the smallest true eigenvalue, that eigenvalue is lost: min_eig ~ -rounding_floor.
        "rounding_floor": EPS * eig[-1],
        "cond": eig[-1] / eig[0] if eig[0] > 0 else np.inf,
        "cholesky_ok": cholesky_ok,
        # Sample variance over model variance, worst variable. ~1 if sampling is correct.
        "worst_var_ratio": (X.var(axis=0, ddof=1) / model_var).max(),
    }


def run():
    configs = [("density=0.2", lambda p: DENSITY * (p - 1))]
    configs += [(f"avg_deg={d}", lambda p, d=d: d) for d in AVG_DEGREES]
    rows = []
    for graph in GRAPHS:
        for label, degree_of in configs:
            for p in SIZES:
                if degree_of(p) > p - 1:  # a node cannot have more than p-1 parents
                    continue
                for seed in SEEDS:
                    W = make_weights(graph, p, degree_of(p), seed)
                    X, Sigma, warned = sample(W, seed)
                    rows.append({"graph": graph, "config": label, "p": p, "seed": seed,
                                 "max_in_degree": int((W != 0).sum(axis=0).max()),  # hubs matter for sf
                                 NUMPY_WARNING: warned, **diagnose(X, Sigma)})
                    print(f"  {graph} {label:<12} p={p:<5} seed={seed}  worst_var_ratio={rows[-1]['worst_var_ratio']:.2f}")
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby(["graph", "config", "p"], sort=False)
    table = pd.DataFrame({
        "max_in_degree": g["max_in_degree"].max(),
        "var_span (decades)": g["var_span_log10"].median().round(1),
        "min_eig": g["min_eig"].median().map("{:.1e}".format),
        "rounding_floor": g["rounding_floor"].median().map("{:.1e}".format),
        "cond(S)": g["cond"].median().map("{:.1e}".format),
        "worst_var_ratio [min,max]": g["worst_var_ratio"].agg(lambda s: f"[{s.min():.2g}, {s.max():.2g}]"),
        "cholesky_ok": g["cholesky_ok"].mean().map("{:.0%}".format),
        NUMPY_WARNING: g[NUMPY_WARNING].mean().map("{:.0%}".format),
    })
    # correct: every seed reproduces the model variances and gives a usable covariance.
    # WARNING: still correct, but numpy already complains, i.e. no margin left.
    correct = g.apply(lambda s: bool((s.worst_var_ratio < 1.5).all() & s.cholesky_ok.all()))
    warned = g[NUMPY_WARNING].any()
    table["verdict"] = np.select([~correct, warned], ["BROKEN", "WARNING"], "OK")
    return table


def plot(df):
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(len(GRAPHS), 2, figsize=(10, 4 * len(GRAPHS)), squeeze=False)
    # left: the symptom (samples no longer follow the model); right: the cause (dynamic range)
    panels = [("worst_var_ratio", "sample var / model var (worst variable)", "log", 1.0, "correct sampling"),
              ("var_span_log10", "orders of magnitude spanned by model variances", "linear", 16, "double precision (16 digits)")]
    for row, graph in zip(axes, GRAPHS):
        for ax, (col, title, yscale, ref, ref_label) in zip(row, panels):
            for color, (label, sub) in zip(colors, df[df.graph == graph].groupby("config", sort=False)):
                ax.scatter(sub.p, sub[col], color=color, s=14, alpha=0.4)
                med = sub.groupby("p")[col].median()
                ax.plot(med.index, med.values, color=color, lw=2, label=label)
            ax.axhline(ref, color="gray", lw=1, ls="--")
            ax.text(SIZES[0], ref, ref_label, color="gray", fontsize=8, va="bottom")
            ax.set(xscale="log", yscale=yscale, xlabel="number of variables p", title=f"{graph}: {title}")
            ax.set_xticks(SIZES, [str(p) for p in SIZES])
            ax.grid(alpha=0.3)
        row[0].legend(frameon=False)
    fig.suptitle(f"sempler LGANM sampling, n={N_OBS}, {len(SEEDS)} seeds per point")
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run()
    summary = summarise(df)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n", summary)
    df.to_csv(OUT.with_name(OUT.name + "_runs.csv"), index=False)
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"))
    plot(df)
    print(f"\noutputs in {OUT_DIR}")
