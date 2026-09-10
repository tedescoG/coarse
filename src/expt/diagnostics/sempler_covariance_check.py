# Numerics of the synthetic generator (an iSCM sampled through sempler, see _common.make_dataset)
# and of COARSE's scoring on it, at sizes beyond what the suite runs.
#
# Every variable has population variance 1, so nothing can snowball with p; this script checks
# that at each density and size. Three things could still break, in this order:
#   1. sampling: sempler hands numpy a population covariance it cannot represent in double
#      precision; numpy rounds it into a non-covariance and the samples are wrong.
#   2. scoring: COARSE regresses each variable on its parents through a Schur complement,
#      Var(X_j) - explained; when Var(X_j)/noise_j is too large the difference has no digits.
#   3. power: individual edges too weak for BIC to keep on the true parent set.
#
#   uv run python src/expt/diagnostics/sempler_covariance_check.py

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng

warnings.filterwarnings("ignore", message="No module named 'rpy2'.*")
from sempler import LGANM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflow" / "scripts"))
from _common import draw_noise_params, iscm_standardize, make_weights
from coarse.scoring import _block_bic_env_from_sigma, _block_regression_from_sigma

N_OBS = 5000
GRAPHS = ["er", "sf"]  # Erdos-Renyi (sempler) and scale-free (Barabasi-Albert), as in the suite
SIZES = [10, 100, 200, 500, 1000]  # 10 is where the density sweeps run; the scalability sweeps stop at 100
SEEDS = range(20)
DENSITIES = [0.2, 0.5, 0.8]  # the suite's density sweep; 0.2 is what the scalability sweeps use
N_EDGE_CHECK = 200  # true edges per run on which the BIC gain is evaluated
EPS = np.finfo(float).eps  # 2.2e-16, relative rounding error of a double
NUMPY_WARNING = "numpy 'covariance is not positive-semidefinite'"
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "diagnostics"
OUT = OUT_DIR / "sempler_covariance_check"  # _er.pdf, _sf.pdf, _summary.csv, _runs.csv


# The suite's model for this cell (standardized W and noise variances), sampled as the suite does.
def sample(graph, p, density, seed):
    W, noise_var = iscm_standardize(make_weights(graph, p, density * (p - 1), seed), draw_noise_params(p, seed)[0])
    model = LGANM(W, means=(-2, 2), variances=noise_var, random_state=seed)
    # Same closed form sempler uses internally before calling np.random.multivariate_normal.
    A = np.linalg.inv(np.eye(model.p) - model.W.T)
    Sigma = A @ np.diag(model.variances) @ A.T
    # generate.py silences this warning; here it is recorded because it is the symptom.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X = model.sample(N_OBS)
    warned = any("positive-semidefinite" in str(w.message) for w in caught)
    return model, X, Sigma, warned


# Residual variance of X_j regressed on its true parents, computed as COARSE does it, over the true noise variance.
def residual_ratio(S, model, j):
    parents = np.flatnonzero(model.W[:, j])
    try:
        _, resid = _block_regression_from_sigma(np.array([j]), parents, S)
    except np.linalg.LinAlgError:
        return np.inf
    resid = resid.item() * N_OBS / (N_OBS - parents.size)  # unbiased for the number of regressors
    return resid / model.variances[j] if resid > 0 else np.inf


# BIC of the true parent set of j minus BIC without the edge i -> j (obs env, lambda=1).
def bic_gain(S, model, i, j):
    parents = np.flatnonzero(model.W[:, j])
    full = _block_bic_env_from_sigma(np.array([j]), parents, S, N_OBS)
    without = _block_bic_env_from_sigma(np.array([j]), parents[parents != i], S, N_OBS)
    return full - without


def diagnose(X, Sigma, model, seed):
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / len(X)
    eig = np.linalg.eigvalsh(S)
    try:
        np.linalg.cholesky(S)
        cholesky_ok = True
    except np.linalg.LinAlgError:
        cholesky_ok = False
    model_var = np.diag(Sigma)
    resid = np.array([residual_ratio(S, model, j) for j in range(model.p)])
    edges = np.argwhere(model.W != 0)
    edges = edges[default_rng(seed).choice(len(edges), min(N_EDGE_CHECK, len(edges)), replace=False)]
    gains = np.array([bic_gain(S, model, i, j) for i, j in edges])
    return {
        "var_span_log10": np.log10(model_var.max() / model_var.min()),  # digits the model needs across variables
        "snr_log10": np.log10(model_var / model.variances).max(),  # digits one regression must resolve: Var(X_j) over its noise, worst variable
        "min_eig": eig[0],  # smallest eigenvalue of the sample covariance; negative means rounding already broke it
        # Size of the rounding error made while forming a matrix whose largest entry is eig[-1].
        # When it exceeds the smallest true eigenvalue, that eigenvalue is lost: min_eig ~ -rounding_floor.
        "rounding_floor": EPS * eig[-1],
        "cond": eig[-1] / eig[0] if eig[0] > 0 else np.inf,  # condition number of the sample covariance
        "cholesky_ok": cholesky_ok,  # whether the sample covariance is numerically positive definite
        "worst_var_ratio": (X.var(axis=0, ddof=1) / model_var).max(),  # sample over model variance, worst variable; ~1 if sampling is correct
        "worst_resid_ratio": np.abs(resid - 1).max(),  # |residual / noise variance - 1|, worst variable; ~0 if COARSE's regression is correct
        "edge_drop_frac": np.mean(~(gains > 0)),  # fraction of true edges COARSE's shrink step would drop from the true parent set
    }


def run():
    rows = []
    for graph in GRAPHS:
        for density in DENSITIES:
            for p in SIZES:
                for seed in SEEDS:
                    model, X, Sigma, warned = sample(graph, p, density, seed)
                    rows.append({"graph": graph, "density": density, "p": p, "seed": seed,
                                 "max_in_degree": int((model.W != 0).sum(axis=0).max()),  # hubs matter for sf
                                 NUMPY_WARNING: warned, **diagnose(X, Sigma, model, seed)})
                    r = rows[-1]
                    print(f"  {graph} density={density} p={p:<5} seed={seed}  worst_var_ratio={r['worst_var_ratio']:.2f}"
                          f"  worst_resid_ratio={r['worst_resid_ratio']:.2g}  edge_drop_frac={r['edge_drop_frac']:.2f}")
    return pd.DataFrame(rows)


def summarise(df):
    g = df.groupby(["graph", "density", "p"], sort=False)
    sci = "{:.1e}".format
    span = lambda s: f"[{s.min():.2g}, {s.max():.2g}]"
    table = pd.DataFrame({
        "max_in_degree": g["max_in_degree"].max(),
        "var_span (decades)": g["var_span_log10"].median().round(1),
        "snr (decades)": g["snr_log10"].median().round(1),
        "min_eig": g["min_eig"].median().map(sci),
        "rounding_floor": g["rounding_floor"].median().map(sci),
        "cond(S)": g["cond"].median().map(sci),
        "cholesky_ok": g["cholesky_ok"].mean().map("{:.0%}".format),
        "worst_var_ratio [min,max]": g["worst_var_ratio"].agg(span),
        "worst_resid_ratio [min,max]": g["worst_resid_ratio"].agg(span),
        "edge_drop_frac": g["edge_drop_frac"].median().round(2),
        NUMPY_WARNING: g[NUMPY_WARNING].mean().map("{:.0%}".format),
    })
    # SAMPLING: the data does not follow the model. SCORING: COARSE's regression on the true parents
    # is off. WEAK: BIC would drop more than 10% of the true edges. WARNING: all correct but numpy
    # already complains, i.e. no margin left. OK: none of the above, on every seed.
    sampling = g["worst_var_ratio"].max() > 1.5
    scoring = g["worst_resid_ratio"].max() > 0.5
    weak = g["edge_drop_frac"].median() > 0.1
    warned = g[NUMPY_WARNING].any()
    table["verdict"] = np.select([sampling, scoring, weak, warned], ["SAMPLING", "SCORING", "WEAK", "WARNING"], "OK")
    return table


def plot(df):
    panels = [("var_span_log10", "orders of magnitude spanned by model variances", "linear", 16, "double precision (16 digits)"),
              ("worst_var_ratio", "sample var / model var (worst variable)", "log", 1.0, "correct sampling"),
              ("worst_resid_ratio", "|residual var / noise var - 1| (worst variable)", "log", 0.5, "SCORING threshold"),
              ("edge_drop_frac", "fraction of true edges dropped by BIC", "linear", 0.1, "WEAK threshold")]
    colors = dict(zip(DENSITIES, plt.cm.tab10.colors))
    for graph in GRAPHS:
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5))
        for ax, (col, title, yscale, ref, ref_label) in zip(axes, panels):
            for density, sub in df[df.graph == graph].groupby("density", sort=False):
                ax.scatter(sub.p, sub[col], color=colors[density], s=14, alpha=0.4)
                med = sub.groupby("p")[col].median()
                ax.plot(med.index, med.values, color=colors[density], lw=2, label=f"density={density}")
            ax.axhline(ref, color="gray", lw=1, ls="--")
            ax.text(SIZES[0], ref, ref_label, color="gray", fontsize=8, va="bottom")
            ax.set(xscale="log", yscale=yscale, xlabel="number of variables p")
            ax.set_title(title, fontsize=10)
            ax.set_xticks(SIZES, [str(p) for p in SIZES])
            ax.grid(alpha=0.3)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{graph}: iSCM sampling and COARSE scoring, n={N_OBS}, {len(SEEDS)} seeds per point")
        fig.tight_layout()
        fig.savefig(OUT.with_name(f"{OUT.name}_{graph}.pdf"))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run()
    summary = summarise(df)
    with pd.option_context("display.width", 300, "display.max_columns", None, "display.max_rows", None):
        print("\n", summary)
    df.to_csv(OUT.with_name(OUT.name + "_runs.csv"), index=False)
    summary.to_csv(OUT.with_name(OUT.name + "_summary.csv"))
    plot(df)
    print(f"\noutputs in {OUT_DIR}")
