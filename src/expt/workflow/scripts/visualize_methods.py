"""Per-method 3-panel DAG visualization for the multi-target experiment.

Produces three PDFs (one per method): COARSE, COARSE-CV, COARSE-1PC. Each
PDF has one page per ``VIZ_CONFIGS`` entry (declared in multitarget.smk).

Each page shows:
  1. **Atomic ground-truth DAG** — fine-grained, one node per variable, with
     a red ring on every node that is part of an intervention target set.
  2. **True partition DAG** — atomic DAG quotiented by the oracle partition
     under the multi-target intervention scheme.
  3. **Learned partition DAG** — the partition returned by the method,
     with edges coloured green / red / light-grey for true-positive /
     false-positive / false-negative edges.

Polished for thesis use:
  - Big top title ("METHOD — multi-target intervention recovery").
  - Compact metric-bearing subtitle (one line, separator '|').
  - Panel titles only carry the panel *role*, not metrics.

Reuses the rendering helpers from ``coarse/diagnostics/visualize_dags.py`` —
that file is the most-edited version, so importing keeps the two views from
drifting.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from _common import build_data_dict, build_oracle_partition
from coarse.coarse import COARSE
from coarse.cv import cv_coarse

# Import the rendering helpers from the diagnostic. The relative path from
# this script to diagnostics/ is `../../../../diagnostics/`; resolving via
# Path keeps the dependency explicit and machine-portable.
_DIAG_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent.parent / "diagnostics"
)
sys.path.insert(0, str(_DIAG_DIR))
from visualize_dags import (  # noqa: E402
    _color_edges,
    draw_atomic,
    draw_partition,
    true_partition_dag,
)


def _extend_layout(base_layout: dict, target_pdag) -> dict:
    """Layout adapter: reuse the true-partition node positions for the
    learned-partition panel so blocks that exist in both line up. Any nodes
    unique to the learned panel are stacked to the right at evenly spaced
    y-positions. Same logic as ``_shared_layout`` (a nested closure in
    diagnostics/visualize_dags.py); inlined here because closures aren't
    importable.
    """
    out = dict(base_layout)
    unmatched = [n for n in target_pdag.nodes if n not in out]
    if unmatched and out:
        xs = [p[0] for p in out.values()]
        ys = [p[1] for p in out.values()]
        x_off = max(xs) + 60
        y_min, y_max = min(ys), max(ys)
        step = (y_max - y_min) / max(1, len(unmatched))
        for i, n in enumerate(unmatched):
            out[n] = (x_off, y_min + i * step)
    return out


METHOD_INFO = {
    "coarse": {
        "label": "COARSE",
        "title": "COARSE — multi-target intervention recovery",
    },
    "cv": {
        "label": "COARSE-CV",
        "title": "COARSE-CV — multi-target intervention recovery",
    },
    "onepc": {
        "label": "COARSE-1PC",
        "title": "COARSE-1PC — multi-target intervention recovery",
    },
}


def _fit_method(method: str, data_dict):
    """Fit the requested method using the same params as multitarget.smk.

    Returning ``(model, runtime_seconds)`` mirrors the contract of
    ``fit.py`` / ``fit_cv.py`` so the rest of the script is method-agnostic.
    """
    t0 = time.perf_counter()
    if method == "coarse":
        model = COARSE().fit(
            data_dict, alpha=1e-4, lambda_pen=1.0, refine_test="welch",
        )
    elif method == "cv":
        model = cv_coarse(
            data_dict, lambda_pen=1.0, refine_test="welch",
        )
    elif method == "onepc":
        model = COARSE().fit(
            data_dict, alpha=1e-4, lambda_pen=1.0,
            refine_test="welch", k=1, pca_pooled=False,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")
    return model, time.perf_counter() - t0


def _partition_ari(model_dag, true_partition, num_nodes):
    """Same ARI formula as evaluate.py — inlined to avoid a snakemake input
    dependency on a per-config metrics CSV that would not otherwise exist."""
    from sklearn.metrics import adjusted_rand_score

    true_labels = np.zeros(num_nodes, dtype=int)
    for lbl, part in enumerate(true_partition):
        true_labels[list(part)] = lbl
    est_labels = np.zeros(num_nodes, dtype=int)
    for lbl, part in enumerate(model_dag.nodes):
        est_labels[list(part)] = lbl
    return adjusted_rand_score(true_labels, est_labels)


def _edge_fscore(model_dag, true_atomic):
    """Partition-level edge F-score — same logic as evaluate.py:49-78."""
    import networkx as nx

    def _is_adj(pa, ch):
        for a in pa:
            for b in ch:
                if true_atomic.has_edge(a, b):
                    return True
        return False

    skel = nx.create_empty_copy(model_dag)
    nl = list(skel.nodes)
    for i, pa in enumerate(nl[:-1]):
        for ch in nl[i + 1:]:
            if _is_adj(pa, ch):
                skel.add_edge(pa, ch)
    tp = sum(1 for e in model_dag.edges if e in skel.edges)
    prec = tp / model_dag.number_of_edges() if model_dag.number_of_edges() else 1.0
    rec = tp / skel.number_of_edges() if skel.number_of_edges() else 1.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _format_subtitle(cfg, ari, f1, runtime_sec):
    tpi_raw = str(cfg["targets_per_interv"])
    if "to" in tpi_raw:
        tpi_label = f"{tpi_raw.replace('to', '–')} (mix)"
    else:
        tpi_label = f"{tpi_raw} (fixed)"
    return (
        f"graph={cfg['graph']}   p={cfg['num_nodes']}   "
        f"density={cfg['density']}   targets/intervention = {tpi_label}   "
        f"n={cfg['samp_size']}   seed={cfg['seed']}   |   "
        f"ARI={ari:.2f}   F={f1:.2f}   rt={runtime_sec*1000:.0f}ms"
    )


def _render_page(pdf, method, cfg, npz_path):
    data = np.load(npz_path, allow_pickle=True)
    weights = data["weights"]
    targets = data["targets"]
    num_nodes = int(cfg["num_nodes"])
    data_dict = build_data_dict(data, targets, "soft")

    model, runtime_sec = _fit_method(method, data_dict)
    true_atomic, _, _, true_partition = build_oracle_partition(
        weights, targets, num_nodes
    )
    true_pdag = true_partition_dag(weights, true_partition)
    interv_set = {int(n) for t in targets for n in np.atleast_1d(t)}

    ari = _partition_ari(model.dag, true_partition, num_nodes)
    f1 = _edge_fscore(model.dag, true_atomic)
    decorated, edge_colors = _color_edges(model.dag, true_pdag)

    fig = plt.figure(figsize=(18, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0])
    ax_atomic = fig.add_subplot(gs[0])
    ax_true = fig.add_subplot(gs[1])
    ax_learned = fig.add_subplot(gs[2])

    draw_atomic(
        ax_atomic, true_atomic, true_partition, interv_set,
        "Atomic ground truth"
    )
    layout_mid = draw_partition(
        ax_true, true_pdag, true_partition, "True partition DAG"
    )
    draw_partition(
        ax_learned, decorated, true_partition, "Learned partition DAG",
        edge_colors=edge_colors,
        layout=_extend_layout(layout_mid, decorated),
    )

    legend = [
        Patch(facecolor="white", edgecolor="green", label="True positive edge"),
        Patch(facecolor="white", edgecolor="red", label="False positive edge"),
        Patch(facecolor="white", edgecolor="lightgray", label="False negative edge"),
    ]
    ax_learned.legend(handles=legend, loc="lower right", fontsize=8, frameon=True)

    fig.suptitle(
        METHOD_INFO[method]["title"],
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.92, _format_subtitle(cfg, ari, f1, runtime_sec),
        ha="center", fontsize=10, style="italic", color="#333",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    pdf.savefig(fig)
    plt.close(fig)
    print(f"  [{method}] {cfg}  ARI={ari:.2f}  F={f1:.2f}  rt={runtime_sec*1000:.0f}ms")


configs = list(snakemake.params.configs)
npz_paths = list(snakemake.input.datasets)
assert len(configs) == len(npz_paths), "configs / dataset paths must align 1:1"

method_to_output = {
    "coarse": snakemake.output.coarse,
    "cv": snakemake.output.cv,
    "onepc": snakemake.output.onepc,
}

for method, out_path in method_to_output.items():
    print(f"rendering {len(configs)} pages → {out_path}")
    with PdfPages(out_path) as pdf:
        for cfg, npz_path in zip(configs, npz_paths):
            _render_page(pdf, method, cfg, npz_path)
    print(f"wrote {out_path}")
