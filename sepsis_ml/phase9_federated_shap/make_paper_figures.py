"""
make_paper_figures.py

Generates the figures the paper actually references, straight from your
pipeline's JSON outputs. Run once you have a completed, CURRENT run folder
(i.e. after re-running aggregate_shap.py / sensitivity_analysis.py on the
5-node data -- don't point this at stale files):

    python make_paper_figures.py --run-dir "D:\\Projects\\pediatric-sepsis-PIC\\sepsis_ml\\phase9_federated_shap\\results\\uni_session_1" --out-dir "D:\\Projects\\FL_SHAP\\figures"

Produces (as vector PDFs -- sharpest option for LaTeX, no rasterization at
print size):
    fig_node_shap_heatmap.pdf   -> nodes x top-feature mean|SHAP|
    fig_node_agreement.pdf      -> Spearman rho per node vs. federated aggregate
    fig_fedavg_vs_fedprox.pdf   -> ranking agreement between the two algorithms

Copy the three PDFs into your paper's figures/ folder -- main.tex already
points \includegraphics at figures/<filename>.pdf relative to itself, so no
LaTeX edits needed once the files are there.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_WIDTH = 3.5  # IEEE single column width, inches
NAVY = "#1F3A5F"
MAROON = "#BD3430"
TEAL = "#2E86AB"
GREY = "#6B7280"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "font.family": "serif",
    "pdf.fonttype": 42,  # embed as real fonts, not Type 3 bitmaps
})


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig_node_shap_heatmap(run_dir, algo, out_dir, top_n=10):
    """Nodes (rows) x top-N federated features (cols), mean|SHAP| per cell."""
    fed_path = os.path.join(run_dir, f"federated_ranking_{algo}.csv")
    if not os.path.exists(fed_path):
        print(f"  [skip] {fed_path} not found")
        return
    df = pd.read_csv(fed_path).head(top_n)

    node_cols = [c for c in df.columns if c.endswith("_mean_abs_shap") and not c.startswith("federated")]
    nodes = [c.replace("_mean_abs_shap", "") for c in node_cols]

    matrix = df[node_cols].values.T  # nodes x features
    features = df["feature"].tolist()

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    im = ax.imshow(matrix, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean |SHAP|", fontsize=7)
    ax.set_title(f"Per-node SHAP importance ({algo.upper()})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig_node_shap_heatmap.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path} (nodes included: {nodes})")


def fig_node_agreement(run_dir, algo, out_dir):
    """Bar chart: Spearman rho of each node's ranking vs. the federated aggregate."""
    fed_path = os.path.join(run_dir, f"federated_ranking_{algo}.json")
    if not os.path.exists(fed_path):
        print(f"  [skip] {fed_path} not found")
        return
    data = load_json(fed_path)
    corr = data["node_vs_federated_spearman"]

    nodes = list(corr.keys())
    rhos = [corr[n]["spearman_rho"] for n in nodes]
    colors = [MAROON if r < 0.5 else (TEAL if r < 0.8 else NAVY) for r in rhos]

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.0))
    ax.bar(nodes, rhos, color=colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(min(0, min(rhos) - 0.1), 1.0)
    ax.set_title(f"Node agreement with federated ranking ({algo.upper()})")
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig_node_agreement.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path} (nodes included: {nodes})")
    if len(nodes) < 5:
        print(f"  !! WARNING: only {len(nodes)} nodes in this figure -- "
              f"expected 5 (CICU, NICU, PICU, SICU, General). "
              f"This looks like stale pre-fix data.")


def fig_fedavg_vs_fedprox(run_dir, out_dir, top_n=10):
    """Grouped bar: top-N federated features, FedAvg score vs FedProx score, side by side."""
    fa_path = os.path.join(run_dir, "federated_ranking_fedavg.json")
    fp_path = os.path.join(run_dir, "federated_ranking_fedprox.json")
    if not (os.path.exists(fa_path) and os.path.exists(fp_path)):
        print(f"  [skip] need both federated_ranking_fedavg.json and _fedprox.json")
        return
    fa_data = load_json(fa_path)
    fp_data = load_json(fp_path)
    fa = {d["feature"]: d["federated_mean_abs_shap"] for d in fa_data["federated_ranking"]}
    fp = {d["feature"]: d["federated_mean_abs_shap"] for d in fp_data["federated_ranking"]}

    top_features = sorted(fa, key=fa.get, reverse=True)[:top_n]
    fa_vals = [fa[f] for f in top_features]
    fp_vals = [fp.get(f, 0.0) for f in top_features]

    x = np.arange(len(top_features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.2))
    ax.bar(x - width / 2, fa_vals, width, label="FedAvg", color=NAVY)
    ax.bar(x + width / 2, fp_vals, width, label="FedProx", color=MAROON)
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("federated mean |SHAP|")
    ax.set_title("FedAvg vs. FedProx feature ranking")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(out_dir, "fig_fedavg_vs_fedprox.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    n_fa = len(fa_data.get("nodes_included", []))
    n_fp = len(fp_data.get("nodes_included", []))
    if n_fa < 5 or n_fp < 5:
        print(f"  !! WARNING: fedavg used {n_fa} nodes, fedprox used {n_fp} nodes -- "
              f"expected 5 each. This looks like stale pre-fix data.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="e.g. phase9_federated_shap/results/uni_session_1")
    parser.add_argument("--out-dir", required=True, help="where to write the figure PDFs, e.g. your paper's figures/ folder")
    parser.add_argument("--algo", default="fedavg", choices=["fedavg", "fedprox"],
                         help="which algorithm's rankings to use for the single-algo figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Node SHAP heatmap...")
    fig_node_shap_heatmap(args.run_dir, args.algo, args.out_dir)

    print("Node agreement bar chart...")
    fig_node_agreement(args.run_dir, args.algo, args.out_dir)

    print("FedAvg vs FedProx comparison...")
    fig_fedavg_vs_fedprox(args.run_dir, args.out_dir)

    print(f"\nDone. Copy {args.out_dir} into your paper folder if it isn't already there.")


if __name__ == "__main__":
    main()
