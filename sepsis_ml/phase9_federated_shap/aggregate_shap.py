"""
aggregate_shap.py

Takes the per-node SHAP JSONs produced by federated_shap.py and:
  1. Computes the federated_SHAP_j = sum_k (n_k / N) * mean|SHAP|_k,j ranking
     (same sample-size weighting FedAvg itself uses).
  2. Computes Spearman rank correlation of each node's own ranking against
     the federated aggregate -> answers "does the General ICU node dominate
     the explanation?"
  3. Writes a clean top-5 features summary for the dashboard.

Run AFTER federated_shap.py has produced node_shap/<algo>_<node>.json files
for every node under the same run folder.
"""
import argparse
import glob
import json
import os

import pandas as pd
from scipy.stats import spearmanr

import fl_shap_config as config
from fl_shap_results_io import RunIO


def load_node_results(run_dir: str, algo: str):
    pattern = os.path.join(run_dir, "node_shap", f"{algo}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No node SHAP files found matching {pattern}. "
            f"Run federated_shap.py --algo {algo} first."
        )
    results = {}
    for fp in files:
        with open(fp) as f:
            r = json.load(f)
        results[r["node"]] = r
    return results


def aggregate(node_results: dict):
    """federated_SHAP_j = sum_k (n_k / N) * mean|SHAP|_k,j"""
    all_features = node_results[next(iter(node_results))]["feature_ranking"]
    feature_order = [f["feature"] for f in all_features]

    per_node_scores = {}  # node -> {feature: mean_abs_shap}
    for node, r in node_results.items():
        per_node_scores[node] = {f["feature"]: f["mean_abs_shap"] for f in r["feature_ranking"]}

    federated = {}
    for feat in feature_order:
        total = 0.0
        for node, scores in per_node_scores.items():
            weight = config.NODES[node] / config.TOTAL_N
            total += weight * scores.get(feat, 0.0)
        federated[feat] = total

    ranking = sorted(federated.items(), key=lambda t: t[1], reverse=True)
    return ranking, per_node_scores


def rank_correlations(ranking, per_node_scores):
    """Spearman correlation of each node's own ranking vs. the federated aggregate."""
    fed_features = [f for f, _ in ranking]
    fed_scores_ordered = [federated for _, federated in ranking]

    correlations = {}
    for node, scores in per_node_scores.items():
        node_scores_ordered = [scores.get(f, 0.0) for f in fed_features]
        rho, pval = spearmanr(fed_scores_ordered, node_scores_ordered)
        correlations[node] = {"spearman_rho": float(rho), "p_value": float(pval)}
    return correlations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--run-dir", required=True, help="run folder produced by federated_shap.py")
    args = parser.parse_args()

    io = RunIO(run_name=os.path.basename(args.run_dir))  # continue writing into the same run
    io.run_dir = args.run_dir  # reuse the existing folder instead of making a new one
    io.log("aggregate_shap_start", algo=args.algo)

    node_results = load_node_results(args.run_dir, args.algo)
    ranking, per_node_scores = aggregate(node_results)
    correlations = rank_correlations(ranking, per_node_scores)

    top5 = ranking[:config.TOP_K]

    output = {
        "algo": args.algo,
        "federated_ranking": [{"feature": f, "federated_mean_abs_shap": v} for f, v in ranking],
        "top_5_features": [{"feature": f, "federated_mean_abs_shap": v} for f, v in top5],
        "node_vs_federated_spearman": correlations,
        "nodes_included": list(node_results.keys()),
    }
    io.save_json(f"federated_ranking_{args.algo}.json", output)

    # a flat CSV too, for quick dashboard/Excel loading
    df = pd.DataFrame(ranking, columns=["feature", "federated_mean_abs_shap"])
    for node in per_node_scores:
        df[f"{node}_mean_abs_shap"] = df["feature"].map(per_node_scores[node])
    io.save_csv(f"federated_ranking_{args.algo}.csv", df)

    print(f"\nTop {config.TOP_K} federated features ({args.algo}):")
    for f, v in top5:
        print(f"  {f}: {v:.4f}")
    print("\nNode vs. federated rank correlation (low rho = that node disagrees with the aggregate):")
    for node, c in correlations.items():
        print(f"  {node}: rho={c['spearman_rho']:.3f} (p={c['p_value']:.4f})")

    io.finish()


if __name__ == "__main__":
    main()
