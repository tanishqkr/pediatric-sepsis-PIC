"""
export_dashboard_data.py

Consolidates a completed run folder into ONE clean dashboard.json — so the
Streamlit app (see mentor's dashboard requirement) never touches raw run
internals, just this single file. Copy dashboard.json alongside the
dashboard's app.py when moving to a different machine; no other run files
are needed to demo it.
"""
import argparse
import json
import os

import fl_shap_config as config
from fl_shap_results_io import RunIO


def load_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    fedavg = load_if_exists(os.path.join(args.run_dir, "federated_ranking_fedavg.json"))
    fedprox = load_if_exists(os.path.join(args.run_dir, "federated_ranking_fedprox.json"))
    sensitivity = load_if_exists(os.path.join(args.run_dir, "sensitivity", "summary.json"))

    dashboard = {
        "run_id": os.path.basename(args.run_dir),
        "nodes": config.NODES,
        "top_k": config.TOP_K,
        "fedavg": {
            "top_5_features": fedavg["top_5_features"] if fedavg else None,
            "node_vs_federated_spearman": fedavg["node_vs_federated_spearman"] if fedavg else None,
            "federated_ranking": fedavg["federated_ranking"] if fedavg else None,
        } if fedavg else None,
        "fedprox": {
            "top_5_features": fedprox["top_5_features"] if fedprox else None,
            "node_vs_federated_spearman": fedprox["node_vs_federated_spearman"] if fedprox else None,
            "federated_ranking": fedprox["federated_ranking"] if fedprox else None,
        } if fedprox else None,
        "sensitivity_analysis": sensitivity,
    }

    out_path = os.path.join(args.run_dir, "dashboard.json")
    with open(out_path, "w") as f:
        json.dump(dashboard, f, indent=2, default=str)

    print(f"Dashboard data written: {out_path}")
    print("Copy this ONE file to the dashboard machine — that's all app.py needs to read.")


if __name__ == "__main__":
    main()
