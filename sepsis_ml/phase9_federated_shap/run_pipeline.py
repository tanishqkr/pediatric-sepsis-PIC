"""
run_pipeline.py

The ONE script to run tomorrow, end to end, after phase7_federated_learning.py
has produced checkpoints (see checkpoint_patch.py).

    python run_pipeline.py --run-name uni_session_1

This will:
  1. Run federated_shap.py for FedAvg
  2. Run federated_shap.py for FedProx
  3. Aggregate both into federated rankings
  4. Run the sensitivity analysis (axis 3 always; axes 1/2 if you've filled
     in their stubs by then)
  5. Export a single clean JSON for the Streamlit dashboard

Each stage is wrapped in try/except so one failure doesn't kill the whole
overnight/afternoon run — check run_log.jsonl in the run folder for exactly
where it stopped.
"""
import argparse
import subprocess
import sys
import os

import fl_shap_config as config
from fl_shap_results_io import RunIO


def run_step(io, name, cmd):
    io.log("step_start", step=name, cmd=" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        io.log("step_done", step=name)
        return True
    except subprocess.CalledProcessError as e:
        io.log("step_failed", step=name, error=str(e))
        print(f"\n!! {name} failed — see run_log.jsonl. Continuing to next step. !!\n")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    io = RunIO(run_name=args.run_name)
    io.log("pipeline_start")
    py = sys.executable

    run_step(io, "federated_shap_fedavg", [py, "federated_shap.py", "--algo", "fedavg", "--run-name", io.run_id])
    run_step(io, "federated_shap_fedprox", [py, "federated_shap.py", "--algo", "fedprox", "--run-name", io.run_id])
    run_step(io, "aggregate_fedavg", [py, "aggregate_shap.py", "--algo", "fedavg", "--run-dir", io.run_dir])
    run_step(io, "aggregate_fedprox", [py, "aggregate_shap.py", "--algo", "fedprox", "--run-dir", io.run_dir])
    run_step(io, "sensitivity_analysis", [py, "sensitivity_analysis.py", "--run-dir", io.run_dir])
    run_step(io, "export_dashboard_data", [py, "export_dashboard_data.py", "--run-dir", io.run_dir])

    io.finish()
    print(f"\nDone. Zip and carry this folder: {io.run_dir}")


if __name__ == "__main__":
    main()
