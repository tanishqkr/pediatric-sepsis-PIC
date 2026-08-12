"""
fl_shap_results_io.py

Your parent project ALREADY has a results-logging utility:
    sepsis_ml/logger.py -> log_experiment(), load_all_results(), print_leaderboard()
Use that, don't build a second one from scratch. This module is a thin
wrapper around it that adds the one thing it doesn't do yet: a single,
self-contained, per-run folder that's trivial to zip and carry to another
PC (your dashboard machine, your laptop for the write-up, etc.).

Every new script in this batch (federated_shap.py, aggregate_shap.py,
sensitivity_analysis.py) calls into this module instead of writing files
directly, so all of today's output ends up in one place:

    <RESULTS_ROOT>/run_<timestamp>/
        manifest.json          <- index of every file this run produced
        node_shap/*.json       <- per-node mean|SHAP| vectors
        federated_ranking.json <- aggregated top-features table
        sensitivity/*.json     <- the 3-axis robustness results
        run_log.jsonl          <- append-only event log (human-readable)

BEFORE RUNNING: open sepsis_ml/logger.py and check log_experiment()'s exact
keyword arguments — the call in log_to_leaderboard() below is a best guess
based on its signature (phase/dataset/model_name/metrics-shaped) and print_leaderboard(dataset=None, phase=None).
Adjust the kwarg names there if they don't match.
"""
import os
import sys
import json
import datetime as dt

sys.path.insert(0, "D:/pediatric-sepsis-PIC/sepsis_ml")  # ADAPT if needed
try:
    from logger import log_experiment  # your existing utility
    HAVE_PARENT_LOGGER = True
except ImportError:
    HAVE_PARENT_LOGGER = False
    print("[fl_shap_results_io] WARNING: could not import sepsis_ml/logger.py — "
          "falling back to local-only logging. Fix the sys.path line above.")

import fl_shap_config as config


class RunIO:
    def __init__(self, run_name=None):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_name or f"run_{ts}"
        self.run_dir = os.path.join(config.RESULTS_ROOT, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        for sub in ("node_shap", "sensitivity"):
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)
        self.manifest = {"run_id": self.run_id, "started": ts, "files": []}
        self._log_path = os.path.join(self.run_dir, "run_log.jsonl")

    # ---- event log: append-only, human-readable, safe to tail while running ----
    def log(self, event, **fields):
        entry = {"ts": dt.datetime.now().isoformat(), "event": event, **fields}
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{entry['ts']}] {event} :: {fields}")

    # ---- save any JSON-serializable result under a relative path in the run dir ----
    def save_json(self, rel_path, obj):
        full = os.path.join(self.run_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        self.manifest["files"].append(rel_path)
        self._write_manifest()
        return full

    def save_csv(self, rel_path, df):
        full = os.path.join(self.run_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        df.to_csv(full, index=False)
        self.manifest["files"].append(rel_path)
        self._write_manifest()
        return full

    def _write_manifest(self):
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    # ---- also push a row into the parent project's existing leaderboard, if available ----
    def log_to_leaderboard(self, phase, model_name, metrics, extra=None):
        if not HAVE_PARENT_LOGGER:
            return
        try:
            log_experiment(
                phase=phase,
                dataset="B",  # Option B / infection-only cohort, per model_datasets/README.md
                model_name=model_name,
                metrics=metrics,
                **(extra or {}),
            )
        except TypeError as e:
            print(f"[fl_shap_results_io] log_experiment() signature mismatch: {e}\n"
                  f"  -> open sepsis_ml/logger.py and fix the kwargs in log_to_leaderboard().")

    def finish(self):
        self.manifest["finished"] = dt.datetime.now().isoformat()
        self._write_manifest()
        print(f"\nRun complete. Zip this folder to move it to another PC:\n  {self.run_dir}\n")
