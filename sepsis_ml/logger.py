"""
logger.py
---------
Experiment results logger. Every model run calls log_experiment().
Appends to experiment_log.json (full detail) and results_summary.csv (paper table).
Never overwrites — always appends so no results are lost.
"""

import json
import csv
import datetime
from pathlib import Path
from config import EXPERIMENT_LOG, RESULTS_SUMMARY, EVAL_METRICS


def log_experiment(
    experiment_id: str,
    dataset: str,
    model_name: str,
    phase: str,
    params: dict,
    metrics: dict,
    notes: str = "",
    extra: dict = None,
) -> None:
    """
    Log a single experiment run.

    Parameters
    ----------
    experiment_id : str
        Unique ID, e.g. "phase2_catboost_B_baseline"
    dataset : str
        "A" or "B"
    model_name : str
        e.g. "catboost", "xgboost"
    phase : str
        e.g. "baseline", "tuned", "ablation_no_symptoms"
    params : dict
        Hyperparameters used (full dict)
    metrics : dict
        Must contain keys from EVAL_METRICS. Also accepts CI bounds:
        e.g. {"auroc": 0.91, "auroc_ci_low": 0.88, "auroc_ci_high": 0.93, ...}
    notes : str
        Free-text note for the paper methods section
    extra : dict
        Any additional fields (threshold used, n_train, n_test, etc.)
    """
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    record = {
        "timestamp"      : timestamp,
        "experiment_id"  : experiment_id,
        "dataset"        : dataset,
        "model_name"     : model_name,
        "phase"          : phase,
        "params"         : params,
        "metrics"        : metrics,
        "notes"          : notes,
    }
    if extra:
        record["extra"] = extra

    # ── Append to JSON log (full detail) ──────────────────────────────────────
    existing = []
    if EXPERIMENT_LOG.exists():
        try:
            with open(EXPERIMENT_LOG, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = []

    existing.append(record)
    with open(EXPERIMENT_LOG, "w") as f:
        json.dump(existing, f, indent=2, default=str)

    # ── Append to CSV summary (flat row for easy Excel/paper table) ───────────
    flat = {
        "timestamp"     : timestamp,
        "experiment_id" : experiment_id,
        "dataset"       : dataset,
        "model_name"    : model_name,
        "phase"         : phase,
        "notes"         : notes,
    }
    for metric in EVAL_METRICS:
        flat[metric]            = metrics.get(metric, "")
        flat[f"{metric}_ci_low"]  = metrics.get(f"{metric}_ci_low", "")
        flat[f"{metric}_ci_high"] = metrics.get(f"{metric}_ci_high", "")

    if extra:
        for k, v in extra.items():
            flat[k] = v

    write_header = not RESULTS_SUMMARY.exists()
    with open(RESULTS_SUMMARY, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    print(f"  [logger] saved → {experiment_id} | AUROC={metrics.get('auroc','?'):.4f}"
          f" | AUPRC={metrics.get('auprc','?'):.4f}")


def load_all_results() -> list:
    """Return all logged experiments as a list of dicts."""
    if not EXPERIMENT_LOG.exists():
        return []
    with open(EXPERIMENT_LOG, "r") as f:
        return json.load(f)


def print_leaderboard(dataset: str = None, phase: str = None) -> None:
    """Print a sorted leaderboard of all logged experiments."""
    records = load_all_results()
    if dataset:
        records = [r for r in records if r.get("dataset") == dataset]
    if phase:
        records = [r for r in records if r.get("phase") == phase]

    records = sorted(
        records,
        key=lambda r: r.get("metrics", {}).get("auprc", 0),
        reverse=True,
    )

    print(f"\n{'='*80}")
    print(f"  LEADERBOARD  |  dataset={dataset or 'all'}  |  phase={phase or 'all'}")
    print(f"{'='*80}")
    header = f"{'Rank':<5} {'Model':<22} {'Dataset':<8} {'Phase':<20} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Sens':>6} {'Spec':>6}"
    print(header)
    print("-" * 80)
    for i, r in enumerate(records, 1):
        m = r.get("metrics", {})
        print(
            f"{i:<5} {r.get('model_name','?'):<22} {r.get('dataset','?'):<8} "
            f"{r.get('phase','?'):<20} "
            f"{m.get('auroc','?'):>7.4f} {m.get('auprc','?'):>7.4f} "
            f"{m.get('f1','?'):>6.4f} {m.get('sensitivity','?'):>6.4f} "
            f"{m.get('specificity','?'):>6.4f}"
        )
    print(f"{'='*80}\n")
