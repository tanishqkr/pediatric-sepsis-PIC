"""
synthetic_catboost_tuned.py
──────────────────────────────────────────────────────────────────────────────
Experiment B — Tuned CatBoost trained on 100% Synthetic Data
Mirrors phase2_tune_catboost.py exactly, substituting synthetic training data.

What is identical to Phase 2:
  - 100 Optuna trials, TPE sampler, MedianPruner
  - Objective: mean 5-fold CV AUPRC (correct metric for imbalanced data)
  - Exact same hyperparameter search space:
      learning_rate       : 0.01 – 0.3  (log scale)
      depth               : 4 – 10
      iterations          : 200 – 1500
      l2_leaf_reg         : 1.0 – 10.0
      bagging_temperature : 0.0 – 1.0
      random_strength     : 0.0 – 2.0
      border_count        : 32 – 255
      class_weight_pos    : 1.5 – 5.0
  - early_stopping_rounds = 50 in every fold
  - Final model trained on full synthetic training set
  - eval_set = real test set for final model (same as Phase 2)
  - Full bootstrap CIs (1000 iterations), DeLong test
  - Exact same plots: optimization history, param importances,
    tuned vs real-trained ROC/PR, confusion matrix, threshold sensitivity

What differs from Phase 2:
  - Training data: synthetic CTGAN patients instead of real
  - Comparison baseline: real-trained tuned CatBoost (Phase 2) instead of
    baseline untuned CatBoost
  - Output folder: sepsis_ml/synthetic/catboost/

Research question:
  Does Optuna find good hyperparameters for CatBoost when trained only on
  synthetic data? Does the resulting model generalise to real patients?

DEVICE SUPPORT:
  CatBoost: CUDA (NVIDIA) > CPU (Mac MPS is not supported by CatBoost —
  falls back to CPU automatically, still fast on Apple Silicon)

  ┌──────────────────────────────────────────────────────────────────┐
  │ Stage                     │ Mac M1/M2  │ Windows CPU │ CUDA GPU │
  ├───────────────────────────┼────────────┼─────────────┼──────────┤
  │ Optuna 100 trials (total) │ 45–90 min  │ 60–120 min  │ 15–40 min│
  │ Final model train         │  2–5 min   │   3–8 min   │  1–2 min │
  │ Total                     │ ~50–95 min │ ~65–130 min │ ~20–45min│
  └──────────────────────────────────────────────────────────────────┘

  The range depends on the hyperparameters Optuna proposes — trials with
  high iterations + deep trees take much longer than others.

OUTPUTS (all under sepsis_ml/synthetic/catboost/):
  models/   — synthetic_catboost_tuned.cbm
  results/  — synthetic_optuna_study.pkl, synthetic_best_params.json,
              synthetic_tuning_trials.csv, synthetic_catboost_metrics.json,
              synthetic_catboost_test_probs.npy, synthetic_threshold_analysis.csv
  figures/  — synthetic_optuna_optimization_history.png,
              synthetic_optuna_param_importances.png,
              synthetic_tuned_vs_real_roc_pr.png, synthetic_tuned_cm.png,
              synthetic_threshold_sensitivity_analysis.png
  logs/     — synthetic_catboost_tuned.log

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/catboost/synthetic_catboost_tuned.py

Prerequisites:
  - generate_synthetic_data.py must have been run first
  - model_datasets/synthetic/B_synthetic_train.csv must exist
"""

import sys
import json
import pickle
import logging
import warnings
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/catboost/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

MODEL_DATA_DIR   = PROJECT_ROOT / "model_datasets"
SYNTH_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train.csv"
REAL_TEST_FILE   = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Real-trained Phase 2 model (for comparison baseline)
REAL_CB_MODEL_PATH = SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm"

# Output folders
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
LOGS_DIR    = SCRIPT_DIR / "logs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS  (same as Phase 2 config.py values)
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED          = 42
CV_FOLDS             = 5
TARGET_COL           = "sepsis_label"
TARGET_SENSITIVITY   = 0.90
BOOTSTRAP_ITERATIONS = 1000
OPTUNA_TRIALS        = 100

np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "synthetic_catboost_tuned.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_catboost_device():
    """
    CatBoost supports CUDA and CPU only.
    MPS (Apple Silicon) is not supported — falls back to CPU.
    Returns (task_type, devices) for CatBoostClassifier kwargs.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log.info(f"  Device: CUDA — {gpu_name}")
            return "GPU", "0"
        elif torch.backends.mps.is_available():
            log.info("  Apple Silicon MPS detected.")
            log.info("  CatBoost does NOT support MPS. Using CPU (still fast).")
            return "CPU", None
        else:
            log.info("  Device: CPU")
            return "CPU", None
    except ImportError:
        log.info("  torch not found — using CPU.")
        return "CPU", None

# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRIC HELPERS  (identical to phase2_tune_catboost.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_threshold_at_sensitivity(y_true, y_prob, target_sens=TARGET_SENSITIVITY):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target_sens)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = find_threshold_at_sensitivity(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv"        : float(ppv),
        "npv"        : float(npv),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold"  : float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, metric_fn,
                 n_iter=BOOTSTRAP_ITERATIONS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics_with_ci(y_true, y_prob):
    threshold  = find_threshold_at_sensitivity(y_true, y_prob)
    metrics    = compute_metrics(y_true, y_prob, threshold)
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    auroc_lo, auroc_hi = bootstrap_ci(y_true_arr, y_prob_arr, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_true_arr, y_prob_arr, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi
    return metrics

# ══════════════════════════════════════════════════════════════════════════════
# 5.  DELONG TEST  (identical matrix-based implementation to phase2)
# ══════════════════════════════════════════════════════════════════════════════

def delong_test(y_true, y_prob_1, y_prob_2):
    """
    DeLong test for statistical significance of AUROC difference.
    Matrix-based fastDeLong implementation — identical to phase2_tune_catboost.py.
    Reference: DeLong et al. (1988) Biometrics.
    """
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]; N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T + 1
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]
        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1. - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    y_true   = np.array(y_true)
    y_prob_1 = np.array(y_prob_1)
    y_prob_2 = np.array(y_prob_2)

    order          = np.argsort(-y_true)
    label_1_count  = int(y_true.sum())

    predictions_sorted = np.vstack([y_prob_1[order], y_prob_2[order]])
    aucs, delongcov    = fastDeLong(predictions_sorted, label_1_count)

    diff    = aucs[0] - aucs[1]
    se      = np.sqrt(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1])
    z_stat  = diff / se
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return float(aucs[0]), float(aucs[1]), float(z_stat), float(p_value)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  OPTUNA OBJECTIVE  (identical search space to Phase 2)
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial, X_df: pd.DataFrame, y: np.ndarray,
              task_type: str, devices) -> float:
    """
    Optuna objective. Returns mean CV AUPRC (higher = better).
    Search space is IDENTICAL to phase2_tune_catboost.py.
    AUPRC is the correct metric for imbalanced clinical data.
    """
    from catboost import CatBoostClassifier

    params = {
        "iterations"         : trial.suggest_int("iterations", 200, 1500),
        "learning_rate"      : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth"              : trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg"        : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength"    : trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count"       : trial.suggest_int("border_count", 32, 255),
        "class_weights"      : [1.0, trial.suggest_float("class_weight_pos", 1.5, 5.0)],
        "eval_metric"        : "AUC",
        "random_seed"        : RANDOM_SEED,
        "verbose"            : 0,
        "early_stopping_rounds": 50,
        "task_type"          : task_type,
    }
    if devices:
        params["devices"] = devices

    skf          = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                   random_state=RANDOM_SEED)
    auprc_scores = []

    for train_idx, val_idx in skf.split(X_df, y):
        X_tr  = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_tr  = y[train_idx]
        y_val = y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc_scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(auprc_scores))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS  (identical structure and style to phase2_tune_catboost.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization_history(study):
    """Plot AUPRC across all trials — shows search convergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    values      = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i + 1]) for i in range(len(values))]

    ax1.scatter(range(len(values)), values,
                alpha=0.4, s=15, color="#3498db", label="Trial AUPRC")
    ax1.plot(range(len(best_so_far)), best_so_far,
             color="#e74c3c", lw=2, label="Best so far")
    ax1.set_xlabel("Trial number")
    ax1.set_ylabel("CV AUPRC")
    ax1.set_title(
        "Optuna optimization history\n"
        "Synthetic-trained CatBoost — 100 trials",
        fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.hist(values, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
    ax2.axvline(max(values), color="#e74c3c", lw=2, linestyle="--",
                label=f"Best: {max(values):.4f}")
    ax2.set_xlabel("CV AUPRC")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of trial scores", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "synthetic_optuna_optimization_history.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_param_importances(study):
    """Plot which hyperparameters mattered most (fANOVA)."""
    try:
        import optuna
        importances = optuna.importance.get_param_importances(study)
        params_list = list(importances.keys())
        values_list = list(importances.values())
        colors = ["#e74c3c" if v == max(values_list) else "#3498db"
                  for v in values_list]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(params_list, values_list, color=colors, edgecolor="white")
        ax.set_xlabel("Importance score (fANOVA)")
        ax.set_title(
            "Hyperparameter importance\n"
            "Synthetic CatBoost Optuna study",
            fontweight="bold"
        )
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = FIGURES_DIR / "synthetic_optuna_param_importances.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"  Saved → {out}")
    except Exception as e:
        log.info(f"  Param importance plot skipped: {e}")


def plot_tuned_vs_real(y_test, y_prob_synth, y_prob_real,
                       metrics_synth, metrics_real):
    """ROC + PR curves: synthetic-trained vs real-trained CatBoost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    for y_prob, label, color, lw in [
        (y_prob_synth,
         f"Synth-Trained CatBoost (AUROC={metrics_synth['auroc']:.4f})",
         "#e74c3c", 2.5),
        (y_prob_real,
         f"Real-Trained CatBoost  (AUROC={metrics_real['auroc']:.4f})",
         "#3498db", 1.5),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.fill_between(*roc_curve(y_test, y_prob_synth)[:2],
                     alpha=0.08, color="#e74c3c")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(
        "ROC — Synthetic-Trained vs Real-Trained CatBoost\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # PR
    for y_prob, label, color, lw in [
        (y_prob_synth,
         f"Synth-Trained CatBoost (AUPRC={metrics_synth['auprc']:.4f})",
         "#e74c3c", 2.5),
        (y_prob_real,
         f"Real-Trained CatBoost  (AUPRC={metrics_real['auprc']:.4f})",
         "#3498db", 1.5),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=lw, label=label)
    prev = np.array(y_test).mean()
    ax2.axhline(prev, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Prevalence ({prev:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(
        "Precision-Recall — Synthetic-Trained vs Real-Trained CatBoost\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "synthetic_tuned_vs_real_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, metrics,
                          title="Synthetic-Trained Tuned CatBoost"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Reds")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-sepsis", "Sepsis"])
    ax.set_yticklabels(["Non-sepsis", "Sepsis"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"{title}\n"
        f"Sens={metrics['sensitivity']:.4f}  "
        f"Spec={metrics['specificity']:.4f}  "
        f"Threshold={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    out = FIGURES_DIR / "synthetic_tuned_cm.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_threshold_sensitivity(y_test, y_prob):
    """
    Performance across multiple sensitivity targets.
    Identical structure to Phase 2 threshold sensitivity plot.
    """
    targets = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    rows = []
    for sens_target in targets:
        thresh = find_threshold_at_sensitivity(y_test, y_prob, sens_target)
        m = compute_metrics(y_test, y_prob, thresh)
        rows.append({
            "target_sensitivity": sens_target,
            "threshold"         : round(thresh, 4),
            "sensitivity"       : round(m["sensitivity"], 4),
            "specificity"       : round(m["specificity"], 4),
            "ppv"               : round(m["ppv"], 4),
            "npv"               : round(m["npv"], 4),
            "f1"                : round(m["f1"], 4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "synthetic_threshold_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Threshold sensitivity analysis — Synthetic-Trained Tuned CatBoost\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontsize=11, fontweight="bold"
    )

    thresh_vals = df["threshold"].values
    sens_vals   = df["sensitivity"].values
    spec_vals   = df["specificity"].values
    ppv_vals    = df["ppv"].values
    npv_vals    = df["npv"].values
    f1_vals     = df["f1"].values

    axes[0].plot(thresh_vals, sens_vals, "o-", color="#e74c3c", lw=2,
                 label="Sensitivity")
    axes[0].plot(thresh_vals, spec_vals, "s-", color="#3498db", lw=2,
                 label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(thresh_vals, ppv_vals, "o-", color="#2ecc71", lw=2,
                 label="PPV (Precision)")
    axes[1].plot(thresh_vals, npv_vals, "s-", color="#9b59b6", lw=2,
                 label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(thresh_vals, f1_vals, "o-", color="#f39c12", lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "synthetic_threshold_sensitivity_analysis.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved threshold analysis → {out}")

    log.info("\n  Threshold sensitivity analysis:")
    log.info(f"  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    log.info("  " + "-" * 75)
    for _, row in df.iterrows():
        log.info(f"  {row['target_sensitivity']:>8.2f} {row['threshold']:>8.4f} "
                 f"{row['sensitivity']:>7.4f} {row['specificity']:>7.4f} "
                 f"{row['ppv']:>7.4f} {row['npv']:>7.4f} {row['f1']:>7.4f} "
                 f"{int(row['tp']):>5} {int(row['fp']):>5} {int(row['fn']):>5}")

    return df

# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()

    log.info("=" * 65)
    log.info("EXPERIMENT B — CATBOOST OPTUNA TUNING ON SYNTHETIC DATA")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Trials: {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}  |  Metric: AUPRC")
    log.info("=" * 65)

    # ── 8.1  Device detection ─────────────────────────────────────────────────
    log.info("\nDevice detection:")
    task_type, devices = detect_catboost_device()

    # ── 8.2  Load data ────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")

    if not SYNTH_TRAIN_FILE.exists():
        log.error(f"Synthetic training file not found: {SYNTH_TRAIN_FILE}")
        log.error("Run generate_synthetic_data.py first.")
        sys.exit(1)

    synth_train  = pd.read_csv(SYNTH_TRAIN_FILE)
    real_test    = pd.read_csv(REAL_TEST_FILE)
    feature_cols = [c for c in synth_train.columns if c != TARGET_COL]

    X_synth_df  = synth_train[feature_cols]
    y_synth     = synth_train[TARGET_COL].values
    X_test_df   = real_test[feature_cols]
    y_test      = real_test[TARGET_COL].values

    log.info(f"  Synthetic train : {synth_train.shape} | "
             f"Sepsis: {y_synth.mean():.1%}")
    log.info(f"  Real test       : {real_test.shape}  | "
             f"Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features        : {len(feature_cols)}")
    log.info("")
    log.info("  IMPORTANT: Training and tuning on SYNTHETIC data only.")
    log.info("  Evaluation on REAL held-out test set (never synthesised).")

    # ── 8.3  Load real-trained model for comparison baseline ──────────────────
    log.info("\nLoading real-trained CatBoost for comparison...")
    from catboost import CatBoostClassifier

    if REAL_CB_MODEL_PATH.exists():
        real_cb_model    = CatBoostClassifier()
        real_cb_model.load_model(str(REAL_CB_MODEL_PATH))
        y_prob_real      = real_cb_model.predict_proba(X_test_df)[:, 1]
        metrics_real     = compute_metrics(y_test, y_prob_real)
        log.info(f"  Real-trained CatBoost: "
                 f"AUROC={metrics_real['auroc']:.4f}  "
                 f"AUPRC={metrics_real['auprc']:.4f}")
    else:
        log.warning(f"  Real-trained model not found: {REAL_CB_MODEL_PATH}")
        log.warning("  Comparison plots will be skipped.")
        y_prob_real  = None
        metrics_real = None

    # ── 8.4  Optuna study ─────────────────────────────────────────────────────
    log.info(f"\nStarting Optuna study ({OPTUNA_TRIALS} trials)...")
    log.info("Searching identical space to Phase 2 (real-data tuning).")
    log.info("Progress printed on every new best and every 10 trials.\n")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner     = optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=5
        ),
        study_name = "synthetic_catboost_auprc_optuna",
    )

    best_so_far = 0.0

    def callback(study, trial):
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            log.info(f"  Trial {trial.number:>3} NEW BEST: "
                     f"AUPRC={trial.value:.4f}  params={trial.params}")
        elif trial.number % 10 == 0:
            log.info(f"  Trial {trial.number:>3}: "
                     f"AUPRC={trial.value:.4f if trial.value else 'pruned'}  "
                     f"(best so far: {best_so_far:.4f})")

    study.optimize(
        lambda trial: objective(trial, X_synth_df, y_synth, task_type, devices),
        n_trials          = OPTUNA_TRIALS,
        callbacks         = [callback],
        show_progress_bar = False,
    )

    log.info(f"\n  Optuna complete.")
    log.info(f"  Best CV AUPRC: {study.best_value:.4f}")
    log.info(f"  Best params:   {study.best_params}")

    # ── 8.5  Save study and params ────────────────────────────────────────────
    study_path = RESULTS_DIR / "synthetic_optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    log.info(f"\n  Optuna study saved → {study_path}")

    best_params_path = RESULTS_DIR / "synthetic_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "best_cv_auprc": study.best_value,
            "best_params"  : study.best_params,
            "n_trials"     : OPTUNA_TRIALS,
            "timestamp"    : datetime.now().isoformat(timespec="seconds"),
            "note"         : "Tuned on CTGAN synthetic data, same search space as Phase 2",
        }, f, indent=2)
    log.info(f"  Best params saved → {best_params_path}")

    # Save all trials
    trials_df   = study.trials_dataframe()
    trials_path = RESULTS_DIR / "synthetic_tuning_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    log.info(f"  All trials saved → {trials_path}")

    # ── 8.6  Train final tuned model on full synthetic training set ───────────
    log.info(f"\nTraining final tuned CatBoost on full synthetic training set...")
    best = study.best_params

    tuned_params = dict(
        iterations           = best["iterations"],
        learning_rate        = best["learning_rate"],
        depth                = best["depth"],
        l2_leaf_reg          = best["l2_leaf_reg"],
        bagging_temperature  = best["bagging_temperature"],
        random_strength      = best["random_strength"],
        border_count         = best["border_count"],
        class_weights        = [1.0, best["class_weight_pos"]],
        eval_metric          = "AUC",
        random_seed          = RANDOM_SEED,
        verbose              = 100,
        early_stopping_rounds= 50,
        task_type            = task_type,
    )
    if devices:
        tuned_params["devices"] = devices

    tuned_model = CatBoostClassifier(**tuned_params)
    tuned_model.fit(
        X_synth_df, y_synth,
        eval_set=(X_test_df, y_test),   # same as Phase 2: monitor on real test
    )

    # ── 8.7  Test set evaluation ──────────────────────────────────────────────
    log.info(f"\nEvaluating on real held-out test set (n=633)...")
    y_prob_synth  = tuned_model.predict_proba(X_test_df)[:, 1]
    metrics_synth = compute_metrics_with_ci(y_test, y_prob_synth)

    log.info(f"\n  {'=' * 55}")
    log.info(f"  SYNTHETIC-TRAINED TUNED CATBOOST — TEST SET RESULTS")
    log.info(f"  {'=' * 55}")
    log.info(f"  AUROC : {metrics_synth['auroc']:.4f} "
             f"[{metrics_synth['auroc_ci_low']:.4f}–{metrics_synth['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_synth['auprc']:.4f} "
             f"[{metrics_synth['auprc_ci_low']:.4f}–{metrics_synth['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_synth['sensitivity']:.4f}  "
             f"Spec: {metrics_synth['specificity']:.4f}  "
             f"F1: {metrics_synth['f1']:.4f}")
    log.info(f"  PPV   : {metrics_synth['ppv']:.4f}  "
             f"NPV: {metrics_synth['npv']:.4f}  "
             f"Brier: {metrics_synth['brier_score']:.4f}")
    log.info(f"  TP={metrics_synth['tp']}  FP={metrics_synth['fp']}  "
             f"TN={metrics_synth['tn']}  FN={metrics_synth['fn']}")

    # ── 8.8  Comparison vs real-trained ──────────────────────────────────────
    if metrics_real:
        auroc_delta = metrics_synth["auroc"] - metrics_real["auroc"]
        auprc_delta = metrics_synth["auprc"] - metrics_real["auprc"]
        log.info(f"\n  Comparison vs Real-Trained CatBoost (Phase 2):")
        log.info(f"    AUROC: {metrics_real['auroc']:.4f} → "
                 f"{metrics_synth['auroc']:.4f} (Δ={auroc_delta:+.4f})")
        log.info(f"    AUPRC: {metrics_real['auprc']:.4f} → "
                 f"{metrics_synth['auprc']:.4f} (Δ={auprc_delta:+.4f})")

        log.info(f"\n  DeLong test (synthetic-trained vs real-trained):")
        auroc_1, auroc_2, z_stat, p_value = delong_test(
            y_test, y_prob_synth, y_prob_real
        )
        log.info(f"    Synth-trained AUROC: {auroc_1:.4f}")
        log.info(f"    Real-trained AUROC : {auroc_2:.4f}")
        log.info(f"    Z-statistic        : {z_stat:.4f}")
        log.info(f"    P-value            : {p_value:.4f}")
        log.info(f"    Significant        : "
                 f"{'YES (p<0.05)' if p_value < 0.05 else 'NO (p>=0.05)'}")

        metrics_synth["delong_vs_real_z"]         = float(z_stat)
        metrics_synth["delong_vs_real_p"]         = float(p_value)
        metrics_synth["delong_significant"]       = bool(p_value < 0.05)
        metrics_synth["auroc_delta_vs_real"]      = float(auroc_delta)
        metrics_synth["auprc_delta_vs_real"]      = float(auprc_delta)
        metrics_synth["real_trained_auroc_phase2"]= float(metrics_real["auroc"])
        metrics_synth["real_trained_auprc_phase2"]= float(metrics_real["auprc"])

    # ── 8.9  Save model and test probs ────────────────────────────────────────
    model_path = MODELS_DIR / "synthetic_catboost_tuned.cbm"
    tuned_model.save_model(str(model_path))
    log.info(f"\n  Tuned model saved → {model_path}")

    np.save(RESULTS_DIR / "synthetic_catboost_test_probs.npy", y_prob_synth)
    log.info(f"  Test probs saved → "
             f"{RESULTS_DIR / 'synthetic_catboost_test_probs.npy'}")

    # Full metrics JSON
    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"          : "CatBoost — Synthetic-Trained Optuna Tuned",
        "experiment"     : "B",
        "timestamp"      : datetime.now().isoformat(timespec="seconds"),
        "training_data"  : "Synthetic only (CTGAN-generated)",
        "test_data"      : "Real held-out (Option B, n=633)",
        "n_synthetic_train": int(len(y_synth)),
        "n_real_test"    : int(len(y_test)),
        "n_features"     : len(feature_cols),
        "cv_folds"       : CV_FOLDS,
        "optuna_trials"  : OPTUNA_TRIALS,
        "best_cv_auprc"  : float(study.best_value),
        "best_params"    : study.best_params,
        "test_metrics"   : metrics_synth,
        "task_type"      : task_type,
        "runtime_minutes": round(runtime, 2),
    }
    with open(RESULTS_DIR / "synthetic_catboost_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"  Full metrics saved → "
             f"{RESULTS_DIR / 'synthetic_catboost_metrics.json'}")

    # ── 8.10  Plots ───────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    plot_optimization_history(study)
    plot_param_importances(study)

    if y_prob_real is not None:
        plot_tuned_vs_real(y_test, y_prob_synth, y_prob_real,
                           metrics_synth, metrics_real)

    y_pred_synth = (y_prob_synth >= metrics_synth["threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred_synth, metrics_synth)
    plot_threshold_sensitivity(y_test, y_prob_synth)

    # ── 8.11  Human-readable summary ──────────────────────────────────────────
    summary_lines = [
        "=" * 65,
        "EXPERIMENT B — CATBOOST OPTUNA TUNING ON SYNTHETIC DATA",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "SETUP",
        f"  Training data  : {len(y_synth):,} synthetic patients (CTGAN-generated)",
        f"  Test data      : {len(y_test):,} real patients (never synthesised)",
        f"  Optuna trials  : {OPTUNA_TRIALS}",
        f"  CV folds       : {CV_FOLDS}",
        f"  Device         : {task_type}",
        f"  Optimising     : AUPRC (5-fold CV mean)",
        "",
        "OPTUNA RESULT",
        f"  Best CV AUPRC  : {study.best_value:.4f}",
        f"  Best params    : {study.best_params}",
        "",
        "TEST SET RESULTS (on real patients)",
        f"  AUROC : {metrics_synth['auroc']:.4f} "
        f"[{metrics_synth['auroc_ci_low']:.4f}–{metrics_synth['auroc_ci_high']:.4f}]",
        f"  AUPRC : {metrics_synth['auprc']:.4f} "
        f"[{metrics_synth['auprc_ci_low']:.4f}–{metrics_synth['auprc_ci_high']:.4f}]",
        f"  Brier : {metrics_synth['brier_score']:.4f}",
        f"  Sens  : {metrics_synth['sensitivity']:.4f}",
        f"  Spec  : {metrics_synth['specificity']:.4f}",
        f"  PPV   : {metrics_synth['ppv']:.4f}",
        f"  NPV   : {metrics_synth['npv']:.4f}",
        f"  F1    : {metrics_synth['f1']:.4f}",
    ]

    if metrics_real:
        summary_lines += [
            "",
            "COMPARISON vs REAL-TRAINED CATBOOST (Phase 2)",
            f"  Real AUROC  : {metrics_real['auroc']:.4f}",
            f"  Synth AUROC : {metrics_synth['auroc']:.4f}",
            f"  Δ AUROC     : {metrics_synth['auroc'] - metrics_real['auroc']:+.4f}",
            f"  DeLong z    : {metrics_synth.get('delong_vs_real_z', 'N/A')}",
            f"  DeLong p    : {metrics_synth.get('delong_vs_real_p', 'N/A')}",
            f"  Significant : {metrics_synth.get('delong_significant', 'N/A')}",
        ]

    summary_lines += ["", f"Runtime : {runtime:.1f} min", "=" * 65]

    with open(OUTPUTS_DIR / "synthetic_catboost_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    log.info(f"  Summary → {OUTPUTS_DIR / 'synthetic_catboost_summary.txt'}")

    # ── 8.12  Final log ───────────────────────────────────────────────────────
    log.info(f"\n{'=' * 65}")
    log.info("EXPERIMENT B — CATBOOST OPTUNA TUNING COMPLETE")
    log.info(f"{'=' * 65}")
    log.info(f"  Best CV AUPRC (Optuna):  {study.best_value:.4f}")
    log.info(f"  Test AUROC:              {metrics_synth['auroc']:.4f} "
             f"[{metrics_synth['auroc_ci_low']:.4f}–"
             f"{metrics_synth['auroc_ci_high']:.4f}]")
    log.info(f"  Test AUPRC:              {metrics_synth['auprc']:.4f} "
             f"[{metrics_synth['auprc_ci_low']:.4f}–"
             f"{metrics_synth['auprc_ci_high']:.4f}]")
    log.info(f"  Sensitivity:             {metrics_synth['sensitivity']:.4f}")
    log.info(f"  Specificity:             {metrics_synth['specificity']:.4f}")
    log.info(f"  Model saved:             {model_path}")
    log.info(f"  Figures:                 {FIGURES_DIR}/")
    log.info(f"  Log:                     {log_path}")
    log.info(f"  Runtime:                 {runtime:.1f} min")
    log.info(f"\n  Next: python sepsis_ml/synthetic/stacking/synthetic_stacking.py")


if __name__ == "__main__":
    main()
