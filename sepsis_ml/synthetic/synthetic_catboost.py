"""
synthetic_catboost.py
──────────────────────────────────────────────────────────────────────────────
Experiment B — CatBoost trained on 100% Synthetic Data
Project : Pediatric Sepsis Prediction (PIC Database)

Research Question:
  Can a CatBoost model trained ONLY on CTGAN-generated synthetic patients
  generalise to real held-out patients?

  This addresses a privacy / data-sharing use case:
  "If a hospital cannot share real patient data, can they share a synthetic
   version that still produces a clinically useful model?"

Pipeline:
  1. Load synthetic training data  (model_datasets/synthetic/B_synthetic_train.csv)
  2. Load REAL test data           (model_datasets/B_test_model_ready.csv)
     → Test set is ALWAYS real. Never synthetic.
  3. Load tuned CatBoost hyperparameters from Phase 2 (run_tuned)
     → We reuse the same architecture/hyperparams for a fair comparison.
     → We do NOT load the trained weights — we train from scratch on synth data.
  4. Train CatBoost on synthetic data only
  5. Evaluate on real held-out test set (n=633)
  6. Compare against real-data baseline (Phase 2 tuned CatBoost, AUROC≈0.987)
  7. DeLong test: synthetic-trained vs real-trained
  8. Full metric suite + threshold analysis + calibration + figures

DEVICE SUPPORT:
  CatBoost supports CPU, CUDA (NVIDIA GPU), and MPS-style CPU fallback.
  This script auto-detects the best available device.

  ┌──────────────────────────────────────────────────────────────┐
  │ Device              │  Training Time  │  Notes               │
  ├─────────────────────┼─────────────────┼──────────────────────┤
  │ CUDA (NVIDIA GPU)   │   1–3 min       │  Fastest             │
  │ CPU (Mac M1/M2/M3)  │   3–8 min       │  CatBoost CPU is fast│
  │ CPU (Windows)       │   3–10 min      │  Depends on cores    │
  └──────────────────────────────────────────────────────────────┘
  Note: MPS (Apple Silicon GPU) is NOT supported by CatBoost directly.
  On Mac, CatBoost uses CPU — which is still fast due to its implementation.

FOLDER OUTPUTS (under sepsis_ml/synthetic/catboost/):
  models/   — synthetic_catboost_model.cbm, synthetic_catboost_params.json
  results/  — synthetic_catboost_metrics.json, synthetic_catboost_threshold_table.csv
              synthetic_catboost_test_probs.npy, synthetic_vs_real_comparison.json
  figures/  — synthetic_catboost_roc_pr.png, synthetic_catboost_calibration.png
              synthetic_catboost_confusion_matrix.png,
              synthetic_catboost_threshold_sensitivity.png
  logs/     — synthetic_catboost.log
  outputs/  — synthetic_catboost_summary.txt (human-readable summary)

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/catboost/synthetic_catboost.py
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/catboost/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

# Input data
MODEL_DATA_DIR     = PROJECT_ROOT / "model_datasets"
SYNTH_TRAIN_FILE   = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train.csv"
REAL_TEST_FILE     = MODEL_DATA_DIR / "B_test_model_ready.csv"
REAL_TRAIN_FILE    = MODEL_DATA_DIR / "B_train_model_ready.csv"   # for baseline comparison

# Existing real-data tuned CatBoost artifacts (Phase 2)
REAL_CB_MODEL_PATH = SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm"
REAL_CB_PROBS_PATH = SEPSIS_ML / "results" / "catboost_test_probs.npy"

# Output folders
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
LOGS_DIR    = SCRIPT_DIR / "logs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "synthetic_catboost.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED        = 42
TARGET_COL         = "sepsis_label"
TARGET_SENSITIVITY = 0.90

# Real-data baseline AUROC (Phase 2 tuned CatBoost result)
# Used for DeLong comparison and reporting
REAL_BASELINE_AUROC = 0.9868

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_catboost_device():
    """
    CatBoost device selection:
      - CUDA: if NVIDIA GPU available via torch
      - CPU:  fallback (includes Mac MPS — CatBoost doesn't support MPS natively)
    Returns (task_type, devices) tuple for CatBoostClassifier.
    """
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            log.info(f"  Device: CUDA — {gpu_name}")
            log.info(f"  CatBoost will use GPU acceleration.")
            return "GPU", "0"
        elif torch.backends.mps.is_available():
            log.info("  Device: Apple Silicon MPS detected.")
            log.info("  NOTE: CatBoost does NOT support MPS. Falling back to CPU.")
            log.info("  CatBoost CPU on Apple Silicon is still fast (3–8 min).")
            return "CPU", None
        else:
            log.info("  Device: CPU")
            return "CPU", None
    except ImportError:
        log.info("  torch not found for device detection — using CPU.")
        return "CPU", None

# ══════════════════════════════════════════════════════════════════════════════
# 4.  METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def find_threshold_at_sensitivity(y_true, y_prob, target=TARGET_SENSITIVITY):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = find_threshold_at_sensitivity(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "ppv"        : float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "npv"        : float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "threshold"  : float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_iter=1000, seed=RANDOM_SEED):
    rng    = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def delong_test(y_true, probs_a, probs_b):
    """DeLong test — returns (auroc_a, auroc_b, z, p)."""
    def compute_midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x)
        T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1); i = j
        T2 = np.empty(N); T2[J] = T + 1
        return T2

    def fast_delong(yt, ys):
        m = int(yt.sum()); n = len(yt) - m
        pos = ys[yt == 1]; neg = ys[yt == 0]
        ranks    = compute_midrank(np.concatenate([pos, neg]))
        pos_ranks = ranks[:m]
        auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)
        v10 = (pos_ranks - np.arange(1, m + 1)) / n
        v01 = np.array([(pos > ns).mean() + 0.5 * (pos == ns).mean() for ns in neg])
        var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
        return auc, var, v10, v01

    yt = np.array(y_true); pa = np.array(probs_a); pb = np.array(probs_b)
    auc_a, var_a, v10_a, v01_a = fast_delong(yt, pa)
    auc_b, var_b, v10_b, v01_b = fast_delong(yt, pb)
    m = int(yt.sum()); n = len(yt) - m
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / m +
           np.cov(v01_a, v01_b, ddof=1)[0, 1] / n)
    z = (auc_a - auc_b) / np.sqrt(max(var_a + var_b - 2 * cov, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "synth_cb" : "#E53935",   # red   — synthetic-trained CatBoost (primary)
    "real_cb"  : "#1565C0",   # blue  — real-trained CatBoost (baseline)
    "random"   : "#9E9E9E",   # grey
}


def plot_roc_pr(y_test, synth_probs, real_probs, metrics, real_auroc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Experiment B — CatBoost: Synthetic-Trained vs Real-Trained\n"
        f"Evaluated on Real Test Set (n={len(y_test)})",
        fontsize=12, fontweight="bold"
    )

    # ROC
    for probs, label, color, lw, ls in [
        (synth_probs,
         f"Synth-Trained  AUROC={metrics['auroc']:.4f}",
         COLORS["synth_cb"], 2.5, "-"),
        (real_probs,
         f"Real-Trained   AUROC={real_auroc:.4f}",
         COLORS["real_cb"],  1.5, "--"),
    ]:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)

    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, synth_probs)[:2],
                         alpha=0.06, color=COLORS["synth_cb"])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # PR
    real_auprc = average_precision_score(y_test, real_probs)
    for probs, label, color, lw, ls in [
        (synth_probs,
         f"Synth-Trained  AUPRC={metrics['auprc']:.4f}",
         COLORS["synth_cb"], 2.5, "-"),
        (real_probs,
         f"Real-Trained   AUPRC={real_auprc:.4f}",
         COLORS["real_cb"],  1.5, "--"),
    ]:
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls, label=label)

    prev = y_test.mean()
    axes[1].axhline(prev, color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({prev:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_catboost_roc_pr.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_catboost_roc_pr.png")


def plot_calibration(y_test, synth_probs, real_probs, metrics):
    fig, ax = plt.subplots(figsize=(7, 6))

    real_brier = brier_score_loss(y_test, real_probs)
    for probs, label, color in [
        (synth_probs,
         f"Synth-Trained (Brier={metrics['brier_score']:.4f})",
         COLORS["synth_cb"]),
        (real_probs,
         f"Real-Trained  (Brier={real_brier:.4f})",
         COLORS["real_cb"]),
    ]:
        prob_true_, prob_pred_ = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(prob_pred_, prob_true_, "o-", color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration — Synthetic-Trained vs Real-Trained CatBoost",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_catboost_calibration.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_catboost_calibration.png")


def plot_confusion_matrix(y_test, synth_probs, metrics):
    y_pred = (synth_probs >= metrics["threshold"]).astype(int)
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Reds", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-sepsis", "Sepsis"])
    ax.set_yticklabels(["Non-sepsis", "Sepsis"])
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh_cm else "black",
                    fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Synthetic-Trained CatBoost — Confusion Matrix\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_catboost_confusion_matrix.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_catboost_confusion_matrix.png")


def plot_threshold_sensitivity(y_test, synth_probs, thresh_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Synthetic-Trained CatBoost — Threshold Sensitivity Analysis",
        fontsize=12, fontweight="bold"
    )

    thr_vals  = thresh_df["threshold"].values
    sens_vals = thresh_df["sensitivity"].values
    spec_vals = thresh_df["specificity"].values
    ppv_vals  = thresh_df["ppv"].values
    npv_vals  = thresh_df["npv"].values
    f1_vals   = thresh_df["f1"].values

    axes[0].plot(thr_vals, sens_vals, "o-", color=COLORS["synth_cb"], lw=2,
                 label="Sensitivity")
    axes[0].plot(thr_vals, spec_vals, "s-", color=COLORS["real_cb"],  lw=2,
                 label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thr_vals, ppv_vals, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, npv_vals, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(thr_vals, f1_vals, "o-", color="#6A1B9A", lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_catboost_threshold_sensitivity.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_catboost_threshold_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("EXPERIMENT B — CatBoost: Trained on Synthetic, Tested on Real")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── 6.1  Device detection ─────────────────────────────────────────────────
    log.info("\nDevice detection:")
    task_type, devices = detect_catboost_device()

    # ── 6.2  Load data ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Loading data")
    log.info("=" * 60)

    # Check synthetic data exists
    if not SYNTH_TRAIN_FILE.exists():
        log.error(f"Synthetic training file not found: {SYNTH_TRAIN_FILE}")
        log.error("Please run generate_synthetic_data.py first.")
        sys.exit(1)

    synth_train_df = pd.read_csv(SYNTH_TRAIN_FILE)
    test_df        = pd.read_csv(REAL_TEST_FILE)
    feature_cols   = [c for c in synth_train_df.columns if c != TARGET_COL]

    X_synth = synth_train_df[feature_cols]
    y_synth = synth_train_df[TARGET_COL].values
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL].values

    log.info(f"  Synthetic train : {X_synth.shape} | "
             f"Sepsis: {y_synth.mean():.1%}")
    log.info(f"  Real test       : {X_test.shape}  | "
             f"Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features        : {len(feature_cols)}")
    log.info("")
    log.info("  IMPORTANT: Training on SYNTHETIC data only.")
    log.info("  Evaluation on REAL held-out test set.")

    # ── 6.3  Load existing tuned CatBoost params ──────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Loading tuned CatBoost hyperparameters")
    log.info("=" * 60)

    if REAL_CB_MODEL_PATH.exists():
        real_cb = CatBoostClassifier()
        real_cb.load_model(str(REAL_CB_MODEL_PATH))
        params = real_cb.get_params()
        log.info("  Loaded params from existing tuned model (catboost_tuned.cbm)")
        log.info(f"  Key params: iterations={params.get('iterations')}, "
                 f"depth={params.get('depth')}, "
                 f"learning_rate={params.get('learning_rate'):.4f}")
        # Remove device-specific params before re-setting
        params.pop("task_type", None)
        params.pop("devices",   None)
    else:
        log.warning("  Tuned CatBoost model not found. Using sensible defaults.")
        log.warning(f"  Expected: {REAL_CB_MODEL_PATH}")
        params = {
            "iterations"       : 1000,
            "depth"            : 6,
            "learning_rate"    : 0.05,
            "l2_leaf_reg"      : 3.0,
            "bagging_temperature": 1.0,
            "random_strength"  : 1.0,
            "border_count"     : 128,
            "loss_function"    : "Logloss",
            "eval_metric"      : "AUC",
            "random_seed"      : RANDOM_SEED,
            "verbose"          : 0,
        }

    # Set class weights to match synthetic prevalence
    synth_prevalence = y_synth.mean()
    class_weight_ratio = (1 - synth_prevalence) / synth_prevalence
    params["class_weights"]  = [1.0, round(class_weight_ratio, 2)]
    params["random_seed"]    = RANDOM_SEED
    params["verbose"]        = 100
    params["task_type"]      = task_type
    if devices:
        params["devices"] = devices

    log.info(f"  Class weights: [1.0, {class_weight_ratio:.2f}] "
             f"(based on synthetic prevalence {synth_prevalence:.1%})")
    log.info(f"  Task type: {task_type}")

    # ── 6.4  Train CatBoost on synthetic data ─────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Training CatBoost on synthetic data")
    log.info("=" * 60)
    log.info(f"  Training on {len(X_synth):,} synthetic patients...")

    t_train = time.time()
    synth_cb = CatBoostClassifier(**params)
    synth_cb.fit(X_synth, y_synth)
    train_time = (time.time() - t_train) / 60
    log.info(f"  Training complete in {train_time:.1f} minutes")

    # Save model
    model_path = MODELS_DIR / "synthetic_catboost_model.cbm"
    synth_cb.save_model(str(model_path))
    log.info(f"  Model saved -> {model_path}")

    # Save params used
    params_save = {k: v for k, v in params.items() if k != "verbose"}
    with open(MODELS_DIR / "synthetic_catboost_params.json", "w") as f:
        json.dump(params_save, f, indent=2, default=str)

    # ── 6.5  Load real-trained CatBoost test probabilities ────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 4 — Loading real-trained CatBoost baseline probabilities")
    log.info("=" * 60)

    if REAL_CB_PROBS_PATH.exists():
        real_cb_probs = np.load(str(REAL_CB_PROBS_PATH))
        real_auroc    = roc_auc_score(y_test, real_cb_probs)
        log.info(f"  Loaded real-trained CatBoost probs from {REAL_CB_PROBS_PATH}")
        log.info(f"  Real-trained CatBoost AUROC: {real_auroc:.4f}")
    else:
        log.warning("  Real CatBoost test probs not found. Re-predicting from model.")
        if REAL_CB_MODEL_PATH.exists():
            real_cb_probs = real_cb.predict_proba(X_test)[:, 1]
            real_auroc    = roc_auc_score(y_test, real_cb_probs)
            log.info(f"  Re-predicted real-trained AUROC: {real_auroc:.4f}")
        else:
            log.warning("  Cannot load real-trained model. Using known AUROC=0.9868.")
            real_cb_probs = None
            real_auroc    = REAL_BASELINE_AUROC

    # ── 6.6  Evaluate on real test set ────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 5 — Evaluating on real held-out test set")
    log.info("=" * 60)

    synth_probs = synth_cb.predict_proba(X_test)[:, 1]

    # Save test probabilities
    np.save(RESULTS_DIR / "synthetic_catboost_test_probs.npy", synth_probs)

    # Compute metrics
    metrics  = compute_metrics(y_test, synth_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, synth_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, synth_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  {'=' * 50}")
    log.info(f"  SYNTHETIC-TRAINED CatBoost — TEST SET RESULTS")
    log.info(f"  {'=' * 50}")
    log.info(f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier Score : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}")
    log.info(f"  NPV         : {metrics['npv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"  TP={metrics['tp']} FP={metrics['fp']} "
             f"TN={metrics['tn']} FN={metrics['fn']}")

    log.info(f"\n  Baseline comparison:")
    log.info(f"    Real-trained CatBoost AUROC  : {real_auroc:.4f}")
    log.info(f"    Synth-trained CatBoost AUROC : {metrics['auroc']:.4f}")
    auroc_diff = metrics["auroc"] - real_auroc
    log.info(f"    Difference                   : "
             f"{auroc_diff:+.4f} "
             f"({'synth better' if auroc_diff > 0 else 'real better'})")

    # ── 6.7  DeLong test: synthetic vs real ───────────────────────────────────
    log.info("\n  DeLong test: Synthetic-trained vs Real-trained CatBoost")

    if real_cb_probs is not None:
        auc_s, auc_r, z_stat, p_val = delong_test(y_test, synth_probs, real_cb_probs)
        log.info(f"    Synth-trained AUROC : {auc_s:.4f}")
        log.info(f"    Real-trained AUROC  : {auc_r:.4f}")
        log.info(f"    Z-statistic         : {z_stat:.4f}")
        log.info(f"    P-value             : {p_val:.4f}")
        log.info(f"    Significant         : "
                 f"{'YES (p<0.05)' if p_val < 0.05 else 'NO (p>=0.05)'}")
        delong_results = {
            "synth_auroc" : auc_s,
            "real_auroc"  : auc_r,
            "z_statistic" : z_stat,
            "p_value"     : p_val,
            "significant" : p_val < 0.05,
            "direction"   : "Synth better" if auc_s > auc_r else "Real better",
        }
    else:
        log.warning("  DeLong test skipped — real CatBoost probs not available.")
        delong_results = {"note": "DeLong skipped — real probs not available"}

    # ── 6.8  Threshold sensitivity ────────────────────────────────────────────
    log.info("\n  Threshold sensitivity analysis:")
    thresh_rows = []
    targets     = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    log.info(f"  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    log.info("  " + "-" * 70)
    for target in targets:
        thr = find_threshold_at_sensitivity(y_test, synth_probs, target)
        m   = compute_metrics(y_test, synth_probs, thr)
        thresh_rows.append({
            "target_sensitivity": target,
            "threshold"         : round(thr, 4),
            "sensitivity"       : round(m["sensitivity"], 4),
            "specificity"       : round(m["specificity"], 4),
            "ppv"               : round(m["ppv"], 4),
            "npv"               : round(m["npv"], 4),
            "f1"                : round(m["f1"], 4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
        log.info(f"  {target:>8.2f} {thr:>8.4f} {m['sensitivity']:>7.4f} "
                 f"{m['specificity']:>7.4f} {m['ppv']:>7.4f} {m['npv']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "synthetic_catboost_threshold_table.csv",
                     index=False)

    # ── 6.9  Save results ─────────────────────────────────────────────────────
    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"            : "CatBoost (Synthetic-Trained, Experiment B)",
        "experiment"       : "B",
        "timestamp"        : datetime.now().isoformat(),
        "training_data"    : "Synthetic only (CTGAN-generated)",
        "test_data"        : "Real held-out (Option B, n=633)",
        "n_synthetic_train": int(len(X_synth)),
        "n_real_test"      : int(len(X_test)),
        "n_features"       : len(feature_cols),
        "synth_prevalence" : float(y_synth.mean()),
        "real_prevalence"  : float(y_test.mean()),
        "test_metrics"     : metrics,
        "delong"           : delong_results,
        "real_baseline_auroc" : float(real_auroc),
        "auroc_difference"    : float(metrics["auroc"] - real_auroc),
        "task_type"        : task_type,
        "train_time_min"   : round(train_time, 2),
        "runtime_minutes"  : round(runtime, 2),
    }

    with open(RESULTS_DIR / "synthetic_catboost_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Metrics saved -> {RESULTS_DIR / 'synthetic_catboost_metrics.json'}")

    # ── 6.10  Figures ─────────────────────────────────────────────────────────
    log.info("\nGenerating figures...")

    if real_cb_probs is not None:
        plot_roc_pr(y_test, synth_probs, real_cb_probs, metrics, real_auroc)
    plot_calibration(y_test, synth_probs,
                     real_cb_probs if real_cb_probs is not None else synth_probs,
                     metrics)
    plot_confusion_matrix(y_test, synth_probs, metrics)
    plot_threshold_sensitivity(y_test, synth_probs, thresh_df)

    # ── 6.11  Human-readable summary ──────────────────────────────────────────
    summary_lines = [
        "=" * 70,
        "EXPERIMENT B — CatBoost: Synthetic-Trained vs Real-Trained",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "SETUP",
        f"  Training data : {len(X_synth):,} synthetic patients (CTGAN-generated)",
        f"  Test data     : {len(X_test):,} real patients (never synthesised)",
        f"  Features      : {len(feature_cols)}",
        f"  Device        : {task_type}",
        "",
        "SYNTHETIC-TRAINED CatBoost RESULTS (on real test set)",
        f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]",
        f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]",
        f"  Brier Score : {metrics['brier_score']:.4f}",
        f"  Sensitivity : {metrics['sensitivity']:.4f}",
        f"  Specificity : {metrics['specificity']:.4f}",
        f"  PPV         : {metrics['ppv']:.4f}",
        f"  NPV         : {metrics['npv']:.4f}",
        f"  F1          : {metrics['f1']:.4f}",
        "",
        "COMPARISON",
        f"  Real-trained CatBoost AUROC  : {real_auroc:.4f}",
        f"  Synth-trained CatBoost AUROC : {metrics['auroc']:.4f}",
        f"  Difference                   : {metrics['auroc'] - real_auroc:+.4f}",
    ]

    if real_cb_probs is not None and "z_statistic" in delong_results:
        summary_lines += [
            "",
            "DeLong TEST",
            f"  Z-statistic : {delong_results['z_statistic']:.4f}",
            f"  P-value     : {delong_results['p_value']:.4f}",
            f"  Significant : {'YES' if delong_results['significant'] else 'NO'}",
            f"  Direction   : {delong_results['direction']}",
        ]

    summary_lines += [
        "",
        f"Runtime : {runtime:.1f} min",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    with open(OUTPUTS_DIR / "synthetic_catboost_summary.txt", "w") as f:
        f.write(summary_text)
    log.info(f"  Summary saved -> {OUTPUTS_DIR / 'synthetic_catboost_summary.txt'}")

    # ── 6.12  Final log ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT B — CatBoost COMPLETE")
    log.info("=" * 70)
    log.info(f"  AUROC      : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC      : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier      : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  vs Real-trained CatBoost: AUROC diff = "
             f"{metrics['auroc'] - real_auroc:+.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")
    log.info(f"\n  Models   -> {MODELS_DIR}")
    log.info(f"  Results  -> {RESULTS_DIR}")
    log.info(f"  Figures  -> {FIGURES_DIR}")
    log.info(f"  Outputs  -> {OUTPUTS_DIR}")
    log.info(f"  Log      -> {log_path}")
    log.info("\n  Next: Run synthetic_stacking.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
