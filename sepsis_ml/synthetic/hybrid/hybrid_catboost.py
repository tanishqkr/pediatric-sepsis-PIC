"""
hybrid_catboost.py
──────────────────────────────────────────────────────────────────────────────
Experiment C — Tuned CatBoost trained on Real + Synthetic (Hybrid) Data

Mirrors synthetic_catboost_tuned.py exactly with two changes:
  1. Training data: C_hybrid_train.csv (real + synthetic combined)
  2. Optuna trials: 5 (quick run — hyperparams likely similar to Phase 2)

Rationale for 5 trials:
  We already have well-tuned hyperparameters from Phase 2 (real data) and
  Experiment B (synthetic data). Hybrid data is close to real in distribution.
  5 trials confirm whether a different setting helps, without wasting compute.

What is identical to synthetic_catboost_tuned.py:
  - Same hyperparameter search space
  - Same CV objective: mean 5-fold AUPRC
  - Same early_stopping_rounds = 50
  - Final model trained on full hybrid training set
  - eval_set = real test set
  - Full bootstrap CIs (1000 iterations), DeLong test vs real-trained
  - Same plots: optimization history, tuned vs real ROC/PR,
    confusion matrix, threshold sensitivity

Output folder: sepsis_ml/synthetic/hybrid/

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/hybrid/hybrid_catboost.py

Prerequisites:
  - make_hybrid_dataset.py must have been run first
  - model_datasets/synthetic/C_hybrid_train.csv must exist
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

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/hybrid/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

MODEL_DATA_DIR    = PROJECT_ROOT / "model_datasets"
HYBRID_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "C_hybrid_train.csv"
REAL_TEST_FILE    = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Real-trained Phase 2 model (comparison baseline)
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
# 1.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED          = 42
CV_FOLDS             = 5
TARGET_COL           = "sepsis_label"
TARGET_SENSITIVITY   = 0.90
BOOTSTRAP_ITERATIONS = 1000
OPTUNA_TRIALS        = 5          # Quick — hyperparams already well-known

np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "hybrid_catboost.log"
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
# 4.  METRIC HELPERS
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
# 5.  DELONG TEST
# ══════════════════════════════════════════════════════════════════════════════

def delong_test(y_true, y_prob_1, y_prob_2):
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

    order         = np.argsort(-y_true)
    label_1_count = int(y_true.sum())

    predictions_sorted = np.vstack([y_prob_1[order], y_prob_2[order]])
    aucs, delongcov    = fastDeLong(predictions_sorted, label_1_count)

    diff    = aucs[0] - aucs[1]
    se      = np.sqrt(delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1])
    z_stat  = diff / se
    p_value = 2 * stats.norm.sf(abs(z_stat))

    return float(aucs[0]), float(aucs[1]), float(z_stat), float(p_value)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  OPTUNA OBJECTIVE  (identical search space to Phase 2 / Experiment B)
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial, X_df: pd.DataFrame, y: np.ndarray,
              task_type: str, devices) -> float:
    from catboost import CatBoostClassifier

    params = {
        "iterations"          : trial.suggest_int("iterations", 200, 1500),
        "learning_rate"       : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth"               : trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg"         : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature" : trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength"     : trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count"        : trial.suggest_int("border_count", 32, 255),
        "class_weights"       : [1.0, trial.suggest_float("class_weight_pos", 1.5, 5.0)],
        "eval_metric"         : "Logloss",
        "custom_metric"       : ["AUC"],
        "random_seed"         : RANDOM_SEED,
        "verbose"             : 0,
        "early_stopping_rounds": 50,
        "task_type"           : task_type,
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
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc_scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(auprc_scores))

# ══════════════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization_history(study):
    values      = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(values)), values,
               alpha=0.6, s=60, color="#3498db", label="Trial AUPRC", zorder=3)
    ax.plot(range(len(best_so_far)), best_so_far,
            color="#e74c3c", lw=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("CV AUPRC")
    ax.set_title(
        "Optuna optimization history\n"
        "Hybrid-trained CatBoost — 5 trials",
        fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_optuna_optimization_history.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_three_way_roc_pr(y_test, y_prob_hybrid, y_prob_synth, y_prob_real,
                          metrics_hybrid, metrics_synth, metrics_real):
    """
    Three-way comparison: Hybrid vs Synthetic-only vs Real-only.
    This is the key figure for the paper.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    curves = [
        (y_prob_hybrid, f"Hybrid (Real+Synth)  AUROC={metrics_hybrid['auroc']:.4f}", "#2ecc71", 2.5),
        (y_prob_real,   f"Real-only (Phase 2)  AUROC={metrics_real['auroc']:.4f}",   "#3498db", 1.8),
        (y_prob_synth,  f"Synth-only (Exp B)   AUROC={metrics_synth['auroc']:.4f}",  "#e74c3c", 1.2),
    ]

    # ROC
    for y_prob, label, color, lw in curves:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(
        "ROC — Hybrid vs Synthetic-only vs Real-only\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # PR
    pr_curves = [
        (y_prob_hybrid, f"Hybrid (Real+Synth)  AUPRC={metrics_hybrid['auprc']:.4f}", "#2ecc71", 2.5),
        (y_prob_real,   f"Real-only (Phase 2)  AUPRC={metrics_real['auprc']:.4f}",   "#3498db", 1.8),
        (y_prob_synth,  f"Synth-only (Exp B)   AUPRC={metrics_synth['auprc']:.4f}",  "#e74c3c", 1.2),
    ]
    for y_prob, label, color, lw in pr_curves:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=lw, label=label)
    prev = np.array(y_test).mean()
    ax2.axhline(prev, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Prevalence ({prev:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(
        "Precision-Recall — Hybrid vs Synthetic-only vs Real-only\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_three_way_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, metrics,
                          title="Hybrid-Trained Tuned CatBoost"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Greens")
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
    out = FIGURES_DIR / "hybrid_cm.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_threshold_sensitivity(y_test, y_prob):
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
    df.to_csv(RESULTS_DIR / "hybrid_threshold_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Threshold sensitivity analysis — Hybrid-Trained Tuned CatBoost\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontsize=11, fontweight="bold"
    )

    thresh_vals = df["threshold"].values

    axes[0].plot(thresh_vals, df["sensitivity"].values, "o-", color="#e74c3c",
                 lw=2, label="Sensitivity")
    axes[0].plot(thresh_vals, df["specificity"].values, "s-", color="#3498db",
                 lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(thresh_vals, df["ppv"].values, "o-", color="#2ecc71",
                 lw=2, label="PPV (Precision)")
    axes[1].plot(thresh_vals, df["npv"].values, "s-", color="#9b59b6",
                 lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(thresh_vals, df["f1"].values, "o-", color="#f39c12", lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "hybrid_threshold_sensitivity_analysis.png"
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
    log.info("EXPERIMENT C — CATBOOST TUNING ON HYBRID DATA (Real + Synthetic)")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Trials: {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}  |  Metric: AUPRC")
    log.info("=" * 65)

    # ── 8.1  Device ───────────────────────────────────────────────────────────
    log.info("\nDevice detection:")
    task_type, devices = detect_catboost_device()

    # ── 8.2  Load data ────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")

    if not HYBRID_TRAIN_FILE.exists():
        log.error(f"Hybrid training file not found: {HYBRID_TRAIN_FILE}")
        log.error("Run make_hybrid_dataset.py first.")
        sys.exit(1)

    hybrid_train = pd.read_csv(HYBRID_TRAIN_FILE)
    real_test    = pd.read_csv(REAL_TEST_FILE)
    feature_cols = [c for c in hybrid_train.columns if c != TARGET_COL]

    X_hybrid_df = hybrid_train[feature_cols]
    y_hybrid    = hybrid_train[TARGET_COL].values
    X_test_df   = real_test[feature_cols]
    y_test      = real_test[TARGET_COL].values

    n_real  = len(pd.read_csv(MODEL_DATA_DIR / "B_train_model_ready.csv"))
    n_synth = len(hybrid_train) - n_real

    log.info(f"  Hybrid train  : {hybrid_train.shape} | "
             f"Sepsis: {y_hybrid.mean():.1%} "
             f"({y_hybrid.sum()} / {len(y_hybrid)})")
    log.info(f"    ↳ Real rows   : {n_real:,}")
    log.info(f"    ↳ Synth rows  : {n_synth:,}")
    log.info(f"  Real test     : {real_test.shape}  | "
             f"Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features      : {len(feature_cols)}")
    log.info("")
    log.info("  IMPORTANT: CV runs on HYBRID data; eval is on REAL test set only.")

    # ── 8.3  Load baselines for comparison ────────────────────────────────────
    log.info("\nLoading comparison baselines...")
    from catboost import CatBoostClassifier

    # Real-trained (Phase 2)
    if REAL_CB_MODEL_PATH.exists():
        real_cb_model = CatBoostClassifier()
        real_cb_model.load_model(str(REAL_CB_MODEL_PATH))
        y_prob_real   = real_cb_model.predict_proba(X_test_df)[:, 1]
        metrics_real  = compute_metrics(y_test, y_prob_real)
        log.info(f"  Real-trained  : AUROC={metrics_real['auroc']:.4f}  "
                 f"AUPRC={metrics_real['auprc']:.4f}")
    else:
        log.warning("  Real-trained model not found — comparison skipped.")
        y_prob_real  = None
        metrics_real = None

    # Synth-trained (Experiment B)
    synth_model_path = (SYNTHETIC_ML / "catboost" / "models"
                        / "synthetic_catboost_tuned.cbm")
    if synth_model_path.exists():
        synth_cb_model = CatBoostClassifier()
        synth_cb_model.load_model(str(synth_model_path))
        y_prob_synth   = synth_cb_model.predict_proba(X_test_df)[:, 1]
        metrics_synth  = compute_metrics(y_test, y_prob_synth)
        log.info(f"  Synth-trained : AUROC={metrics_synth['auroc']:.4f}  "
                 f"AUPRC={metrics_synth['auprc']:.4f}")
    else:
        log.warning("  Synthetic-trained model not found — three-way plot skipped.")
        y_prob_synth  = None
        metrics_synth = None

    # ── 8.4  Optuna study ─────────────────────────────────────────────────────
    log.info(f"\nStarting Optuna study ({OPTUNA_TRIALS} trials)...")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner     = optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=3
        ),
        study_name = "hybrid_catboost_auprc_optuna",
    )

    best_so_far = 0.0

    def callback(study, trial):
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            log.info(f"  Trial {trial.number:>2} NEW BEST: "
                     f"AUPRC={trial.value:.4f}  params={trial.params}")
        else:
            val_str = f"{trial.value:.4f}" if trial.value is not None else "pruned"
            log.info(f"  Trial {trial.number:>2}: AUPRC={val_str}  "
                     f"(best: {best_so_far:.4f})")

    study.optimize(
        lambda trial: objective(trial, X_hybrid_df, y_hybrid, task_type, devices),
        n_trials          = OPTUNA_TRIALS,
        callbacks         = [callback],
        show_progress_bar = False,
    )

    log.info(f"\n  Optuna complete.")
    log.info(f"  Best CV AUPRC : {study.best_value:.4f}")
    log.info(f"  Best params   : {study.best_params}")

    # ── 8.5  Save study ───────────────────────────────────────────────────────
    with open(RESULTS_DIR / "hybrid_optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    with open(RESULTS_DIR / "hybrid_best_params.json", "w") as f:
        json.dump({
            "best_cv_auprc": study.best_value,
            "best_params"  : study.best_params,
            "n_trials"     : OPTUNA_TRIALS,
            "timestamp"    : datetime.now().isoformat(timespec="seconds"),
            "note"         : "5-trial Optuna on hybrid (real+synthetic) data",
        }, f, indent=2)

    study.trials_dataframe().to_csv(
        RESULTS_DIR / "hybrid_tuning_trials.csv", index=False
    )

    # ── 8.6  Train final model on full hybrid training set ────────────────────
    log.info(f"\nTraining final tuned CatBoost on full hybrid training set...")
    best = study.best_params

    tuned_params = dict(
        iterations            = best["iterations"],
        learning_rate         = best["learning_rate"],
        depth                 = best["depth"],
        l2_leaf_reg           = best["l2_leaf_reg"],
        bagging_temperature   = best["bagging_temperature"],
        random_strength       = best["random_strength"],
        border_count          = best["border_count"],
        class_weights         = [1.0, best["class_weight_pos"]],
        eval_metric           = "Logloss",
        custom_metric         = ["AUC"],
        random_seed           = RANDOM_SEED,
        verbose               = False,
        allow_writing_files   = False,
        early_stopping_rounds = 50,
        task_type             = task_type,
    )
    if devices:
        tuned_params["devices"] = devices

    tuned_model = CatBoostClassifier(**tuned_params)
    tuned_model.fit(
        X_hybrid_df, y_hybrid,
        eval_set=(X_test_df, y_test),
    )

    # ── 8.7  Evaluate on real test set ────────────────────────────────────────
    log.info(f"\nEvaluating on real held-out test set (n=633)...")
    y_prob_hybrid  = tuned_model.predict_proba(X_test_df)[:, 1]
    metrics_hybrid = compute_metrics_with_ci(y_test, y_prob_hybrid)

    log.info(f"\n  {'=' * 55}")
    log.info(f"  HYBRID-TRAINED TUNED CATBOOST — TEST SET RESULTS")
    log.info(f"  {'=' * 55}")
    log.info(f"  AUROC : {metrics_hybrid['auroc']:.4f} "
             f"[{metrics_hybrid['auroc_ci_low']:.4f}–{metrics_hybrid['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_hybrid['auprc']:.4f} "
             f"[{metrics_hybrid['auprc_ci_low']:.4f}–{metrics_hybrid['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_hybrid['sensitivity']:.4f}  "
             f"Spec: {metrics_hybrid['specificity']:.4f}  "
             f"F1: {metrics_hybrid['f1']:.4f}")
    log.info(f"  PPV   : {metrics_hybrid['ppv']:.4f}  "
             f"NPV: {metrics_hybrid['npv']:.4f}  "
             f"Brier: {metrics_hybrid['brier_score']:.4f}")
    log.info(f"  TP={metrics_hybrid['tp']}  FP={metrics_hybrid['fp']}  "
             f"TN={metrics_hybrid['tn']}  FN={metrics_hybrid['fn']}")

    # ── 8.8  Comparisons + DeLong ─────────────────────────────────────────────
    if metrics_real:
        auroc_delta = metrics_hybrid["auroc"] - metrics_real["auroc"]
        auprc_delta = metrics_hybrid["auprc"] - metrics_real["auprc"]
        log.info(f"\n  Comparison vs Real-Trained (Phase 2):")
        log.info(f"    AUROC: {metrics_real['auroc']:.4f} → "
                 f"{metrics_hybrid['auroc']:.4f} (Δ={auroc_delta:+.4f})")
        log.info(f"    AUPRC: {metrics_real['auprc']:.4f} → "
                 f"{metrics_hybrid['auprc']:.4f} (Δ={auprc_delta:+.4f})")

        log.info(f"\n  DeLong test (hybrid vs real-trained):")
        a1, a2, z, p = delong_test(y_test, y_prob_hybrid, y_prob_real)
        log.info(f"    Hybrid AUROC     : {a1:.4f}")
        log.info(f"    Real AUROC       : {a2:.4f}")
        log.info(f"    Z-statistic      : {z:.4f}")
        log.info(f"    P-value          : {p:.4f}")
        log.info(f"    Significant      : "
                 f"{'YES (p<0.05)' if p < 0.05 else 'NO (p>=0.05)'}")

        metrics_hybrid.update({
            "delong_vs_real_z"         : float(z),
            "delong_vs_real_p"         : float(p),
            "delong_significant"       : bool(p < 0.05),
            "auroc_delta_vs_real"      : float(auroc_delta),
            "auprc_delta_vs_real"      : float(auprc_delta),
            "real_trained_auroc_phase2": float(metrics_real["auroc"]),
            "real_trained_auprc_phase2": float(metrics_real["auprc"]),
        })

    if metrics_synth:
        auprc_delta_vs_synth = metrics_hybrid["auprc"] - metrics_synth["auprc"]
        log.info(f"\n  Comparison vs Synth-Trained (Exp B):")
        log.info(f"    AUPRC: {metrics_synth['auprc']:.4f} → "
                 f"{metrics_hybrid['auprc']:.4f} "
                 f"(Δ={auprc_delta_vs_synth:+.4f})")
        metrics_hybrid["auprc_delta_vs_synth"] = float(auprc_delta_vs_synth)

    # ── 8.9  Save model + probs ───────────────────────────────────────────────
    model_path = MODELS_DIR / "hybrid_catboost_tuned.cbm"
    tuned_model.save_model(str(model_path))
    np.save(RESULTS_DIR / "hybrid_catboost_test_probs.npy", y_prob_hybrid)

    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"           : "CatBoost — Hybrid-Trained Optuna Tuned",
        "experiment"      : "C",
        "timestamp"       : datetime.now().isoformat(timespec="seconds"),
        "training_data"   : "Hybrid: Real + Synthetic (CTGAN)",
        "test_data"       : "Real held-out (Option B, n=633)",
        "n_hybrid_train"  : int(len(y_hybrid)),
        "n_real_rows"     : int(n_real),
        "n_synth_rows"    : int(n_synth),
        "n_real_test"     : int(len(y_test)),
        "n_features"      : len(feature_cols),
        "cv_folds"        : CV_FOLDS,
        "optuna_trials"   : OPTUNA_TRIALS,
        "best_cv_auprc"   : float(study.best_value),
        "best_params"     : study.best_params,
        "test_metrics"    : metrics_hybrid,
        "task_type"       : task_type,
        "runtime_minutes" : round(runtime, 2),
    }
    with open(RESULTS_DIR / "hybrid_catboost_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Model saved   → {model_path}")
    log.info(f"  Metrics saved → {RESULTS_DIR / 'hybrid_catboost_metrics.json'}")

    # ── 8.10  Plots ───────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    plot_optimization_history(study)

    if y_prob_real is not None and y_prob_synth is not None:
        plot_three_way_roc_pr(y_test, y_prob_hybrid, y_prob_synth, y_prob_real,
                              metrics_hybrid, metrics_synth, metrics_real)
    elif y_prob_real is not None:
        # Two-way fallback if synth model missing
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for y_prob, label, color, lw, ax_roc, ax_pr in [
            (y_prob_hybrid, f"Hybrid AUROC={metrics_hybrid['auroc']:.4f}", "#2ecc71", 2.5, ax1, ax2),
            (y_prob_real,   f"Real   AUROC={metrics_real['auroc']:.4f}",   "#3498db", 1.5, ax1, ax2),
        ]:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax_roc.plot(fpr, tpr, color=color, lw=lw, label=label)
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ax_pr.plot(rec, prec, color=color, lw=lw, label=label)
        ax1.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC", fontweight="bold")
        ax1.legend(fontsize=9); ax1.spines[["top","right"]].set_visible(False)
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR", fontweight="bold")
        ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        out = FIGURES_DIR / "hybrid_two_way_roc_pr.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.close()

    y_pred_hybrid = (y_prob_hybrid >= metrics_hybrid["threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred_hybrid, metrics_hybrid)
    plot_threshold_sensitivity(y_test, y_prob_hybrid)

    # ── 8.11  Summary ─────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 65,
        "EXPERIMENT C — CATBOOST ON HYBRID DATA (Real + Synthetic)",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "SETUP",
        f"  Real train rows: {n_real:,}",
        f"  Synth train rows: {n_synth:,}",
        f"  Total hybrid   : {len(y_hybrid):,}",
        f"  Optuna trials  : {OPTUNA_TRIALS}",
        f"  CV folds       : {CV_FOLDS}",
        f"  Optimising     : AUPRC (5-fold CV mean)",
        "",
        "OPTUNA RESULT",
        f"  Best CV AUPRC  : {study.best_value:.4f}",
        f"  Best params    : {study.best_params}",
        "",
        "TEST SET RESULTS (on real patients)",
        f"  AUROC : {metrics_hybrid['auroc']:.4f} "
        f"[{metrics_hybrid['auroc_ci_low']:.4f}–{metrics_hybrid['auroc_ci_high']:.4f}]",
        f"  AUPRC : {metrics_hybrid['auprc']:.4f} "
        f"[{metrics_hybrid['auprc_ci_low']:.4f}–{metrics_hybrid['auprc_ci_high']:.4f}]",
        f"  Brier : {metrics_hybrid['brier_score']:.4f}",
        f"  Sens  : {metrics_hybrid['sensitivity']:.4f}",
        f"  Spec  : {metrics_hybrid['specificity']:.4f}",
        f"  PPV   : {metrics_hybrid['ppv']:.4f}",
        f"  NPV   : {metrics_hybrid['npv']:.4f}",
        f"  F1    : {metrics_hybrid['f1']:.4f}",
    ]

    if metrics_real:
        summary_lines += [
            "",
            "COMPARISON SUMMARY",
            f"  Exp A (Real only)   AUPRC: {metrics_real['auprc']:.4f}",
            f"  Exp B (Synth only)  AUPRC: {metrics_synth['auprc']:.4f}" if metrics_synth else "",
            f"  Exp C (Hybrid)      AUPRC: {metrics_hybrid['auprc']:.4f}",
            f"  Δ Hybrid vs Real         : {metrics_hybrid.get('auprc_delta_vs_real', 'N/A'):+.4f}",
            f"  DeLong p (vs Real)       : {metrics_hybrid.get('delong_vs_real_p', 'N/A')}",
        ]

    summary_lines += ["", f"Runtime : {runtime:.1f} min", "=" * 65]

    with open(OUTPUTS_DIR / "hybrid_catboost_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    # ── 8.12  Final log ───────────────────────────────────────────────────────
    log.info(f"\n{'=' * 65}")
    log.info("EXPERIMENT C — HYBRID CATBOOST COMPLETE")
    log.info(f"{'=' * 65}")
    log.info(f"  Test AUROC : {metrics_hybrid['auroc']:.4f} "
             f"[{metrics_hybrid['auroc_ci_low']:.4f}–{metrics_hybrid['auroc_ci_high']:.4f}]")
    log.info(f"  Test AUPRC : {metrics_hybrid['auprc']:.4f} "
             f"[{metrics_hybrid['auprc_ci_low']:.4f}–{metrics_hybrid['auprc_ci_high']:.4f}]")
    log.info(f"  Sensitivity: {metrics_hybrid['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics_hybrid['specificity']:.4f}")
    if metrics_real:
        log.info(f"  Δ AUPRC vs Real: {metrics_hybrid.get('auprc_delta_vs_real', 'N/A'):+.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")


if __name__ == "__main__":
    main()
