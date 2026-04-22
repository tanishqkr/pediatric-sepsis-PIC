"""
weighted_hybrid_catboost.py
──────────────────────────────────────────────────────────────────────────────
Experiment D — Tuned CatBoost trained on Weighted Hybrid Data

Motivation:
  Experiment C (unweighted hybrid) was significantly worse than real-only
  (ΔAUPRC = -0.0452, DeLong p<0.0001). Root cause: 10k synthetic rows
  dominated 2.5k real rows by sheer volume (80% synthetic influence).

  This experiment rebalances influence to 50:50 by assigning:
    real samples      -> sample_weight = 4.5
    synthetic samples -> sample_weight = 1.0

  Effective influence:
    Real    : 2,530 × 4.5 ≈ 11,385  (~53% of total influence)
    Synthetic: 10,000 × 1.0 = 10,000  (~47% of total influence)

  Research question:
    Does equalising real vs synthetic influence close the gap vs real-only,
    or does the quality difference persist regardless of weighting?

What is identical to hybrid_catboost.py (Exp C):
  - Same C_hybrid_train.csv (same concatenated data)
  - Same hyperparameter search space
  - Same 5 Optuna trials, 5-fold CV, AUPRC objective
  - Same evaluation on real held-out test set
  - Same plots: optimization history, three-way ROC/PR, CM, threshold analysis
  - Full bootstrap CIs, DeLong test vs real-trained and vs unweighted hybrid

What differs:
  - sample_weight vector passed to model.fit() in both CV and final training
  - Output folder: sepsis_ml/synthetic/whybrid/

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/whybrid/weighted_hybrid_catboost.py

Prerequisites:
  - make_hybrid_dataset.py must have been run (C_hybrid_train.csv must exist)
  - Exp B model (synthetic_catboost_tuned.cbm) for three-way plot (optional)
  - Exp C model (hybrid_catboost_tuned.cbm) for four-way comparison (optional)
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

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/whybrid/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

MODEL_DATA_DIR    = PROJECT_ROOT / "model_datasets"
HYBRID_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "C_hybrid_train.csv"
REAL_TRAIN_FILE   = MODEL_DATA_DIR / "B_train_model_ready.csv"
REAL_TEST_FILE    = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Comparison baselines
REAL_CB_MODEL_PATH  = SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm"
SYNTH_CB_MODEL_PATH = SYNTHETIC_ML / "catboost" / "models" / "synthetic_catboost_tuned.cbm"
HYBRID_CB_MODEL_PATH= SYNTHETIC_ML / "hybrid" / "models" / "hybrid_catboost_tuned.cbm"

# Output folders — all created here, no manual mkdir needed
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
OPTUNA_TRIALS        = 10

# 50:50 influence balance
# Real: 2530 × 4.5 ≈ 11,385  |  Synthetic: 10,000 × 1.0 = 10,000
REAL_SAMPLE_WEIGHT  = 4
SYNTH_SAMPLE_WEIGHT = 1.0

np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "weighted_hybrid_catboost.log"
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
# 4.  SAMPLE WEIGHT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sample_weights(n_total: int, n_real: int) -> np.ndarray:
    """
    First n_real rows = real patients  -> weight REAL_SAMPLE_WEIGHT
    Remaining rows    = synthetic      -> weight SYNTH_SAMPLE_WEIGHT

    C_hybrid_train.csv is built by make_hybrid_dataset.py which concatenates
    real FIRST, then synthetic — so row order is guaranteed.
    """
    weights = np.full(n_total, SYNTH_SAMPLE_WEIGHT, dtype=float)
    weights[:n_real] = REAL_SAMPLE_WEIGHT
    return weights

# ══════════════════════════════════════════════════════════════════════════════
# 5.  METRIC HELPERS
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
    return (float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


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
# 6.  DELONG TEST
# ══════════════════════════════════════════════════════════════════════════════

def delong_test(y_true, y_prob_1, y_prob_2):
    def compute_midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N, dtype=float); T2[J] = T + 1
        return T2

    def fastDeLong(pst, label_1_count):
        m = label_1_count; n = pst.shape[1] - m
        k = pst.shape[0]
        tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
        for r in range(k):
            tx[r] = compute_midrank(pst[r, :m])
            ty[r] = compute_midrank(pst[r, m:])
            tz[r] = compute_midrank(pst[r])
        aucs = (tz[:, :m].sum(1) - tx.sum(1)) / (m * n)
        v01  = (tz[:, :m] - tx) / n
        v10  = 1. - (tz[:, m:] - ty) / m
        delongcov = np.cov(v01) / m + np.cov(v10) / n
        return aucs, delongcov

    y_true = np.array(y_true); y_prob_1 = np.array(y_prob_1); y_prob_2 = np.array(y_prob_2)
    order  = np.argsort(-y_true)
    aucs, delongcov = fastDeLong(
        np.vstack([y_prob_1[order], y_prob_2[order]]), int(y_true.sum())
    )
    diff    = aucs[0] - aucs[1]
    se      = np.sqrt(delongcov[0,0] + delongcov[1,1] - 2*delongcov[0,1])
    z_stat  = diff / se
    p_value = 2 * stats.norm.sf(abs(z_stat))
    return float(aucs[0]), float(aucs[1]), float(z_stat), float(p_value)

# ══════════════════════════════════════════════════════════════════════════════
# 7.  OPTUNA OBJECTIVE  — sample_weight passed in CV folds
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial, X_df: pd.DataFrame, y: np.ndarray,
              sample_weights: np.ndarray, task_type: str, devices) -> float:
    from catboost import CatBoostClassifier, Pool

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
        w_tr  = sample_weights[train_idx]   # weights only on training fold
        # val fold is unweighted — we want unbiased AUPRC estimate

        train_pool = Pool(X_tr, y_tr, weight=w_tr)
        val_pool   = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc_scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(auprc_scores))

# ══════════════════════════════════════════════════════════════════════════════
# 8.  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization_history(study):
    values      = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i + 1]) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(len(values)), values,
               alpha=0.6, s=60, color="#9b59b6", label="Trial AUPRC", zorder=3)
    ax.plot(range(len(best_so_far)), best_so_far,
            color="#e74c3c", lw=2, label="Best so far")
    ax.set_xlabel("Trial number"); ax.set_ylabel("CV AUPRC")
    ax.set_title(
        "Optuna optimization history\n"
        f"Weighted Hybrid CatBoost — {len(values)} trials  "
        f"(real weight={REAL_SAMPLE_WEIGHT})",
        fontweight="bold"
    )
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIGURES_DIR / "whybrid_optuna_optimization_history.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close(); log.info(f"  Saved -> {out}")


def plot_four_way_roc_pr(y_test,
                         y_prob_whybrid, y_prob_hybrid,
                         y_prob_real, y_prob_synth,
                         m_whybrid, m_hybrid, m_real, m_synth):
    """
    Four-way comparison:
      Weighted Hybrid (D) vs Unweighted Hybrid (C) vs Real-only (A) vs Synth-only (B)
    This is the definitive synthetic data story figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    curves = [
        (y_prob_real,     f"A: Real-only       AUROC={m_real['auroc']:.4f}",     "#3498db", 2.0),
        (y_prob_whybrid,  f"D: Weighted Hybrid AUROC={m_whybrid['auroc']:.4f}",  "#9b59b6", 2.5),
        (y_prob_hybrid,   f"C: Hybrid          AUROC={m_hybrid['auroc']:.4f}",   "#2ecc71", 1.5),
        (y_prob_synth,    f"B: Synth-only      AUROC={m_synth['auroc']:.4f}",    "#e74c3c", 1.2),
    ]

    for y_prob, label, color, lw in curves:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, label=label)
    ax1.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title(
        "ROC — A: Real-only vs B: Synth vs C: Hybrid vs D: Weighted Hybrid\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.spines[["top","right"]].set_visible(False)

    pr_curves = [
        (y_prob_real,    f"A: Real-only       AUPRC={m_real['auprc']:.4f}",    "#3498db", 2.0),
        (y_prob_whybrid, f"D: Weighted Hybrid AUPRC={m_whybrid['auprc']:.4f}", "#9b59b6", 2.5),
        (y_prob_hybrid,  f"C: Hybrid          AUPRC={m_hybrid['auprc']:.4f}",  "#2ecc71", 1.5),
        (y_prob_synth,   f"B: Synth-only      AUPRC={m_synth['auprc']:.4f}",   "#e74c3c", 1.2),
    ]
    for y_prob, label, color, lw in pr_curves:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=lw, label=label)
    prev = np.array(y_test).mean()
    ax2.axhline(prev, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Prevalence ({prev:.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(
        "PR — A: Real-only vs B: Synth vs C: Hybrid vs D: Weighted Hybrid\n"
        "Evaluated on Real Test Set (Option B, n=633)",
        fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "whybrid_four_way_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close(); log.info(f"  Saved -> {out}")


def plot_confusion_matrix(y_true, y_pred, metrics):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"])
    ax.set_yticklabels(["Non-sepsis","Sepsis"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Weighted Hybrid-Trained CatBoost\n"
        f"Sens={metrics['sensitivity']:.4f}  "
        f"Spec={metrics['specificity']:.4f}  "
        f"Threshold={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    out = FIGURES_DIR / "whybrid_cm.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close(); log.info(f"  Saved -> {out}")


def plot_threshold_sensitivity(y_test, y_prob):
    targets = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    rows = []
    for t in targets:
        thresh = find_threshold_at_sensitivity(y_test, y_prob, t)
        m = compute_metrics(y_test, y_prob, thresh)
        rows.append({
            "target_sensitivity": t,
            "threshold": round(thresh, 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "ppv": round(m["ppv"], 4), "npv": round(m["npv"], 4),
            "f1": round(m["f1"], 4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "whybrid_threshold_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Threshold sensitivity — Weighted Hybrid CatBoost\n"
        f"(real weight={REAL_SAMPLE_WEIGHT})  Real Test Set (n=633)",
        fontsize=11, fontweight="bold"
    )
    tv = df["threshold"].values
    axes[0].plot(tv, df["sensitivity"].values, "o-", color="#e74c3c", lw=2, label="Sensitivity")
    axes[0].plot(tv, df["specificity"].values, "s-", color="#3498db", lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(tv, df["ppv"].values, "o-", color="#2ecc71", lw=2, label="PPV")
    axes[1].plot(tv, df["npv"].values, "s-", color="#9b59b6", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)

    axes[2].plot(tv, df["f1"].values, "o-", color="#f39c12", lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIGURES_DIR / "whybrid_threshold_sensitivity_analysis.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close(); log.info(f"  Saved -> {out}")

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
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()

    log.info("=" * 68)
    log.info("EXPERIMENT D — WEIGHTED HYBRID CATBOOST  (real weight=4)")
    log.info(f"Started : {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Trials  : {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}  |  Metric: AUPRC")
    log.info(f"Weights : real={REAL_SAMPLE_WEIGHT}  synthetic={SYNTH_SAMPLE_WEIGHT}")
    log.info("=" * 68)

    # ── 9.1  Device ───────────────────────────────────────────────────────────
    log.info("\nDevice detection:")
    task_type, devices = detect_catboost_device()

    # ── 9.2  Load data ────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")

    if not HYBRID_TRAIN_FILE.exists():
        log.error(f"Hybrid training file not found: {HYBRID_TRAIN_FILE}")
        log.error("Run make_hybrid_dataset.py first.")
        sys.exit(1)

    hybrid_train = pd.read_csv(HYBRID_TRAIN_FILE)
    real_test    = pd.read_csv(REAL_TEST_FILE)
    n_real       = len(pd.read_csv(REAL_TRAIN_FILE))   # exact real row count
    n_synth      = len(hybrid_train) - n_real
    feature_cols = [c for c in hybrid_train.columns if c != TARGET_COL]

    X_hybrid_df = hybrid_train[feature_cols]
    y_hybrid    = hybrid_train[TARGET_COL].values
    X_test_df   = real_test[feature_cols]
    y_test      = real_test[TARGET_COL].values

    # Build sample weights
    sample_weights = build_sample_weights(len(hybrid_train), n_real)

    real_influence  = n_real  * REAL_SAMPLE_WEIGHT
    synth_influence = n_synth * SYNTH_SAMPLE_WEIGHT
    total_influence = real_influence + synth_influence

    log.info(f"  Hybrid train  : {hybrid_train.shape} | "
             f"Sepsis: {y_hybrid.mean():.1%}")
    log.info(f"    -> Real rows   : {n_real:,}  (weight={REAL_SAMPLE_WEIGHT}  "
             f"influence={real_influence:,.0f}  "
             f"{real_influence/total_influence:.1%} of total)")
    log.info(f"    -> Synth rows  : {n_synth:,}  (weight={SYNTH_SAMPLE_WEIGHT}  "
             f"influence={synth_influence:,.0f}  "
             f"{synth_influence/total_influence:.1%} of total)")
    log.info(f"  Real test     : {real_test.shape}  | Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features      : {len(feature_cols)}")

    # ── 9.3  Load comparison baselines ───────────────────────────────────────
    log.info("\nLoading comparison baselines...")
    from catboost import CatBoostClassifier, Pool

    def load_cb(path, name):
        if path.exists():
            m = CatBoostClassifier(); m.load_model(str(path))
            probs   = m.predict_proba(X_test_df)[:, 1]
            metrics = compute_metrics(y_test, probs)
            log.info(f"  {name:20s}: AUROC={metrics['auroc']:.4f}  "
                     f"AUPRC={metrics['auprc']:.4f}")
            return probs, metrics
        else:
            log.warning(f"  {name:20s}: not found at {path}")
            return None, None

    y_prob_real,   m_real   = load_cb(REAL_CB_MODEL_PATH,   "Real-trained (A)")
    y_prob_synth,  m_synth  = load_cb(SYNTH_CB_MODEL_PATH,  "Synth-trained (B)")
    y_prob_hybrid, m_hybrid = load_cb(HYBRID_CB_MODEL_PATH, "Hybrid (C)")

    # ── 9.4  Optuna study ─────────────────────────────────────────────────────
    log.info(f"\nStarting Optuna study ({OPTUNA_TRIALS} trials)...")
    log.info("Note: CV val folds are unweighted for unbiased AUPRC estimates.\n")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        study_name = "whybrid_catboost_auprc_optuna",
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
        lambda trial: objective(
            trial, X_hybrid_df, y_hybrid, sample_weights, task_type, devices
        ),
        n_trials=OPTUNA_TRIALS, callbacks=[callback], show_progress_bar=False,
    )

    log.info(f"\n  Optuna complete.")
    log.info(f"  Best CV AUPRC : {study.best_value:.4f}")
    log.info(f"  Best params   : {study.best_params}")

    # ── 9.5  Save study ───────────────────────────────────────────────────────
    with open(RESULTS_DIR / "whybrid_optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)
    with open(RESULTS_DIR / "whybrid_best_params.json", "w") as f:
        json.dump({
            "best_cv_auprc"     : study.best_value,
            "best_params"       : study.best_params,
            "n_trials"          : OPTUNA_TRIALS,
            "real_sample_weight": REAL_SAMPLE_WEIGHT,
            "synth_sample_weight": SYNTH_SAMPLE_WEIGHT,
            "timestamp"         : datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    study.trials_dataframe().to_csv(
        RESULTS_DIR / "whybrid_tuning_trials.csv", index=False
    )

    # ── 9.6  Train final model on full weighted hybrid set ────────────────────
    log.info(f"\nTraining final model on full weighted hybrid set...")
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

    train_pool = Pool(X_hybrid_df, y_hybrid, weight=sample_weights)
    eval_pool  = Pool(X_test_df, y_test)

    tuned_model = CatBoostClassifier(**tuned_params)
    tuned_model.fit(train_pool, eval_set=eval_pool)

    # ── 9.7  Evaluate ─────────────────────────────────────────────────────────
    log.info(f"\nEvaluating on real held-out test set (n=633)...")
    y_prob_whybrid  = tuned_model.predict_proba(X_test_df)[:, 1]
    metrics_whybrid = compute_metrics_with_ci(y_test, y_prob_whybrid)

    log.info(f"\n  {'=' * 58}")
    log.info(f"  WEIGHTED HYBRID CATBOOST — TEST SET RESULTS")
    log.info(f"  {'=' * 58}")
    log.info(f"  AUROC : {metrics_whybrid['auroc']:.4f} "
             f"[{metrics_whybrid['auroc_ci_low']:.4f}–{metrics_whybrid['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_whybrid['auprc']:.4f} "
             f"[{metrics_whybrid['auprc_ci_low']:.4f}–{metrics_whybrid['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_whybrid['sensitivity']:.4f}  "
             f"Spec: {metrics_whybrid['specificity']:.4f}  "
             f"F1: {metrics_whybrid['f1']:.4f}")
    log.info(f"  PPV   : {metrics_whybrid['ppv']:.4f}  "
             f"NPV: {metrics_whybrid['npv']:.4f}  "
             f"Brier: {metrics_whybrid['brier_score']:.4f}")
    log.info(f"  TP={metrics_whybrid['tp']}  FP={metrics_whybrid['fp']}  "
             f"TN={metrics_whybrid['tn']}  FN={metrics_whybrid['fn']}")

    # ── 9.8  DeLong comparisons ───────────────────────────────────────────────
    if y_prob_real is not None:
        log.info(f"\n  DeLong test (weighted hybrid vs real-trained):")
        a1, a2, z, p = delong_test(y_test, y_prob_whybrid, y_prob_real)
        log.info(f"    Weighted Hybrid AUROC : {a1:.4f}")
        log.info(f"    Real-trained AUROC    : {a2:.4f}")
        log.info(f"    Z-statistic           : {z:.4f}")
        log.info(f"    P-value               : {p:.4f}")
        log.info(f"    Significant           : "
                 f"{'YES (p<0.05)' if p < 0.05 else 'NO (p>=0.05)'}")
        metrics_whybrid.update({
            "delong_vs_real_z"         : float(z),
            "delong_vs_real_p"         : float(p),
            "delong_significant_vs_real": bool(p < 0.05),
            "auroc_delta_vs_real"      : float(a1 - a2),
            "auprc_delta_vs_real"      : float(metrics_whybrid["auprc"] - m_real["auprc"]),
        })

    if y_prob_hybrid is not None:
        log.info(f"\n  DeLong test (weighted hybrid vs unweighted hybrid):")
        a1, a2, z, p = delong_test(y_test, y_prob_whybrid, y_prob_hybrid)
        log.info(f"    Weighted Hybrid AUROC   : {a1:.4f}")
        log.info(f"    Unweighted Hybrid AUROC : {a2:.4f}")
        log.info(f"    Z-statistic             : {z:.4f}")
        log.info(f"    P-value                 : {p:.4f}")
        log.info(f"    Significant             : "
                 f"{'YES (p<0.05)' if p < 0.05 else 'NO (p>=0.05)'}")
        metrics_whybrid.update({
            "delong_vs_hybrid_z"            : float(z),
            "delong_vs_hybrid_p"            : float(p),
            "delong_significant_vs_hybrid"  : bool(p < 0.05),
            "auprc_delta_vs_hybrid"         : float(
                metrics_whybrid["auprc"] - m_hybrid["auprc"]
            ),
        })

    # ── 9.9  Save model + artifacts ───────────────────────────────────────────
    model_path = MODELS_DIR / "weighted_hybrid_catboost_tuned.cbm"
    tuned_model.save_model(str(model_path))
    np.save(RESULTS_DIR / "whybrid_catboost_test_probs.npy", y_prob_whybrid)

    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"              : "CatBoost — Weighted Hybrid (Exp D)",
        "experiment"         : "D",
        "timestamp"          : datetime.now().isoformat(timespec="seconds"),
        "training_data"      : "Weighted Hybrid: Real (w=4.5) + Synthetic (w=1.0)",
        "real_sample_weight" : REAL_SAMPLE_WEIGHT,
        "synth_sample_weight": SYNTH_SAMPLE_WEIGHT,
        "n_real_train"       : int(n_real),
        "n_synth_train"      : int(n_synth),
        "n_real_test"        : int(len(y_test)),
        "n_features"         : len(feature_cols),
        "optuna_trials"      : OPTUNA_TRIALS,
        "best_cv_auprc"      : float(study.best_value),
        "best_params"        : study.best_params,
        "test_metrics"       : metrics_whybrid,
        "runtime_minutes"    : round(runtime, 2),
    }
    with open(RESULTS_DIR / "whybrid_catboost_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Model saved   -> {model_path}")
    log.info(f"  Metrics saved -> {RESULTS_DIR / 'whybrid_catboost_metrics.json'}")

    # ── 9.10  Plots ───────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    plot_optimization_history(study)

    if all(v is not None for v in [y_prob_hybrid, y_prob_real, y_prob_synth]):
        plot_four_way_roc_pr(
            y_test,
            y_prob_whybrid, y_prob_hybrid, y_prob_real, y_prob_synth,
            metrics_whybrid, m_hybrid, m_real, m_synth,
        )
    elif y_prob_real is not None:
        log.warning("  One or more baselines missing — skipping four-way plot.")

    y_pred_whybrid = (y_prob_whybrid >= metrics_whybrid["threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred_whybrid, metrics_whybrid)
    plot_threshold_sensitivity(y_test, y_prob_whybrid)

    # ── 9.11  Summary ─────────────────────────────────────────────────────────
    summary_lines = [
        "=" * 68,
        "EXPERIMENT D — WEIGHTED HYBRID CATBOOST",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 68,
        "",
        "WEIGHTING",
        f"  Real rows  : {n_real:,} × {REAL_SAMPLE_WEIGHT} = "
        f"{n_real*REAL_SAMPLE_WEIGHT:,.0f} influence  "
        f"({n_real*REAL_SAMPLE_WEIGHT/total_influence:.1%})",
        f"  Synth rows : {n_synth:,} × {SYNTH_SAMPLE_WEIGHT} = "
        f"{n_synth*SYNTH_SAMPLE_WEIGHT:,.0f} influence  "
        f"({n_synth*SYNTH_SAMPLE_WEIGHT/total_influence:.1%})",
        "",
        "TEST SET RESULTS",
        f"  AUROC : {metrics_whybrid['auroc']:.4f} "
        f"[{metrics_whybrid['auroc_ci_low']:.4f}–{metrics_whybrid['auroc_ci_high']:.4f}]",
        f"  AUPRC : {metrics_whybrid['auprc']:.4f} "
        f"[{metrics_whybrid['auprc_ci_low']:.4f}–{metrics_whybrid['auprc_ci_high']:.4f}]",
        f"  Sens  : {metrics_whybrid['sensitivity']:.4f}",
        f"  Spec  : {metrics_whybrid['specificity']:.4f}",
        f"  F1    : {metrics_whybrid['f1']:.4f}",
        f"  Brier : {metrics_whybrid['brier_score']:.4f}",
        "",
        "EXPERIMENT COMPARISON (AUPRC on real test set)",
        f"  A: Real-only       : {m_real['auprc']:.4f}" if m_real else "",
        f"  B: Synth-only      : {m_synth['auprc']:.4f}" if m_synth else "",
        f"  C: Hybrid (unwt)   : {m_hybrid['auprc']:.4f}" if m_hybrid else "",
        f"  D: Weighted Hybrid : {metrics_whybrid['auprc']:.4f}",
        f"  Δ D vs A           : {metrics_whybrid.get('auprc_delta_vs_real', 'N/A')}",
        f"  Δ D vs C           : {metrics_whybrid.get('auprc_delta_vs_hybrid', 'N/A')}",
        "",
        f"Runtime : {runtime:.1f} min",
        "=" * 68,
    ]

    with open(OUTPUTS_DIR / "whybrid_catboost_summary.txt", "w") as f:
        f.write("\n".join(l for l in summary_lines if l is not None))

    # ── 9.12  Final log ───────────────────────────────────────────────────────
    log.info(f"\n{'=' * 68}")
    log.info("EXPERIMENT D — WEIGHTED HYBRID CATBOOST COMPLETE")
    log.info(f"{'=' * 68}")
    log.info(f"  Test AUROC : {metrics_whybrid['auroc']:.4f} "
             f"[{metrics_whybrid['auroc_ci_low']:.4f}–{metrics_whybrid['auroc_ci_high']:.4f}]")
    log.info(f"  Test AUPRC : {metrics_whybrid['auprc']:.4f} "
             f"[{metrics_whybrid['auprc_ci_low']:.4f}–{metrics_whybrid['auprc_ci_high']:.4f}]")
    if m_real:
        log.info(f"  Δ AUPRC vs Real-only  : "
                 f"{metrics_whybrid.get('auprc_delta_vs_real', 'N/A'):+.4f}")
    if m_hybrid:
        log.info(f"  Δ AUPRC vs Hybrid (C) : "
                 f"{metrics_whybrid.get('auprc_delta_vs_hybrid', 'N/A'):+.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")


if __name__ == "__main__":
    main()
