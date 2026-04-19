"""
synthetic_stacking.py
──────────────────────────────────────────────────────────────────────────────
Experiment B — Stacking Ensemble (CatBoost + FT-Transformer) trained on
100% Synthetic Data

Research Question:
  Can a stacking ensemble (CatBoost + FT-Transformer) trained ONLY on
  CTGAN-generated synthetic patients generalise to real held-out patients?
  How does it compare to the same ensemble trained on real data (Phase 7)?

Pipeline:
  1. Load synthetic training data  (model_datasets/synthetic/B_synthetic_train.csv)
  2. Load REAL test data           (model_datasets/B_test_model_ready.csv)
     → Test set is ALWAYS real. Never synthetic.
  3. Generate OOF predictions from CatBoost and FT-Transformer via 5-fold CV
     on the synthetic training set
  4. Train Logistic Regression meta-learner on synthetic OOF matrix
  5. Evaluate stacking ensemble on REAL held-out test set (n=633)
  6. DeLong tests:
       a) Synthetic stack vs Real stack (Phase 7, AUROC=0.9851)
       b) Synthetic stack vs Real CatBoost (Phase 2, AUROC=0.9868)
  7. Full metric suite + threshold analysis + figures

DEVICE SUPPORT:
  This script auto-detects and supports all three device types:
  CatBoost:       CUDA > CPU (MPS falls back to CPU — expected)
  FT-Transformer: CUDA > MPS > CPU

  ┌────────────────────────────────────────────────────────────────────────┐
  │ Device              │ CatBoost   │ FT-T (per fold) │ Total (5 folds) │
  ├─────────────────────┼────────────┼─────────────────┼─────────────────┤
  │ CUDA (NVIDIA GPU)   │  1–3 min   │  3–8 min        │  ~20–45 min     │
  │ MPS  (Apple Silicon)│  3–8 min*  │  5–12 min       │  ~35–65 min     │
  │ CPU  (Mac/Windows)  │  3–8 min   │  8–20 min       │  ~50–110 min    │
  └────────────────────────────────────────────────────────────────────────┘
  * CatBoost falls back to CPU even on MPS (CatBoost does not support MPS).
    FT-Transformer uses MPS on Apple Silicon for acceleration.

  Note: These are estimates for 10,000 synthetic training patients.
  Larger synthetic datasets will take proportionally longer.

FOLDER OUTPUTS (under sepsis_ml/synthetic/stacking/):
  models/   — synthetic_meta_learner.pkl, synthetic_catboost_fold.cbm,
              synthetic_fttransformer_weights/
  results/  — synthetic_oof_probs.csv, synthetic_test_probs.csv,
              synthetic_meta_coefficients.json, synthetic_stacking_metrics.json,
              synthetic_delong_results.json, synthetic_threshold_table.csv
  figures/  — synthetic_stacking_roc_pr.png, synthetic_stacking_calibration.png,
              synthetic_stacking_confusion_matrix.png,
              synthetic_stacking_threshold_sensitivity.png
  logs/     — synthetic_stacking.log
  outputs/  — synthetic_stacking_summary.txt

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/stacking/synthetic_stacking.py
"""

import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats

from catboost import CatBoostClassifier
from rtdl_revisiting_models import FTTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/stacking/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

# Input data
MODEL_DATA_DIR   = PROJECT_ROOT / "model_datasets"
SYNTH_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train.csv"
REAL_TEST_FILE   = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Phase 7 real-trained stacking artifacts (for baseline comparison)
PHASE7_DIR            = SEPSIS_ML / "phase7_stacking"
PHASE7_CB_PROBS       = SEPSIS_ML / "results" / "catboost_test_probs.npy"
PHASE7_TEST_PROBS_CSV = PHASE7_DIR / "results" / "phase7_test_probs.csv"

# FT-Transformer hyperparams from Phase 6 (reuse architecture)
FTT_PARAMS_PATH = SEPSIS_ML / "dl" / "models" / "run_phase6_fttransformer" / "best_params_final.json"

# Phase 7 real-trained ensemble test probs (for DeLong)
PHASE7_ENS_PROBS_PATH = PHASE7_DIR / "results" / "phase7_test_probs.csv"

# Output folders
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
LOGS_DIR    = SCRIPT_DIR / "logs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

FTT_WEIGHTS_DIR = MODELS_DIR / "synthetic_fttransformer_weights"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OUTPUTS_DIR,
          FTT_WEIGHTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "synthetic_stacking.log"
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
CV_FOLDS           = 5
TARGET_COL         = "sepsis_label"
TARGET_SENSITIVITY = 0.90

# Real-data baseline results (Phase 7) — for comparison reporting
REAL_STACK_AUROC  = 0.9851
REAL_CB_AUROC     = 0.9868

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 3.  DEVICE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_devices():
    """
    Detects best available device for CatBoost and FT-Transformer separately.
    Returns:
      cb_task_type  : "GPU" or "CPU"     (for CatBoostClassifier)
      cb_devices    : "0" or None        (for CatBoostClassifier)
      torch_device  : torch.device       (for FT-Transformer)
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  CUDA detected : {gpu_name} ({vram_gb:.1f} GB)")
        log.info(f"  CatBoost      : GPU")
        log.info(f"  FT-Transformer: CUDA")
        return "GPU", "0", torch.device("cuda")

    elif torch.backends.mps.is_available():
        log.info("  Apple Silicon MPS detected.")
        log.info("  CatBoost      : CPU (CatBoost does not support MPS — expected)")
        log.info("  FT-Transformer: MPS (Apple Silicon GPU — accelerated)")
        return "CPU", None, torch.device("mps")

    else:
        log.info("  Device: CPU only")
        log.info("  CatBoost      : CPU")
        log.info("  FT-Transformer: CPU")
        return "CPU", None, torch.device("cpu")

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
        ranks     = compute_midrank(np.concatenate([pos, neg]))
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
# 5.  FT-TRANSFORMER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_fttransformer(params, n_features, device):
    d_block = params["d_block"]
    n_heads = params.get("actual_attention_n_heads", params["attention_n_heads"])
    while d_block % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1:
            n_heads = 1
            break
    model = FTTransformer(
        n_cont_features         = n_features,
        cat_cardinalities       = None,
        d_out                   = 1,
        n_blocks                = params["n_blocks"],
        d_block                 = d_block,
        attention_n_heads       = n_heads,
        attention_dropout       = params["attention_dropout"],
        ffn_d_hidden_multiplier = params["ffn_d_hidden_multiplier"],
        ffn_dropout             = params["ffn_dropout"],
        residual_dropout        = params["residual_dropout"],
    ).to(device)
    return model


def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_epoch_ftt(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b, None).squeeze(-1)
        loss   = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probs_ftt(model, loader, device):
    model.eval()
    all_probs = []
    for X_b, _ in loader:
        X_b    = X_b.to(device)
        logits = model(X_b, None).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def train_ftt_fold(X_tr_s, y_tr, X_val_s, y_val, params, device,
                   patience=15, max_epochs=100):
    """Train one FT-T fold on synthetic data. Returns val probabilities."""
    n_features = X_tr_s.shape[1]
    model      = build_fttransformer(params, n_features, device)
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32
    ).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.AdamW(model.parameters(),
                             lr=params["lr"],
                             weight_decay=params["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    batch_size = params["batch_size"]
    tr_loader  = make_loader(X_tr_s, y_tr.astype(np.float32), batch_size, shuffle=True)
    val_loader = make_loader(X_val_s, y_val.astype(np.float32), batch_size, shuffle=False)

    best_auprc   = 0.0
    best_weights = None
    no_improve   = 0

    for epoch in range(max_epochs):
        train_epoch_ftt(model, tr_loader, optimizer, criterion, device)
        scheduler.step()
        val_probs = get_probs_ftt(model, val_loader, device)
        auprc     = average_precision_score(y_val, val_probs)
        if auprc > best_auprc:
            best_auprc   = auprc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_weights:
        model.load_state_dict(best_weights)
    val_probs = get_probs_ftt(model, val_loader, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't have explicit cache clear — just del is fine
        pass

    return val_probs

# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "synth_stack" : "#E53935",   # red    — synthetic stacking (primary)
    "real_stack"  : "#1565C0",   # blue   — real stacking (Phase 7 baseline)
    "real_cb"     : "#6A1B9A",   # purple — real CatBoost (Phase 2 baseline)
    "random"      : "#9E9E9E",   # grey
}


def plot_roc_pr(y_test, synth_ens_probs, real_stack_probs, real_cb_probs,
                metrics, real_stack_auroc, real_cb_auroc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Experiment B — Stacking Ensemble: Synthetic-Trained vs Real-Trained\n"
        f"Evaluated on Real Test Set (n={len(y_test)})",
        fontsize=12, fontweight="bold"
    )

    roc_items = [
        (synth_ens_probs,
         f"Synth Stack   AUROC={metrics['auroc']:.4f}",
         COLORS["synth_stack"], 2.5, "-"),
        (real_stack_probs,
         f"Real Stack    AUROC={real_stack_auroc:.4f}",
         COLORS["real_stack"],  1.5, "--"),
    ]
    if real_cb_probs is not None:
        roc_items.append((
            real_cb_probs,
            f"Real CatBoost AUROC={real_cb_auroc:.4f}",
            COLORS["real_cb"], 1.2, ":"
        ))

    for probs, label, color, lw, ls in roc_items:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)

    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, synth_ens_probs)[:2],
                         alpha=0.06, color=COLORS["synth_stack"])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    pr_items = [
        (synth_ens_probs,
         f"Synth Stack   AUPRC={metrics['auprc']:.4f}",
         COLORS["synth_stack"], 2.5, "-"),
        (real_stack_probs,
         f"Real Stack    AUPRC={average_precision_score(y_test, real_stack_probs):.4f}",
         COLORS["real_stack"],  1.5, "--"),
    ]
    if real_cb_probs is not None:
        pr_items.append((
            real_cb_probs,
            f"Real CatBoost AUPRC={average_precision_score(y_test, real_cb_probs):.4f}",
            COLORS["real_cb"], 1.2, ":"
        ))

    for probs, label, color, lw, ls in pr_items:
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls, label=label)

    prev = y_test.mean()
    axes[1].axhline(prev, color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({prev:.2f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_stacking_roc_pr.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_roc_pr.png")


def plot_calibration(y_test, synth_ens_probs, real_stack_probs, real_cb_probs, metrics):
    fig, ax = plt.subplots(figsize=(7, 6))

    items = [
        (synth_ens_probs,
         f"Synth Stack  (Brier={metrics['brier_score']:.4f})",
         COLORS["synth_stack"]),
        (real_stack_probs,
         f"Real Stack   (Brier={brier_score_loss(y_test, real_stack_probs):.4f})",
         COLORS["real_stack"]),
    ]
    if real_cb_probs is not None:
        items.append((
            real_cb_probs,
            f"Real CB      (Brier={brier_score_loss(y_test, real_cb_probs):.4f})",
            COLORS["real_cb"]
        ))

    for probs, label, color in items:
        prob_true_, prob_pred_ = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(prob_pred_, prob_true_, "o-", color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration — Synthetic Stack vs Real Stack vs Real CatBoost",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_stacking_calibration.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_calibration.png")


def plot_confusion_matrix(y_test, synth_ens_probs, metrics):
    y_pred = (synth_ens_probs >= metrics["threshold"]).astype(int)
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
        f"Synthetic-Trained Stack — Confusion Matrix\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(FIGURES_DIR / f"synthetic_stacking_confusion_matrix.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_confusion_matrix.png")


def plot_threshold_sensitivity(thresh_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Synthetic-Trained Stacking — Threshold Sensitivity Analysis",
        fontsize=12, fontweight="bold"
    )

    thr_vals  = thresh_df["threshold"].values
    sens_vals = thresh_df["sensitivity"].values
    spec_vals = thresh_df["specificity"].values
    ppv_vals  = thresh_df["ppv"].values
    npv_vals  = thresh_df["npv"].values
    f1_vals   = thresh_df["f1"].values

    axes[0].plot(thr_vals, sens_vals, "o-", color=COLORS["synth_stack"], lw=2,
                 label="Sensitivity")
    axes[0].plot(thr_vals, spec_vals, "s-", color=COLORS["real_stack"],  lw=2,
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
        plt.savefig(FIGURES_DIR / f"synthetic_stacking_threshold_sensitivity.{ext}",
                    dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_threshold_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("EXPERIMENT B — Stacking Ensemble: Synthetic-Trained, Real-Tested")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── 7.1  Device detection ─────────────────────────────────────────────────
    log.info("\nDevice detection:")
    cb_task_type, cb_devices, torch_device = detect_devices()
    log.info(f"  CatBoost task_type : {cb_task_type}")
    log.info(f"  FT-T torch device  : {torch_device}")

    # ── 7.2  Load data ────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Loading data")
    log.info("=" * 60)

    if not SYNTH_TRAIN_FILE.exists():
        log.error(f"Synthetic training file not found: {SYNTH_TRAIN_FILE}")
        log.error("Please run generate_synthetic_data.py first.")
        sys.exit(1)

    synth_train_df = pd.read_csv(SYNTH_TRAIN_FILE)
    test_df        = pd.read_csv(REAL_TEST_FILE)
    feature_cols   = [c for c in synth_train_df.columns if c != TARGET_COL]

    X_synth    = synth_train_df[feature_cols].values
    y_synth    = synth_train_df[TARGET_COL].values
    X_test_df  = test_df[feature_cols]
    X_test     = X_test_df.values
    y_test     = test_df[TARGET_COL].values

    log.info(f"  Synthetic train : {X_synth.shape} | "
             f"Sepsis: {y_synth.mean():.1%}")
    log.info(f"  Real test       : {X_test.shape}  | "
             f"Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features        : {len(feature_cols)}")

    # ── 7.3  Identify binary vs numerical columns (for FT-T scaling) ──────────
    binary_cols = [c for c in feature_cols
                   if set(synth_train_df[c].dropna().unique()).issubset(
                       {0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]
    num_idx     = [feature_cols.index(c) for c in num_cols]
    bin_idx     = [feature_cols.index(c) for c in binary_cols]

    log.info(f"  Numerical cols : {len(num_cols)} | Binary cols: {len(binary_cols)}")

    def scale_for_ftt(X_arr, scaler=None, fit=False):
        """Scale numerical columns only; leave binary as-is."""
        from sklearn.preprocessing import StandardScaler
        X_scaled = X_arr.copy().astype(np.float32)
        if fit:
            scaler = StandardScaler()
            X_scaled[:, num_idx] = scaler.fit_transform(X_arr[:, num_idx])
        else:
            X_scaled[:, num_idx] = scaler.transform(X_arr[:, num_idx])
        return X_scaled, scaler

    # ── 7.4  Load FT-T hyperparameters ────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Loading FT-Transformer hyperparameters")
    log.info("=" * 60)

    if FTT_PARAMS_PATH.exists():
        with open(FTT_PARAMS_PATH) as f:
            ftt_params = json.load(f)
        log.info(f"  Loaded from: {FTT_PARAMS_PATH}")
        log.info(f"  FT-T params: {ftt_params}")
    else:
        log.warning("  FT-T params not found. Using sensible defaults.")
        ftt_params = {
            "n_blocks"              : 3,
            "d_block"               : 192,
            "attention_n_heads"     : 8,
            "attention_dropout"     : 0.2,
            "ffn_d_hidden_multiplier": 1.333,
            "ffn_dropout"           : 0.1,
            "residual_dropout"      : 0.0,
            "lr"                    : 1e-4,
            "weight_decay"          : 1e-5,
            "batch_size"            : 256,
        }

    # ── 7.5  Load real-trained baseline probs ─────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Loading real-trained baseline probabilities")
    log.info("=" * 60)

    # Real-trained stacking ensemble probs (Phase 7)
    real_stack_probs = None
    if PHASE7_ENS_PROBS_PATH.exists():
        p7_df            = pd.read_csv(PHASE7_ENS_PROBS_PATH)
        real_stack_probs = p7_df["ensemble_prob"].values
        real_stack_auroc = roc_auc_score(y_test, real_stack_probs)
        log.info(f"  Real stack probs loaded. AUROC: {real_stack_auroc:.4f}")
    else:
        log.warning(f"  Phase 7 test probs not found at {PHASE7_ENS_PROBS_PATH}")
        log.warning(f"  Will use known AUROC={REAL_STACK_AUROC} for comparison text.")
        real_stack_probs = None
        real_stack_auroc = REAL_STACK_AUROC

    # Real CatBoost probs (Phase 2)
    real_cb_probs = None
    if PHASE7_CB_PROBS.exists():
        real_cb_probs = np.load(str(PHASE7_CB_PROBS))
        real_cb_auroc = roc_auc_score(y_test, real_cb_probs)
        log.info(f"  Real CatBoost probs loaded. AUROC: {real_cb_auroc:.4f}")
    else:
        real_cb_auroc = REAL_CB_AUROC
        log.warning(f"  Real CB probs not found. Using known AUROC={real_cb_auroc}.")

    # ── 7.6  5-Fold CV: OOF predictions on synthetic data ─────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 4 — 5-Fold CV: OOF predictions on synthetic training data")
    log.info("=" * 60)

    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                             random_state=RANDOM_SEED)
    oof_cb  = np.zeros(len(y_synth))
    oof_ftt = np.zeros(len(y_synth))

    # CatBoost class weight based on synthetic prevalence
    synth_prev     = y_synth.mean()
    cb_class_ratio = (1 - synth_prev) / synth_prev

    # Load real CB params for architecture reuse
    real_cb_for_params = None
    if (SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm").exists():
        real_cb_for_params = CatBoostClassifier()
        real_cb_for_params.load_model(
            str(SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm")
        )

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_synth, y_synth)):
        fold_start = time.time()
        log.info(f"\n  Fold {fold_idx + 1}/{CV_FOLDS}")

        X_tr, X_val = X_synth[tr_idx], X_synth[val_idx]
        y_tr, y_val = y_synth[tr_idx], y_synth[val_idx]
        log.info(f"    Train: {len(y_tr):,} | Val: {len(y_val):,} | "
                 f"Sepsis in val: {y_val.mean():.1%}")

        # ── CatBoost fold ──────────────────────────────────────────────────
        log.info(f"    Training CatBoost fold {fold_idx + 1}...")
        cb_params = {}
        if real_cb_for_params:
            cb_params = {k: v for k, v in real_cb_for_params.get_params().items()
                         if k not in ("task_type", "devices")}
        else:
            cb_params = {
                "iterations"   : 1000,
                "depth"        : 6,
                "learning_rate": 0.05,
                "l2_leaf_reg"  : 3.0,
                "loss_function": "Logloss",
                "eval_metric"  : "AUC",
            }
        cb_params["class_weights"] = [1.0, round(cb_class_ratio, 2)]
        cb_params["random_seed"]   = RANDOM_SEED
        cb_params["verbose"]       = 0
        cb_params["task_type"]     = cb_task_type
        if cb_devices:
            cb_params["devices"] = cb_devices

        fold_cb = CatBoostClassifier(**cb_params)

        # Use DataFrame to preserve feature names
        X_tr_df  = pd.DataFrame(X_tr,  columns=feature_cols)
        X_val_df = pd.DataFrame(X_val, columns=feature_cols)
        fold_cb.fit(X_tr_df, y_tr)
        oof_cb[val_idx] = fold_cb.predict_proba(X_val_df)[:, 1]
        cb_val_auroc    = roc_auc_score(y_val, oof_cb[val_idx])
        log.info(f"    CatBoost val AUROC: {cb_val_auroc:.4f}")

        # Save fold model
        fold_cb.save_model(str(MODELS_DIR / f"synthetic_catboost_fold{fold_idx+1}.cbm"))

        # ── FT-Transformer fold ────────────────────────────────────────────
        log.info(f"    Training FT-Transformer fold {fold_idx + 1}...")
        X_tr_s, fold_scaler = scale_for_ftt(X_tr,   fit=True)
        X_val_s, _          = scale_for_ftt(X_val,  scaler=fold_scaler)

        oof_ftt[val_idx] = train_ftt_fold(
            X_tr_s, y_tr, X_val_s, y_val, ftt_params, torch_device,
            patience=15, max_epochs=100
        )
        ftt_val_auroc = roc_auc_score(y_val, oof_ftt[val_idx])
        log.info(f"    FT-T val AUROC: {ftt_val_auroc:.4f}")

        fold_time = (time.time() - fold_start) / 60
        log.info(f"    Fold {fold_idx + 1} complete in {fold_time:.1f} min")

    # OOF summary
    oof_cb_auroc  = roc_auc_score(y_synth, oof_cb)
    oof_ftt_auroc = roc_auc_score(y_synth, oof_ftt)
    log.info(f"\n  OOF AUROC — CatBoost       : {oof_cb_auroc:.4f}")
    log.info(f"  OOF AUROC — FT-Transformer : {oof_ftt_auroc:.4f}")

    oof_df = pd.DataFrame({
        "y_true"           : y_synth,
        "oof_catboost"     : oof_cb,
        "oof_fttransformer": oof_ftt,
    })
    oof_df.to_csv(RESULTS_DIR / "synthetic_oof_probs.csv", index=False)
    log.info(f"  OOF probs saved -> {RESULTS_DIR / 'synthetic_oof_probs.csv'}")

    # ── 7.7  Train meta-learner ────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 5 — Training Logistic Regression meta-learner")
    log.info("=" * 60)

    X_meta_train = np.column_stack([oof_cb, oof_ftt])
    meta_learner = LogisticRegression(
        C=1.0, random_state=RANDOM_SEED, max_iter=1000, solver="lbfgs"
    )
    meta_learner.fit(X_meta_train, y_synth)

    coef = meta_learner.coef_[0]
    log.info(f"  Meta-learner coefficients:")
    log.info(f"    CatBoost weight      : {coef[0]:.4f}")
    log.info(f"    FT-Transformer weight: {coef[1]:.4f}")
    log.info(f"    Intercept            : {meta_learner.intercept_[0]:.4f}")
    log.info(f"    Note: Higher = meta-learner trusts that model more")

    with open(MODELS_DIR / "synthetic_meta_learner.pkl", "wb") as f:
        pickle.dump(meta_learner, f)

    coef_dict = {
        "catboost_coefficient"     : float(coef[0]),
        "fttransformer_coefficient": float(coef[1]),
        "intercept"                : float(meta_learner.intercept_[0]),
        "note": "Higher coefficient = meta-learner trusts this model more",
    }
    with open(RESULTS_DIR / "synthetic_meta_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=2)

    # ── 7.8  Test set evaluation ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 6 — Evaluating on real held-out test set (n=633)")
    log.info("=" * 60)

    # Get test probs from each base model
    # CatBoost: use the last fold model on test — but for proper stacking,
    # we need test predictions from models trained on the full synthetic set.
    # Standard practice: retrain once on full synthetic data for test inference.
    log.info("  Retraining CatBoost on full synthetic data for test inference...")
    cb_full_params = {k: v for k, v in cb_params.items()}
    cb_full_params["verbose"] = 100
    cb_full = CatBoostClassifier(**cb_full_params)
    cb_full.fit(pd.DataFrame(X_synth, columns=feature_cols), y_synth)
    cb_test_probs = cb_full.predict_proba(X_test_df)[:, 1]
    log.info(f"  CatBoost test AUROC: {roc_auc_score(y_test, cb_test_probs):.4f}")

    log.info("  Retraining FT-Transformer on full synthetic data for test inference...")
    X_synth_s, full_scaler = scale_for_ftt(X_synth, fit=True)
    X_test_s,  _           = scale_for_ftt(X_test,  scaler=full_scaler)

    # Full FT-T train (single pass, no CV)
    from sklearn.model_selection import train_test_split
    X_ftt_tr, X_ftt_val, y_ftt_tr, y_ftt_val = train_test_split(
        X_synth_s, y_synth, test_size=0.1, stratify=y_synth,
        random_state=RANDOM_SEED
    )
    ftt_test_probs_list = train_ftt_fold(
        X_ftt_tr, y_ftt_tr, X_ftt_val, y_ftt_val,
        ftt_params, torch_device, patience=15, max_epochs=100
    )
    # Re-use: get test probs from full-trained model
    # Re-run on test set properly
    ftt_full = build_fttransformer(ftt_params, X_synth_s.shape[1], torch_device)
    ftt_loader_full = make_loader(X_synth_s, y_synth.astype(np.float32),
                                   ftt_params["batch_size"], shuffle=True)
    pos_weight = torch.tensor(
        [(y_synth == 0).sum() / max((y_synth == 1).sum(), 1)],
        dtype=torch.float32
    ).to(torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_full  = optim.AdamW(ftt_full.parameters(),
                             lr=ftt_params["lr"],
                             weight_decay=ftt_params["weight_decay"])
    sched_full = optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=50, eta_min=1e-6)

    log.info("  FT-T full training (50 epochs)...")
    for ep in range(50):
        train_epoch_ftt(ftt_full, ftt_loader_full, opt_full, criterion, torch_device)
        sched_full.step()

    test_loader = make_loader(X_test_s, y_test.astype(np.float32),
                               ftt_params["batch_size"], shuffle=False)
    ftt_test_probs = get_probs_ftt(ftt_full, test_loader, torch_device)
    ftt_auroc_test = roc_auc_score(y_test, ftt_test_probs)
    log.info(f"  FT-T test AUROC: {ftt_auroc_test:.4f}")

    # Stacking ensemble
    X_meta_test    = np.column_stack([cb_test_probs, ftt_test_probs])
    ensemble_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

    # Save test probs
    test_probs_df = pd.DataFrame({
        "y_true"           : y_test,
        "catboost_prob"    : cb_test_probs,
        "fttransformer_prob": ftt_test_probs,
        "ensemble_prob"    : ensemble_probs,
    })
    test_probs_df.to_csv(RESULTS_DIR / "synthetic_test_probs.csv", index=False)

    # Metrics
    metrics  = compute_metrics(y_test, ensemble_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, ensemble_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, ensemble_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  {'=' * 50}")
    log.info(f"  SYNTHETIC-TRAINED STACK — TEST SET RESULTS")
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

    # ── 7.9  DeLong tests ─────────────────────────────────────────────────────
    log.info("\n  DeLong tests:")
    delong_results = {}

    if real_stack_probs is not None:
        log.info("  a) Synthetic Stack vs Real Stack (Phase 7):")
        auc_s, auc_r, z1, p1 = delong_test(y_test, ensemble_probs, real_stack_probs)
        log.info(f"    Synth Stack : {auc_s:.4f}")
        log.info(f"    Real Stack  : {auc_r:.4f}")
        log.info(f"    Z={z1:.4f}  P={p1:.4f}  "
                 f"{'SIGNIFICANT' if p1 < 0.05 else 'not significant'}")
        delong_results["vs_real_stack"] = {
            "synth_auroc" : auc_s,
            "real_auroc"  : auc_r,
            "z_statistic" : z1,
            "p_value"     : p1,
            "significant" : p1 < 0.05,
            "direction"   : "Synth better" if auc_s > auc_r else "Real better",
        }

    if real_cb_probs is not None:
        log.info("  b) Synthetic Stack vs Real CatBoost (Phase 2):")
        auc_s2, auc_cb, z2, p2 = delong_test(y_test, ensemble_probs, real_cb_probs)
        log.info(f"    Synth Stack : {auc_s2:.4f}")
        log.info(f"    Real CB     : {auc_cb:.4f}")
        log.info(f"    Z={z2:.4f}  P={p2:.4f}  "
                 f"{'SIGNIFICANT' if p2 < 0.05 else 'not significant'}")
        delong_results["vs_real_catboost"] = {
            "synth_auroc" : auc_s2,
            "real_auroc"  : auc_cb,
            "z_statistic" : z2,
            "p_value"     : p2,
            "significant" : p2 < 0.05,
            "direction"   : "Synth better" if auc_s2 > auc_cb else "Real better",
        }

    with open(RESULTS_DIR / "synthetic_delong_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)

    # ── 7.10  Threshold sensitivity ───────────────────────────────────────────
    log.info("\n  Threshold sensitivity analysis:")
    thresh_rows = []
    targets     = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    for target in targets:
        thr = find_threshold_at_sensitivity(y_test, ensemble_probs, target)
        m   = compute_metrics(y_test, ensemble_probs, thr)
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

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "synthetic_threshold_table.csv", index=False)

    # ── 7.11  Save full metrics ────────────────────────────────────────────────
    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"            : "Stacking Ensemble (CatBoost + FT-T) — Synthetic-Trained",
        "experiment"       : "B",
        "timestamp"        : datetime.now().isoformat(),
        "training_data"    : "Synthetic only (CTGAN-generated)",
        "test_data"        : "Real held-out (Option B, n=633)",
        "n_synthetic_train": int(len(X_synth)),
        "n_real_test"      : int(len(X_test)),
        "n_features"       : len(feature_cols),
        "cv_folds"         : CV_FOLDS,
        "synth_prevalence" : float(y_synth.mean()),
        "real_prevalence"  : float(y_test.mean()),
        "oof_auroc_catboost"     : float(oof_cb_auroc),
        "oof_auroc_fttransformer": float(oof_ftt_auroc),
        "test_metrics"           : metrics,
        "meta_learner"           : coef_dict,
        "delong"                 : delong_results,
        "real_stack_auroc_phase7": real_stack_auroc,
        "real_cb_auroc_phase2"   : real_cb_auroc if real_cb_probs is not None
                                   else REAL_CB_AUROC,
        "torch_device"           : str(torch_device),
        "cb_task_type"           : cb_task_type,
        "runtime_minutes"        : round(runtime, 2),
    }

    with open(RESULTS_DIR / "synthetic_stacking_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Metrics saved -> {RESULTS_DIR / 'synthetic_stacking_metrics.json'}")

    # ── 7.12  Figures ─────────────────────────────────────────────────────────
    log.info("\nGenerating figures...")
    rsp = real_stack_probs if real_stack_probs is not None else ensemble_probs
    plot_roc_pr(y_test, ensemble_probs, rsp, real_cb_probs,
                metrics, real_stack_auroc,
                roc_auc_score(y_test, real_cb_probs) if real_cb_probs is not None
                else real_cb_auroc)
    plot_calibration(y_test, ensemble_probs, rsp, real_cb_probs, metrics)
    plot_confusion_matrix(y_test, ensemble_probs, metrics)
    plot_threshold_sensitivity(thresh_df)

    # ── 7.13  Human-readable summary ──────────────────────────────────────────
    summary_lines = [
        "=" * 70,
        "EXPERIMENT B — Stacking Ensemble: Synthetic-Trained vs Real-Trained",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "SETUP",
        f"  Training data : {len(X_synth):,} synthetic patients (CTGAN-generated)",
        f"  Test data     : {len(X_test):,} real patients (never synthesised)",
        f"  Features      : {len(feature_cols)}",
        f"  CV folds      : {CV_FOLDS}",
        f"  CatBoost      : {cb_task_type}",
        f"  FT-Transformer: {torch_device}",
        "",
        "SYNTHETIC-TRAINED STACK RESULTS (on real test set)",
        f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]",
        f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]",
        f"  Brier Score : {metrics['brier_score']:.4f}",
        f"  Sensitivity : {metrics['sensitivity']:.4f}",
        f"  Specificity : {metrics['specificity']:.4f}",
        f"  PPV         : {metrics['ppv']:.4f}",
        f"  NPV         : {metrics['npv']:.4f}",
        f"  F1          : {metrics['f1']:.4f}",
        f"  Meta weights: CatBoost={coef[0]:.4f}, FT-T={coef[1]:.4f}",
        "",
        "COMPARISON",
        f"  Real Stack  (Phase 7) AUROC : {real_stack_auroc:.4f}",
        f"  Real CB     (Phase 2) AUROC : {real_cb_auroc if real_cb_probs is not None else REAL_CB_AUROC:.4f}",
        f"  Synth Stack           AUROC : {metrics['auroc']:.4f}",
        f"  vs Real Stack diff          : {metrics['auroc'] - real_stack_auroc:+.4f}",
        "",
        f"  OOF AUROC CatBoost          : {oof_cb_auroc:.4f}",
        f"  OOF AUROC FT-Transformer    : {oof_ftt_auroc:.4f}",
        "",
        f"Runtime : {runtime:.1f} min",
        "=" * 70,
    ]

    summary_text = "\n".join(summary_lines)
    with open(OUTPUTS_DIR / "synthetic_stacking_summary.txt", "w") as f:
        f.write(summary_text)
    log.info(f"  Summary saved -> {OUTPUTS_DIR / 'synthetic_stacking_summary.txt'}")

    # ── 7.14  Final log ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT B — STACKING COMPLETE")
    log.info("=" * 70)
    log.info(f"  AUROC      : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC      : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier      : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  vs Real Stack (Phase 7): AUROC diff = "
             f"{metrics['auroc'] - real_stack_auroc:+.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")
    log.info(f"\n  Models   -> {MODELS_DIR}")
    log.info(f"  Results  -> {RESULTS_DIR}")
    log.info(f"  Figures  -> {FIGURES_DIR}")
    log.info(f"  Outputs  -> {OUTPUTS_DIR}")
    log.info(f"  Log      -> {log_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
