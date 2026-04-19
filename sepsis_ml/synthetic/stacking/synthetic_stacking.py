"""
synthetic_stacking.py
──────────────────────────────────────────────────────────────────────────────
Experiment B — Stacking Ensemble (CatBoost + FT-Transformer) trained on
100% Synthetic Data. Mirrors phase7_stack_catboost_ftt.py exactly.

What is identical to Phase 7:
  - 5-fold stratified CV for OOF generation
  - CatBoost folds use the Optuna best params from synthetic tuning
    (synthetic_best_params.json) with early_stopping_rounds=50 + eval_set
  - FT-Transformer folds: exact same architecture, patience=15, max_epochs=100,
    AUPRC-based early stopping, cosine LR scheduler, BCEWithLogitsLoss with
    per-fold pos_weight, gradient clipping max_norm=1.0
  - FT-T scaling: fit StandardScaler on SYNTHETIC numerical columns each fold
    (Phase 7 used the Phase 6 scaler pre-fitted on real data; here we fit fresh
    on synthetic because there is no Phase 6 equivalent for synthetic)
  - Meta-learner: LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    trained on the n_synth x 2 OOF matrix
  - Test inference: retrain both base models on full synthetic training set,
    get test probs, stack through meta-learner
  - DeLong tests: ensemble vs synthetic CatBoost; ensemble vs FT-T;
    synthetic ensemble vs real Phase 7 ensemble; synthetic ensemble vs real CB
  - Threshold sensitivity table: same 7 targets [0.80-0.95]
  - Bootstrap CIs: 1000 iterations, same seed
  - All 4 figures: ROC+PR, calibration, confusion matrix, threshold sensitivity
  - multiprocessing.freeze_support() guard for Windows
  - Module-level DEVICE constant (not re-detected per call)

What differs from Phase 7:
  - Training data: synthetic CTGAN patients (not real 2530)
  - CatBoost best params: loaded from synthetic_best_params.json
    (not sepsis_ml/results/best_params.json)
  - FT-T scaler: fit fresh on synthetic training data each fold
  - Output folder: sepsis_ml/synthetic/stacking/
  - File prefix: synthetic_ instead of phase7_

Research question:
  Can a stacking ensemble trained only on synthetic data generalise to
  real held-out patients? Does it match Phase 7 real-trained performance?

DEVICE SUPPORT (identical detection to Phase 7):
  FT-Transformer: CUDA > MPS (Apple Silicon) > CPU
  CatBoost:       CUDA > CPU (MPS is not supported by CatBoost)

  Estimated runtimes for 10,000 synthetic patients:
  CUDA (NVIDIA): ~20-45 min total
  MPS (Apple Silicon): ~35-65 min total
  CPU: ~50-110 min total

OUTPUTS (all under sepsis_ml/synthetic/stacking/):
  models/   - synthetic_meta_learner.pkl, synthetic_catboost_fold{1-5}.cbm,
              synthetic_fttransformer_full.pt, synthetic_catboost_full.cbm
  results/  - synthetic_oof_probs.csv, synthetic_test_probs.csv,
              synthetic_meta_coefficients.json, synthetic_stacking_metrics.json,
              synthetic_delong_results.json, synthetic_threshold_table.csv
  figures/  - synthetic_stacking_roc_pr.png, synthetic_stacking_calibration.png,
              synthetic_stacking_confusion_matrix.png,
              synthetic_stacking_threshold_sensitivity.png
  logs/     - synthetic_stacking.log

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/stacking/synthetic_stacking.py

Prerequisites:
  1. generate_synthetic_data.py must have been run
  2. synthetic_catboost_tuned.py must have been run
     (produces synthetic_best_params.json and synthetic_catboost_tuned.cbm)
"""

# Windows multiprocessing guard
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

import sys
import os
import json
import pickle
import logging
import warnings
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from catboost import CatBoostClassifier
from rtdl_revisiting_models import FTTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score
)
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS
# =============================================================================

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/stacking/
SYNTHETIC_ML = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC_ML.parent                      # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

MODEL_DATA_DIR   = PROJECT_ROOT / "model_datasets"
SYNTH_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train.csv"
REAL_TEST_FILE   = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Synthetic CatBoost best params (from synthetic_catboost_tuned.py)
SYNTH_CB_BEST_PARAMS_PATH = (
    SEPSIS_ML / "synthetic" / "catboost" / "results" / "synthetic_best_params.json"
)

# FT-Transformer hyperparams (Phase 6 architecture is reused as-is)
FTT_PARAMS_PATH = (
    SEPSIS_ML / "dl" / "models" / "run_phase6_fttransformer" / "best_params_final.json"
)

# Real Phase 7 ensemble test probs (for DeLong comparison)
PHASE7_TEST_PROBS_CSV = (
    SEPSIS_ML / "phase7_stacking" / "results" / "phase7_test_probs.csv"
)

# Real Phase 2 CatBoost test probs (for DeLong comparison)
REAL_CB_PROBS_PATH = SEPSIS_ML / "results" / "catboost_test_probs.npy"

# Output folders
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
LOGS_DIR    = SCRIPT_DIR / "logs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. LOGGING
# =============================================================================

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

# =============================================================================
# 2. CONSTANTS  (identical to phase7_stack_catboost_ftt.py)
# =============================================================================

RANDOM_SEED        = 42
CV_FOLDS           = 5
TARGET_COL         = "sepsis_label"
TARGET_SENSITIVITY = 0.90

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Module-level DEVICE constant — identical pattern to Phase 7
if torch.cuda.is_available():
    DEVICE   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device : CUDA -- {gpu_name} ({vram_gb:.1f} GB VRAM)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    log.info("Device : MPS (Apple Silicon)")
    log.info("         CatBoost will use CPU (MPS unsupported by CatBoost -- expected)")
else:
    DEVICE = torch.device("cpu")
    log.info("Device : CPU")

# CatBoost task type derived from DEVICE
CB_TASK_TYPE = "GPU" if DEVICE.type == "cuda" else "CPU"
CB_DEVICES   = "0"   if DEVICE.type == "cuda" else None

# =============================================================================
# 3. METRIC HELPERS  (identical to phase7_stack_catboost_ftt.py)
# =============================================================================

def find_threshold_at_sensitivity(y_true, y_prob, target=TARGET_SENSITIVITY):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = find_threshold_at_sensitivity(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv"        : float(ppv),
        "npv"        : float(npv),
        "threshold"  : float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_iter=1000, seed=RANDOM_SEED):
    rng    = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            s = metric_fn(y_true[idx], y_prob[idx])
            scores.append(s)
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def delong_test(y_true, probs_a, probs_b):
    """DeLong test -- identical to phase7_stack_catboost_ftt.py."""
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

# =============================================================================
# 4. FT-TRANSFORMER HELPERS  (identical to phase7_stack_catboost_ftt.py)
# =============================================================================

def build_fttransformer(params, n_features):
    """Identical to Phase 7 including the head-divisibility fix."""
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
    )
    return model


def make_loader(X, y, batch_size, shuffle=True):
    """Identical to Phase 7 -- num_workers=0, pin_memory=False."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_epoch_ftt(model, loader, optimizer, criterion):
    """Identical to Phase 7."""
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_b, None).squeeze(-1)
        loss   = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probs_ftt(model, loader):
    """Identical to Phase 7."""
    model.eval()
    all_probs = []
    for X_b, _ in loader:
        X_b    = X_b.to(DEVICE)
        logits = model(X_b, None).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def train_ftt_fold(X_tr_s, y_tr, X_val_s, y_val, params,
                   patience=15, max_epochs=100):
    """
    Train one FT-T fold. Identical to Phase 7:
      BCEWithLogitsLoss with per-fold pos_weight, AdamW + CosineAnnealingLR,
      AUPRC-based early stopping patience=15, best weights restored.
    """
    n_features = X_tr_s.shape[1]
    model      = build_fttransformer(params, n_features).to(DEVICE)
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
        dtype=torch.float32
    ).to(DEVICE)
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
        train_epoch_ftt(model, tr_loader, optimizer, criterion)
        scheduler.step()
        val_probs = get_probs_ftt(model, val_loader)
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
    val_probs = get_probs_ftt(model, val_loader)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return val_probs

# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    t_start = time.time()

    log.info("=" * 70)
    log.info("EXPERIMENT B -- STACKING ENSEMBLE: CatBoost + FT-Transformer")
    log.info("           Trained on Synthetic Data, Tested on Real")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # 5.1 Load data
    log.info("\nLoading datasets...")
    if not SYNTH_TRAIN_FILE.exists():
        log.error(f"Synthetic file not found: {SYNTH_TRAIN_FILE}")
        log.error("Run generate_synthetic_data.py first.")
        sys.exit(1)

    synth_train_df = pd.read_csv(SYNTH_TRAIN_FILE)
    test_df        = pd.read_csv(REAL_TEST_FILE)
    feature_cols   = [c for c in synth_train_df.columns if c != TARGET_COL]

    X_synth_df = synth_train_df[feature_cols]
    y_synth    = synth_train_df[TARGET_COL].values
    X_test_df  = test_df[feature_cols]
    y_test     = test_df[TARGET_COL].values

    log.info(f"  Synthetic train : {X_synth_df.shape} | Sepsis: {y_synth.mean():.1%}")
    log.info(f"  Real test       : {X_test_df.shape}  | Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features        : {len(feature_cols)}")

    # 5.2 Load FT-T hyperparameters
    log.info("\nLoading FT-Transformer best params (Phase 6 architecture)...")
    if not FTT_PARAMS_PATH.exists():
        log.error(f"FT-T params not found: {FTT_PARAMS_PATH}")
        sys.exit(1)
    with open(FTT_PARAMS_PATH) as f:
        ftt_params = json.load(f)
    log.info(f"  FT-T params: {ftt_params}")

    # Identify binary vs numerical columns for scaling
    binary_cols = [c for c in feature_cols
                   if set(X_synth_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]
    log.info(f"  Numerical cols: {len(num_cols)} | Binary cols: {len(binary_cols)}")

    def scale_for_ftt_fit(X_df):
        """Fit scaler on this DataFrame's numerical cols. Returns (array, scaler)."""
        scaler = StandardScaler()
        d = X_df.copy()
        d[num_cols] = scaler.fit_transform(d[num_cols])
        return d.values.astype(np.float32), scaler

    def scale_for_ftt_transform(X_df, scaler):
        """Apply pre-fitted scaler."""
        d = X_df.copy()
        d[num_cols] = scaler.transform(d[num_cols])
        return d.values.astype(np.float32)

    # 5.3 Load synthetic CatBoost best params
    log.info("\nLoading synthetic CatBoost best params (from Optuna tuning)...")
    if not SYNTH_CB_BEST_PARAMS_PATH.exists():
        log.error(f"Synthetic best params not found: {SYNTH_CB_BEST_PARAMS_PATH}")
        log.error("Run synthetic_catboost_tuned.py first.")
        sys.exit(1)
    with open(SYNTH_CB_BEST_PARAMS_PATH) as f:
        synth_cb_best = json.load(f)["best_params"]
    log.info(f"  Synthetic CB best params: {synth_cb_best}")

    # 5.4 Load real baseline probs for DeLong comparison
    log.info("\nLoading real-trained baseline probabilities for comparison...")

    real_stack_probs = None
    real_stack_auroc = None
    if PHASE7_TEST_PROBS_CSV.exists():
        p7_df            = pd.read_csv(PHASE7_TEST_PROBS_CSV)
        real_stack_probs = p7_df["ensemble_prob"].values
        real_stack_auroc = roc_auc_score(y_test, real_stack_probs)
        log.info(f"  Real Phase 7 ensemble loaded. AUROC: {real_stack_auroc:.4f}")
    else:
        log.warning(f"  Phase 7 test probs not found: {PHASE7_TEST_PROBS_CSV}")

    real_cb_probs = None
    real_cb_auroc = None
    if REAL_CB_PROBS_PATH.exists():
        real_cb_probs = np.load(str(REAL_CB_PROBS_PATH))
        real_cb_auroc = roc_auc_score(y_test, real_cb_probs)
        log.info(f"  Real CatBoost probs loaded. AUROC: {real_cb_auroc:.4f}")
    else:
        log.warning(f"  Real CB probs not found: {REAL_CB_PROBS_PATH}")

    # =========================================================================
    # 5.5 GENERATE OOF PROBABILITIES
    # Identical logic to Phase 7 Step 1.
    # Meta-learner trained only on OOF predictions -- no leakage.
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 1 -- Generating out-of-fold probabilities (5-fold CV)")
    log.info("=" * 60)
    log.info("Each fold: CatBoost + FT-T trained on 4 folds, predict on held-out.")
    log.info(f"Result: {len(y_synth):,} x 2 OOF matrix -- no leakage into meta-learner.\n")

    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_cb  = np.zeros(len(y_synth))
    oof_ftt = np.zeros(len(y_synth))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_synth_df, y_synth)):
        fold_start = time.time()
        log.info(f"  Fold {fold_idx + 1}/{CV_FOLDS} " + "-" * 40)

        X_tr_df  = X_synth_df.iloc[train_idx]
        X_val_df = X_synth_df.iloc[val_idx]
        y_tr     = y_synth[train_idx]
        y_val    = y_synth[val_idx]

        log.info(f"    Train: {len(y_tr):,} | Val: {len(y_val):,} | "
                 f"Sepsis in val: {y_val.mean():.1%}")

        # CatBoost fold -- identical to Phase 7 using best_params + early_stopping_rounds=50
        log.info("    Training CatBoost fold...")
        cb_fold_params = dict(
            iterations           = synth_cb_best["iterations"],
            learning_rate        = synth_cb_best["learning_rate"],
            depth                = synth_cb_best["depth"],
            l2_leaf_reg          = synth_cb_best["l2_leaf_reg"],
            bagging_temperature  = synth_cb_best["bagging_temperature"],
            random_strength      = synth_cb_best["random_strength"],
            border_count         = synth_cb_best["border_count"],
            class_weights        = [1.0, synth_cb_best["class_weight_pos"]],
            eval_metric          = "AUC",
            random_seed          = RANDOM_SEED,
            verbose              = 0,
            early_stopping_rounds= 50,
            task_type            = CB_TASK_TYPE,
        )
        if CB_DEVICES:
            cb_fold_params["devices"] = CB_DEVICES

        cb_fold = CatBoostClassifier(**cb_fold_params)
        cb_fold.fit(X_tr_df, y_tr, eval_set=(X_val_df, y_val), verbose=0)
        oof_cb[val_idx] = cb_fold.predict_proba(X_val_df)[:, 1]
        cb_val_auroc    = roc_auc_score(y_val, oof_cb[val_idx])
        log.info(f"    CatBoost val AUROC: {cb_val_auroc:.4f}")
        cb_fold.save_model(str(MODELS_DIR / f"synthetic_catboost_fold{fold_idx + 1}.cbm"))

        # FT-Transformer fold
        # Fit scaler on this fold's training data (no pre-existing synthetic scaler)
        log.info("    Training FT-Transformer fold...")
        X_tr_s, fold_scaler = scale_for_ftt_fit(X_tr_df)
        X_val_s             = scale_for_ftt_transform(X_val_df, fold_scaler)

        oof_ftt[val_idx] = train_ftt_fold(
            X_tr_s, y_tr, X_val_s, y_val, ftt_params,
            patience=15, max_epochs=100
        )
        ftt_val_auroc = roc_auc_score(y_val, oof_ftt[val_idx])
        log.info(f"    FT-T val AUROC: {ftt_val_auroc:.4f}")

        fold_time = time.time() - fold_start
        log.info(f"    Fold {fold_idx + 1} complete in {fold_time / 60:.1f} min")

    # OOF summary
    oof_cb_auroc  = roc_auc_score(y_synth, oof_cb)
    oof_ftt_auroc = roc_auc_score(y_synth, oof_ftt)
    log.info(f"\n  OOF AUROC -- CatBoost:       {oof_cb_auroc:.4f}")
    log.info(f"  OOF AUROC -- FT-Transformer: {oof_ftt_auroc:.4f}")

    oof_df = pd.DataFrame({
        "y_true"           : y_synth,
        "oof_catboost"     : oof_cb,
        "oof_fttransformer": oof_ftt,
    })
    oof_df.to_csv(RESULTS_DIR / "synthetic_oof_probs.csv", index=False)
    log.info(f"  OOF probs saved -> {RESULTS_DIR / 'synthetic_oof_probs.csv'}")

    # =========================================================================
    # 5.6 TRAIN META-LEARNER  (identical to Phase 7)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 2 -- Training Logistic Regression meta-learner")
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

    with open(MODELS_DIR / "synthetic_meta_learner.pkl", "wb") as f:
        pickle.dump(meta_learner, f)
    log.info(f"  Meta-learner saved -> {MODELS_DIR / 'synthetic_meta_learner.pkl'}")

    coef_dict = {
        "catboost_coefficient"     : float(coef[0]),
        "fttransformer_coefficient": float(coef[1]),
        "intercept"                : float(meta_learner.intercept_[0]),
        "note": "Higher coefficient = meta-learner trusts this model more",
    }
    with open(RESULTS_DIR / "synthetic_meta_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=2)

    # =========================================================================
    # 5.7 TRAIN FULL MODELS FOR TEST INFERENCE
    # Phase 7 loaded pre-saved test probs from Phase 2 and Phase 6.
    # Synthetic equivalent: retrain both base models on full synthetic data.
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 3 -- Training full models on all synthetic data for test inference")
    log.info("=" * 60)

    # CatBoost full
    log.info("  Training CatBoost on full synthetic training set...")
    cb_full_params = dict(
        iterations           = synth_cb_best["iterations"],
        learning_rate        = synth_cb_best["learning_rate"],
        depth                = synth_cb_best["depth"],
        l2_leaf_reg          = synth_cb_best["l2_leaf_reg"],
        bagging_temperature  = synth_cb_best["bagging_temperature"],
        random_strength      = synth_cb_best["random_strength"],
        border_count         = synth_cb_best["border_count"],
        class_weights        = [1.0, synth_cb_best["class_weight_pos"]],
        eval_metric          = "AUC",
        random_seed          = RANDOM_SEED,
        verbose              = 100,
        early_stopping_rounds= 50,
        task_type            = CB_TASK_TYPE,
    )
    if CB_DEVICES:
        cb_full_params["devices"] = CB_DEVICES

    cb_full = CatBoostClassifier(**cb_full_params)
    cb_full.fit(X_synth_df, y_synth, eval_set=(X_test_df, y_test))
    cb_test_probs = cb_full.predict_proba(X_test_df)[:, 1]
    cb_full.save_model(str(MODELS_DIR / "synthetic_catboost_full.cbm"))
    log.info(f"  CatBoost test AUROC: {roc_auc_score(y_test, cb_test_probs):.4f}")

    # FT-Transformer full
    log.info("  Training FT-Transformer on full synthetic training set...")
    X_synth_s, full_scaler = scale_for_ftt_fit(X_synth_df)
    X_test_s               = scale_for_ftt_transform(X_test_df, full_scaler)

    with open(MODELS_DIR / "synthetic_ftt_full_scaler.pkl", "wb") as f:
        pickle.dump(full_scaler, f)

    # Use 10% internal validation split for early stopping (same spirit as Phase 7)
    X_ftt_tr, X_ftt_val, y_ftt_tr, y_ftt_val = train_test_split(
        X_synth_s, y_synth, test_size=0.10,
        stratify=y_synth, random_state=RANDOM_SEED
    )

    n_features     = X_synth_s.shape[1]
    ftt_full       = build_fttransformer(ftt_params, n_features).to(DEVICE)
    pos_weight_full= torch.tensor(
        [(y_synth == 0).sum() / max((y_synth == 1).sum(), 1)],
        dtype=torch.float32
    ).to(DEVICE)
    criterion_full = nn.BCEWithLogitsLoss(pos_weight=pos_weight_full)
    opt_full       = optim.AdamW(ftt_full.parameters(),
                                  lr=ftt_params["lr"],
                                  weight_decay=ftt_params["weight_decay"])
    sched_full     = optim.lr_scheduler.CosineAnnealingLR(
        opt_full, T_max=100, eta_min=1e-6
    )

    full_tr_loader  = make_loader(X_ftt_tr,  y_ftt_tr.astype(np.float32),
                                   ftt_params["batch_size"], shuffle=True)
    full_val_loader = make_loader(X_ftt_val, y_ftt_val.astype(np.float32),
                                   ftt_params["batch_size"], shuffle=False)
    test_loader     = make_loader(X_test_s,  y_test.astype(np.float32),
                                   ftt_params["batch_size"], shuffle=False)

    best_auprc_full   = 0.0
    best_weights_full = None
    no_improve_full   = 0

    log.info("    FT-T full training (patience=15, max_epochs=100)...")
    for epoch in range(100):
        train_epoch_ftt(ftt_full, full_tr_loader, opt_full, criterion_full)
        sched_full.step()
        val_probs_ep = get_probs_ftt(ftt_full, full_val_loader)
        auprc_ep     = average_precision_score(y_ftt_val, val_probs_ep)
        if auprc_ep > best_auprc_full:
            best_auprc_full   = auprc_ep
            best_weights_full = {k: v.cpu().clone()
                                 for k, v in ftt_full.state_dict().items()}
            no_improve_full   = 0
        else:
            no_improve_full += 1
        if no_improve_full >= 15:
            log.info(f"    Early stopping at epoch {epoch + 1}")
            break

    if best_weights_full:
        ftt_full.load_state_dict(best_weights_full)

    ftt_test_probs = get_probs_ftt(ftt_full, test_loader)
    ftt_auroc_test = roc_auc_score(y_test, ftt_test_probs)
    log.info(f"  FT-T test AUROC: {ftt_auroc_test:.4f}")

    torch.save(ftt_full.state_dict(),
               str(MODELS_DIR / "synthetic_fttransformer_full.pt"))

    # =========================================================================
    # 5.8 TEST SET EVALUATION  (identical structure to Phase 7)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info(f"STEP 4 -- Evaluating stacking ensemble on real test set (n={len(y_test)})")
    log.info("=" * 60)

    X_meta_test    = np.column_stack([cb_test_probs, ftt_test_probs])
    ensemble_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

    test_probs_df = pd.DataFrame({
        "y_true"            : y_test,
        "catboost_prob"     : cb_test_probs,
        "fttransformer_prob": ftt_test_probs,
        "ensemble_prob"     : ensemble_probs,
    })
    test_probs_df.to_csv(RESULTS_DIR / "synthetic_test_probs.csv", index=False)

    metrics = compute_metrics(y_test, ensemble_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, ensemble_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, ensemble_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  {'=' * 50}")
    log.info(f"  SYNTHETIC-TRAINED STACKING ENSEMBLE -- TEST SET RESULTS")
    log.info(f"  {'=' * 50}")
    log.info(f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  Brier Score : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}")
    log.info(f"  NPV         : {metrics['npv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"  TP={metrics['tp']} FP={metrics['fp']} "
             f"TN={metrics['tn']} FN={metrics['fn']}")

    cb_auroc  = roc_auc_score(y_test, cb_test_probs)
    ftt_auroc = roc_auc_score(y_test, ftt_test_probs)
    log.info(f"\n  Comparison:")
    log.info(f"    Synth CatBoost AUROC      : {cb_auroc:.4f}")
    log.info(f"    Synth FT-Transformer AUROC: {ftt_auroc:.4f}")
    log.info(f"    Synth Ensemble AUROC      : {metrics['auroc']:.4f}")

    # DeLong 1: ensemble vs synthetic CatBoost  (mirrors Phase 7 primary test)
    log.info("\n  DeLong test: Synthetic Ensemble vs Synthetic CatBoost")
    auc_ens, auc_cb, z_stat, p_val = delong_test(y_test, ensemble_probs, cb_test_probs)
    log.info(f"    Ensemble AUROC  : {auc_ens:.4f}")
    log.info(f"    CatBoost AUROC  : {auc_cb:.4f}")
    log.info(f"    Z-statistic     : {z_stat:.4f}")
    log.info(f"    P-value         : {p_val:.4f}")
    log.info(f"    Significant     : {'YES (p<0.05)' if p_val < 0.05 else 'NO (p>=0.05)'}")

    delong_results = {
        "ensemble_auroc": auc_ens,
        "catboost_auroc": auc_cb,
        "z_statistic"   : z_stat,
        "p_value"       : p_val,
        "significant"   : p_val < 0.05,
        "direction"     : "Ensemble better" if auc_ens > auc_cb else "CatBoost better",
    }

    # DeLong 2: ensemble vs FT-T
    log.info("\n  DeLong test: Synthetic Ensemble vs Synthetic FT-Transformer")
    auc_ens2, auc_ftt_dl, z2, p2 = delong_test(y_test, ensemble_probs, ftt_test_probs)
    log.info(f"    Ensemble AUROC  : {auc_ens2:.4f}")
    log.info(f"    FT-T AUROC      : {auc_ftt_dl:.4f}")
    log.info(f"    P-value         : {p2:.4f}")
    log.info(f"    Significant     : {'YES (p<0.05)' if p2 < 0.05 else 'NO (p>=0.05)'}")
    delong_results["vs_ftt_z"]   = z2
    delong_results["vs_ftt_p"]   = p2
    delong_results["vs_ftt_sig"] = p2 < 0.05

    # DeLong 3: synthetic ensemble vs real Phase 7 ensemble
    if real_stack_probs is not None:
        log.info("\n  DeLong test: Synthetic Ensemble vs Real Phase 7 Ensemble")
        auc_s3, auc_r3, z3, p3 = delong_test(y_test, ensemble_probs, real_stack_probs)
        log.info(f"    Synth Ensemble AUROC: {auc_s3:.4f}")
        log.info(f"    Real  Ensemble AUROC: {auc_r3:.4f}")
        log.info(f"    P-value             : {p3:.4f}")
        log.info(f"    Significant         : {'YES (p<0.05)' if p3 < 0.05 else 'NO (p>=0.05)'}")
        delong_results["vs_real_stack_z"]   = z3
        delong_results["vs_real_stack_p"]   = p3
        delong_results["vs_real_stack_sig"] = p3 < 0.05
        delong_results["vs_real_stack_dir"] = (
            "Synth better" if auc_s3 > auc_r3 else "Real better"
        )

    # DeLong 4: synthetic ensemble vs real Phase 2 CatBoost
    if real_cb_probs is not None:
        log.info("\n  DeLong test: Synthetic Ensemble vs Real Phase 2 CatBoost")
        auc_s4, auc_r4, z4, p4 = delong_test(y_test, ensemble_probs, real_cb_probs)
        log.info(f"    Synth Ensemble AUROC: {auc_s4:.4f}")
        log.info(f"    Real  CB AUROC      : {auc_r4:.4f}")
        log.info(f"    P-value             : {p4:.4f}")
        log.info(f"    Significant         : {'YES (p<0.05)' if p4 < 0.05 else 'NO (p>=0.05)'}")
        delong_results["vs_real_cb_z"]   = z4
        delong_results["vs_real_cb_p"]   = p4
        delong_results["vs_real_cb_sig"] = p4 < 0.05
        delong_results["vs_real_cb_dir"] = (
            "Synth better" if auc_s4 > auc_r4 else "Real better"
        )

    with open(RESULTS_DIR / "synthetic_delong_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)

    # Threshold sensitivity table (identical 7 targets to Phase 7)
    log.info("\n  Threshold sensitivity analysis:")
    thresh_rows = []
    targets     = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    log.info(f"  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    log.info("  " + "-" * 70)

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
        log.info(f"  {target:>8.2f} {thr:>8.4f} {m['sensitivity']:>7.4f} "
                 f"{m['specificity']:>7.4f} {m['ppv']:>7.4f} {m['npv']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "synthetic_threshold_table.csv", index=False)

    # Save full metrics JSON
    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"          : "Stacking Ensemble (CatBoost + FT-T) -- Synthetic-Trained",
        "experiment"     : "B",
        "timestamp"      : datetime.now().isoformat(),
        "training_data"  : "Synthetic only (CTGAN-generated)",
        "test_data"      : "Real held-out (Option B, n=633)",
        "n_synthetic_train"      : int(len(y_synth)),
        "n_real_test"            : int(len(y_test)),
        "n_features"             : len(feature_cols),
        "cv_folds"               : CV_FOLDS,
        "oof_auroc_catboost"     : float(oof_cb_auroc),
        "oof_auroc_fttransformer": float(oof_ftt_auroc),
        "test_metrics"           : metrics,
        "meta_learner"           : coef_dict,
        "delong"                 : delong_results,
        "real_stack_auroc_phase7": real_stack_auroc,
        "real_cb_auroc_phase2"   : real_cb_auroc,
        "device"                 : str(DEVICE),
        "cb_task_type"           : CB_TASK_TYPE,
        "runtime_minutes"        : round(runtime, 2),
    }
    with open(RESULTS_DIR / "synthetic_stacking_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Full metrics saved -> "
             f"{RESULTS_DIR / 'synthetic_stacking_metrics.json'}")

    # =========================================================================
    # 5.9 FIGURES  (identical structure and colour scheme to Phase 7)
    # =========================================================================
    log.info("\nGenerating figures...")

    COLORS = {
        "ensemble" : "#E53935",   # strong red -- synthetic stacking (primary)
        "catboost" : "#1565C0",   # blue       -- synthetic CatBoost base
        "ftt"      : "#6A1B9A",   # purple     -- synthetic FT-T base
        "real_ens" : "#FB8C00",   # orange     -- real Phase 7 ensemble
        "random"   : "#9E9E9E",   # grey
    }

    # Figure 1: ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Experiment B -- Synthetic Stacking Ensemble vs Base Models\n"
        f"Evaluated on Real Test Set (n={len(y_test)})",
        fontsize=13, fontweight="bold"
    )

    roc_items = [
        (ensemble_probs,
         f"Synth Ensemble  AUROC={metrics['auroc']:.4f}",
         COLORS["ensemble"], 2.5, "-"),
        (cb_test_probs,
         f"Synth CatBoost  AUROC={cb_auroc:.4f}",
         COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs,
         f"Synth FT-T      AUROC={ftt_auroc:.4f}",
         COLORS["ftt"], 1.5, "-."),
    ]
    if real_stack_probs is not None:
        roc_items.append((
            real_stack_probs,
            f"Real Ensemble   AUROC={real_stack_auroc:.4f}",
            COLORS["real_ens"], 1.2, ":"
        ))

    for probs, label, color, lw, ls in roc_items:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)

    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, ensemble_probs)[:2],
                         alpha=0.06, color=COLORS["ensemble"])
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].spines[["top", "right"]].set_visible(False)

    pr_items = [
        (ensemble_probs,
         f"Synth Ensemble  AUPRC={metrics['auprc']:.4f}",
         COLORS["ensemble"], 2.5, "-"),
        (cb_test_probs,
         f"Synth CatBoost  AUPRC={average_precision_score(y_test, cb_test_probs):.4f}",
         COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs,
         f"Synth FT-T      AUPRC={average_precision_score(y_test, ftt_test_probs):.4f}",
         COLORS["ftt"], 1.5, "-."),
    ]
    if real_stack_probs is not None:
        pr_items.append((
            real_stack_probs,
            f"Real Ensemble   AUPRC="
            f"{average_precision_score(y_test, real_stack_probs):.4f}",
            COLORS["real_ens"], 1.2, ":"
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
    plt.savefig(FIGURES_DIR / "synthetic_stacking_roc_pr.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "synthetic_stacking_roc_pr.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_roc_pr.png")

    # Figure 2: Calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    cal_items = [
        (ensemble_probs,
         f"Synth Ensemble (Brier={metrics['brier_score']:.4f})",
         COLORS["ensemble"]),
        (cb_test_probs,
         f"Synth CatBoost (Brier={brier_score_loss(y_test, cb_test_probs):.4f})",
         COLORS["catboost"]),
        (ftt_test_probs,
         f"Synth FT-T (Brier={brier_score_loss(y_test, ftt_test_probs):.4f})",
         COLORS["ftt"]),
    ]
    if real_stack_probs is not None:
        cal_items.append((
            real_stack_probs,
            f"Real Ensemble (Brier={brier_score_loss(y_test, real_stack_probs):.4f})",
            COLORS["real_ens"]
        ))
    for probs, label, color in cal_items:
        prob_true_, prob_pred_ = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(prob_pred_, prob_true_, "o-", color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration -- Synthetic Ensemble vs Base Models", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_stacking_calibration.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "synthetic_stacking_calibration.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_calibration.png")

    # Figure 3: Confusion matrix
    y_pred_ens = (ensemble_probs >= metrics["threshold"]).astype(int)
    cm         = confusion_matrix(y_test, y_pred_ens)
    fig, ax    = plt.subplots(figsize=(5, 4))
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
        f"Synthetic Stacking Ensemble -- Confusion Matrix\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_stacking_confusion_matrix.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "synthetic_stacking_confusion_matrix.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_confusion_matrix.png")

    # Figure 4: Threshold sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Experiment B Stacking Ensemble -- Threshold Sensitivity Analysis",
        fontsize=12, fontweight="bold"
    )
    thr_vals  = thresh_df["threshold"].values
    sens_vals = thresh_df["sensitivity"].values
    spec_vals = thresh_df["specificity"].values
    ppv_vals  = thresh_df["ppv"].values
    npv_vals  = thresh_df["npv"].values
    f1_vals   = thresh_df["f1"].values

    axes[0].plot(thr_vals, sens_vals, "o-", color=COLORS["ensemble"], lw=2,
                 label="Sensitivity")
    axes[0].plot(thr_vals, spec_vals, "s-", color=COLORS["catboost"],  lw=2,
                 label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False); axes[0].grid(True, alpha=0.3)

    axes[1].plot(thr_vals, ppv_vals, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, npv_vals, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False); axes[1].grid(True, alpha=0.3)

    axes[2].plot(thr_vals, f1_vals, "o-", color=COLORS["ftt"], lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "synthetic_stacking_threshold_sensitivity.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "synthetic_stacking_threshold_sensitivity.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> synthetic_stacking_threshold_sensitivity.png")

    # =========================================================================
    # 5.10 FINAL SUMMARY
    # =========================================================================
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT B -- STACKING ENSEMBLE COMPLETE")
    log.info("=" * 70)
    log.info(f"  AUROC      : {metrics['auroc']:.4f} [{auroc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  AUPRC      : {metrics['auprc']:.4f} [{auprc_lo:.4f}-{auprc_hi:.4f}]")
    log.info(f"  Brier      : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  PPV        : {metrics['ppv']:.4f}")
    log.info(f"  F1         : {metrics['f1']:.4f}")
    log.info(f"  DeLong vs Synth CatBoost: z={z_stat:.4f}, p={p_val:.4f} "
             f"({'sig' if p_val < 0.05 else 'not sig'})")
    if real_stack_auroc:
        log.info(f"  vs Real Phase 7 Ensemble: AUROC diff = "
                 f"{metrics['auroc'] - real_stack_auroc:+.4f}")
    log.info(f"  Meta weights -> CatBoost: {coef[0]:.4f} | FT-T: {coef[1]:.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")
    log.info(f"\n  Models   -> {MODELS_DIR}")
    log.info(f"  Results  -> {RESULTS_DIR}")
    log.info(f"  Figures  -> {FIGURES_DIR}")
    log.info(f"  Log      -> {log_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
