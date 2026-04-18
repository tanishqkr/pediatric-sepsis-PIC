"""
phase7_stack_catboost_ftt.py
─────────────────────────────────────────────────────────────────────────────
Phase 7 — Stacking Ensemble: CatBoost (tuned) + FT-Transformer
Meta-learner: Logistic Regression trained on out-of-fold (OOF) probabilities

Pipeline:
  1. Generate OOF predictions from CatBoost and FT-Transformer via 5-fold CV
     on the 2,530 training patients (Option B, infection-only)
  2. Train Logistic Regression meta-learner on the 2,530 x 2 OOF matrix
  3. Load saved test-set predictions from Phase 2 (CatBoost) and Phase 6 (FT-T)
  4. Evaluate stacking ensemble on 633 held-out test patients
  5. DeLong test: ensemble vs tuned CatBoost (primary model)
  6. Full metric suite + threshold analysis + figures

Windows CUDA notes:
  - num_workers=0 in all DataLoaders
  - multiprocessing.freeze_support() at top
  - All paths use pathlib.Path (handles spaces correctly)
  - Tested with torch + cu121 build

Folder outputs (all under sepsis_ml/phase7_stacking/):
  models/   — phase7_meta_learner.pkl
  results/  — phase7_oof_probs.csv, phase7_test_probs.csv,
               phase7_meta_coefficients.json, phase7_metrics.json,
               phase7_delong_results.json, phase7_threshold_table.csv
  figures/  — phase7_roc_pr.png, phase7_calibration.png,
               phase7_confusion_matrix.png, phase7_threshold_sensitivity.png
  logs/     — phase7_stacking.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase7_stacking/phase7_stack_catboost_ftt.py
"""

# ── Windows multiprocessing guard ─────────────────────────────────────────────
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
from sklearn.model_selection import StratifiedKFold
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

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/phase7_stacking/
SEPSIS_ML    = SCRIPT_DIR.parent                        # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

# Input data
MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Existing model artifacts (Phase 2 + Phase 6)
CATBOOST_MODEL_PATH  = SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm"
CATBOOST_PROBS_PATH  = SEPSIS_ML / "results" / "catboost_test_probs.npy"
FTT_MODEL_DIR        = SEPSIS_ML / "dl" / "models" / "run_phase6_fttransformer"
FTT_WEIGHTS_PATH     = FTT_MODEL_DIR / "fttransformer_best.pt"
FTT_PARAMS_PATH      = FTT_MODEL_DIR / "best_params_final.json"
FTT_SCALER_PATH      = FTT_MODEL_DIR / "scaler.pkl"
FTT_PROBS_PATH       = SEPSIS_ML / "dl" / "results" / "phase6_fttransformer_predictions.npz"

# Phase 7 output folders
PHASE7_DIR   = SCRIPT_DIR
MODELS_DIR   = PHASE7_DIR / "models"
RESULTS_DIR  = PHASE7_DIR / "results"
FIGURES_DIR  = PHASE7_DIR / "figures"
LOGS_DIR     = PHASE7_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "phase7_stacking.log"
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
# 2. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED       = 42
CV_FOLDS          = 5
TARGET_COL        = "sepsis_label"
TARGET_SENSITIVITY = 0.90

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── CUDA device ───────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device : CUDA — {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    DEVICE = torch.device("cpu")
    log.info("Device : CPU")

# ══════════════════════════════════════════════════════════════════════════════
# 3. METRIC HELPERS
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
        ranks = compute_midrank(np.concatenate([pos, neg]))
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
# 4. FT-TRANSFORMER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_fttransformer(params, n_features):
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
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_epoch_ftt(model, loader, optimizer, criterion):
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
    model.eval()
    all_probs = []
    for X_b, _ in loader:
        X_b    = X_b.to(DEVICE)
        logits = model(X_b, None).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


def train_ftt_fold(X_tr_s, y_tr, X_val_s, y_val, params, patience=15, max_epochs=100):
    """Train one FT-T fold. Returns val probabilities."""
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    log.info("=" * 70)
    log.info("PHASE 7 — STACKING ENSEMBLE: CatBoost + FT-Transformer")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── 5.1 Load data ─────────────────────────────────────────────────────────
    log.info("\nLoading Option B datasets...")
    train_df     = pd.read_csv(TRAIN_FILE)
    test_df      = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train_df = train_df[feature_cols]
    y_train    = train_df[TARGET_COL].values
    X_test_df  = test_df[feature_cols]
    y_test     = test_df[TARGET_COL].values

    log.info(f"  Train : {X_train_df.shape} | Sepsis: {y_train.mean():.1%}")
    log.info(f"  Test  : {X_test_df.shape}  | Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features: {len(feature_cols)}")

    # ── 5.2 Load FT-T hyperparameters and scaler ──────────────────────────────
    log.info("\nLoading FT-Transformer best params and scaler...")
    with open(FTT_PARAMS_PATH) as f:
        ftt_params = json.load(f)
    with open(FTT_SCALER_PATH, "rb") as f:
        ftt_scaler = pickle.load(f)
    log.info(f"  FT-T params: {ftt_params}")

    # Identify binary vs numerical columns (must match Phase 6 exactly)
    binary_cols = [c for c in feature_cols
                   if set(X_train_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]
    log.info(f"  Numerical cols: {len(num_cols)} | Binary cols: {len(binary_cols)}")

    def scale_for_ftt(df):
        """Apply Phase 6 scaler — fit already happened on full train, just transform."""
        d = df.copy()
        d[num_cols] = ftt_scaler.transform(d[num_cols])
        return d.values.astype(np.float32)

    # ── 5.3 Load tuned CatBoost ───────────────────────────────────────────────
    log.info("\nLoading tuned CatBoost model...")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model(str(CATBOOST_MODEL_PATH))
    log.info("  CatBoost loaded.")

    # ── 5.4 Load saved test-set probabilities ─────────────────────────────────
    log.info("\nLoading saved test-set probabilities...")

    # CatBoost test probs (Phase 2)
    cb_test_probs = np.load(CATBOOST_PROBS_PATH)
    log.info(f"  CatBoost test probs shape: {cb_test_probs.shape}")

    # FT-T test probs (Phase 6)
    ftt_npz      = np.load(FTT_PROBS_PATH)
    ftt_test_probs = ftt_npz["test_probs"]
    log.info(f"  FT-T test probs shape: {ftt_test_probs.shape}")

    # Sanity check
    assert len(cb_test_probs) == len(y_test), "CatBoost test probs length mismatch"
    assert len(ftt_test_probs) == len(y_test), "FT-T test probs length mismatch"
    log.info("  Sanity checks passed.")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.5 GENERATE OUT-OF-FOLD PROBABILITIES (training set only)
    # This is the critical step — meta-learner trained only on OOF predictions
    # to prevent data leakage into the meta-learner.
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Generating out-of-fold probabilities (5-fold CV)")
    log.info("=" * 60)
    log.info("Each fold: CatBoost + FT-T trained on 4 folds, predict on held-out.")
    log.info("Result: 2530 x 2 OOF matrix — no leakage into meta-learner.\n")

    skf         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_cb      = np.zeros(len(y_train))
    oof_ftt     = np.zeros(len(y_train))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train)):
        fold_start = time.time()
        log.info(f"  Fold {fold_idx + 1}/{CV_FOLDS} -----------------------------")

        X_tr_df  = X_train_df.iloc[train_idx]
        X_val_df = X_train_df.iloc[val_idx]
        y_tr     = y_train[train_idx]
        y_val    = y_train[val_idx]

        log.info(f"    Train: {len(y_tr)} | Val: {len(y_val)} | "
                 f"Sepsis in val: {y_val.mean():.1%}")

        # ── CatBoost fold ────────────────────────────────────────────────────
        log.info("    Training CatBoost fold...")
        with open(SEPSIS_ML / "results" / "best_params.json") as f:
            cb_best = json.load(f)["best_params"]

        cb_fold = CatBoostClassifier(
            iterations           = cb_best["iterations"],
            learning_rate        = cb_best["learning_rate"],
            depth                = cb_best["depth"],
            l2_leaf_reg          = cb_best["l2_leaf_reg"],
            bagging_temperature  = cb_best["bagging_temperature"],
            random_strength      = cb_best["random_strength"],
            border_count         = cb_best["border_count"],
            class_weights        = [1.0, cb_best["class_weight_pos"]],
            eval_metric          = "AUC",
            random_seed          = RANDOM_SEED,
            verbose              = 0,
            early_stopping_rounds= 50,
        )
        cb_fold.fit(
            X_tr_df, y_tr,
            eval_set=(X_val_df, y_val),
            verbose=0,
        )
        oof_cb[val_idx] = cb_fold.predict_proba(X_val_df)[:, 1]
        cb_val_auroc    = roc_auc_score(y_val, oof_cb[val_idx])
        log.info(f"    CatBoost val AUROC: {cb_val_auroc:.4f}")

        # ── FT-Transformer fold ───────────────────────────────────────────────
        log.info("    Training FT-Transformer fold...")

        # Scale using the Phase 6 scaler (fitted on full train — consistent)
        X_tr_s  = scale_for_ftt(X_tr_df)
        X_val_s = scale_for_ftt(X_val_df)

        oof_ftt[val_idx] = train_ftt_fold(
            X_tr_s, y_tr, X_val_s, y_val, ftt_params,
            patience=15, max_epochs=100
        )
        ftt_val_auroc = roc_auc_score(y_val, oof_ftt[val_idx])
        log.info(f"    FT-T val AUROC: {ftt_val_auroc:.4f}")

        fold_time = time.time() - fold_start
        log.info(f"    Fold {fold_idx + 1} complete in {fold_time / 60:.1f} min")

    # ── OOF summary ───────────────────────────────────────────────────────────
    oof_cb_auroc  = roc_auc_score(y_train, oof_cb)
    oof_ftt_auroc = roc_auc_score(y_train, oof_ftt)
    log.info(f"\n  OOF AUROC — CatBoost:       {oof_cb_auroc:.4f}")
    log.info(f"  OOF AUROC — FT-Transformer: {oof_ftt_auroc:.4f}")

    # Save OOF probs
    oof_df = pd.DataFrame({
        "y_true"           : y_train,
        "oof_catboost"     : oof_cb,
        "oof_fttransformer": oof_ftt,
    })
    oof_df.to_csv(RESULTS_DIR / "phase7_oof_probs.csv", index=False)
    log.info(f"  OOF probs saved -> {RESULTS_DIR / 'phase7_oof_probs.csv'}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.6 TRAIN META-LEARNER
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Training Logistic Regression meta-learner")
    log.info("=" * 60)

    X_meta_train = np.column_stack([oof_cb, oof_ftt])
    meta_learner = LogisticRegression(
        C              = 1.0,
        random_state   = RANDOM_SEED,
        max_iter       = 1000,
        solver         = "lbfgs",
    )
    meta_learner.fit(X_meta_train, y_train)

    coef = meta_learner.coef_[0]
    log.info(f"  Meta-learner coefficients:")
    log.info(f"    CatBoost weight     : {coef[0]:.4f}")
    log.info(f"    FT-Transformer weight: {coef[1]:.4f}")
    log.info(f"    Intercept           : {meta_learner.intercept_[0]:.4f}")

    # Save meta-learner
    with open(MODELS_DIR / "phase7_meta_learner.pkl", "wb") as f:
        pickle.dump(meta_learner, f)
    log.info(f"  Meta-learner saved -> {MODELS_DIR / 'phase7_meta_learner.pkl'}")

    # Save coefficients
    coef_dict = {
        "catboost_coefficient"      : float(coef[0]),
        "fttransformer_coefficient" : float(coef[1]),
        "intercept"                 : float(meta_learner.intercept_[0]),
        "note": "Higher coefficient = meta-learner trusts this model more"
    }
    with open(RESULTS_DIR / "phase7_meta_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # 5.7 TEST SET EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Evaluating stacking ensemble on held-out test set (n=633)")
    log.info("=" * 60)

    X_meta_test   = np.column_stack([cb_test_probs, ftt_test_probs])
    ensemble_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

    # Save test probs
    test_probs_df = pd.DataFrame({
        "y_true"              : y_test,
        "catboost_prob"       : cb_test_probs,
        "fttransformer_prob"  : ftt_test_probs,
        "ensemble_prob"       : ensemble_probs,
    })
    test_probs_df.to_csv(RESULTS_DIR / "phase7_test_probs.csv", index=False)

    # Full metrics
    metrics = compute_metrics(y_test, ensemble_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, ensemble_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, ensemble_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  {'=' * 50}")
    log.info(f"  STACKING ENSEMBLE — TEST SET RESULTS")
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
    log.info(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    # Compare against base models
    cb_auroc  = roc_auc_score(y_test, cb_test_probs)
    ftt_auroc = roc_auc_score(y_test, ftt_test_probs)
    log.info(f"\n  Comparison:")
    log.info(f"    Tuned CatBoost AUROC      : {cb_auroc:.4f}")
    log.info(f"    FT-Transformer AUROC      : {ftt_auroc:.4f}")
    log.info(f"    Stacking Ensemble AUROC   : {metrics['auroc']:.4f}")

    # ── DeLong: ensemble vs tuned CatBoost ────────────────────────────────────
    log.info("\n  DeLong test: Ensemble vs Tuned CatBoost")
    auc_ens, auc_cb, z_stat, p_val = delong_test(y_test, ensemble_probs, cb_test_probs)
    log.info(f"    Ensemble AUROC  : {auc_ens:.4f}")
    log.info(f"    CatBoost AUROC  : {auc_cb:.4f}")
    log.info(f"    Z-statistic     : {z_stat:.4f}")
    log.info(f"    P-value         : {p_val:.4f}")
    log.info(f"    Significant     : {'YES (p<0.05)' if p_val < 0.05 else 'NO (p>=0.05)'}")

    delong_results = {
        "ensemble_auroc"  : auc_ens,
        "catboost_auroc"  : auc_cb,
        "z_statistic"     : z_stat,
        "p_value"         : p_val,
        "significant"     : p_val < 0.05,
        "direction"       : "Ensemble better" if auc_ens > auc_cb else "CatBoost better",
    }

    # ── DeLong: ensemble vs FT-Transformer ────────────────────────────────────
    log.info("\n  DeLong test: Ensemble vs FT-Transformer")
    auc_ens2, auc_ftt, z2, p2 = delong_test(y_test, ensemble_probs, ftt_test_probs)
    log.info(f"    Ensemble AUROC  : {auc_ens2:.4f}")
    log.info(f"    FT-T AUROC      : {auc_ftt:.4f}")
    log.info(f"    P-value         : {p2:.4f}")
    log.info(f"    Significant     : {'YES (p<0.05)' if p2 < 0.05 else 'NO (p>=0.05)'}")

    delong_results["vs_ftt_z"]   = z2
    delong_results["vs_ftt_p"]   = p2
    delong_results["vs_ftt_sig"] = p2 < 0.05

    with open(RESULTS_DIR / "phase7_delong_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)

    # ── Threshold sensitivity table ────────────────────────────────────────────
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
    thresh_df.to_csv(RESULTS_DIR / "phase7_threshold_table.csv", index=False)

    # ── Save full metrics JSON ─────────────────────────────────────────────────
    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"         : "Stacking Ensemble (CatBoost + FT-Transformer)",
        "phase"         : "Phase 7",
        "timestamp"     : datetime.now().isoformat(),
        "dataset"       : "Option B (infection-only)",
        "n_train"       : int(len(y_train)),
        "n_test"        : int(len(y_test)),
        "n_features"    : len(feature_cols),
        "cv_folds"      : CV_FOLDS,
        "oof_auroc_catboost"      : float(oof_cb_auroc),
        "oof_auroc_fttransformer" : float(oof_ftt_auroc),
        "test_metrics"  : metrics,
        "meta_learner"  : coef_dict,
        "delong"        : delong_results,
        "runtime_minutes": float(runtime),
    }
    with open(RESULTS_DIR / "phase7_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    log.info(f"\n  Full metrics saved -> {RESULTS_DIR / 'phase7_metrics.json'}")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.8 FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nGenerating figures...")

    COLORS = {
        "ensemble": "#E53935",   # strong red — primary model
        "catboost": "#1565C0",   # blue
        "ftt"     : "#6A1B9A",   # purple
        "random"  : "#9E9E9E",   # grey
    }

    # ── Figure 1: ROC + PR ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Phase 7 — Stacking Ensemble vs Base Models\n"
        f"Option B Infection-Only Cohort (n_test={len(y_test)})",
        fontsize=13, fontweight="bold"
    )

    for probs, label, color, lw, ls in [
        (ensemble_probs, f"Ensemble  AUROC={metrics['auroc']:.4f}", COLORS["ensemble"], 2.5, "-"),
        (cb_test_probs,  f"CatBoost  AUROC={cb_auroc:.4f}",         COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs, f"FT-T      AUROC={ftt_auroc:.4f}",        COLORS["ftt"],      1.5, "-."),
    ]:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)

    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, ensemble_probs)[:2], alpha=0.06,
                         color=COLORS["ensemble"])
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    for probs, label, color, lw, ls in [
        (ensemble_probs, f"Ensemble  AUPRC={metrics['auprc']:.4f}", COLORS["ensemble"], 2.5, "-"),
        (cb_test_probs,  f"CatBoost  AUPRC={average_precision_score(y_test, cb_test_probs):.4f}",
                         COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs, f"FT-T      AUPRC={average_precision_score(y_test, ftt_test_probs):.4f}",
                         COLORS["ftt"], 1.5, "-."),
    ]:
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls, label=label)

    prev = y_test.mean()
    axes[1].axhline(prev, color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({prev:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase7_roc_pr.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> phase7_roc_pr.png")

    # ── Figure 2: Calibration ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    for probs, label, color in [
        (ensemble_probs, f"Ensemble (Brier={metrics['brier_score']:.4f})", COLORS["ensemble"]),
        (cb_test_probs,  f"CatBoost (Brier={brier_score_loss(y_test, cb_test_probs):.4f})",
                         COLORS["catboost"]),
        (ftt_test_probs, f"FT-T (Brier={brier_score_loss(y_test, ftt_test_probs):.4f})",
                         COLORS["ftt"]),
    ]:
        prob_true_, prob_pred_ = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(prob_pred_, prob_true_, "o-", color=color, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves — Ensemble vs Base Models", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_calibration.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase7_calibration.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> phase7_calibration.png")

    # ── Figure 3: Confusion matrix ────────────────────────────────────────────
    y_pred_ens = (ensemble_probs >= metrics["threshold"]).astype(int)
    cm         = confusion_matrix(y_test, y_pred_ens)
    fig, ax    = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-sepsis", "Sepsis"])
    ax.set_yticklabels(["Non-sepsis", "Sepsis"])
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh_cm else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Stacking Ensemble — Confusion Matrix\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase7_confusion_matrix.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> phase7_confusion_matrix.png")

    # ── Figure 4: Threshold sensitivity ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 7 Stacking Ensemble — Threshold Sensitivity Analysis",
                 fontsize=12, fontweight="bold")

    targs     = thresh_df["target_sensitivity"].values
    sens_vals = thresh_df["sensitivity"].values
    spec_vals = thresh_df["specificity"].values
    ppv_vals  = thresh_df["ppv"].values
    npv_vals  = thresh_df["npv"].values
    f1_vals   = thresh_df["f1"].values
    thr_vals  = thresh_df["threshold"].values

    axes[0].plot(thr_vals, sens_vals, "o-", color=COLORS["ensemble"], lw=2, label="Sensitivity")
    axes[0].plot(thr_vals, spec_vals, "s-", color=COLORS["catboost"], lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(thr_vals, ppv_vals, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, npv_vals, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(thr_vals, f1_vals, "o-", color=COLORS["ftt"], lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase7_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase7_threshold_sensitivity.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> phase7_threshold_sensitivity.png")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.9 FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    runtime = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("PHASE 7 COMPLETE — STACKING ENSEMBLE")
    log.info("=" * 70)
    log.info(f"  AUROC      : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC      : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier      : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  PPV        : {metrics['ppv']:.4f}")
    log.info(f"  F1         : {metrics['f1']:.4f}")
    log.info(f"  DeLong vs CatBoost: z={z_stat:.4f}, p={p_val:.4f} "
             f"({'sig' if p_val < 0.05 else 'not sig'})")
    log.info(f"  Meta weights -> CatBoost: {coef[0]:.4f} | FT-T: {coef[1]:.4f}")
    log.info(f"  Runtime    : {runtime:.1f} min")
    log.info(f"\n  Models  -> {MODELS_DIR}")
    log.info(f"  Results -> {RESULTS_DIR}")
    log.info(f"  Figures -> {FIGURES_DIR}")
    log.info(f"  Log     -> {log_path}")
    log.info("\n  Next step: federated learning (Phase 8)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
