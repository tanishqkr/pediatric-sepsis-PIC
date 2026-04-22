"""
synthetic_stacking.py
─────────────────────────────────────────────────────────────────────────────
Stacking Ensemble — Weighted Hybrid Data
CatBoost (Exp D) + FT-Transformer (hybrid weighted)
Meta-learner: Logistic Regression trained on out-of-fold (OOF) probabilities

This is the direct hybrid-data equivalent of phase7_stack_catboost_ftt.py
which stacked on real-only Option B data. Architecture, OOF pipeline,
meta-learner design, evaluation suite, and figure set are identical.

Prerequisites (must run first):
  1. weighted_hybrid_catboost.py  -> whybrid/results/whybrid_catboost_test_probs.npy
  2. hybrid_fttransformer.py      -> ft/results/hybrid_ftt_predictions.npz

Weighting — consistent at every level of the pipeline:
  - CatBoost OOF folds   : Pool(weight=sw) real=4.0, synth=1.0
  - FT-T OOF folds       : WeightedRandomSampler real=4.0, synth=1.0
  - Meta-learner fit      : sample_weight=sw (real=4.0, synth=1.0)
  This ensures real patient signal dominates at every layer, matching
  Exp D philosophy end-to-end. Justifiable in paper as consistent pipeline.

Research question:
  Does a stacking layer on top of weighted-hybrid base models close the
  remaining DELTA-AUPRC gap vs real-only Exp A?

DeLong tests:
  - Ensemble vs Exp D CatBoost (did stacking improve over best hybrid CB?)
  - Ensemble vs Hybrid FT-T (did stacking improve over hybrid FT-T?)
  - Ensemble vs Exp A real-only (key scientific question — gap closed?)

Mac (Apple M1 Pro) notes:
  - CatBoost falls back to CPU on MPS (expected)
  - FT-T uses MPS if available, else CPU
  - num_workers=0 in all DataLoaders

Windows (Uzair — RTX 4500 Ada) notes:
  - multiprocessing.freeze_support() at top
  - num_workers=0 in all DataLoaders
  - CUDA 12.1 build required
  - Use ASCII (-> not unicode arrows) if cp1252 console errors

Folder outputs (all under sepsis_ml/synthetic/stacking/):
  models/   — stacking_meta_learner.pkl
              stacking_meta_coefficients.json
  results/  — stacking_oof_probs.csv
              stacking_test_probs.csv
              stacking_predictions.npz
              stacking_metrics.json
              stacking_delong_results.json
              stacking_threshold_table.csv
  figures/  — stacking_roc_pr.png/.pdf
              stacking_calibration.png/.pdf
              stacking_confusion_matrix.png/.pdf
              stacking_threshold_sensitivity.png/.pdf
              stacking_experiment_comparison.png/.pdf
  logs/     — synthetic_stacking.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/synthetic/stacking/synthetic_stacking.py
"""

# ── Windows multiprocessing guard ────────────────────────────────────────────
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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from catboost import CatBoostClassifier, Pool
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

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/stacking/
SYNTHETIC    = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC.parent                         # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

# ── Input data ────────────────────────────────────────────────────────────────
MODEL_DATA_DIR  = PROJECT_ROOT / "model_datasets"
REAL_TRAIN_FILE = MODEL_DATA_DIR / "B_train_model_ready.csv"
REAL_TEST_FILE  = MODEL_DATA_DIR / "B_test_model_ready.csv"
HYBRID_FILE     = MODEL_DATA_DIR / "synthetic" / "C_hybrid_train.csv"

# ── Prerequisites — must exist before running ─────────────────────────────────
WHYBRID_CB_PROBS_PATH  = SYNTHETIC / "whybrid" / "results" / "whybrid_catboost_test_probs.npy"
WHYBRID_CB_PARAMS_PATH = SYNTHETIC / "whybrid" / "results" / "whybrid_best_params.json"
HYBRID_FTT_PROBS_PATH  = SYNTHETIC / "ft" / "results" / "hybrid_ftt_predictions.npz"
HYBRID_FTT_PARAMS_PATH = SYNTHETIC / "ft" / "models" / "hybrid_ftt_best_params_final.json"
HYBRID_FTT_SCALER_PATH = SYNTHETIC / "ft" / "models" / "hybrid_ftt_scaler.pkl"
EXPA_CB_PROBS_PATH     = SEPSIS_ML / "results" / "catboost_test_probs.npy"

# ── Output folders ────────────────────────────────────────────────────────────
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
LOGS_DIR    = SCRIPT_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "synthetic_stacking.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED        = 42
CV_FOLDS           = 5
TARGET_COL         = "sepsis_label"
TARGET_SENSITIVITY = 0.90
REAL_SAMPLE_WEIGHT  = 4.0
SYNTH_SAMPLE_WEIGHT = 1.0

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    DEVICE   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device : CUDA - {gpu_name} ({vram_gb:.1f} GB VRAM)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    log.info("Device : Apple MPS (M1/M2)")
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
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(y_true[idx]):
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def delong_test(y_true, probs_a, probs_b):
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
    m   = int(yt.sum()); n = len(yt) - m
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
    return FTTransformer(
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


def make_loader_simple(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def make_loader_weighted(X, y, weights, batch_size):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(weights, dtype=torch.float32),
        num_samples = len(ds),
        replacement = True,
    )
    return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                      num_workers=0, pin_memory=False)


def train_epoch_ftt(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0; n_samples = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_b, None).squeeze(-1)
        loss   = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b); n_samples += len(y_b)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def get_probs_ftt(model, loader):
    model.eval()
    all_probs = []
    for X_b, _ in loader:
        logits = model(X_b.to(DEVICE), None).squeeze(-1)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(all_probs)


def train_ftt_fold(X_tr_s, y_tr, sw_tr, X_val_s, y_val,
                   params, patience=15, max_epochs=100):
    """
    Train one FT-T CV fold.
    Train loader: WeightedRandomSampler — real rows 4x influence.
    Val loader  : unweighted — clean unbiased AUPRC signal.
    """
    model     = build_fttransformer(params, X_tr_s.shape[1]).to(DEVICE)
    pw        = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32).to(DEVICE)
    )
    optimizer = optim.AdamW(model.parameters(),
                            lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    tr_loader  = make_loader_weighted(X_tr_s, y_tr.astype(np.float32),
                                      sw_tr, params["batch_size"])
    val_loader = make_loader_simple(X_val_s, y_val.astype(np.float32),
                                    params["batch_size"], shuffle=False)
    best_auprc = 0.0; best_weights = None; no_improve = 0

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
    log.info("STACKING ENSEMBLE — WEIGHTED HYBRID DATA")
    log.info("Base models: Exp D CatBoost + Weighted Hybrid FT-Transformer")
    log.info("Meta-learner: Logistic Regression (weighted fit)")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── 5.1 Verify prerequisites ──────────────────────────────────────────────
    log.info("\nChecking prerequisites...")
    for path, name in [
        (WHYBRID_CB_PROBS_PATH,  "Exp D CatBoost test probs"),
        (WHYBRID_CB_PARAMS_PATH, "Exp D CatBoost params"),
        (HYBRID_FTT_PROBS_PATH,  "Hybrid FT-T test probs"),
        (HYBRID_FTT_PARAMS_PATH, "Hybrid FT-T params"),
        (HYBRID_FTT_SCALER_PATH, "Hybrid FT-T scaler"),
    ]:
        if not path.exists():
            log.error(f"  MISSING: {name} -> {path}")
            log.error("  Run weighted_hybrid_catboost.py and hybrid_fttransformer.py first.")
            sys.exit(1)
        log.info(f"  OK : {name}")

    # ── 5.2 Load data ─────────────────────────────────────────────────────────
    log.info("\nLoading datasets...")
    hybrid_df     = pd.read_csv(HYBRID_FILE)
    real_train_df = pd.read_csv(REAL_TRAIN_FILE)
    real_test_df  = pd.read_csv(REAL_TEST_FILE)
    feature_cols  = [c for c in hybrid_df.columns if c != TARGET_COL]

    X_hybrid_df = hybrid_df[feature_cols]
    y_hybrid    = hybrid_df[TARGET_COL].values
    X_test_df   = real_test_df[feature_cols]
    y_test      = real_test_df[TARGET_COL].values

    n_real  = len(real_train_df)
    n_total = len(hybrid_df)
    n_synth = n_total - n_real

    log.info(f"  Hybrid train : {X_hybrid_df.shape} | Sepsis: {y_hybrid.mean():.1%}")
    log.info(f"  Real rows : {n_real} | Synthetic rows : {n_synth}")
    log.info(f"  Real test : {X_test_df.shape}  | Sepsis: {y_test.mean():.1%}")

    # ── 5.3 Sample weight vector ──────────────────────────────────────────────
    sample_weights          = np.ones(n_total, dtype=np.float32)
    sample_weights[:n_real] = REAL_SAMPLE_WEIGHT
    sample_weights[n_real:] = SYNTH_SAMPLE_WEIGHT

    real_inf = n_real * REAL_SAMPLE_WEIGHT
    pct_real = 100 * real_inf / (real_inf + n_synth * SYNTH_SAMPLE_WEIGHT)
    log.info(f"\n  Weighting : real={REAL_SAMPLE_WEIGHT}, synthetic={SYNTH_SAMPLE_WEIGHT}")
    log.info(f"  Effective influence : real={pct_real:.1f}% | synthetic={100-pct_real:.1f}%")
    log.info(f"  Applied at : CatBoost Pool, FT-T WeightedRandomSampler, LR fit")

    # ── 5.4 Load params ───────────────────────────────────────────────────────
    log.info("\nLoading params and scaler...")
    with open(WHYBRID_CB_PARAMS_PATH) as f:
        cb_best = json.load(f)["best_params"]
    with open(HYBRID_FTT_PARAMS_PATH) as f:
        ftt_params = json.load(f)
    with open(HYBRID_FTT_SCALER_PATH, "rb") as f:
        ftt_scaler = pickle.load(f)
    log.info(f"  CatBoost params : {cb_best}")
    log.info(f"  FT-T params     : {ftt_params}")

    binary_cols = [c for c in feature_cols
                   if set(X_hybrid_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]

    def scale_for_ftt(df):
        d = df.copy()
        d[num_cols] = ftt_scaler.transform(d[num_cols])
        return d.values.astype(np.float32)

    X_hybrid_scaled = scale_for_ftt(X_hybrid_df)
    X_test_scaled   = scale_for_ftt(X_test_df)

    # ── 5.5 Load saved test probs ─────────────────────────────────────────────
    log.info("\nLoading saved test-set probabilities...")
    cb_test_probs  = np.load(WHYBRID_CB_PROBS_PATH)
    ftt_test_probs = np.load(HYBRID_FTT_PROBS_PATH)["test_probs"]

    assert len(cb_test_probs)  == len(y_test), "CatBoost test probs length mismatch"
    assert len(ftt_test_probs) == len(y_test), "FT-T test probs length mismatch"

    log.info(f"  Exp D CatBoost AUROC : {roc_auc_score(y_test, cb_test_probs):.4f}")
    log.info(f"  Hybrid FT-T AUROC    : {roc_auc_score(y_test, ftt_test_probs):.4f}")

    real_cb_test_probs = None
    if EXPA_CB_PROBS_PATH.exists():
        real_cb_test_probs = np.load(EXPA_CB_PROBS_PATH)
        log.info(f"  Exp A real-only AUROC: {roc_auc_score(y_test, real_cb_test_probs):.4f}")
    else:
        log.warning("  Exp A probs not found — 4-way DeLong will be skipped.")

    # ══════════════════════════════════════════════════════════════════════════
    # 5.6 OUT-OF-FOLD PROBABILITIES
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Generating out-of-fold probabilities (5-fold CV)")
    log.info("=" * 60)
    log.info("Train folds: weighted. Val fold: unweighted (clean AUPRC signal).")

    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_cb  = np.zeros(n_total)
    oof_ftt = np.zeros(n_total)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_hybrid_df, y_hybrid)):
        fold_start = time.time()
        log.info(f"\n  Fold {fold_idx + 1}/{CV_FOLDS} " + "-" * 38)

        X_tr_df = X_hybrid_df.iloc[train_idx]
        X_val_df = X_hybrid_df.iloc[val_idx]
        y_tr    = y_hybrid[train_idx]
        y_val   = y_hybrid[val_idx]
        sw_tr   = sample_weights[train_idx]

        log.info(f"    Train: {len(y_tr)} "
                 f"(real={(train_idx < n_real).sum()}, "
                 f"synth={(train_idx >= n_real).sum()}) | "
                 f"Val: {len(y_val)} | Sepsis: {y_val.mean():.1%}")

        # CatBoost fold
        log.info("    [CatBoost] Exp D params + Pool(weight=sw_tr)...")
        train_pool = Pool(X_tr_df, y_tr, weight=sw_tr)
        val_pool   = Pool(X_val_df, y_val)
        cb_fold = CatBoostClassifier(
            iterations            = cb_best["iterations"],
            learning_rate         = cb_best["learning_rate"],
            depth                 = cb_best["depth"],
            l2_leaf_reg           = cb_best["l2_leaf_reg"],
            bagging_temperature   = cb_best["bagging_temperature"],
            random_strength       = cb_best["random_strength"],
            border_count          = cb_best["border_count"],
            class_weights         = [1.0, cb_best["class_weight_pos"]],
            eval_metric           = "AUC",
            random_seed           = RANDOM_SEED,
            verbose               = 0,
            early_stopping_rounds = 50,
        )
        cb_fold.fit(train_pool, eval_set=val_pool, verbose=0)
        oof_cb[val_idx] = cb_fold.predict_proba(X_val_df)[:, 1]
        log.info(f"    CatBoost val AUROC : {roc_auc_score(y_val, oof_cb[val_idx]):.4f}")

        # FT-T fold
        log.info("    [FT-T] Hybrid params + WeightedRandomSampler...")
        oof_ftt[val_idx] = train_ftt_fold(
            X_hybrid_scaled[train_idx], y_tr, sw_tr,
            X_hybrid_scaled[val_idx],   y_val,
            ftt_params, patience=15, max_epochs=100,
        )
        log.info(f"    FT-T val AUROC     : {roc_auc_score(y_val, oof_ftt[val_idx]):.4f}")
        log.info(f"    Fold time          : {(time.time()-fold_start)/60:.1f} min")

    oof_cb_auroc  = roc_auc_score(y_hybrid, oof_cb)
    oof_ftt_auroc = roc_auc_score(y_hybrid, oof_ftt)
    log.info(f"\n  OOF AUROC - CatBoost : {oof_cb_auroc:.4f}")
    log.info(f"  OOF AUROC - FT-T     : {oof_ftt_auroc:.4f}")
    log.info("  NOTE: OOF AUROC on hybrid data is inflated. Not for clinical claims.")

    oof_df = pd.DataFrame({
        "y_true"           : y_hybrid,
        "is_real"          : (np.arange(n_total) < n_real).astype(int),
        "sample_weight"    : sample_weights,
        "oof_catboost"     : oof_cb,
        "oof_fttransformer": oof_ftt,
    })
    oof_df.to_csv(RESULTS_DIR / "stacking_oof_probs.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # 5.7 TRAIN META-LEARNER (weighted fit)
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Training meta-learner (weighted LR fit)")
    log.info("=" * 60)

    X_meta_train = np.column_stack([oof_cb, oof_ftt])
    meta_learner = LogisticRegression(
        C=1.0, random_state=RANDOM_SEED, max_iter=1000, solver="lbfgs"
    )
    # sample_weight ensures real rows drive meta-learner coefficients 4x more
    meta_learner.fit(X_meta_train, y_hybrid, sample_weight=sample_weights)

    coef = meta_learner.coef_[0]
    log.info(f"  CatBoost coefficient : {coef[0]:.4f}")
    log.info(f"  FT-T coefficient     : {coef[1]:.4f}")
    log.info(f"  Intercept            : {meta_learner.intercept_[0]:.4f}")

    with open(MODELS_DIR / "stacking_meta_learner.pkl", "wb") as f:
        pickle.dump(meta_learner, f)

    coef_dict = {
        "catboost_coefficient"     : float(coef[0]),
        "fttransformer_coefficient": float(coef[1]),
        "intercept"                : float(meta_learner.intercept_[0]),
        "meta_learner_weighted"    : True,
        "real_weight"              : REAL_SAMPLE_WEIGHT,
        "synth_weight"             : SYNTH_SAMPLE_WEIGHT,
        "note": "Weighted fit: real rows 4x influence. Consistent with full pipeline.",
    }
    with open(RESULTS_DIR / "stacking_meta_coefficients.json", "w") as f:
        json.dump(coef_dict, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # 5.8 TEST SET EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Evaluating ensemble on real test set (n=633)")
    log.info("=" * 60)

    X_meta_test    = np.column_stack([cb_test_probs, ftt_test_probs])
    ensemble_probs = meta_learner.predict_proba(X_meta_test)[:, 1]

    pd.DataFrame({
        "y_true": y_test, "catboost_prob": cb_test_probs,
        "fttransformer_prob": ftt_test_probs, "ensemble_prob": ensemble_probs,
    }).to_csv(RESULTS_DIR / "stacking_test_probs.csv", index=False)

    np.savez(RESULTS_DIR / "stacking_predictions.npz",
             ensemble_probs=ensemble_probs, catboost_probs=cb_test_probs,
             ftt_probs=ftt_test_probs, y_test=y_test)

    metrics            = compute_metrics(y_test, ensemble_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, ensemble_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, ensemble_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    cb_auroc  = roc_auc_score(y_test, cb_test_probs)
    ftt_auroc = roc_auc_score(y_test, ftt_test_probs)

    log.info(f"\n  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}-{auprc_hi:.4f}]")
    log.info(f"  Brier       : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}")
    log.info(f"  NPV         : {metrics['npv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")

    log.info(f"\n  Exp D CatBoost AUROC     : {cb_auroc:.4f}")
    log.info(f"  Hybrid FT-T AUROC        : {ftt_auroc:.4f}")
    log.info(f"  Stacking Ensemble AUROC  : {metrics['auroc']:.4f}")
    if real_cb_test_probs is not None:
        real_auroc = roc_auc_score(y_test, real_cb_test_probs)
        real_auprc = average_precision_score(y_test, real_cb_test_probs)
        log.info(f"  Exp A Real-only AUROC    : {real_auroc:.4f}")
        log.info(f"  DELTA AUROC vs real-only : {metrics['auroc'] - real_auroc:+.4f}")
        log.info(f"  DELTA AUPRC vs real-only : {metrics['auprc'] - real_auprc:+.4f}")

    # DeLong
    delong_results = {}
    a1, b1, z1, p1 = delong_test(y_test, ensemble_probs, cb_test_probs)
    log.info(f"\n  DeLong vs Exp D CatBoost  : z={z1:.4f}, p={p1:.4f} "
             f"({'sig' if p1 < 0.05 else 'not sig'})")
    delong_results["vs_expD_catboost"] = {
        "ensemble_auroc": a1, "expD_auroc": b1, "z": z1, "p": p1,
        "significant": p1 < 0.05,
        "direction": "Ensemble better" if a1 > b1 else "Exp D CatBoost better",
    }

    a2, b2, z2, p2 = delong_test(y_test, ensemble_probs, ftt_test_probs)
    log.info(f"  DeLong vs Hybrid FT-T     : z={z2:.4f}, p={p2:.4f} "
             f"({'sig' if p2 < 0.05 else 'not sig'})")
    delong_results["vs_hybrid_ftt"] = {
        "ensemble_auroc": a2, "hybrid_ftt_auroc": b2, "z": z2, "p": p2,
        "significant": p2 < 0.05,
        "direction": "Ensemble better" if a2 > b2 else "Hybrid FT-T better",
    }

    if real_cb_test_probs is not None:
        a3, b3, z3, p3 = delong_test(y_test, ensemble_probs, real_cb_test_probs)
        log.info(f"  DeLong vs Exp A real-only : z={z3:.4f}, p={p3:.4f} "
                 f"({'sig' if p3 < 0.05 else 'not sig'})")
        delong_results["vs_expA_real_only"] = {
            "ensemble_auroc": a3, "expA_auroc": b3, "z": z3, "p": p3,
            "significant": p3 < 0.05,
            "direction": "Ensemble better" if a3 > b3 else "Real-only better",
            "delta_auroc_vs_real": float(metrics["auroc"] - b3),
            "delta_auprc_vs_real": float(
                metrics["auprc"] - average_precision_score(y_test, real_cb_test_probs)
            ),
        }

    with open(RESULTS_DIR / "stacking_delong_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)

    # Threshold table
    thresh_rows = []
    log.info(f"\n  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7}")
    log.info("  " + "-" * 60)
    for target in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]:
        thr = find_threshold_at_sensitivity(y_test, ensemble_probs, target)
        m   = compute_metrics(y_test, ensemble_probs, thr)
        thresh_rows.append({
            "target_sensitivity": target, "threshold": round(thr, 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "ppv": round(m["ppv"], 4), "npv": round(m["npv"], 4),
            "f1": round(m["f1"], 4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
        log.info(f"  {target:>8.2f} {thr:>8.4f} {m['sensitivity']:>7.4f} "
                 f"{m['specificity']:>7.4f} {m['ppv']:>7.4f} "
                 f"{m['npv']:>7.4f} {m['f1']:>7.4f}")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "stacking_threshold_table.csv", index=False)

    runtime = (time.time() - t_start) / 60
    full_metrics = {
        "model"                   : "Stacking Ensemble (Weighted Hybrid)",
        "experiment"              : "Synthetic Stacking",
        "timestamp"               : datetime.now().isoformat(),
        "dataset"                 : "Hybrid train (12,530) | Real test (633)",
        "n_hybrid_train"          : int(n_total),
        "n_real_train"            : int(n_real),
        "n_synthetic_train"       : int(n_synth),
        "n_test"                  : int(len(y_test)),
        "n_features"              : len(feature_cols),
        "cv_folds"                : CV_FOLDS,
        "real_sample_weight"      : REAL_SAMPLE_WEIGHT,
        "synth_sample_weight"     : SYNTH_SAMPLE_WEIGHT,
        "real_influence_pct"      : float(pct_real),
        "weighting_applied_at"    : ["CatBoost Pool", "FT-T WeightedRandomSampler",
                                     "LogisticRegression sample_weight"],
        "oof_auroc_catboost"      : float(oof_cb_auroc),
        "oof_auroc_fttransformer" : float(oof_ftt_auroc),
        "oof_note"                : "OOF on hybrid data is inflated — not for clinical claims",
        "test_metrics"            : metrics,
        "meta_learner"            : coef_dict,
        "delong"                  : delong_results,
        "runtime_minutes"         : float(runtime),
    }
    with open(RESULTS_DIR / "stacking_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # 5.9 FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nGenerating figures...")
    COLORS = {
        "ensemble": "#E53935", "expD_cb": "#1565C0",
        "ftt": "#6A1B9A",      "expA": "#2E7D32", "random": "#9E9E9E",
    }

    plot_models = [
        (ensemble_probs,  "Ensemble",       COLORS["ensemble"], 2.5, "-"),
        (cb_test_probs,   "Exp D CatBoost", COLORS["expD_cb"],  1.5, "--"),
        (ftt_test_probs,  "Hybrid FT-T",    COLORS["ftt"],      1.5, "-."),
    ]
    if real_cb_test_probs is not None:
        plot_models.append(
            (real_cb_test_probs, "Exp A Real-only", COLORS["expA"], 1.5, ":")
        )

    # Figure 1: ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Weighted Hybrid Stacking Ensemble vs Base Models\n"
        f"Real Test Set (n={len(y_test)}, prevalence={y_test.mean():.1%})",
        fontsize=13, fontweight="bold"
    )
    for probs, name, color, lw, ls in plot_models:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls,
                     label=f"{name}  AUROC={roc_auc_score(y_test, probs):.4f}")
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls,
                     label=f"{name}  AUPRC={average_precision_score(y_test, probs):.4f}")
    axes[0].plot([0,1],[0,1],"--",color=COLORS["random"],lw=0.8,alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, ensemble_probs)[:2],
                         alpha=0.06, color=COLORS["ensemble"])
    for ax, xlabel, ylabel, title, loc in [
        (axes[0], "False Positive Rate", "True Positive Rate", "ROC Curve", "lower right"),
        (axes[1], "Recall", "Precision", "Precision-Recall Curve", "upper right"),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc=loc, fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
    axes[1].axhline(y_test.mean(), color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({y_test.mean():.2f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stacking_roc_pr.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> stacking_roc_pr.png")

    # Figure 2: Calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    for probs, name, color, *_ in plot_models:
        pt, pp = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(pp, pt, "o-", color=color, lw=2,
                label=f"{name} (Brier={brier_score_loss(y_test, probs):.4f})")
    ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_calibration.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stacking_calibration.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> stacking_calibration.png")

    # Figure 3: Confusion matrix
    cm  = confusion_matrix(y_test, (ensemble_probs >= metrics["threshold"]).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"])
    ax.set_yticklabels(["Non-sepsis","Sepsis"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Stacking Ensemble (Weighted Hybrid)\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stacking_confusion_matrix.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> stacking_confusion_matrix.png")

    # Figure 4: Threshold sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Weighted Hybrid Stacking — Threshold Sensitivity Analysis",
                 fontsize=12, fontweight="bold")
    thr_vals = thresh_df["threshold"].values
    axes[0].plot(thr_vals, thresh_df["sensitivity"].values, "o-",
                 color=COLORS["ensemble"], lw=2, label="Sensitivity")
    axes[0].plot(thr_vals, thresh_df["specificity"].values, "s-",
                 color=COLORS["expD_cb"], lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(thr_vals, thresh_df["ppv"].values, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, thresh_df["npv"].values, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top","right"]].set_visible(False)
    axes[2].plot(thr_vals, thresh_df["f1"].values, "o-", color=COLORS["ftt"], lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].grid(True, alpha=0.3); axes[2].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_threshold_sensitivity.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stacking_threshold_sensitivity.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> stacking_threshold_sensitivity.png")

    # Figure 5: Experiment comparison bar chart
    exp_names  = ["Exp D\nWeighted\nHybrid CB", "Hybrid\nFT-T", "Stacking\nEnsemble"]
    auroc_vals = [cb_auroc, ftt_auroc, metrics["auroc"]]
    auprc_vals = [average_precision_score(y_test, cb_test_probs),
                  average_precision_score(y_test, ftt_test_probs), metrics["auprc"]]
    bar_colors = [COLORS["expD_cb"], COLORS["ftt"], COLORS["ensemble"]]
    if real_cb_test_probs is not None:
        exp_names  = ["Exp A\nReal-only"] + exp_names
        auroc_vals = [roc_auc_score(y_test, real_cb_test_probs)] + auroc_vals
        auprc_vals = [average_precision_score(y_test, real_cb_test_probs)] + auprc_vals
        bar_colors = [COLORS["expA"]] + bar_colors

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Experiment Comparison — All Hybrid Models on Real Test Set (n=633)\n"
        "Weighting: real=4.0, synthetic=1.0 applied consistently",
        fontsize=12, fontweight="bold"
    )
    x = np.arange(len(exp_names))
    for ax_i, (vals, ylabel, title, ylim) in enumerate([
        (auroc_vals, "AUROC", "AUROC Comparison", (0.85, 1.0)),
        (auprc_vals, "AUPRC", "AUPRC Comparison", (0.80, 1.0)),
    ]):
        bars = axes[ax_i].bar(x, vals, color=bar_colors, alpha=0.85, width=0.55)
        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(exp_names, fontsize=9)
        axes[ax_i].set_ylabel(ylabel); axes[ax_i].set_title(title, fontweight="bold")
        axes[ax_i].set_ylim(ylim)
        axes[ax_i].spines[["top","right"]].set_visible(False)
        axes[ax_i].grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            axes[ax_i].text(bar.get_x() + bar.get_width()/2, val + 0.001,
                            f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stacking_experiment_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "stacking_experiment_comparison.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved -> stacking_experiment_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    runtime = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("STACKING ENSEMBLE (WEIGHTED HYBRID) — COMPLETE")
    log.info("=" * 70)
    log.info(f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}-{auroc_hi:.4f}]")
    log.info(f"  Brier       : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"\n  Meta-learner : CatBoost={coef[0]:.4f} | FT-T={coef[1]:.4f}")
    log.info(f"  DeLong vs Exp D : z={z1:.4f}, p={p1:.4f}")
    log.info(f"  DeLong vs FT-T  : z={z2:.4f}, p={p2:.4f}")
    if real_cb_test_probs is not None:
        log.info(f"  DeLong vs Exp A : z={z3:.4f}, p={p3:.4f}")
        log.info(f"  DELTA AUPRC vs real-only : "
                 f"{metrics['auprc'] - average_precision_score(y_test, real_cb_test_probs):+.4f}")
    log.info(f"\n  Runtime  : {runtime:.1f} min")
    log.info(f"  Models   -> {MODELS_DIR}")
    log.info(f"  Results  -> {RESULTS_DIR}")
    log.info(f"  Figures  -> {FIGURES_DIR}")
    log.info(f"  Log      -> {log_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
