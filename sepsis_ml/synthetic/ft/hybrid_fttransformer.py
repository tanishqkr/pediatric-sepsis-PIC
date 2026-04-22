"""
hybrid_fttransformer.py
─────────────────────────────────────────────────────────────────────────────
Weighted Hybrid FT-Transformer
Trains FT-Transformer on hybrid data (2,530 real + 10,000 synthetic = 12,530)
with real rows upweighted via WeightedRandomSampler to match Exp D CatBoost
weighting philosophy (real=4.0, synthetic=1.0).

This is the direct hybrid-data equivalent of Phase 6 (phase6_fttransformer.py)
which trained on real-only Option B data. Architecture, Optuna search space,
evaluation suite, and figure set are identical to Phase 6.

Weighting:
  WeightedRandomSampler assigns real rows 4x higher sampling probability
  per epoch than synthetic rows. This is the PyTorch-native equivalent of
  CatBoost sample_weight=4.0 — same gradient rebalancing, zero data
  duplication. Consistent with Exp D CatBoost weighting throughout.

  Real influence  : 2,530 x 4.0 = 10,120
  Synth influence : 10,000 x 1.0 = 10,000
  Real pct        : ~50.3% of total gradient signal

Key design decisions (mirrors Phase 6 exactly):
  - Optuna 100 trials, AUPRC-optimised, TPE sampler + MedianPruner
  - 80/20 stratified split of hybrid data for Optuna val (unweighted val)
  - WeightedRandomSampler on training split only — val always unweighted
  - Refit StandardScaler on hybrid training set (not real-only scaler)
  - pos_weight in BCEWithLogitsLoss computed from training fold class counts
  - Final model retrained on full hybrid set, val split for early stopping
  - Test set = real held-out 633 patients (never touched, never synthesised)
  - DeLong vs Phase 6 real-only FT-T and vs Exp D CatBoost
  - Saves test_probs .npz for downstream stacking script

Mac (Apple M1 Pro) notes:
  - Uses MPS if available, else CPU
  - num_workers=0 in all DataLoaders

Windows (Uzair — RTX 4500 Ada) notes:
  - multiprocessing.freeze_support() called at top
  - num_workers=0 in all DataLoaders
  - CUDA 12.1 build of PyTorch required
  - Use ASCII (-> not ->) if cp1252 errors in console

Folder outputs (all under sepsis_ml/synthetic/ft/):
  models/   — hybrid_ftt_best.pt
              hybrid_ftt_scaler.pkl
              hybrid_ftt_best_params.json
              hybrid_ftt_best_params_final.json
              hybrid_ftt_optuna_study.pkl
  results/  — hybrid_ftt_results.json
              hybrid_ftt_delong.json
              hybrid_ftt_threshold_table.csv
              hybrid_ftt_predictions.npz        <-- loaded by stacking script
  figures/  — hybrid_ftt_optuna_history.png/.pdf
              hybrid_ftt_roc_pr.png/.pdf
              hybrid_ftt_calibration.png/.pdf
              hybrid_ftt_threshold_sensitivity.png/.pdf
  logs/     — hybrid_fttransformer.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/synthetic/ft/hybrid_fttransformer.py
"""

# ── Windows multiprocessing guard ────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
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
import optuna
from optuna.samplers import TPESampler

from rtdl_revisiting_models import FTTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/ft/
SYNTHETIC    = SCRIPT_DIR.parent                        # sepsis_ml/synthetic/
SEPSIS_ML    = SYNTHETIC.parent                         # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

# ── Input data ────────────────────────────────────────────────────────────────
MODEL_DATA_DIR  = PROJECT_ROOT / "model_datasets"
REAL_TRAIN_FILE = MODEL_DATA_DIR / "B_train_model_ready.csv"
REAL_TEST_FILE  = MODEL_DATA_DIR / "B_test_model_ready.csv"
HYBRID_FILE     = MODEL_DATA_DIR / "synthetic" / "C_hybrid_train.csv"

# ── Comparison artifacts (for DeLong) ────────────────────────────────────────
# Phase 6 real-only FT-T test probs
PHASE6_FTT_PROBS_PATH = SEPSIS_ML / "dl" / "results" / "phase6_fttransformer_predictions.npz"
# Exp D weighted hybrid CatBoost test probs
EXPD_CB_PROBS_PATH    = SYNTHETIC / "whybrid" / "results" / "whybrid_catboost_test_probs.npy"
# Exp A real-only CatBoost test probs
EXPA_CB_PROBS_PATH    = SEPSIS_ML / "results" / "catboost_test_probs.npy"

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

log_path = LOGS_DIR / "hybrid_fttransformer.log"
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

RANDOM_SEED         = 42
TARGET_COL          = "sepsis_label"
TARGET_SENSITIVITY  = 0.90
N_OPTUNA_TRIALS     = 30
N_EPOCHS_MAX        = 50
PATIENCE            = 15
PATIENCE_FINAL      = 20
N_EPOCHS_FINAL      = 150

# Weighting — matches Exp D CatBoost
REAL_SAMPLE_WEIGHT  = 4.0
SYNTH_SAMPLE_WEIGHT = 1.0

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Device ────────────────────────────────────────────────────────────────────
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
# 3. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

log.info("=" * 70)
log.info("WEIGHTED HYBRID FT-TRANSFORMER")
log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

log.info("\nLoading datasets...")
hybrid_df     = pd.read_csv(HYBRID_FILE)
real_train_df = pd.read_csv(REAL_TRAIN_FILE)
test_df       = pd.read_csv(REAL_TEST_FILE)

TARGET       = "sepsis_label"
feature_cols = [c for c in hybrid_df.columns if c != TARGET]

X_hybrid = hybrid_df[feature_cols].copy()
y_hybrid = hybrid_df[TARGET].values.astype(np.float32)
X_test   = test_df[feature_cols].copy()
y_test   = test_df[TARGET].values.astype(np.float32)

n_real  = len(real_train_df)          # 2,530 — real rows are first in hybrid
n_total = len(hybrid_df)              # 12,530
n_synth = n_total - n_real            # 10,000

log.info(f"  Hybrid train : {X_hybrid.shape} | Sepsis: {y_hybrid.mean():.1%}")
log.info(f"  Real rows    : {n_real} (rows 0..{n_real-1})")
log.info(f"  Synthetic    : {n_synth} (rows {n_real}..{n_total-1})")
log.info(f"  Real test    : {X_test.shape}  | Sepsis: {y_test.mean():.1%}")
log.info(f"  Features     : {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SAMPLE WEIGHT VECTOR
# ══════════════════════════════════════════════════════════════════════════════

sample_weights          = np.ones(n_total, dtype=np.float32)
sample_weights[:n_real] = REAL_SAMPLE_WEIGHT
sample_weights[n_real:] = SYNTH_SAMPLE_WEIGHT

real_inf  = n_real  * REAL_SAMPLE_WEIGHT
synth_inf = n_synth * SYNTH_SAMPLE_WEIGHT
pct_real  = 100 * real_inf / (real_inf + synth_inf)
log.info(f"\n  Weighting : real={REAL_SAMPLE_WEIGHT}, synthetic={SYNTH_SAMPLE_WEIGHT}")
log.info(f"  Effective influence : real={pct_real:.1f}% | synthetic={100-pct_real:.1f}%")
log.info(f"  FT-T method : WeightedRandomSampler (no data duplication)")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL SPLIT (for Optuna + final model early stopping)
# ══════════════════════════════════════════════════════════════════════════════

tr_idx, val_idx = train_test_split(
    np.arange(n_total), test_size=0.2,
    stratify=y_hybrid, random_state=RANDOM_SEED
)

X_tr_df  = X_hybrid.iloc[tr_idx]
X_val_df = X_hybrid.iloc[val_idx]
y_tr     = y_hybrid[tr_idx]
y_val    = y_hybrid[val_idx]
sw_tr    = sample_weights[tr_idx]    # weights for training split only

log.info(f"\n  Train split : {len(y_tr)} | Val split: {len(y_val)}")
log.info(f"  Real in train split  : {(tr_idx < n_real).sum()}")
log.info(f"  Synth in train split : {(tr_idx >= n_real).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SCALING
# Refit scaler on hybrid training split (not real-only scaler from Phase 6)
# Hybrid distribution differs — refitting is correct
# ══════════════════════════════════════════════════════════════════════════════

binary_cols = [c for c in feature_cols
               if set(X_hybrid[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
num_cols    = [c for c in feature_cols if c not in binary_cols]

log.info(f"\n  Numerical features : {len(num_cols)}")
log.info(f"  Binary features    : {len(binary_cols)}")
log.info("  Fitting scaler on hybrid training split...")

scaler = StandardScaler()
scaler.fit(X_tr_df[num_cols])

with open(MODELS_DIR / "hybrid_ftt_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
log.info("  Scaler saved.")


def scale_df(df):
    d = df.copy()
    d[num_cols] = scaler.transform(d[num_cols])
    return d.values.astype(np.float32)


X_tr_s  = scale_df(X_tr_df)
X_val_s = scale_df(X_val_df)
X_te_s  = scale_df(X_test)

N_FEATURES = X_tr_s.shape[1]

# ══════════════════════════════════════════════════════════════════════════════
# 7. DATALOADER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_loader_simple(X, y, batch_size, shuffle=True):
    """Standard loader — used for val and test (never weighted)."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def make_loader_weighted(X, y, weights, batch_size):
    """
    WeightedRandomSampler loader.
    Real rows sampled ~4x more often per epoch than synthetic rows.
    Replacement=True ensures every epoch draws len(dataset) samples.
    Applied to training split only — val is always unweighted.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_fttransformer(params):
    d_block = params["d_block"]
    n_heads = params["attention_n_heads"]
    while d_block % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1:
            n_heads = 1
            break
    model = FTTransformer(
        n_cont_features         = N_FEATURES,
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
    return model, n_heads


# ══════════════════════════════════════════════════════════════════════════════
# 9. TRAIN / INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_samples  = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_b, None).squeeze(-1)
        loss   = criterion(logits, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        n_samples  += len(y_b)
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs = []
    for X_b, _ in loader:
        X_b    = X_b.to(DEVICE)
        logits = model(X_b, None).squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ══════════════════════════════════════════════════════════════════════════════
# 10. OPTUNA OBJECTIVE
# Mirrors Phase 6 objective exactly — same search space, same early stopping
# Only difference: WeightedRandomSampler on training loader
# Val loader always unweighted for unbiased AUPRC optimisation signal
# ══════════════════════════════════════════════════════════════════════════════

# pos_weight from training split class counts
pos_weight_val = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1.0)
POS_WEIGHT     = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
log.info(f"\n  pos_weight (training split) : {pos_weight_val:.3f}")

val_loader_optuna = make_loader_simple(X_val_s, y_val, batch_size=256, shuffle=False)


def objective(trial):
    params = {
        "n_blocks"              : trial.suggest_int("n_blocks", 1, 4),
        "d_block"               : trial.suggest_categorical("d_block", [64, 128, 192, 256]),
        "attention_n_heads"     : trial.suggest_categorical("attention_n_heads", [4, 8]),
        "attention_dropout"     : trial.suggest_float("attention_dropout", 0.0, 0.4),
        "ffn_d_hidden_multiplier": trial.suggest_float("ffn_d_hidden_multiplier", 1.0, 4.0),
        "ffn_dropout"           : trial.suggest_float("ffn_dropout", 0.0, 0.4),
        "residual_dropout"      : trial.suggest_float("residual_dropout", 0.0, 0.2),
        "lr"                    : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay"          : trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size"            : trial.suggest_categorical("batch_size", [64, 128, 256]),
    }

    try:
        model, _ = build_fttransformer(params)
        model    = model.to(DEVICE)
    except Exception as e:
        log.warning(f"Trial {trial.number} build failed: {e}")
        raise optuna.exceptions.TrialPruned()

    optimizer  = optim.AdamW(model.parameters(),
                             lr=params["lr"],
                             weight_decay=params["weight_decay"])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_MAX, eta_min=1e-6
    )

    # WeightedRandomSampler on train — val always unweighted
    tr_loader = make_loader_weighted(X_tr_s, y_tr, sw_tr, params["batch_size"])

    best_auprc   = 0.0
    no_improve   = 0
    best_weights = None

    for epoch in range(N_EPOCHS_MAX):
        try:
            train_epoch(model, tr_loader, optimizer, criterion)
        except RuntimeError as e:
            log.warning(f"Trial {trial.number} epoch {epoch} error: {e}")
            raise optuna.exceptions.TrialPruned()

        scheduler.step()
        val_probs = get_probs(model, val_loader_optuna)
        auprc     = average_precision_score(y_val, val_probs)

        if auprc > best_auprc:
            best_auprc   = auprc
            no_improve   = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        trial.report(auprc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if no_improve >= PATIENCE:
            break

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_auprc


# ══════════════════════════════════════════════════════════════════════════════
# 11. RUN OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "=" * 60)
log.info(f"Starting Optuna: {N_OPTUNA_TRIALS} trials (Hybrid FT-Transformer)")
log.info("Objective: AUPRC on unweighted val split")
log.info("Training: WeightedRandomSampler (real=4.0, synthetic=1.0)")
log.info("=" * 60)

study_path = MODELS_DIR / "hybrid_ftt_optuna_study.pkl"
sampler    = TPESampler(seed=RANDOM_SEED)
pruner     = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
study      = optuna.create_study(
    direction  = "maximize",
    sampler    = sampler,
    pruner     = pruner,
    study_name = "hybrid_fttransformer",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, n_jobs=1, show_progress_bar=True)
elapsed = time.time() - t0

with open(study_path, "wb") as f:
    pickle.dump(study, f)

log.info(f"\nOptuna finished in {elapsed/60:.1f} min")
log.info(f"Best val AUPRC : {study.best_value:.4f}")
log.info(f"Best params    : {study.best_params}")

best_params = study.best_params
with open(MODELS_DIR / "hybrid_ftt_best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 12. RETRAIN BEST MODEL ON FULL HYBRID SET
# WeightedRandomSampler applied to full training set
# Val split used for early stopping (unweighted)
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "=" * 60)
log.info("Retraining best model on full hybrid set (12,530 rows)...")
log.info("WeightedRandomSampler: real rows sampled ~4x per epoch.")
log.info("=" * 60)

X_full_s = scale_df(X_hybrid)

final_model, actual_n_heads = build_fttransformer(best_params)
final_model = final_model.to(DEVICE)

# Save actual n_heads (may differ after divisibility enforcement)
best_params_final = dict(best_params)
best_params_final["actual_attention_n_heads"] = actual_n_heads
with open(MODELS_DIR / "hybrid_ftt_best_params_final.json", "w") as f:
    json.dump(best_params_final, f, indent=2)

# pos_weight from full hybrid training set
pos_weight_full = float((y_hybrid == 0).sum()) / max(float((y_hybrid == 1).sum()), 1.0)
POS_WEIGHT_FULL = torch.tensor([pos_weight_full], dtype=torch.float32).to(DEVICE)
log.info(f"  pos_weight (full hybrid) : {pos_weight_full:.3f}")

final_optimizer = optim.AdamW(
    final_model.parameters(),
    lr           = best_params["lr"],
    weight_decay = best_params["weight_decay"],
)
final_criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FULL)
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(
    final_optimizer, T_max=N_EPOCHS_FINAL, eta_min=1e-6
)

# Full hybrid train loader with WeightedRandomSampler
full_tr_loader  = make_loader_weighted(
    X_full_s, y_hybrid, sample_weights, best_params["batch_size"]
)
# Val loader for early stopping — unweighted
val_loader_fin  = make_loader_simple(X_val_s, y_val, best_params["batch_size"],
                                     shuffle=False)

best_auprc_final   = 0.0
best_final_weights = None
no_improve_final   = 0

for epoch in range(N_EPOCHS_FINAL):
    try:
        train_epoch(final_model, full_tr_loader, final_optimizer, final_criterion)
    except RuntimeError as e:
        log.warning(f"Final training error epoch {epoch}: {e}")
        break

    scheduler_final.step()
    val_probs = get_probs(final_model, val_loader_fin)
    auprc     = average_precision_score(y_val, val_probs)

    if auprc > best_auprc_final:
        best_auprc_final   = auprc
        no_improve_final   = 0
        best_final_weights = {k: v.cpu().clone()
                              for k, v in final_model.state_dict().items()}
    else:
        no_improve_final += 1

    if no_improve_final >= PATIENCE_FINAL:
        log.info(f"  Early stopping at epoch {epoch + 1} "
                 f"(best val AUPRC={best_auprc_final:.4f})")
        break

if best_final_weights:
    final_model.load_state_dict(best_final_weights)

torch.save(final_model.state_dict(), MODELS_DIR / "hybrid_ftt_best.pt")
log.info(f"  Final model saved. Best val AUPRC: {best_auprc_final:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "=" * 60)
log.info("Evaluating on real held-out test set (n=633)...")
log.info("=" * 60)

te_loader  = make_loader_simple(X_te_s, y_test, batch_size=256, shuffle=False)
test_probs = get_probs(final_model, te_loader)


def bootstrap_ci(y_true, probs, metric_fn, n=1000, seed=RANDOM_SEED):
    rng    = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(y_true[idx]):
            continue
        try:
            scores.append(metric_fn(y_true[idx], probs[idx]))
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


auroc       = roc_auc_score(y_test, test_probs)
auprc       = average_precision_score(y_test, test_probs)
brier       = brier_score_loss(y_test, test_probs)
auroc_ci    = bootstrap_ci(y_test, test_probs, roc_auc_score)
auprc_ci    = bootstrap_ci(y_test, test_probs, average_precision_score)

log.info(f"  AUROC : {auroc:.4f} [{auroc_ci[0]:.4f}-{auroc_ci[1]:.4f}]")
log.info(f"  AUPRC : {auprc:.4f} [{auprc_ci[0]:.4f}-{auprc_ci[1]:.4f}]")
log.info(f"  Brier : {brier:.4f}")


def find_threshold(y_true, probs, target_sens):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = np.where(tpr >= target_sens)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


threshold_rows = []
for target in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]:
    thr   = find_threshold(y_test, test_probs, target)
    preds = (test_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test.astype(int), preds).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_test.astype(int), preds)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    threshold_rows.append({
        "target_sensitivity": target,
        "threshold"         : round(thr, 4),
        "sensitivity"       : round(sens, 4),
        "specificity"       : round(spec, 4),
        "ppv"               : round(ppv, 4),
        "npv"               : round(npv, 4),
        "f1"                : round(f1, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    })

thresh_df = pd.DataFrame(threshold_rows)
thresh_df.to_csv(RESULTS_DIR / "hybrid_ftt_threshold_table.csv", index=False)
primary   = thresh_df[thresh_df["target_sensitivity"] == 0.90].iloc[0]

log.info(f"\n  Primary threshold (90% sensitivity):")
log.info(f"    Threshold   : {primary['threshold']}")
log.info(f"    Sensitivity : {primary['sensitivity']:.4f}")
log.info(f"    Specificity : {primary['specificity']:.4f}")
log.info(f"    PPV         : {primary['ppv']:.4f}")
log.info(f"    NPV         : {primary['npv']:.4f}")
log.info(f"    F1          : {primary['f1']:.4f}")
log.info(f"    TP={primary['tp']} FP={primary['fp']} "
         f"TN={primary['tn']} FN={primary['fn']}")

# ══════════════════════════════════════════════════════════════════════════════
# 14. DELONG TESTS
# ══════════════════════════════════════════════════════════════════════════════

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


delong_results = {}
log.info("\n  DeLong tests:")

# vs Phase 6 real-only FT-T
if PHASE6_FTT_PROBS_PATH.exists():
    phase6_probs = np.load(PHASE6_FTT_PROBS_PATH)["test_probs"]
    a1, b1, z1, p1 = delong_test(y_test, test_probs, phase6_probs)
    log.info(f"    Hybrid FT-T vs Phase 6 FT-T : z={z1:.4f}, p={p1:.4f} "
             f"({'sig' if p1 < 0.05 else 'not sig'})")
    delong_results["vs_phase6_ftt"] = {
        "hybrid_ftt_auroc": a1, "phase6_ftt_auroc": b1,
        "z": z1, "p": p1, "significant": p1 < 0.05,
        "direction": "Hybrid FT-T better" if a1 > b1 else "Phase 6 FT-T better",
    }

# vs Exp D CatBoost
if EXPD_CB_PROBS_PATH.exists():
    expd_probs  = np.load(EXPD_CB_PROBS_PATH)
    a2, b2, z2, p2 = delong_test(y_test, test_probs, expd_probs)
    log.info(f"    Hybrid FT-T vs Exp D CatBoost : z={z2:.4f}, p={p2:.4f} "
             f"({'sig' if p2 < 0.05 else 'not sig'})")
    delong_results["vs_expD_catboost"] = {
        "hybrid_ftt_auroc": a2, "expD_catboost_auroc": b2,
        "z": z2, "p": p2, "significant": p2 < 0.05,
        "direction": "Hybrid FT-T better" if a2 > b2 else "Exp D CatBoost better",
    }

# vs Exp A real-only CatBoost
if EXPA_CB_PROBS_PATH.exists():
    expa_probs  = np.load(EXPA_CB_PROBS_PATH)
    a3, b3, z3, p3 = delong_test(y_test, test_probs, expa_probs)
    log.info(f"    Hybrid FT-T vs Exp A real-only : z={z3:.4f}, p={p3:.4f} "
             f"({'sig' if p3 < 0.05 else 'not sig'})")
    delong_results["vs_expA_real_only"] = {
        "hybrid_ftt_auroc": a3, "expA_auroc": b3,
        "z": z3, "p": p3, "significant": p3 < 0.05,
        "direction": "Hybrid FT-T better" if a3 > b3 else "Real-only better",
        "delta_auroc": float(a3 - b3),
        "delta_auprc": float(auprc - average_precision_score(y_test, expa_probs)),
    }

with open(RESULTS_DIR / "hybrid_ftt_delong.json", "w") as f:
    json.dump(delong_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 15. SAVE RESULTS + PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

runtime = elapsed / 60   # Optuna time; total runtime logged at end

results = {
    "model"                  : "Weighted Hybrid FT-Transformer",
    "experiment"             : "Hybrid FT-T (real=4.0, synth=1.0)",
    "timestamp"              : datetime.now().isoformat(),
    "dataset"                : "Hybrid train (12,530) | Real test (633)",
    "n_hybrid_train"         : int(n_total),
    "n_real_train"           : int(n_real),
    "n_synthetic_train"      : int(n_synth),
    "n_test"                 : int(len(y_test)),
    "n_features"             : len(feature_cols),
    "real_sample_weight"     : REAL_SAMPLE_WEIGHT,
    "synth_sample_weight"    : SYNTH_SAMPLE_WEIGHT,
    "real_influence_pct"     : float(pct_real),
    "weighting_method"       : "WeightedRandomSampler",
    "n_optuna_trials"        : N_OPTUNA_TRIALS,
    "best_optuna_val_auprc"  : float(study.best_value),
    "test_metrics": {
        "auroc"          : float(auroc),
        "auroc_ci_lower" : float(auroc_ci[0]),
        "auroc_ci_upper" : float(auroc_ci[1]),
        "auprc"          : float(auprc),
        "auprc_ci_lower" : float(auprc_ci[0]),
        "auprc_ci_upper" : float(auprc_ci[1]),
        "brier_score"    : float(brier),
        "threshold_90sens": {
            "threshold"  : float(primary["threshold"]),
            "sensitivity": float(primary["sensitivity"]),
            "specificity": float(primary["specificity"]),
            "ppv"        : float(primary["ppv"]),
            "npv"        : float(primary["npv"]),
            "f1"         : float(primary["f1"]),
            "tp": int(primary["tp"]), "fp": int(primary["fp"]),
            "fn": int(primary["fn"]), "tn": int(primary["tn"]),
        },
    },
    "best_hyperparameters"   : best_params_final,
    "delong"                 : delong_results,
    "optuna_runtime_minutes" : float(runtime),
}

with open(RESULTS_DIR / "hybrid_ftt_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save predictions — loaded by stacking script
np.savez(
    RESULTS_DIR / "hybrid_ftt_predictions.npz",
    test_probs = test_probs,
    y_test     = y_test,
)
log.info(f"\n  Predictions saved -> {RESULTS_DIR / 'hybrid_ftt_predictions.npz'}")
log.info(f"  Results saved     -> {RESULTS_DIR / 'hybrid_ftt_results.json'}")

# ══════════════════════════════════════════════════════════════════════════════
# 16. FIGURES — mirrors Phase 6 figure set exactly
# ══════════════════════════════════════════════════════════════════════════════

log.info("\nGenerating figures...")
COLORS = {"ftt": "#9C27B0", "random": "#9E9E9E"}

# ── Figure 1: Optuna history ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Hybrid FT-Transformer — Optuna Optimisation History ({N_OPTUNA_TRIALS} Trials)\n"
    f"WeightedRandomSampler (real=4.0, synthetic=1.0)",
    fontsize=13, fontweight="bold"
)
trial_nums  = [t.number for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]
trial_vals  = [t.value for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]
best_so_far = np.maximum.accumulate(trial_vals) if trial_vals else []

axes[0].scatter(trial_nums, trial_vals, alpha=0.4, color=COLORS["ftt"], s=20)
if len(best_so_far):
    axes[0].plot(trial_nums, best_so_far, color="black", lw=2, label="Best so far")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Validation AUPRC")
axes[0].set_title("AUPRC per Trial")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

try:
    importances = optuna.importance.get_param_importances(study)
    top5 = dict(list(importances.items())[:5])
    axes[1].barh(list(top5.keys()), list(top5.values()),
                 color=COLORS["ftt"], alpha=0.8)
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Top 5 Hyperparameter Importances")
    axes[1].grid(True, alpha=0.3, axis="x")
except Exception:
    axes[1].text(0.5, 0.5, "Not available", ha="center", va="center")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_ftt_optuna_history.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "hybrid_ftt_optuna_history.pdf", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved -> hybrid_ftt_optuna_history.png")

# ── Figure 2: ROC + PR ────────────────────────────────────────────────────────
fpr_f, tpr_f, _  = roc_curve(y_test, test_probs)
prec_f, rec_f, _ = precision_recall_curve(y_test, test_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Weighted Hybrid FT-Transformer — Test Set Performance\n"
    f"Real held-out patients (n={len(y_test)}, prevalence={y_test.mean():.1%})",
    fontsize=13, fontweight="bold"
)
axes[0].plot(fpr_f, tpr_f, color=COLORS["ftt"], lw=2,
             label=f"Hybrid FT-T  AUROC={auroc:.4f} [{auroc_ci[0]:.4f}-{auroc_ci[1]:.4f}]")
axes[0].plot([0, 1], [0, 1], "--", color=COLORS["random"], lw=1)
axes[0].fill_between(fpr_f, tpr_f, alpha=0.06, color=COLORS["ftt"])
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve", fontweight="bold")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].spines[["top", "right"]].set_visible(False)

axes[1].plot(rec_f, prec_f, color=COLORS["ftt"], lw=2,
             label=f"Hybrid FT-T  AUPRC={auprc:.4f} [{auprc_ci[0]:.4f}-{auprc_ci[1]:.4f}]")
axes[1].axhline(y_test.mean(), color=COLORS["random"], linestyle="--",
                label=f"No-skill = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve", fontweight="bold")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_ftt_roc_pr.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "hybrid_ftt_roc_pr.pdf", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved -> hybrid_ftt_roc_pr.png")

# ── Figure 3: Calibration ──────────────────────────────────────────────────────
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "s-", color=COLORS["ftt"], lw=2,
        label=f"Hybrid FT-T (Brier={brier:.4f})")
ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("Weighted Hybrid FT-Transformer — Calibration Curve", fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_ftt_calibration.png", dpi=150, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "hybrid_ftt_calibration.pdf", dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved -> hybrid_ftt_calibration.png")

# ── Figure 4: Threshold sensitivity ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Weighted Hybrid FT-Transformer — Threshold Sensitivity Analysis",
             fontsize=13, fontweight="bold")

targ_vals = thresh_df["target_sensitivity"].values
axes[0].plot(targ_vals, thresh_df["sensitivity"].values, "o-",
             label="Sensitivity", color="#4CAF50", lw=2)
axes[0].plot(targ_vals, thresh_df["specificity"].values, "s-",
             label="Specificity", color=COLORS["ftt"], lw=2)
axes[0].plot(targ_vals, thresh_df["f1"].values, "^-",
             label="F1", color="#FF9800", lw=2)
axes[0].plot(targ_vals, thresh_df["ppv"].values, "D-",
             label="PPV", color="#9C27B0", lw=2)
axes[0].set_xlabel("Target Sensitivity"); axes[0].set_ylabel("Metric")
axes[0].set_title("Metrics vs Target Sensitivity", fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].spines[["top", "right"]].set_visible(False)

x = np.arange(len(targ_vals))
axes[1].bar(x - 0.25, thresh_df["tp"].values, 0.25,
            label="TP", color="#4CAF50", alpha=0.8)
axes[1].bar(x,         thresh_df["fp"].values, 0.25,
            label="FP", color="#F44336", alpha=0.8)
axes[1].bar(x + 0.25,  thresh_df["fn"].values, 0.25,
            label="FN", color="#FF9800", alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{t:.0%}" for t in targ_vals])
axes[1].set_xlabel("Target Sensitivity"); axes[1].set_ylabel("Count")
axes[1].set_title("TP / FP / FN by Threshold", fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "hybrid_ftt_threshold_sensitivity.png",
            dpi=150, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "hybrid_ftt_threshold_sensitivity.pdf",
            dpi=150, bbox_inches="tight")
plt.close()
log.info("  Saved -> hybrid_ftt_threshold_sensitivity.png")

# ══════════════════════════════════════════════════════════════════════════════
# 17. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

total_runtime = (time.time() - t0) / 60
log.info("\n" + "=" * 70)
log.info("WEIGHTED HYBRID FT-TRANSFORMER — COMPLETE")
log.info("=" * 70)
log.info(f"  AUROC       : {auroc:.4f} [{auroc_ci[0]:.4f}-{auroc_ci[1]:.4f}]")
log.info(f"  AUPRC       : {auprc:.4f} [{auprc_ci[0]:.4f}-{auprc_ci[1]:.4f}]")
log.info(f"  Brier       : {brier:.4f}")
log.info(f"  Sensitivity : {primary['sensitivity']:.4f} @ threshold {primary['threshold']}")
log.info(f"  Specificity : {primary['specificity']:.4f}")
log.info(f"  PPV         : {primary['ppv']:.4f}")
log.info(f"  NPV         : {primary['npv']:.4f}")
log.info(f"  F1          : {primary['f1']:.4f}")
log.info(f"\n  Weighting   : real={REAL_SAMPLE_WEIGHT}, synth={SYNTH_SAMPLE_WEIGHT}")
log.info(f"  Real influence : {pct_real:.1f}%")
log.info(f"\n  Optuna runtime : {elapsed/60:.1f} min")
log.info(f"  Total runtime  : {total_runtime:.1f} min")
log.info(f"\n  Models  -> {MODELS_DIR}")
log.info(f"  Results -> {RESULTS_DIR}")
log.info(f"  Figures -> {FIGURES_DIR}")
log.info(f"  Log     -> {log_path}")
log.info(f"\n  Next -> run synthetic_stacking.py (loads hybrid_ftt_predictions.npz)")
log.info("=" * 70)
