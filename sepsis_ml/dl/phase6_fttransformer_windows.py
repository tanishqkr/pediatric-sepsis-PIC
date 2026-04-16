"""
Phase 6 - Script 2: FT-Transformer (FIXED)
============================================
Fix: rtdl package is deprecated. Using rtdl_revisiting_models instead.
     Correct constructor: FTTransformer(n_cont_features, d_block,
     attention_n_heads, n_blocks, ...) — NOT make_baseline() with n_heads.

Windows note: This script is ready to run on Windows.
  - num_workers=0 in DataLoaders (required on Windows)
  - pathlib.Path handles Windows paths and spaces correctly
  - No MPS/Apple Silicon code paths

Install before running:
    pip install rtdl_revisiting_models

PyTorch must be installed separately (NOT via requirements.txt):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

The model forward pass takes (x_cont, x_cat=None) and returns logits directly.
"""

# Windows multiprocessing guard — required when using PyTorch DataLoader on Windows
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

import os
import sys
import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
from rtdl_revisiting_models import FTTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss, confusion_matrix
)
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT.parent / "model_datasets"
DL_DIR       = SCRIPT_DIR
MODEL_DIR    = DL_DIR / "models" / "run_phase6_fttransformer"
RESULTS_DIR  = DL_DIR / "results"
FIGURES_DIR  = DL_DIR / "figures"
LOGS_DIR     = DL_DIR / "logs"
OPTUNA_DIR   = DL_DIR / "optuna_studies"

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OPTUNA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────
log_path = LOGS_DIR / "phase6_fttransformer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("PHASE 6 — FT-TRANSFORMER (FIXED — rtdl_revisiting_models)")
log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

# ─────────────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device : CUDA GPU — {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    DEVICE = torch.device("cpu")
    log.info("Device : CPU (no CUDA GPU detected — check CUDA installation)")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 3. LOAD DATA
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Loading datasets...")

train_df = pd.read_csv(DATA_DIR / "B_train_model_ready.csv")
test_df  = pd.read_csv(DATA_DIR / "B_test_model_ready.csv")

TARGET       = "sepsis_label"
feature_cols = [c for c in train_df.columns if c != TARGET]

X_train_full = train_df[feature_cols].copy()
y_train_full = train_df[TARGET].values.astype(np.float32)
X_test       = test_df[feature_cols].copy()
y_test       = test_df[TARGET].values.astype(np.float32)

log.info(f"Train shape : {X_train_full.shape} | Sepsis: {y_train_full.mean():.1%}")
log.info(f"Test shape  : {X_test.shape}       | Sepsis: {y_test.mean():.1%}")
log.info(f"Features    : {len(feature_cols)}")

# ─────────────────────────────────────────────
# 4. TRAIN / VALIDATION SPLIT
# ─────────────────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, stratify=y_train_full, random_state=RANDOM_SEED
)
log.info(f"Train split : {X_tr.shape[0]} | Val split: {X_val.shape[0]}")

# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
binary_cols = [c for c in feature_cols
               if set(X_train_full[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
num_cols    = [c for c in feature_cols if c not in binary_cols]

log.info(f"Numerical features to scale : {len(num_cols)}")
log.info(f"Binary features (no scaling): {len(binary_cols)}")

scaler = StandardScaler()
scaler.fit(X_tr[num_cols])

def scale_df(df):
    df = df.copy()
    df[num_cols] = scaler.transform(df[num_cols])
    return df

X_tr_s  = scale_df(X_tr).values.astype(np.float32)
X_val_s = scale_df(X_val).values.astype(np.float32)
X_te_s  = scale_df(X_test).values.astype(np.float32)

with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
log.info("Scaler saved.")

N_FEATURES = X_tr_s.shape[1]

# ─────────────────────────────────────────────
# 6. CLASS WEIGHT
# ─────────────────────────────────────────────
pos_weight_val = (y_tr == 0).sum() / (y_tr == 1).sum()
POS_WEIGHT     = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
log.info(f"pos_weight : {pos_weight_val:.3f}")

# ─────────────────────────────────────────────
# 7. PYTORCH DATASETS
# ─────────────────────────────────────────────
def make_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)

# ─────────────────────────────────────────────
# 8. BUILD FT-TRANSFORMER
# Correct API from rtdl_revisiting_models:
#   FTTransformer(
#     n_cont_features, cat_cardinalities, d_out,
#     n_blocks, d_block, attention_n_heads,
#     attention_dropout, ffn_d_hidden_multiplier, ffn_dropout, residual_dropout
#   )
# Forward: model(x_cont, x_cat=None) → logits of shape (batch, d_out)
# ─────────────────────────────────────────────
def build_fttransformer(params):
    # d_block must be divisible by attention_n_heads
    d_block = params["d_block"]
    n_heads = params["attention_n_heads"]
    # Enforce divisibility
    while d_block % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1:
            n_heads = 1
            break

    model = FTTransformer(
        n_cont_features        = N_FEATURES,
        cat_cardinalities      = None,
        d_out                  = 1,
        n_blocks               = params["n_blocks"],
        d_block                = d_block,
        attention_n_heads      = n_heads,
        attention_dropout      = params["attention_dropout"],
        ffn_d_hidden_multiplier= params["ffn_d_hidden_multiplier"],
        ffn_dropout            = params["ffn_dropout"],
        residual_dropout       = params["residual_dropout"],
    )
    return model, n_heads  # return actual n_heads used

# ─────────────────────────────────────────────
# 9. TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        # rtdl_revisiting_models FTTransformer: forward(x_cont, x_cat)
        logits = model(X_batch, None).squeeze(-1)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

# ─────────────────────────────────────────────
# 10. GET PROBABILITIES
# ─────────────────────────────────────────────
@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        logits  = model(X_batch, None).squeeze(-1)
        probs   = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)

# ─────────────────────────────────────────────
# 11. OPTUNA OBJECTIVE
# ─────────────────────────────────────────────
N_EPOCHS_MAX = 100
PATIENCE     = 15

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

    optimizer = optim.AdamW(model.parameters(),
                            lr=params["lr"],
                            weight_decay=params["weight_decay"])
    criterion  = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    tr_loader  = make_loader(X_tr_s,  y_tr,  params["batch_size"], shuffle=True)
    val_loader = make_loader(X_val_s, y_val, params["batch_size"], shuffle=False)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_MAX, eta_min=1e-6
    )

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
        val_probs = get_probs(model, val_loader)
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

    return best_auprc

# ─────────────────────────────────────────────
# 12. RUN OPTUNA
# ─────────────────────────────────────────────
log.info("=" * 50)
log.info("Starting Optuna: 100 trials (FT-Transformer)")
log.info("Estimated runtime: 2-4 hours on RTX 4500 Ada (24 GB). Monitor GPU usage.")
log.info("=" * 50)

study_path = OPTUNA_DIR / "fttransformer_study.pkl"
sampler    = TPESampler(seed=RANDOM_SEED)
pruner     = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
study      = optuna.create_study(
    direction="maximize", sampler=sampler, pruner=pruner,
    study_name="fttransformer_phase6"
)
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
elapsed = time.time() - t0

with open(study_path, "wb") as f:
    pickle.dump(study, f)

log.info(f"Optuna finished in {elapsed/60:.1f} min")
log.info(f"Best CV AUPRC : {study.best_value:.4f}")
log.info(f"Best params   : {study.best_params}")

best_params = study.best_params
with open(MODEL_DIR / "best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)

# ─────────────────────────────────────────────
# 13. RETRAIN BEST MODEL ON FULL TRAIN SET
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Retraining best model on full training set...")

X_full_s        = scale_df(X_train_full).values.astype(np.float32)
pos_weight_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()
POS_WEIGHT_FULL = torch.tensor([pos_weight_full], dtype=torch.float32).to(DEVICE)

final_model, actual_n_heads = build_fttransformer(best_params)
final_model = final_model.to(DEVICE)

# Save actual n_heads used (may differ if divisibility was enforced)
best_params_saved = dict(best_params)
best_params_saved["actual_attention_n_heads"] = actual_n_heads
with open(MODEL_DIR / "best_params_final.json", "w") as f:
    json.dump(best_params_saved, f, indent=2)

final_optimizer = optim.AdamW(
    final_model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)
final_criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FULL)
full_loader     = make_loader(X_full_s, y_train_full, best_params["batch_size"], shuffle=True)
val_loader_fin  = make_loader(X_val_s, y_val, best_params["batch_size"], shuffle=False)
scheduler_final = optim.lr_scheduler.CosineAnnealingLR(
    final_optimizer, T_max=150, eta_min=1e-6
)

best_auprc_final   = 0.0
best_final_weights = None
no_improve_final   = 0

for epoch in range(150):
    try:
        train_epoch(final_model, full_loader, final_optimizer, final_criterion)
    except RuntimeError as e:
        log.warning(f"Final training error epoch {epoch}: {e}")
        break
    scheduler_final.step()
    val_probs = get_probs(final_model, val_loader_fin)
    auprc = average_precision_score(y_val, val_probs)
    if auprc > best_auprc_final:
        best_auprc_final   = auprc
        no_improve_final   = 0
        best_final_weights = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    else:
        no_improve_final += 1
    if no_improve_final >= 20:
        break

if best_final_weights:
    final_model.load_state_dict(best_final_weights)

torch.save(final_model.state_dict(), MODEL_DIR / "fttransformer_best.pt")
log.info("Final model saved.")

# ─────────────────────────────────────────────
# 14. TEST SET EVALUATION
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Evaluating on test set...")

te_loader  = make_loader(X_te_s, y_test, batch_size=256, shuffle=False)
test_probs = get_probs(final_model, te_loader)

def bootstrap_ci(y_true, probs, metric_fn, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(y_true[idx]):
            continue
        scores.append(metric_fn(y_true[idx], probs[idx]))
    return np.percentile(scores, [2.5, 97.5])

auroc    = roc_auc_score(y_test, test_probs)
auprc    = average_precision_score(y_test, test_probs)
auroc_ci = bootstrap_ci(y_test, test_probs, roc_auc_score)
auprc_ci = bootstrap_ci(y_test, test_probs, average_precision_score)

log.info(f"AUROC : {auroc:.4f} [{auroc_ci[0]:.4f}–{auroc_ci[1]:.4f}]")
log.info(f"AUPRC : {auprc:.4f} [{auprc_ci[0]:.4f}–{auprc_ci[1]:.4f}]")

def find_threshold(y_true, probs, target_sens):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = np.argmin(np.abs(tpr - target_sens))
    return float(thresholds[idx]), float(tpr[idx])

threshold_rows = []
for target in [0.80, 0.85, 0.90, 0.92, 0.95]:
    thr, achieved_sens = find_threshold(y_test, test_probs, target)
    preds = (test_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test.astype(int), preds).ravel()
    spec = tn / (tn + fp)
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    threshold_rows.append({
        "target_sens": target, "threshold": round(thr, 3),
        "achieved_sens": round(achieved_sens, 4), "specificity": round(spec, 4),
        "ppv": round(ppv, 4), "npv": round(npv, 4), "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    })

thresh_df = pd.DataFrame(threshold_rows)
thresh_df.to_csv(RESULTS_DIR / "phase6_fttransformer_threshold_table.csv", index=False)
primary = thresh_df[thresh_df["target_sens"] == 0.90].iloc[0]
brier   = brier_score_loss(y_test, test_probs)

log.info(f"\nPrimary threshold (90% sens): {primary['threshold']}")
log.info(f"Sensitivity : {primary['achieved_sens']:.4f}")
log.info(f"Specificity : {primary['specificity']:.4f}")
log.info(f"PPV         : {primary['ppv']:.4f}")
log.info(f"NPV         : {primary['npv']:.4f}")
log.info(f"F1          : {primary['f1']:.4f}")
log.info(f"Brier Score : {brier:.4f}")

# ─────────────────────────────────────────────
# 15. DELONG TEST vs TUNED CATBOOST
# ─────────────────────────────────────────────
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

    def fast_delong(y_true, y_score):
        m = int(y_true.sum()); n = len(y_true) - m
        pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
        ranks = compute_midrank(np.concatenate([pos, neg]))
        pos_ranks = ranks[:m]
        auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)
        v10 = (pos_ranks - np.arange(1, m + 1)) / n
        v01 = np.array([(pos > ns).mean() + 0.5 * (pos == ns).mean() for ns in neg])
        var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
        return auc, var

    auc_a, var_a = fast_delong(y_true, probs_a)
    auc_b, var_b = fast_delong(y_true, probs_b)
    m = int(y_true.sum()); n = len(y_true) - m
    pos_a = probs_a[y_true == 1]; neg_a = probs_a[y_true == 0]
    pos_b = probs_b[y_true == 1]; neg_b = probs_b[y_true == 0]
    v10_a = np.array([(pa > neg_a).mean() + 0.5 * (pa == neg_a).mean() for pa in pos_a])
    v10_b = np.array([(pb > neg_b).mean() + 0.5 * (pb == neg_b).mean() for pb in pos_b])
    v01_a = np.array([(pos_a > na).mean() + 0.5 * (pos_a == na).mean() for na in neg_a])
    v01_b = np.array([(pos_b > nb).mean() + 0.5 * (pos_b == nb).mean() for nb in neg_b])
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / m +
           np.cov(v01_a, v01_b, ddof=1)[0, 1] / n)
    z = (auc_a - auc_b) / np.sqrt(max(var_a + var_b - 2 * cov, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)

delong_results = {}
cb_preds_path  = PROJECT_ROOT / "results" / "catboost_test_probs.npy"
try:
    if cb_preds_path.exists():
        cb_probs = np.load(cb_preds_path)
        auc_ft, auc_cb, z_stat, p_val = delong_test(y_test, test_probs, cb_probs)
        delong_results = {
            "fttransformer_auroc": auc_ft, "catboost_auroc": auc_cb,
            "z_statistic": z_stat, "p_value": p_val,
            "significant": p_val < 0.05,
            "direction": "FT-Transformer better" if auc_ft > auc_cb else "CatBoost better"
        }
        log.info(f"\nDeLong vs CatBoost: z={z_stat:.4f}, p={p_val:.4f}")
        log.info(f"Significant: {p_val < 0.05} | {delong_results['direction']}")
    else:
        delong_results = {"note": "CatBoost predictions not found"}
except Exception as e:
    delong_results = {"error": str(e)}

# ─────────────────────────────────────────────
# 16. SAVE RESULTS
# ─────────────────────────────────────────────
results = {
    "model": "FT-Transformer", "phase": "Phase 6",
    "timestamp": datetime.now().isoformat(),
    "dataset": "Option B (infection-only)",
    "n_train": int(len(y_train_full)), "n_test": int(len(y_test)),
    "n_features": len(feature_cols),
    "best_optuna_auprc": float(study.best_value),
    "test_metrics": {
        "auroc": float(auroc), "auroc_ci_lower": float(auroc_ci[0]),
        "auroc_ci_upper": float(auroc_ci[1]),
        "auprc": float(auprc), "auprc_ci_lower": float(auprc_ci[0]),
        "auprc_ci_upper": float(auprc_ci[1]),
        "brier_score": float(brier),
        "threshold_90sens": {
            "threshold": float(primary["threshold"]),
            "sensitivity": float(primary["achieved_sens"]),
            "specificity": float(primary["specificity"]),
            "ppv": float(primary["ppv"]), "npv": float(primary["npv"]),
            "f1": float(primary["f1"]), "tp": int(primary["tp"]),
            "fp": int(primary["fp"]), "fn": int(primary["fn"]),
        }
    },
    "best_hyperparameters": best_params_saved,
    "delong_vs_catboost": delong_results,
    "runtime_minutes": float(elapsed / 60)
}

with open(RESULTS_DIR / "phase6_fttransformer_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(RESULTS_DIR / "phase6_fttransformer_delong.json", "w") as f:
    json.dump(delong_results, f, indent=2)
np.savez(RESULTS_DIR / "phase6_fttransformer_predictions.npz",
         test_probs=test_probs, y_test=y_test)
log.info("Results saved.")

# ─────────────────────────────────────────────
# 17. FIGURES
# ─────────────────────────────────────────────
log.info("Generating figures...")
COLORS = {"ftt": "#9C27B0", "random": "#9E9E9E"}

# ── Figure 1: Optuna history ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FT-Transformer — Optuna Optimisation History (100 Trials)",
             fontsize=14, fontweight="bold")
trial_nums  = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_vals  = [t.value  for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
best_so_far = np.maximum.accumulate(trial_vals) if trial_vals else []
axes[0].scatter(trial_nums, trial_vals, alpha=0.4, color=COLORS["ftt"], s=20)
if len(best_so_far):
    axes[0].plot(trial_nums, best_so_far, color="black", lw=2, label="Best so far")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Validation AUPRC")
axes[0].set_title("AUPRC per Trial"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
try:
    importances = optuna.importance.get_param_importances(study)
    top5 = dict(list(importances.items())[:5])
    axes[1].barh(list(top5.keys()), list(top5.values()), color=COLORS["ftt"], alpha=0.8)
    axes[1].set_xlabel("Importance"); axes[1].set_title("Top 5 Hyperparameter Importances")
    axes[1].grid(True, alpha=0.3, axis="x")
except Exception:
    axes[1].text(0.5, 0.5, "Not available", ha="center", va="center")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fttransformer_optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 2: ROC + PR ──
fpr_f, tpr_f, _  = roc_curve(y_test, test_probs)
prec_f, rec_f, _ = precision_recall_curve(y_test, test_probs)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"FT-Transformer — Test Set Performance (n={len(y_test)})",
             fontsize=14, fontweight="bold")
axes[0].plot(fpr_f, tpr_f, color=COLORS["ftt"], lw=2,
             label=f"FT-Transformer AUROC = {auroc:.4f}")
axes[0].plot([0,1],[0,1], "--", color=COLORS["random"], lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
axes[1].plot(rec_f, prec_f, color=COLORS["ftt"], lw=2,
             label=f"FT-Transformer AUPRC = {auprc:.4f}")
axes[1].axhline(y_test.mean(), color=COLORS["random"], linestyle="--",
                label=f"No-skill = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fttransformer_roc_pr.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3: Calibration ──
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "s-", color=COLORS["ftt"], lw=2,
        label=f"FT-Transformer (Brier={brier:.4f})")
ax.plot([0,1],[0,1], "--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("FT-Transformer — Calibration Curve"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fttransformer_calibration.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: Threshold sensitivity ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("FT-Transformer — Threshold Sensitivity Analysis", fontsize=14, fontweight="bold")
targets = thresh_df["target_sens"].values
axes[0].plot(targets, thresh_df["achieved_sens"].values, "o-", label="Sensitivity", color="#4CAF50")
axes[0].plot(targets, thresh_df["specificity"].values,   "s-", label="Specificity", color=COLORS["ftt"])
axes[0].plot(targets, thresh_df["f1"].values,            "^-", label="F1",          color="#FF9800")
axes[0].plot(targets, thresh_df["ppv"].values,           "D-", label="PPV",         color="#9C27B0")
axes[0].set_xlabel("Target Sensitivity"); axes[0].set_ylabel("Metric")
axes[0].set_title("Metrics vs Target Sensitivity"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
x = np.arange(len(targets))
axes[1].bar(x - 0.25, thresh_df["tp"].values, 0.25, label="TP", color="#4CAF50", alpha=0.8)
axes[1].bar(x,         thresh_df["fp"].values, 0.25, label="FP", color="#F44336", alpha=0.8)
axes[1].bar(x + 0.25,  thresh_df["fn"].values, 0.25, label="FN", color="#FF9800", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([f"{t:.0%}" for t in targets])
axes[1].set_xlabel("Target Sensitivity"); axes[1].set_ylabel("Count")
axes[1].set_title("TP / FP / FN by Threshold"); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fttransformer_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()

log.info("All figures saved.")

# ─────────────────────────────────────────────
# 18. FINAL SUMMARY
# ─────────────────────────────────────────────
log.info("=" * 70)
log.info("PHASE 6 FT-TRANSFORMER — COMPLETE")
log.info("=" * 70)
log.info(f"AUROC  : {auroc:.4f} [{auroc_ci[0]:.4f}–{auroc_ci[1]:.4f}]")
log.info(f"AUPRC  : {auprc:.4f} [{auprc_ci[0]:.4f}–{auprc_ci[1]:.4f}]")
log.info(f"Brier  : {brier:.4f}")
log.info(f"Sens   : {primary['achieved_sens']:.4f} @ threshold {primary['threshold']}")
log.info(f"Spec   : {primary['specificity']:.4f}")
log.info(f"F1     : {primary['f1']:.4f}")
log.info(f"Runtime: {elapsed/60:.1f} min")
log.info("-" * 70)
log.info(f"  Model   : {MODEL_DIR}")
log.info(f"  Results : {RESULTS_DIR}")
log.info(f"  Figures : {FIGURES_DIR}")
log.info(f"  Log     : {log_path}")
log.info("=" * 70)