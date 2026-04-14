"""
Phase 6 - Script 1: Tabular ResNet
===================================
Project : Early Prediction of Pediatric Sepsis (PIC Database)
Authors : Tanish Porwal & Uzair | NMIMS
Device  : Apple M1 Pro (MPS) with CPU fallback

What this script does:
  - Loads leakage-clean Option B dataset (225 features)
  - StandardScaler on numerical features (fit on train only)
  - Optuna Bayesian hyperparameter tuning: 100 trials
  - Trains best ResNet, evaluates on test set
  - DeLong test vs tuned CatBoost
  - Saves all outputs to sepsis_ml/dl/

Outputs
-------
Models   : dl/models/run_phase6_resnet/resnet_best.pt
           dl/models/run_phase6_resnet/best_params.json
           dl/models/run_phase6_resnet/scaler.pkl
Results  : dl/results/phase6_resnet_results.json
           dl/results/phase6_resnet_delong.json
           dl/results/phase6_resnet_threshold_table.csv
           dl/results/phase6_resnet_predictions.npz
Figures  : dl/figures/resnet_optuna_history.png
           dl/figures/resnet_roc_pr.png
           dl/figures/resnet_calibration.png
           dl/figures/resnet_threshold_sensitivity.png
Logs     : dl/logs/phase6_resnet.log
Optuna   : dl/optuna_studies/resnet_study.pkl
"""

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
import rtdl
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
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/dl/
PROJECT_ROOT = SCRIPT_DIR.parent                         # sepsis_ml/
DATA_DIR     = PROJECT_ROOT.parent / "model_datasets"
DL_DIR       = SCRIPT_DIR                                # sepsis_ml/dl/
MODEL_DIR    = DL_DIR / "models" / "run_phase6_resnet"
RESULTS_DIR  = DL_DIR / "results"
FIGURES_DIR  = DL_DIR / "figures"
LOGS_DIR     = DL_DIR / "logs"
OPTUNA_DIR   = DL_DIR / "optuna_studies"
CATBOOST_DIR = PROJECT_ROOT / "models" / "run_tuned"

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OPTUNA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────
log_path = LOGS_DIR / "phase6_resnet.log"
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
log.info("PHASE 6 — TABULAR RESNET")
log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

# ─────────────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    log.info("Device : Apple MPS (M1 Pro)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    log.info("Device : CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    log.info("Device : CPU")

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

TARGET = "sepsis_label"
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
    test_size=0.2,
    stratify=y_train_full,
    random_state=RANDOM_SEED
)
log.info(f"Train split : {X_tr.shape[0]} | Val split: {X_val.shape[0]}")

# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
# Identify binary columns (only 0/1 values) - skip scaling these
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

# Save scaler
with open(MODEL_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
log.info("Scaler saved.")

# ─────────────────────────────────────────────
# 6. CLASS WEIGHT
# ─────────────────────────────────────────────
pos_weight_val = (y_tr == 0).sum() / (y_tr == 1).sum()
POS_WEIGHT = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
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
# 8. RESNET MODEL (via rtdl)
# ─────────────────────────────────────────────
def build_resnet(trial_params, n_features):
    model = rtdl.ResNet.make_baseline(
        d_in          = n_features,
        d_main        = trial_params["d_main"],
        d_hidden      = trial_params["d_hidden"],
        dropout_first  = trial_params["dropout_first"],
        dropout_second = trial_params["dropout_second"],
        n_blocks      = trial_params["n_blocks"],
        d_out         = 1
    )
    return model

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
        logits = model(X_batch).squeeze(-1)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)

# ─────────────────────────────────────────────
# 10. EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_probs = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        logits  = model(X_batch).squeeze(-1)
        probs   = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)

def compute_auprc(y_true, probs):
    return average_precision_score(y_true, probs)

# ─────────────────────────────────────────────
# 11. OPTUNA OBJECTIVE
# ─────────────────────────────────────────────
N_EPOCHS_MAX  = 100
PATIENCE      = 15

def objective(trial):
    params = {
        "n_blocks"      : trial.suggest_int("n_blocks", 2, 8),
        "d_main"        : trial.suggest_categorical("d_main", [64, 128, 256, 512]),
        "d_hidden"      : trial.suggest_categorical("d_hidden", [64, 128, 256, 512]),
        "dropout_first" : trial.suggest_float("dropout_first", 0.0, 0.5),
        "dropout_second": trial.suggest_float("dropout_second", 0.0, 0.5),
        "lr"            : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay"  : trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size"    : trial.suggest_categorical("batch_size", [64, 128, 256]),
    }

    model = build_resnet(params, n_features=X_tr_s.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=params["lr"],
                            weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)

    tr_loader  = make_loader(X_tr_s,  y_tr,  params["batch_size"], shuffle=True)
    val_loader = make_loader(X_val_s, y_val, params["batch_size"], shuffle=False)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_MAX, eta_min=1e-6
    )

    best_auprc   = 0.0
    no_improve   = 0
    best_weights = None

    for epoch in range(N_EPOCHS_MAX):
        train_epoch(model, tr_loader, optimizer, criterion)
        scheduler.step()

        val_probs = get_probs(model, val_loader)
        auprc     = compute_auprc(y_val, val_probs)

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

    # Restore best weights for scoring
    if best_weights:
        model.load_state_dict(best_weights)

    return best_auprc

# ─────────────────────────────────────────────
# 12. RUN OPTUNA
# ─────────────────────────────────────────────
log.info("=" * 50)
log.info("Starting Optuna: 100 trials (ResNet)")
log.info("=" * 50)

study_path = OPTUNA_DIR / "resnet_study.pkl"

sampler = TPESampler(seed=RANDOM_SEED)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
study   = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
    study_name="resnet_phase6"
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
study.optimize(objective, n_trials=100, n_jobs=1, show_progress_bar=True)
elapsed = time.time() - t0

# Save study
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

X_full_s = scale_df(X_train_full).values.astype(np.float32)
pos_weight_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()
POS_WEIGHT_FULL = torch.tensor([pos_weight_full], dtype=torch.float32).to(DEVICE)

final_model = build_resnet(best_params, n_features=X_full_s.shape[1]).to(DEVICE)
final_optimizer = optim.AdamW(
    final_model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)
final_criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FULL)
full_loader = make_loader(X_full_s, y_train_full, best_params["batch_size"], shuffle=True)
val_loader_final = make_loader(X_val_s, y_val, best_params["batch_size"], shuffle=False)

scheduler_final = optim.lr_scheduler.CosineAnnealingLR(
    final_optimizer, T_max=150, eta_min=1e-6
)

best_auprc_final = 0.0
best_final_weights = None
no_improve_final = 0

for epoch in range(150):
    train_epoch(final_model, full_loader, final_optimizer, final_criterion)
    scheduler_final.step()
    val_probs = get_probs(final_model, val_loader_final)
    auprc = compute_auprc(y_val, val_probs)
    if auprc > best_auprc_final:
        best_auprc_final = auprc
        no_improve_final = 0
        best_final_weights = {k: v.cpu().clone() for k, v in final_model.state_dict().items()}
    else:
        no_improve_final += 1
    if no_improve_final >= 20:
        break

if best_final_weights:
    final_model.load_state_dict(best_final_weights)

torch.save(final_model.state_dict(), MODEL_DIR / "resnet_best.pt")
log.info("Final model saved.")

# ─────────────────────────────────────────────
# 14. TEST SET EVALUATION
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Evaluating on test set...")

te_loader = make_loader(X_te_s, y_test, batch_size=256, shuffle=False)
test_probs = get_probs(final_model, te_loader)

# Bootstrap AUROC and AUPRC CIs
def bootstrap_ci(y_true, probs, metric_fn, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == len(y_true[idx]):
            continue
        scores.append(metric_fn(y_true[idx], probs[idx]))
    return np.percentile(scores, [2.5, 97.5])

auroc = roc_auc_score(y_test, test_probs)
auprc = average_precision_score(y_test, test_probs)
auroc_ci = bootstrap_ci(y_test, test_probs, roc_auc_score)
auprc_ci = bootstrap_ci(y_test, test_probs, average_precision_score)

log.info(f"AUROC : {auroc:.4f} [{auroc_ci[0]:.4f}–{auroc_ci[1]:.4f}]")
log.info(f"AUPRC : {auprc:.4f} [{auprc_ci[0]:.4f}–{auprc_ci[1]:.4f}]")

# Threshold search: find threshold closest to target sensitivities
def find_threshold(y_true, probs, target_sens):
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    idx = np.argmin(np.abs(tpr - target_sens))
    return float(thresholds[idx]), float(tpr[idx])

threshold_rows = []
for target in [0.80, 0.85, 0.90, 0.92, 0.95]:
    thr, achieved_sens = find_threshold(y_test, test_probs, target)
    preds = (test_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    spec  = tn / (tn + fp)
    ppv   = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv   = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1    = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    threshold_rows.append({
        "target_sens": target, "threshold": round(thr, 3),
        "achieved_sens": round(achieved_sens, 4),
        "specificity": round(spec, 4), "ppv": round(ppv, 4),
        "npv": round(npv, 4), "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    })

thresh_df = pd.DataFrame(threshold_rows)
thresh_df.to_csv(RESULTS_DIR / "phase6_resnet_threshold_table.csv", index=False)

# Primary metrics at 90% sensitivity threshold
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
    """DeLong et al. 1988 — compare two AUROCs."""
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.empty(N)
        T2[J] = T + 1
        return T2

    def fast_delong(y_true, y_score):
        m = int(y_true.sum())
        n = len(y_true) - m
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        all_scores = np.concatenate([pos_scores, neg_scores])
        ranks      = compute_midrank(all_scores)
        pos_ranks  = ranks[:m]
        auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)
        v10 = (pos_ranks - np.arange(1, m + 1)) / n
        v01 = np.zeros(n)
        for k, ns in enumerate(neg_scores):
            v01[k] = (pos_scores > ns).mean() + 0.5 * (pos_scores == ns).mean()
        var = (np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n)
        return auc, var

    auc_a, var_a = fast_delong(y_true, probs_a)
    auc_b, var_b = fast_delong(y_true, probs_b)

    # Covariance via placements
    m = int(y_true.sum())
    n = len(y_true) - m
    pos_a = probs_a[y_true == 1]; neg_a = probs_a[y_true == 0]
    pos_b = probs_b[y_true == 1]; neg_b = probs_b[y_true == 0]

    v10_a = (np.array([(pa > neg_a).mean() + 0.5 * (pa == neg_a).mean() for pa in pos_a]))
    v10_b = (np.array([(pb > neg_b).mean() + 0.5 * (pb == neg_b).mean() for pb in pos_b]))
    v01_a = np.array([(pos_a > na).mean() + 0.5 * (pos_a == na).mean() for na in neg_a])
    v01_b = np.array([(pos_b > nb).mean() + 0.5 * (pos_b == nb).mean() for nb in neg_b])

    cov = np.cov(v10_a, v10_b, ddof=1)[0, 1] / m + np.cov(v01_a, v01_b, ddof=1)[0, 1] / n
    z   = (auc_a - auc_b) / np.sqrt(max(var_a + var_b - 2 * cov, 1e-12))
    p   = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)

# Load CatBoost predictions
catboost_pred_path = PROJECT_ROOT / "results" / "phase3_delong_results.csv"
delong_results = {}

try:
    # Try to load saved test predictions from CatBoost
    cb_preds_path = PROJECT_ROOT / "results" / "catboost_test_probs.npy"
    if cb_preds_path.exists():
        cb_probs = np.load(cb_preds_path)
        auc_resnet, auc_cb, z_stat, p_val = delong_test(y_test, test_probs, cb_probs)
        delong_results = {
            "resnet_auroc": auc_resnet,
            "catboost_auroc": auc_cb,
            "z_statistic": z_stat,
            "p_value": p_val,
            "significant": p_val < 0.05,
            "direction": "ResNet better" if auc_resnet > auc_cb else "CatBoost better"
        }
        log.info(f"\nDeLong vs CatBoost: z={z_stat:.4f}, p={p_val:.4f}")
        log.info(f"Significant: {p_val < 0.05} | {delong_results['direction']}")
    else:
        log.warning("CatBoost test probs not found. Skipping DeLong test.")
        log.warning(f"Expected at: {cb_preds_path}")
        delong_results = {"note": "CatBoost predictions not found for DeLong test"}
except Exception as e:
    log.warning(f"DeLong test failed: {e}")
    delong_results = {"error": str(e)}

# ─────────────────────────────────────────────
# 16. SAVE RESULTS
# ─────────────────────────────────────────────
results = {
    "model": "Tabular ResNet",
    "phase": "Phase 6",
    "timestamp": datetime.now().isoformat(),
    "dataset": "Option B (infection-only)",
    "n_train": int(len(y_train_full)),
    "n_test": int(len(y_test)),
    "n_features": len(feature_cols),
    "best_optuna_auprc": float(study.best_value),
    "test_metrics": {
        "auroc": float(auroc),
        "auroc_ci_lower": float(auroc_ci[0]),
        "auroc_ci_upper": float(auroc_ci[1]),
        "auprc": float(auprc),
        "auprc_ci_lower": float(auprc_ci[0]),
        "auprc_ci_upper": float(auprc_ci[1]),
        "brier_score": float(brier),
        "threshold_90sens": {
            "threshold": float(primary["threshold"]),
            "sensitivity": float(primary["achieved_sens"]),
            "specificity": float(primary["specificity"]),
            "ppv": float(primary["ppv"]),
            "npv": float(primary["npv"]),
            "f1": float(primary["f1"]),
            "tp": int(primary["tp"]),
            "fp": int(primary["fp"]),
            "fn": int(primary["fn"]),
        }
    },
    "best_hyperparameters": best_params,
    "delong_vs_catboost": delong_results,
    "runtime_minutes": float(elapsed / 60)
}

with open(RESULTS_DIR / "phase6_resnet_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open(RESULTS_DIR / "phase6_resnet_delong.json", "w") as f:
    json.dump(delong_results, f, indent=2)

np.savez(
    RESULTS_DIR / "phase6_resnet_predictions.npz",
    test_probs=test_probs,
    y_test=y_test
)

log.info("Results saved.")

# ─────────────────────────────────────────────
# 17. FIGURES
# ─────────────────────────────────────────────
log.info("Generating figures...")

COLORS = {
    "resnet"   : "#2196F3",
    "catboost" : "#FF5722",
    "random"   : "#9E9E9E"
}

# ── Figure 1: Optuna optimisation history ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet — Optuna Optimisation History (100 Trials)", fontsize=14, fontweight="bold")

trial_nums  = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_vals  = [t.value  for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
best_so_far = np.maximum.accumulate(trial_vals)

axes[0].scatter(trial_nums, trial_vals, alpha=0.4, color=COLORS["resnet"], s=20, label="Trial AUPRC")
axes[0].plot(trial_nums, best_so_far, color="black", linewidth=2, label="Best so far")
axes[0].set_xlabel("Trial Number"); axes[0].set_ylabel("Validation AUPRC")
axes[0].set_title("AUPRC per Trial"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Parameter importance (top 5)
try:
    importances = optuna.importance.get_param_importances(study)
    top5 = dict(list(importances.items())[:5])
    axes[1].barh(list(top5.keys()), list(top5.values()), color=COLORS["resnet"], alpha=0.8)
    axes[1].set_xlabel("Importance Score"); axes[1].set_title("Top 5 Hyperparameter Importances")
    axes[1].grid(True, alpha=0.3, axis="x")
except Exception:
    axes[1].text(0.5, 0.5, "Importance not available", ha="center", va="center")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "resnet_optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 2: ROC + PR curves ──
fpr_r, tpr_r, _ = roc_curve(y_test, test_probs)
prec_r, rec_r, _ = precision_recall_curve(y_test, test_probs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"ResNet — Test Set Performance (n={len(y_test)})", fontsize=14, fontweight="bold")

axes[0].plot(fpr_r, tpr_r, color=COLORS["resnet"], lw=2,
             label=f"ResNet AUROC = {auroc:.4f}")
axes[0].plot([0,1],[0,1], "--", color=COLORS["random"], lw=1)
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)

axes[1].plot(rec_r, prec_r, color=COLORS["resnet"], lw=2,
             label=f"ResNet AUPRC = {auprc:.4f}")
axes[1].axhline(y_test.mean(), color=COLORS["random"], linestyle="--",
                label=f"No-skill = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "resnet_roc_pr.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3: Calibration curve ──
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "s-", color=COLORS["resnet"], lw=2, label=f"ResNet (Brier={brier:.4f})")
ax.plot([0,1],[0,1], "--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("ResNet — Calibration Curve"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "resnet_calibration.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: Threshold sensitivity ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet — Threshold Sensitivity Analysis", fontsize=14, fontweight="bold")

targets    = thresh_df["target_sens"].values
sens_vals  = thresh_df["achieved_sens"].values
spec_vals  = thresh_df["specificity"].values
f1_vals    = thresh_df["f1"].values
ppv_vals   = thresh_df["ppv"].values

axes[0].plot(targets, sens_vals, "o-", label="Sensitivity", color="#4CAF50")
axes[0].plot(targets, spec_vals, "s-", label="Specificity", color="#2196F3")
axes[0].plot(targets, f1_vals,   "^-", label="F1",          color="#FF9800")
axes[0].plot(targets, ppv_vals,  "D-", label="PPV",         color="#9C27B0")
axes[0].set_xlabel("Target Sensitivity"); axes[0].set_ylabel("Metric Value")
axes[0].set_title("Metrics vs Target Sensitivity"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

tp_vals = thresh_df["tp"].values
fp_vals = thresh_df["fp"].values
fn_vals = thresh_df["fn"].values
x = np.arange(len(targets))
axes[1].bar(x - 0.25, tp_vals, 0.25, label="TP", color="#4CAF50", alpha=0.8)
axes[1].bar(x,         fp_vals, 0.25, label="FP", color="#F44336", alpha=0.8)
axes[1].bar(x + 0.25,  fn_vals, 0.25, label="FN", color="#FF9800", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([f"{t:.0%}" for t in targets])
axes[1].set_xlabel("Target Sensitivity"); axes[1].set_ylabel("Count")
axes[1].set_title("TP / FP / FN by Threshold"); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(FIGURES_DIR / "resnet_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()

log.info("All figures saved.")

# ─────────────────────────────────────────────
# 18. FINAL SUMMARY
# ─────────────────────────────────────────────
log.info("=" * 70)
log.info("PHASE 6 RESNET — COMPLETE")
log.info("=" * 70)
log.info(f"AUROC  : {auroc:.4f} [{auroc_ci[0]:.4f}–{auroc_ci[1]:.4f}]")
log.info(f"AUPRC  : {auprc:.4f} [{auprc_ci[0]:.4f}–{auprc_ci[1]:.4f}]")
log.info(f"Brier  : {brier:.4f}")
log.info(f"Sens   : {primary['achieved_sens']:.4f} @ threshold {primary['threshold']}")
log.info(f"Spec   : {primary['specificity']:.4f}")
log.info(f"F1     : {primary['f1']:.4f}")
log.info(f"Runtime: {elapsed/60:.1f} min")
log.info("-" * 70)
log.info("Outputs saved to:")
log.info(f"  Model   : {MODEL_DIR}")
log.info(f"  Results : {RESULTS_DIR}")
log.info(f"  Figures : {FIGURES_DIR}")
log.info(f"  Log     : {log_path}")
log.info("=" * 70)
