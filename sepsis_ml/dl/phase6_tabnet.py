"""
Phase 6 - Script 3: TabNet (FIXED)
====================================
Fix: batch_size and virtual_batch_size removed from TabNetClassifier()
     constructor — they belong only in .fit() call.
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
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
from optuna.samplers import TPESampler
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
MODEL_DIR    = DL_DIR / "models" / "run_phase6_tabnet"
RESULTS_DIR  = DL_DIR / "results"
FIGURES_DIR  = DL_DIR / "figures"
LOGS_DIR     = DL_DIR / "logs"
OPTUNA_DIR   = DL_DIR / "optuna_studies"

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, OPTUNA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOGGING
# ─────────────────────────────────────────────
log_path = LOGS_DIR / "phase6_tabnet.log"
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
log.info("PHASE 6 — TABNET (FIXED)")
log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info("=" * 70)

# ─────────────────────────────────────────────
# 2. DEVICE
# ─────────────────────────────────────────────
DEVICE = "cpu"
log.info("Device : CPU (pytorch-tabnet, MPS support limited)")

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
y_train_full = train_df[TARGET].values.astype(np.int64)
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

# ─────────────────────────────────────────────
# 6. CLASS WEIGHT
# ─────────────────────────────────────────────
pos_weight_val = (y_tr == 0).sum() / (y_tr == 1).sum()
WEIGHTS = {0: 1.0, 1: float(pos_weight_val)}
log.info(f"Class weights : {WEIGHTS}")

# ─────────────────────────────────────────────
# 7. OPTUNA OBJECTIVE
# FIX: batch_size / virtual_batch_size ONLY in .fit(), NOT in constructor
# ─────────────────────────────────────────────
def objective(trial):
    n_d        = trial.suggest_categorical("n_d", [8, 16, 32, 64])
    n_a        = trial.suggest_categorical("n_a", [8, 16, 32, 64])
    n_steps    = trial.suggest_int("n_steps", 3, 10)
    gamma      = trial.suggest_float("gamma", 1.0, 2.0)
    lam_sparse = trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True)
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    momentum   = trial.suggest_float("momentum", 0.01, 0.4)
    mask_type  = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    vbs_opts   = [bs for bs in [32, 64, 128] if bs <= batch_size]
    vbs        = trial.suggest_categorical("virtual_batch_size",
                                           vbs_opts if vbs_opts else [32])

    clf = TabNetClassifier(
        n_d              = n_d,
        n_a              = n_a,
        n_steps          = n_steps,
        gamma            = gamma,
        lambda_sparse    = lam_sparse,
        optimizer_fn     = torch.optim.Adam,
        optimizer_params = {"lr": lr},
        scheduler_fn     = torch.optim.lr_scheduler.StepLR,
        scheduler_params = {"step_size": 50, "gamma": 0.9},
        mask_type        = mask_type,
        momentum         = momentum,
        device_name      = DEVICE,
        verbose          = 0,
        seed             = RANDOM_SEED
    )

    try:
        clf.fit(
            X_train            = X_tr_s,
            y_train            = y_tr,
            eval_set           = [(X_val_s, y_val)],
            eval_name          = ["val"],
            eval_metric        = ["auc"],
            max_epochs         = 300,
            patience           = 30,
            weights            = WEIGHTS,
            batch_size         = batch_size,
            virtual_batch_size = vbs,
            drop_last          = False
        )
    except Exception as e:
        log.warning(f"Trial {trial.number} fit failed: {e}")
        return 0.0

    val_probs = clf.predict_proba(X_val_s)[:, 1]
    return average_precision_score(y_val, val_probs)

# ─────────────────────────────────────────────
# 8. RUN OPTUNA
# ─────────────────────────────────────────────
log.info("=" * 50)
log.info("Starting Optuna: 100 trials (TabNet)")
log.info("=" * 50)

study_path = OPTUNA_DIR / "tabnet_study.pkl"
sampler    = TPESampler(seed=RANDOM_SEED)
study      = optuna.create_study(direction="maximize", sampler=sampler,
                                  study_name="tabnet_phase6")
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
# 9. RETRAIN BEST MODEL ON FULL TRAIN SET
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Retraining best model on full training set...")

X_full_s        = scale_df(X_train_full).values.astype(np.float32)
pos_weight_full = (y_train_full == 0).sum() / (y_train_full == 1).sum()
WEIGHTS_FULL    = {0: 1.0, 1: float(pos_weight_full)}

bs  = best_params.get("batch_size", 256)
vbs = best_params.get("virtual_batch_size", 64)
if vbs > bs:
    vbs = bs

final_clf = TabNetClassifier(
    n_d              = best_params["n_d"],
    n_a              = best_params["n_a"],
    n_steps          = best_params["n_steps"],
    gamma            = best_params["gamma"],
    lambda_sparse    = best_params["lambda_sparse"],
    optimizer_fn     = torch.optim.Adam,
    optimizer_params = {"lr": best_params["lr"]},
    scheduler_fn     = torch.optim.lr_scheduler.StepLR,
    scheduler_params = {"step_size": 50, "gamma": 0.9},
    mask_type        = best_params.get("mask_type", "sparsemax"),
    momentum         = best_params["momentum"],
    device_name      = DEVICE,
    verbose          = 0,
    seed             = RANDOM_SEED
)

final_clf.fit(
    X_train            = X_full_s,
    y_train            = y_train_full,
    eval_set           = [(X_val_s, y_val)],
    eval_name          = ["val"],
    eval_metric        = ["auc"],
    max_epochs         = 1000,
    patience           = 50,
    weights            = WEIGHTS_FULL,
    batch_size         = bs,
    virtual_batch_size = vbs,
    drop_last          = False
)

save_path = str(MODEL_DIR / "tabnet_best")
final_clf.save_model(save_path)
log.info(f"Final model saved to {save_path}.zip")

# ─────────────────────────────────────────────
# 10. TEST SET EVALUATION
# ─────────────────────────────────────────────
log.info("-" * 50)
log.info("Evaluating on test set...")

test_probs = final_clf.predict_proba(X_te_s)[:, 1]

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
thresh_df.to_csv(RESULTS_DIR / "phase6_tabnet_threshold_table.csv", index=False)
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
# 11. TABNET FEATURE IMPORTANCE
# ─────────────────────────────────────────────
log.info("Extracting TabNet feature importances...")
fi_df = None
try:
    fi_df = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": final_clf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(RESULTS_DIR / "phase6_tabnet_feature_importance.csv", index=False)
    log.info("Top 10 features:")
    for _, row in fi_df.head(10).iterrows():
        log.info(f"  {row['feature']:<40} {row['importance']:.4f}")
except Exception as e:
    log.warning(f"Feature importance failed: {e}")

# ─────────────────────────────────────────────
# 12. DELONG TEST vs TUNED CATBOOST
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
        auc_tn, auc_cb, z_stat, p_val = delong_test(y_test, test_probs, cb_probs)
        delong_results = {
            "tabnet_auroc": auc_tn, "catboost_auroc": auc_cb,
            "z_statistic": z_stat, "p_value": p_val,
            "significant": p_val < 0.05,
            "direction": "TabNet better" if auc_tn > auc_cb else "CatBoost better"
        }
        log.info(f"\nDeLong vs CatBoost: z={z_stat:.4f}, p={p_val:.4f}")
        log.info(f"Significant: {p_val < 0.05} | {delong_results['direction']}")
    else:
        delong_results = {"note": "CatBoost predictions not found"}
except Exception as e:
    delong_results = {"error": str(e)}

# ─────────────────────────────────────────────
# 13. SAVE RESULTS
# ─────────────────────────────────────────────
results = {
    "model": "TabNet", "phase": "Phase 6",
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
    "best_hyperparameters": best_params,
    "delong_vs_catboost": delong_results,
    "runtime_minutes": float(elapsed / 60)
}

with open(RESULTS_DIR / "phase6_tabnet_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(RESULTS_DIR / "phase6_tabnet_delong.json", "w") as f:
    json.dump(delong_results, f, indent=2)
np.savez(RESULTS_DIR / "phase6_tabnet_predictions.npz",
         test_probs=test_probs, y_test=y_test)
log.info("Results saved.")

# ─────────────────────────────────────────────
# 14. FIGURES
# ─────────────────────────────────────────────
log.info("Generating figures...")
COLORS = {"tabnet": "#4CAF50", "random": "#9E9E9E"}

# ── Figure 1: Optuna history ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TabNet — Optuna Optimisation History (100 Trials)",
             fontsize=14, fontweight="bold")
trial_nums  = [t.number for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_vals  = [t.value  for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
best_so_far = np.maximum.accumulate(trial_vals) if trial_vals else []
axes[0].scatter(trial_nums, trial_vals, alpha=0.4, color=COLORS["tabnet"], s=20)
if len(best_so_far):
    axes[0].plot(trial_nums, best_so_far, color="black", lw=2, label="Best so far")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Validation AUPRC")
axes[0].set_title("AUPRC per Trial"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
try:
    importances = optuna.importance.get_param_importances(study)
    top5 = dict(list(importances.items())[:5])
    axes[1].barh(list(top5.keys()), list(top5.values()), color=COLORS["tabnet"], alpha=0.8)
    axes[1].set_xlabel("Importance"); axes[1].set_title("Top 5 Hyperparameter Importances")
    axes[1].grid(True, alpha=0.3, axis="x")
except Exception:
    axes[1].text(0.5, 0.5, "Not available", ha="center", va="center")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tabnet_optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 2: ROC + PR ──
fpr_t, tpr_t, _  = roc_curve(y_test, test_probs)
prec_t, rec_t, _ = precision_recall_curve(y_test, test_probs)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"TabNet — Test Set Performance (n={len(y_test)})",
             fontsize=14, fontweight="bold")
axes[0].plot(fpr_t, tpr_t, color=COLORS["tabnet"], lw=2,
             label=f"TabNet AUROC = {auroc:.4f}")
axes[0].plot([0,1],[0,1], "--", color=COLORS["random"], lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend(loc="lower right"); axes[0].grid(True, alpha=0.3)
axes[1].plot(rec_t, prec_t, color=COLORS["tabnet"], lw=2,
             label=f"TabNet AUPRC = {auprc:.4f}")
axes[1].axhline(y_test.mean(), color=COLORS["random"], linestyle="--",
                label=f"No-skill = {y_test.mean():.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tabnet_roc_pr.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3: Calibration ──
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=10)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(prob_pred, prob_true, "s-", color=COLORS["tabnet"], lw=2,
        label=f"TabNet (Brier={brier:.4f})")
ax.plot([0,1],[0,1], "--", color="gray", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
ax.set_title("TabNet — Calibration Curve"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "tabnet_calibration.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4: Threshold sensitivity ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("TabNet — Threshold Sensitivity Analysis", fontsize=14, fontweight="bold")
targets = thresh_df["target_sens"].values
axes[0].plot(targets, thresh_df["achieved_sens"].values, "o-", label="Sensitivity", color="#4CAF50")
axes[0].plot(targets, thresh_df["specificity"].values,   "s-", label="Specificity", color=COLORS["tabnet"])
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
plt.savefig(FIGURES_DIR / "tabnet_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 5: Feature Importance ──
if fi_df is not None and len(fi_df) > 0:
    top20 = fi_df.head(20).copy()
    top20["feature_clean"] = top20["feature"].str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top20)), top20["importance"].values,
                   color=COLORS["tabnet"], alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature_clean"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("TabNet Feature Importance (Attention Mask Weight)")
    ax.set_title("TabNet — Top 20 Feature Importances", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    for i, (bar, val) in enumerate(zip(bars, top20["importance"].values)):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tabnet_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

log.info("All figures saved.")

# ─────────────────────────────────────────────
# 15. FINAL SUMMARY
# ─────────────────────────────────────────────
log.info("=" * 70)
log.info("PHASE 6 TABNET — COMPLETE")
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