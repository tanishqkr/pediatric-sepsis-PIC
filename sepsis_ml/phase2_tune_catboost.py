"""
phase2_tune_catboost.py
-----------------------
Optuna Bayesian hyperparameter optimization for CatBoost.
100 trials, optimizing AUPRC on 5-fold stratified CV.
This is the primary model for the paper.

Search space:
  - learning_rate       : 0.01 – 0.3
  - depth               : 4 – 10
  - iterations          : 200 – 1500
  - l2_leaf_reg         : 1 – 10
  - bagging_temperature : 0.0 – 1.0
  - random_strength     : 0.0 – 2.0
  - border_count        : 32 – 255
  - class_weight_pos    : 1.5 – 5.0  (weight for sepsis class)

Outputs:
  models/run_tuned/catboost_tuned.cbm        — best tuned model
  results/optuna_study.pkl                   — full Optuna study object
  results/best_params.json                   — best hyperparameters
  results/tuning_trials.csv                  — all 100 trial results
  figures/phase2_tuning/                     — optimization plots
  logs/phase2_tune_catboost.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase2_tune_catboost.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output (paste)
  - figures/phase2_tuning/optuna_optimization_history.png
  - figures/phase2_tuning/optuna_param_importances.png
  - figures/phase2_tuning/tuned_vs_baseline_roc_pr.png
  - figures/phase2_tuning/tuned_cm.png
  - results/best_params.json
  - results/tuning_trials.csv
"""

import sys
import json
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TARGET_COL, RANDOM_SEED, CV_FOLDS,
    MODELS_DIR, FIG_TUNING, LOGS_DIR, RESULTS_DIR,
    TARGET_SENSITIVITY, BOOTSTRAP_ITERATIONS, OPTUNA_TRIALS
)
from logger import log_experiment

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

# ── Tuned model output folder ─────────────────────────────────────────────────
TUNED_MODEL_DIR = MODELS_DIR / "run_tuned"
TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase2_tune_catboost.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()
np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers (same as phase 1 for consistency)
# ══════════════════════════════════════════════════════════════════════════════

def find_threshold_at_sensitivity(y_true, y_prob, target_sens=TARGET_SENSITIVITY):
    from sklearn.metrics import roc_curve
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


def bootstrap_ci(y_true, y_prob, metric_fn, n_iter=BOOTSTRAP_ITERATIONS, seed=RANDOM_SEED):
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
# Optuna objective
# ══════════════════════════════════════════════════════════════════════════════

def objective(trial, X_df: pd.DataFrame, y: np.ndarray) -> float:
    """
    Optuna objective. Returns mean CV AUPRC (higher = better).
    AUPRC is the correct metric for imbalanced clinical data —
    it is more sensitive to minority class performance than AUROC.
    """
    from catboost import CatBoostClassifier

    params = {
        "iterations"        : trial.suggest_int("iterations", 200, 1500),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth"             : trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg"       : trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength"   : trial.suggest_float("random_strength", 0.0, 2.0),
        "border_count"      : trial.suggest_int("border_count", 32, 255),
        "class_weights"     : [1.0, trial.suggest_float("class_weight_pos", 1.5, 5.0)],
        "eval_metric"       : "AUC",
        "random_seed"       : RANDOM_SEED,
        "verbose"           : 0,
        "early_stopping_rounds": 50,
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    auprc_scores = []

    for train_idx, val_idx in skf.split(X_df, y):
        X_tr  = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_tr  = y[train_idx]
        y_val = y[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            verbose=0,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc_scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(auprc_scores))


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_optimization_history(study):
    """Plot AUPRC across all trials — shows search convergence."""
    trials_df = study.trials_dataframe()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Trial values over time
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i+1]) for i in range(len(values))]
    ax1.scatter(range(len(values)), values, alpha=0.4, s=15, color="#3498db", label="Trial AUPRC")
    ax1.plot(range(len(best_so_far)), best_so_far, color="#e74c3c", lw=2, label="Best so far")
    ax1.set_xlabel("Trial number")
    ax1.set_ylabel("CV AUPRC")
    ax1.set_title("Optuna optimization history\nCatBoost — 100 trials", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # Distribution of trial AUPRCs
    ax2.hist(values, bins=20, color="#3498db", alpha=0.7, edgecolor="white")
    ax2.axvline(max(values), color="#e74c3c", lw=2, linestyle="--",
                label=f"Best: {max(values):.4f}")
    ax2.set_xlabel("CV AUPRC")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of trial scores", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_TUNING / "optuna_optimization_history.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_param_importances(study):
    """Plot which hyperparameters mattered most."""
    try:
        import optuna.visualization.matplotlib as optuna_mpl
        fig, ax = plt.subplots(figsize=(8, 5))
        importances = optuna.importance.get_param_importances(study)
        params  = list(importances.keys())
        values  = list(importances.values())
        colors  = ["#e74c3c" if v == max(values) else "#3498db" for v in values]
        ax.barh(params, values, color=colors, edgecolor="white")
        ax.set_xlabel("Importance score (fANOVA)")
        ax.set_title("Hyperparameter importance\nCatBoost Optuna study", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = FIG_TUNING / "optuna_param_importances.png"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"  Saved → {out}")
    except Exception as e:
        log.info(f"  Param importance plot skipped: {e}")


def plot_tuned_vs_baseline(y_test, y_prob_tuned, y_prob_baseline, metrics_tuned, metrics_baseline):
    """ROC + PR curves comparing tuned vs baseline CatBoost."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    for y_prob, label, color, lw in [
        (y_prob_tuned,    f"Tuned CatBoost (AUROC={metrics_tuned['auroc']:.4f})",    "#e74c3c", 2.5),
        (y_prob_baseline, f"Baseline CatBoost (AUROC={metrics_baseline['auroc']:.4f})", "#3498db", 1.5),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, label=label)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.fill_between(*roc_curve(y_test, y_prob_tuned)[:2],
                     alpha=0.08, color="#e74c3c")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC — Tuned vs Baseline CatBoost\nOption B (Infection-only)",
                  fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    # PR
    for y_prob, label, color, lw in [
        (y_prob_tuned,    f"Tuned CatBoost (AUPRC={metrics_tuned['auprc']:.4f})",    "#e74c3c", 2.5),
        (y_prob_baseline, f"Baseline CatBoost (AUPRC={metrics_baseline['auprc']:.4f})", "#3498db", 1.5),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=lw, label=label)
    baseline_prev = y_test.mean()
    ax2.axhline(baseline_prev, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Prevalence ({baseline_prev:.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall — Tuned vs Baseline CatBoost\nOption B (Infection-only)",
                  fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_TUNING / "tuned_vs_baseline_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_confusion_matrix(y_true, y_pred, metrics, title="Tuned CatBoost"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-sepsis", "Sepsis"])
    ax.set_yticklabels(["Non-sepsis", "Sepsis"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\n"
                 f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
                 f"Threshold={metrics['threshold']:.4f}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = FIG_TUNING / "tuned_cm.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_threshold_sensitivity(y_test, y_prob_tuned):
    """
    Show performance across multiple sensitivity targets.
    Critical for paper — shows clinicians the tradeoff space.
    """
    targets = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    rows = []
    for sens_target in targets:
        thresh = find_threshold_at_sensitivity(y_test, y_prob_tuned, sens_target)
        m = compute_metrics(y_test, y_prob_tuned, thresh)
        rows.append({
            "target_sensitivity": sens_target,
            "threshold"         : round(thresh, 4),
            "sensitivity"       : round(m["sensitivity"], 4),
            "specificity"       : round(m["specificity"], 4),
            "ppv"               : round(m["ppv"], 4),
            "npv"               : round(m["npv"], 4),
            "f1"                : round(m["f1"], 4),
            "tp"                : m["tp"],
            "fp"                : m["fp"],
            "tn"                : m["tn"],
            "fn"                : m["fn"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Threshold sensitivity analysis — Tuned CatBoost\n"
                 "Option B (Infection-only)", fontsize=11, fontweight="bold")

    sens_vals = df["sensitivity"].values
    spec_vals = df["specificity"].values
    ppv_vals  = df["ppv"].values
    npv_vals  = df["npv"].values
    f1_vals   = df["f1"].values
    thresh_vals = df["threshold"].values

    axes[0].plot(thresh_vals, sens_vals, "o-", color="#e74c3c", lw=2, label="Sensitivity")
    axes[0].plot(thresh_vals, spec_vals, "s-", color="#3498db", lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(thresh_vals, ppv_vals, "o-", color="#2ecc71", lw=2, label="PPV (Precision)")
    axes[1].plot(thresh_vals, npv_vals, "s-", color="#9b59b6", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)

    axes[2].plot(thresh_vals, f1_vals, "o-", color="#f39c12", lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_TUNING / "threshold_sensitivity_analysis.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved threshold analysis → {out}")

    log.info("\n  Threshold sensitivity analysis:")
    log.info(f"  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    log.info("  " + "-"*75)
    for _, row in df.iterrows():
        log.info(f"  {row['target_sensitivity']:>8.2f} {row['threshold']:>8.4f} "
                 f"{row['sensitivity']:>7.4f} {row['specificity']:>7.4f} "
                 f"{row['ppv']:>7.4f} {row['npv']:>7.4f} {row['f1']:>7.4f} "
                 f"{int(row['tp']):>5} {int(row['fp']):>5} {int(row['fn']):>5}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# DeLong test for AUROC comparison
# ══════════════════════════════════════════════════════════════════════════════

def delong_test(y_true, y_prob_1, y_prob_2):
    """
    DeLong test for statistical significance of AUROC difference.
    Returns z-statistic and p-value.
    Reference: DeLong et al. (1988) Biometrics.
    """
    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
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

    y_true  = np.array(y_true)
    y_prob_1 = np.array(y_prob_1)
    y_prob_2 = np.array(y_prob_2)

    order   = np.argsort(-y_true)
    label_1_count = int(y_true.sum())

    predictions_sorted = np.vstack([y_prob_1[order], y_prob_2[order]])
    aucs, delongcov = fastDeLong(predictions_sorted, label_1_count)

    diff      = aucs[0] - aucs[1]
    se        = np.sqrt(delongcov[0,0] + delongcov[1,1] - 2*delongcov[0,1])
    z_stat    = diff / se
    from scipy import stats
    p_value   = 2 * stats.norm.sf(abs(z_stat))

    return float(aucs[0]), float(aucs[1]), float(z_stat), float(p_value)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("=" * 65)
    log.info("PHASE 2A — CATBOOST OPTUNA TUNING")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Trials: {OPTUNA_TRIALS}  |  CV folds: {CV_FOLDS}  |  Metric: AUPRC")
    log.info("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("\nLoading model-ready datasets...")
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train.columns if c != TARGET_COL]
    X_train_df   = train[feature_cols]
    y_train      = train[TARGET_COL].values
    X_test_df    = test[feature_cols]
    y_test       = test[TARGET_COL].values
    log.info(f"  Train: {train.shape}  |  Test: {test.shape}  |  Features: {len(feature_cols)}")

    # ── Load baseline CatBoost for comparison ─────────────────────────────────
    from catboost import CatBoostClassifier
    baseline_model_path = MODELS_DIR / "run_003" / "catboost.cbm"
    if baseline_model_path.exists():
        baseline_model = CatBoostClassifier()
        baseline_model.load_model(str(baseline_model_path))
        y_prob_baseline = baseline_model.predict_proba(X_test_df)[:, 1]
        metrics_baseline = compute_metrics(y_test, y_prob_baseline)
        log.info(f"\n  Baseline CatBoost: AUROC={metrics_baseline['auroc']:.4f}  "
                 f"AUPRC={metrics_baseline['auprc']:.4f}")
    else:
        log.info("  Baseline model not found — skipping comparison")
        y_prob_baseline  = None
        metrics_baseline = None

    # ── Optuna study ──────────────────────────────────────────────────────────
    log.info(f"\nStarting Optuna study ({OPTUNA_TRIALS} trials)...")
    log.info("This will take approximately 20-40 minutes.")
    log.info("Progress printed every 10 trials.\n")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name = "catboost_auprc_optuna",
    )

    best_so_far = 0.0

    def callback(study, trial):
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            log.info(f"  Trial {trial.number:>3} NEW BEST: AUPRC={trial.value:.4f}  "
                     f"params={trial.params}")
        elif trial.number % 10 == 0:
            log.info(f"  Trial {trial.number:>3}: AUPRC={trial.value:.4f}  "
                     f"(best so far: {best_so_far:.4f})")

    study.optimize(
        lambda trial: objective(trial, X_train_df, y_train),
        n_trials  = OPTUNA_TRIALS,
        callbacks = [callback],
        show_progress_bar = False,
    )

    log.info(f"\n  Optuna complete.")
    log.info(f"  Best CV AUPRC: {study.best_value:.4f}")
    log.info(f"  Best params:   {study.best_params}")

    # ── Save study and params ─────────────────────────────────────────────────
    study_path = RESULTS_DIR / "optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    log.info(f"\n  Optuna study saved → {study_path}")

    best_params_path = RESULTS_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "best_cv_auprc": study.best_value,
            "best_params"  : study.best_params,
            "n_trials"     : OPTUNA_TRIALS,
            "timestamp"    : datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    log.info(f"  Best params saved → {best_params_path}")

    # ── Save all trials to CSV ────────────────────────────────────────────────
    trials_df = study.trials_dataframe()
    trials_path = RESULTS_DIR / "tuning_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    log.info(f"  All trials saved → {trials_path}")

    # ── Train final tuned model on full training set ───────────────────────────
    log.info(f"\nTraining final tuned CatBoost on full training set...")
    best = study.best_params

    tuned_model = CatBoostClassifier(
        iterations          = best["iterations"],
        learning_rate       = best["learning_rate"],
        depth               = best["depth"],
        l2_leaf_reg         = best["l2_leaf_reg"],
        bagging_temperature = best["bagging_temperature"],
        random_strength     = best["random_strength"],
        border_count        = best["border_count"],
        class_weights       = [1.0, best["class_weight_pos"]],
        eval_metric         = "AUC",
        random_seed         = RANDOM_SEED,
        verbose             = 100,
        early_stopping_rounds = 50,
    )
    tuned_model.fit(
        X_train_df, y_train,
        eval_set=(X_test_df, y_test),
    )

    # ── Test set evaluation with bootstrap CIs ────────────────────────────────
    log.info(f"\nEvaluating tuned model on test set...")
    y_prob_tuned = tuned_model.predict_proba(X_test_df)[:, 1]
    metrics_tuned = compute_metrics_with_ci(y_test, y_prob_tuned)

    log.info(f"\n  {'='*55}")
    log.info(f"  TUNED CATBOOST — TEST SET RESULTS")
    log.info(f"  {'='*55}")
    log.info(f"  AUROC : {metrics_tuned['auroc']:.4f} "
             f"[{metrics_tuned['auroc_ci_low']:.4f}–{metrics_tuned['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_tuned['auprc']:.4f} "
             f"[{metrics_tuned['auprc_ci_low']:.4f}–{metrics_tuned['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_tuned['sensitivity']:.4f}  "
             f"Spec: {metrics_tuned['specificity']:.4f}  "
             f"F1: {metrics_tuned['f1']:.4f}")
    log.info(f"  PPV   : {metrics_tuned['ppv']:.4f}  NPV: {metrics_tuned['npv']:.4f}  "
             f"Brier: {metrics_tuned['brier_score']:.4f}")
    log.info(f"  TP={metrics_tuned['tp']}  FP={metrics_tuned['fp']}  "
             f"TN={metrics_tuned['tn']}  FN={metrics_tuned['fn']}")

    # ── Improvement over baseline ─────────────────────────────────────────────
    if metrics_baseline:
        auroc_delta = metrics_tuned["auroc"] - metrics_baseline["auroc"]
        auprc_delta = metrics_tuned["auprc"] - metrics_baseline["auprc"]
        log.info(f"\n  Improvement over baseline:")
        log.info(f"    AUROC: {metrics_baseline['auroc']:.4f} → {metrics_tuned['auroc']:.4f} "
                 f"(Δ={auroc_delta:+.4f})")
        log.info(f"    AUPRC: {metrics_baseline['auprc']:.4f} → {metrics_tuned['auprc']:.4f} "
                 f"(Δ={auprc_delta:+.4f})")

        # DeLong test
        log.info(f"\n  DeLong test (tuned vs baseline AUROC):")
        auroc_1, auroc_2, z_stat, p_value = delong_test(y_test, y_prob_tuned, y_prob_baseline)
        log.info(f"    Tuned AUROC:   {auroc_1:.4f}")
        log.info(f"    Baseline AUROC:{auroc_2:.4f}")
        log.info(f"    Z-statistic:   {z_stat:.4f}")
        log.info(f"    P-value:       {p_value:.4f}")
        log.info(f"    Significant:   {'YES (p<0.05)' if p_value < 0.05 else 'NO (p>=0.05)'}")
        metrics_tuned["delong_z"]       = z_stat
        metrics_tuned["delong_p"]       = p_value
        metrics_tuned["auroc_delta_vs_baseline"] = auroc_delta
        metrics_tuned["auprc_delta_vs_baseline"] = auprc_delta

    # ── Save tuned model ──────────────────────────────────────────────────────
    tuned_model_path = TUNED_MODEL_DIR / "catboost_tuned.cbm"
    tuned_model.save_model(str(tuned_model_path))
    log.info(f"\n  Tuned model saved → {tuned_model_path}")

    # ── Log to experiment tracker ─────────────────────────────────────────────
    log_experiment(
        experiment_id = "phase2_catboost_tuned_B",
        dataset       = "B",
        model_name    = "catboost",
        phase         = "optuna_tuned",
        params        = best,
        metrics       = metrics_tuned,
        notes         = f"Optuna {OPTUNA_TRIALS} trials, AUPRC optimized, "
                        f"5-fold CV, early stopping 50 rounds",
        extra         = {
            "best_cv_auprc" : study.best_value,
            "n_trials"      : OPTUNA_TRIALS,
            "n_features"    : len(feature_cols),
        }
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    plot_optimization_history(study)
    plot_param_importances(study)

    if y_prob_baseline is not None:
        plot_tuned_vs_baseline(y_test, y_prob_tuned, y_prob_baseline,
                               metrics_tuned, metrics_baseline)

    y_pred_tuned = (y_prob_tuned >= metrics_tuned["threshold"]).astype(int)
    plot_confusion_matrix(y_test, y_pred_tuned, metrics_tuned)
    threshold_df = plot_threshold_sensitivity(y_test, y_prob_tuned)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("PHASE 2A COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  Best CV AUPRC (Optuna):  {study.best_value:.4f}")
    log.info(f"  Test AUROC:              {metrics_tuned['auroc']:.4f} "
             f"[{metrics_tuned['auroc_ci_low']:.4f}–{metrics_tuned['auroc_ci_high']:.4f}]")
    log.info(f"  Test AUPRC:              {metrics_tuned['auprc']:.4f} "
             f"[{metrics_tuned['auprc_ci_low']:.4f}–{metrics_tuned['auprc_ci_high']:.4f}]")
    log.info(f"  Sensitivity:             {metrics_tuned['sensitivity']:.4f}")
    log.info(f"  Specificity:             {metrics_tuned['specificity']:.4f}")
    log.info(f"  Tuned model:             {tuned_model_path}")
    log.info(f"  Figures:                 {FIG_TUNING}/")
    log.info(f"  Log:                     {log_path}")
    log.info(f"\n  Next step: python sepsis_ml/phase2b_evaluate_tuned.py")


if __name__ == "__main__":
    main()
