"""
phase1_train_all_models.py
--------------------------
Trains all 6 models on Option B (infection-only) with class weights.
5-fold stratified CV on train, final evaluation on held-out test.

Each run creates a versioned subfolder:
  models/run_001/         — saved model files
  figures/phase1_baseline/run_001/  — all plots for this run
  results/experiment_log.json       — appended (never overwritten)
  results/results_summary.csv       — appended (never overwritten)

Fix vs previous version:
  - DataFrames (not numpy arrays) passed to all models so feature names
    are embedded in CatBoost and available for SHAP later
  - Versioned run folders — rerunning never overwrites previous results

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase1_train_all_models.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output (paste here)
  - figures/phase1_baseline/run_XXX/phase1_roc_pr_all_models.png
  - figures/phase1_baseline/run_XXX/phase1_calibration_all_models.png
  - figures/phase1_baseline/run_XXX/phase1_metrics_comparison.png
  - figures/phase1_baseline/run_XXX/phase1_cm_catboost.png
  - figures/phase1_baseline/run_XXX/phase1_cm_xgboost.png
  - figures/phase1_baseline/run_XXX/phase1_cm_lightgbm.png
  - figures/phase1_baseline/run_XXX/phase1_cm_random_forest.png
  - figures/phase1_baseline/run_XXX/phase1_cm_logistic_regression.png
  - figures/phase1_baseline/run_XXX/phase1_cm_mlp.png
  - results/results_summary.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TARGET_COL, RANDOM_SEED, CV_FOLDS,
    MODELS_DIR, FIG_BASELINE, LOGS_DIR,
    TARGET_SENSITIVITY, BOOTSTRAP_ITERATIONS
)
from logger import log_experiment, print_leaderboard

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

# ── Versioned run folder ──────────────────────────────────────────────────────
def get_run_dir(base_dir: Path) -> Path:
    """Create next versioned run folder: run_001, run_002, etc."""
    existing = sorted(base_dir.glob("run_*"))
    next_n   = len(existing) + 1
    run_dir  = base_dir / f"run_{next_n:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

MODELS_RUN_DIR = get_run_dir(MODELS_DIR)
FIG_RUN_DIR    = get_run_dir(FIG_BASELINE)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase1_train_all_models.log"
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

# ── Class weights for Option B (2.7:1 negative:positive) ─────────────────────
CLASS_WEIGHT     = {0: 1.0, 1: 2.7}
SCALE_POS_WEIGHT = 2.7


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions
# ══════════════════════════════════════════════════════════════════════════════

def get_models():
    from catboost import CatBoostClassifier
    import xgboost as xgb
    import lightgbm as lgb

    return {
        "catboost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            class_weights=[1.0, 2.7],
            eval_metric="AUC",
            random_seed=RANDOM_SEED,
            verbose=0,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=SCALE_POS_WEIGHT,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
            verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=SCALE_POS_WEIGHT,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight=CLASS_WEIGHT,
                max_iter=1000,
                random_state=RANDOM_SEED,
                solver="lbfgs",
                C=1.0,
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=200,
                random_state=RANDOM_SEED,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
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
# Cross-validation — passes DataFrame to preserve feature names
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(model, X_df: pd.DataFrame, y: np.ndarray, model_name: str):
    """5-fold stratified CV. Passes DataFrame so CatBoost stores feature names."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
        X_tr  = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_tr  = y[train_idx]
        y_val = y[val_idx]

        # sklearn Pipelines need numpy; CatBoost works with both
        if model_name in ("logistic_regression", "mlp"):
            model.fit(X_tr.values, y_tr)
            y_prob = model.predict_proba(X_val.values)[:, 1]
        else:
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]

        m = compute_metrics(y_val, y_prob)
        fold_metrics.append(m)
        log.info(f"    Fold {fold}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                 f"Sens={m['sensitivity']:.4f}  Spec={m['specificity']:.4f}")

    mean_metrics = {}
    for key in ["auroc", "auprc", "f1", "sensitivity", "specificity",
                "ppv", "npv", "brier_score"]:
        vals = [f[key] for f in fold_metrics]
        mean_metrics[f"cv_{key}_mean"] = float(np.mean(vals))
        mean_metrics[f"cv_{key}_std"]  = float(np.std(vals))

    log.info(f"    CV mean: AUROC={mean_metrics['cv_auroc_mean']:.4f}"
             f"±{mean_metrics['cv_auroc_std']:.4f}  "
             f"AUPRC={mean_metrics['cv_auprc_mean']:.4f}"
             f"±{mean_metrics['cv_auprc_std']:.4f}")
    return mean_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Plots — all saved to versioned run folder
# ══════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name: str):
    out = FIG_RUN_DIR / name
    fig.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"  Saved → {out}")


def plot_roc_pr_all(results: dict, y_test: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    for (name, res), color in zip(results.items(), colors):
        y_prob     = res["y_prob_test"]
        label_name = name.replace("_", " ").title()
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=1.5,
                 label=f"{label_name} (AUROC={res['test_metrics']['auroc']:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(rec, prec, color=color, lw=1.5,
                 label=f"{label_name} (AUPRC={res['test_metrics']['auprc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves — All Models\nOption B", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)
    baseline = y_test.mean()
    ax2.axhline(baseline, color="k", linestyle="--", lw=0.8, alpha=0.5,
                label=f"Baseline (prevalence={baseline:.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves — All Models\nOption B", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "phase1_roc_pr_all_models.png")


def plot_calibration_all(results: dict, y_test: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    for (name, res), color in zip(results.items(), colors):
        fraction_pos, mean_pred = calibration_curve(y_test, res["y_prob_test"], n_bins=10)
        ax.plot(mean_pred, fraction_pos, "s-", color=color, lw=1.5, markersize=4,
                label=f"{name.replace('_',' ').title()} "
                      f"(Brier={res['test_metrics']['brier_score']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves — All Models\nOption B", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "phase1_calibration_all_models.png")


def plot_metrics_comparison(results: dict):
    model_names  = [n.replace("_", "\n").title() for n in results.keys()]
    metrics_plot = ["auroc", "auprc", "f1", "sensitivity", "specificity"]
    colors_map   = {"auroc": "#2ecc71", "auprc": "#e74c3c", "f1": "#3498db",
                    "sensitivity": "#9b59b6", "specificity": "#f39c12"}
    fig, axes = plt.subplots(1, len(metrics_plot), figsize=(16, 5))
    fig.suptitle("Model Comparison — Option B — Test Set", fontsize=12, fontweight="bold")
    for ax, metric in zip(axes, metrics_plot):
        vals = [res["test_metrics"][metric] for res in results.values()]
        bars = ax.bar(model_names, vals, color=colors_map[metric],
                      alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(metric.upper(), fontweight="bold", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "phase1_metrics_comparison.png")


def plot_confusion_matrix(model_name: str, y_true, y_pred, metrics: dict):
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
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{model_name.replace('_',' ').title()}\n"
                 f"Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}  "
                 f"Threshold={metrics['threshold']:.3f}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, f"phase1_cm_{model_name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("PHASE 1 — BASELINE TRAINING (All 6 Models, Option B)")
    log.info(f"Started:   {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Models dir: {MODELS_RUN_DIR}")
    log.info(f"Figs dir:   {FIG_RUN_DIR}")
    log.info("=" * 60)

    # ── Load data — keep as DataFrames ────────────────────────────────────────
    log.info("\nLoading model-ready datasets...")
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    log.info(f"  Train: {train.shape}  |  Test: {test.shape}")

    feature_cols = [c for c in train.columns if c != TARGET_COL]
    X_train_df   = train[feature_cols]          # DataFrame — preserves feature names
    y_train      = train[TARGET_COL].values
    X_test_df    = test[feature_cols]
    y_test       = test[TARGET_COL].values

    log.info(f"  Features: {len(feature_cols)}")
    log.info(f"  Train — sepsis: {y_train.sum()}  non-sepsis: {(y_train==0).sum()}")
    log.info(f"  Test  — sepsis: {y_test.sum()}   non-sepsis: {(y_test==0).sum()}")

    # ── Train all models ──────────────────────────────────────────────────────
    all_models  = get_models()
    all_results = {}

    for model_name, model in all_models.items():
        log.info(f"\n{'='*50}")
        log.info(f"  MODEL: {model_name.upper()}")
        log.info(f"{'='*50}")

        # Cross-validation
        log.info(f"  Running {CV_FOLDS}-fold stratified CV...")
        cv_metrics = cross_validate(model, X_train_df, y_train, model_name)

        # Final fit on full training set
        log.info(f"  Fitting on full training set...")
        if model_name in ("logistic_regression", "mlp"):
            model.fit(X_train_df.values, y_train)
            y_prob_test = model.predict_proba(X_test_df.values)[:, 1]
        else:
            model.fit(X_train_df, y_train)
            y_prob_test = model.predict_proba(X_test_df)[:, 1]

        # Test set evaluation with bootstrap CIs
        test_metrics = compute_metrics_with_ci(y_test, y_prob_test)

        log.info(f"  Test AUROC : {test_metrics['auroc']:.4f} "
                 f"[{test_metrics['auroc_ci_low']:.4f}–{test_metrics['auroc_ci_high']:.4f}]")
        log.info(f"  Test AUPRC : {test_metrics['auprc']:.4f} "
                 f"[{test_metrics['auprc_ci_low']:.4f}–{test_metrics['auprc_ci_high']:.4f}]")
        log.info(f"  Test Sens  : {test_metrics['sensitivity']:.4f}  "
                 f"Spec: {test_metrics['specificity']:.4f}  "
                 f"F1: {test_metrics['f1']:.4f}  "
                 f"PPV: {test_metrics['ppv']:.4f}  NPV: {test_metrics['npv']:.4f}")
        log.info(f"  Threshold  : {test_metrics['threshold']:.4f} "
                 f"(at ~{TARGET_SENSITIVITY*100:.0f}% sensitivity)")
        log.info(f"  TP={test_metrics['tp']}  FP={test_metrics['fp']}  "
                 f"TN={test_metrics['tn']}  FN={test_metrics['fn']}")

        # Confusion matrix plot
        y_pred_test = (y_prob_test >= test_metrics["threshold"]).astype(int)
        plot_confusion_matrix(model_name, y_test, y_pred_test, test_metrics)

        # Save model to versioned run folder
        if model_name == "catboost":
            model_path = MODELS_RUN_DIR / f"{model_name}.cbm"
            model.save_model(str(model_path))
        else:
            model_path = MODELS_RUN_DIR / f"{model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        log.info(f"  Saved model → {model_path}")

        # Log to experiment tracker
        combined_metrics = {**test_metrics, **cv_metrics}
        log_experiment(
            experiment_id = f"phase1_{model_name}_B_baseline_{MODELS_RUN_DIR.name}",
            dataset       = "B",
            model_name    = model_name,
            phase         = "baseline",
            params        = {"class_weight": "0:1 1:2.7", "run": MODELS_RUN_DIR.name},
            metrics       = combined_metrics,
            notes         = f"Baseline {model_name}, class_weights={{0:1, 1:2.7}}, "
                            f"5-fold CV, threshold at {TARGET_SENSITIVITY*100:.0f}% sensitivity",
            extra         = {
                "n_train"   : int(len(y_train)),
                "n_test"    : int(len(y_test)),
                "n_features": int(len(feature_cols)),
                "run_folder": str(MODELS_RUN_DIR.name),
            }
        )

        all_results[model_name] = {
            "model"       : model,
            "y_prob_test" : y_prob_test,
            "test_metrics": test_metrics,
            "cv_metrics"  : cv_metrics,
        }

    # ── Combined plots ────────────────────────────────────────────────────────
    log.info("\n--- Generating combined figures ---")
    plot_roc_pr_all(all_results, y_test)
    plot_calibration_all(all_results, y_test)
    plot_metrics_comparison(all_results)

    # ── Leaderboard ───────────────────────────────────────────────────────────
    print_leaderboard(dataset="B", phase="baseline")

    # ── Summary table ─────────────────────────────────────────────────────────
    log.info("\n--- RESULTS SUMMARY TABLE (Test Set) ---")
    log.info(f"{'Model':<22} {'AUROC':>7} {'95% CI':<16} {'AUPRC':>7} "
             f"{'F1':>6} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'Brier':>7}")
    log.info("-" * 100)
    for name, res in all_results.items():
        m = res["test_metrics"]
        ci = f"[{m['auroc_ci_low']:.3f}-{m['auroc_ci_high']:.3f}]"
        log.info(f"{name:<22} {m['auroc']:>7.4f} {ci:<16} {m['auprc']:>7.4f} "
                 f"{m['f1']:>6.4f} {m['sensitivity']:>6.4f} {m['specificity']:>6.4f} "
                 f"{m['ppv']:>6.4f} {m['npv']:>6.4f} {m['brier_score']:>7.4f}")

    log.info(f"\n  Run folder (models): {MODELS_RUN_DIR}")
    log.info(f"  Run folder (figs):   {FIG_RUN_DIR}")
    log.info(f"  Experiment log:      results/experiment_log.json")
    log.info(f"  Results CSV:         results/results_summary.csv")
    log.info(f"\nPhase 1 complete.")
    log.info(f"Next step: python sepsis_ml/phase2_tune_catboost.py")


if __name__ == "__main__":
    main()