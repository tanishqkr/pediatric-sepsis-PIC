"""
phase2_ensemble.py
------------------
Two novel contributions combined in one script:

PART A — Feature Interaction Engineering (Option 3)
  Clinically motivated composite features encoding domain knowledge:
  - lactate_max × platelets_min       : cardiovascular-coagulation burden
  - ddimer_max × inr_max              : coagulopathy composite index
  - base_excess_min / (ph_min + 1e-6) : acid-base severity ratio
  - pt_mean × inr_mean                : coagulation cascade severity
  - lactate_max / (platelets_min + 1) : shock-coagulopathy index
  - glucose_max - glucose_min         : glucose instability range
  - anion_gap_max × base_excess_min   : metabolic crisis index (both negative)
  - ptt_max × pt_max                  : dual coagulation pathway burden
  - (lactate_mean - lactate_first)    : lactate trajectory (already have trend
                                        but this is normalised differently)
  - crp_max × ddimer_max              : inflammatory-coagulation interaction

PART B — Stacking Ensemble (Option 1)
  Base learners: tuned CatBoost + LightGBM (baseline) + XGBoost (baseline)
  Meta-learner: Logistic Regression trained on out-of-fold base learner probs
  Uses both original features AND interaction features for base learners.
  Meta-learner sees only the 3 probability columns — keeps it clean.

Both parts evaluated with bootstrap CIs. DeLong test vs tuned CatBoost.
All results logged to experiment tracker.

Outputs:
  model_datasets/B_train_with_interactions.csv
  model_datasets/B_test_with_interactions.csv
  models/run_ensemble/stacking_meta.pkl
  models/run_ensemble/interaction_feature_list.json
  figures/phase2_tuning/ensemble_roc_pr.png
  figures/phase2_tuning/ensemble_cm.png
  figures/phase2_tuning/interaction_feature_correlations.png
  results/ensemble_oof_probs.csv
  logs/phase2_ensemble.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase2_ensemble.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output
  - figures/phase2_tuning/ensemble_roc_pr.png
  - figures/phase2_tuning/ensemble_cm.png
  - figures/phase2_tuning/interaction_feature_correlations.png
  - figures/phase2_tuning/all_models_final_comparison.png
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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
    TARGET_SENSITIVITY, BOOTSTRAP_ITERATIONS
)
from logger import log_experiment

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"
ENSEMBLE_DIR   = MODELS_DIR / "run_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

log_path = LOGS_DIR / "phase2_ensemble.log"
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
# Evaluation helpers
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
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "ppv"        : float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "npv"        : float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
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
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics_with_ci(y_true, y_prob):
    threshold = find_threshold_at_sensitivity(y_true, y_prob)
    metrics   = compute_metrics(y_true, y_prob, threshold)
    y_true_arr, y_prob_arr = np.array(y_true), np.array(y_prob)
    auroc_lo, auroc_hi = bootstrap_ci(y_true_arr, y_prob_arr, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_true_arr, y_prob_arr, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi
    return metrics


def delong_test(y_true, y_prob_1, y_prob_2):
    """DeLong test for AUROC comparison."""
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

    def fastDeLong(pst, label_1_count):
        m = label_1_count
        n = pst.shape[1] - m
        pos = pst[:, :m]
        neg = pst[:, m:]
        k = pst.shape[0]
        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)
        for r in range(k):
            tx[r, :] = compute_midrank(pos[r, :])
            ty[r, :] = compute_midrank(neg[r, :])
            tz[r, :] = compute_midrank(pst[r, :])
        aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
        v01  = (tz[:, :m] - tx[:, :]) / n
        v10  = 1. - (tz[:, m:] - ty[:, :]) / m
        delongcov = np.cov(v01) / m + np.cov(v10) / n
        return aucs, delongcov

    y_true = np.array(y_true)
    order  = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    pst = np.vstack([np.array(y_prob_1)[order], np.array(y_prob_2)[order]])
    aucs, cov = fastDeLong(pst, label_1_count)
    diff = aucs[0] - aucs[1]
    se   = np.sqrt(cov[0,0] + cov[1,1] - 2*cov[0,1])
    z    = diff / se
    from scipy import stats
    p    = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Feature Interaction Engineering
# ══════════════════════════════════════════════════════════════════════════════

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically motivated interaction features.
    All interactions are based on published sepsis pathophysiology.
    Safe division uses epsilon to avoid division by zero.
    """
    df = df.copy()
    eps = 1e-6

    def safe_col(name, default=0.0):
        return df[name].values if name in df.columns else np.full(len(df), default)

    lactate_max   = safe_col("lactate_max")
    platelets_min = safe_col("platelets_min")
    ddimer_max    = safe_col("ddimer_max")
    inr_max       = safe_col("inr_max")
    inr_mean      = safe_col("inr_mean")
    base_excess_min = safe_col("base_excess_min")
    ph_min        = safe_col("ph_min")
    pt_mean       = safe_col("pt_mean")
    pt_max        = safe_col("pt_max")
    ptt_max       = safe_col("ptt_max")
    glucose_max   = safe_col("glucose_max")
    glucose_min   = safe_col("glucose_min")
    anion_gap_max = safe_col("anion_gap_max")
    crp_max       = safe_col("crp_max")
    lactate_mean  = safe_col("lactate_mean")
    lactate_first = safe_col("lactate_first")

    # Interaction 1: Cardiovascular-coagulation burden
    # High lactate + low platelets = multi-organ failure signature
    df["ix_lactate_x_platelets"] = lactate_max * (1.0 / (platelets_min + eps))

    # Interaction 2: Coagulopathy composite index
    # Both D-dimer and INR elevated = DIC (disseminated intravascular coagulation)
    df["ix_ddimer_x_inr"] = ddimer_max * inr_max

    # Interaction 3: Acid-base severity ratio
    # Base excess and pH together encode metabolic acidosis depth
    df["ix_base_excess_ph_ratio"] = np.abs(base_excess_min) / (ph_min + eps)

    # Interaction 4: Coagulation cascade severity
    # PT and INR measure overlapping but distinct coagulation pathways
    df["ix_pt_x_inr"] = pt_mean * inr_mean

    # Interaction 5: Shock-coagulopathy index (novel composite)
    # Combines cardiovascular failure (lactate) with coagulation failure (platelets)
    # Normalised to be unitless
    df["ix_shock_coagulopathy"] = lactate_max / (platelets_min / 100.0 + eps)

    # Interaction 6: Glucose instability range
    # Wide glucose swing = stress hyperglycemia + insulin resistance in sepsis
    df["ix_glucose_range"] = glucose_max - glucose_min

    # Interaction 7: Metabolic crisis index
    # Anion gap and base excess both reflect metabolic acidosis but from different angles
    df["ix_anion_gap_x_base_excess"] = anion_gap_max * np.abs(base_excess_min)

    # Interaction 8: Dual coagulation pathway burden
    # PT measures extrinsic pathway, PTT measures intrinsic pathway
    # Both elevated = global coagulation failure
    df["ix_pt_x_ptt"] = pt_max * ptt_max

    # Interaction 9: Lactate trajectory normalised
    # Different from lactate_trend (last-first) — captures relative change
    df["ix_lactate_relative_change"] = (lactate_mean - lactate_first) / (lactate_first + eps)

    # Interaction 10: Inflammatory-coagulation interaction
    # CRP (inflammation) × D-dimer (coagulation) = sepsis-DIC axis
    df["ix_crp_x_ddimer"] = crp_max * ddimer_max

    return df


def log_interaction_correlations(train_with_ix: pd.DataFrame):
    """Log and plot correlations of interaction features with label."""
    ix_cols = [c for c in train_with_ix.columns if c.startswith("ix_")]
    corr = train_with_ix[ix_cols].corrwith(train_with_ix[TARGET_COL]).abs()\
                                  .sort_values(ascending=False)

    log.info(f"\n  Interaction feature correlations with sepsis label:")
    for col, r in corr.items():
        log.info(f"    {col:<45} |r|={r:.4f}")

    # Compare with parent features
    log.info(f"\n  Key parent feature correlations for context:")
    for col in ["lactate_max", "ddimer_max", "platelets_min", "inr_max", "pt_mean"]:
        if col in train_with_ix.columns:
            r = abs(train_with_ix[col].corr(train_with_ix[TARGET_COL]))
            log.info(f"    {col:<45} |r|={r:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if r > 0.35 else "#f39c12" if r > 0.25 else "#3498db"
              for r in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white")
    ax.axvline(0.35, color="#e74c3c", linestyle="--", lw=1, alpha=0.7,
               label="|r|=0.35 (strong)")
    ax.axvline(0.25, color="#f39c12", linestyle="--", lw=1, alpha=0.7,
               label="|r|=0.25 (moderate)")
    ax.set_xlabel("Pearson |r| with sepsis label")
    ax.set_title("Interaction feature correlations with sepsis label\n"
                 "Red=strong (>0.35), Orange=moderate (>0.25), Blue=weak",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_TUNING / "interaction_feature_correlations.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"\n  Saved → {out}")

    return corr


# ══════════════════════════════════════════════════════════════════════════════
# PART B — Stacking Ensemble
# ══════════════════════════════════════════════════════════════════════════════

def get_base_models(best_params: dict):
    """Return the three base learners."""
    from catboost import CatBoostClassifier
    import xgboost as xgb
    import lightgbm as lgb

    catboost = CatBoostClassifier(
        iterations          = best_params["iterations"],
        learning_rate       = best_params["learning_rate"],
        depth               = best_params["depth"],
        l2_leaf_reg         = best_params["l2_leaf_reg"],
        bagging_temperature = best_params["bagging_temperature"],
        random_strength     = best_params["random_strength"],
        border_count        = best_params["border_count"],
        class_weights       = [1.0, best_params["class_weight_pos"]],
        eval_metric         = "AUC",
        random_seed         = RANDOM_SEED,
        verbose             = 0,
    )
    xgboost = xgb.XGBClassifier(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        scale_pos_weight = 2.7,
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        verbosity        = 0,
    )
    lightgbm = lgb.LGBMClassifier(
        n_estimators     = 500,
        learning_rate    = 0.05,
        max_depth        = 6,
        scale_pos_weight = 2.7,
        random_state     = RANDOM_SEED,
        verbose          = -1,
    )
    return {
        "catboost_tuned": catboost,
        "xgboost"       : xgboost,
        "lightgbm"      : lightgbm,
    }


def generate_oof_predictions(base_models: dict,
                              X_df: pd.DataFrame,
                              y: np.ndarray) -> np.ndarray:
    """
    Generate out-of-fold predictions for all base models.
    These are used to train the meta-learner — ensures no leakage.
    Each sample's prediction comes from a model that never saw it during training.
    """
    skf     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof     = np.zeros((len(y), len(base_models)))
    names   = list(base_models.keys())

    log.info(f"\n  Generating OOF predictions ({CV_FOLDS} folds × {len(base_models)} models)...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_df, y), 1):
        X_tr  = X_df.iloc[train_idx]
        X_val = X_df.iloc[val_idx]
        y_tr  = y[train_idx]
        y_val = y[val_idx]

        for m_idx, (name, model) in enumerate(base_models.items()):
            import copy
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            oof[val_idx, m_idx] = m.predict_proba(X_val)[:, 1]

        fold_aurocs = [roc_auc_score(y_val, oof[val_idx, i]) for i in range(len(names))]
        log.info(f"    Fold {fold_idx}: " +
                 "  ".join(f"{n}={a:.4f}" for n, a in zip(names, fold_aurocs)))

    log.info(f"\n  OOF AUROC per model:")
    for i, name in enumerate(names):
        auroc = roc_auc_score(y, oof[:, i])
        log.info(f"    {name:<20} OOF AUROC = {auroc:.4f}")

    return oof


def train_meta_learner(oof_probs: np.ndarray, y: np.ndarray):
    """Train logistic regression meta-learner on OOF base model probabilities."""
    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter     = 1000,
            random_state = RANDOM_SEED,
            C            = 1.0,
        ))
    ])
    meta.fit(oof_probs, y)
    log.info(f"\n  Meta-learner trained on OOF probs shape: {oof_probs.shape}")
    log.info(f"  Meta-learner coefficients: {meta.named_steps['clf'].coef_[0]}")
    return meta


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_ensemble_vs_all(y_test, results_dict):
    """ROC + PR curves for ensemble vs all baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    styles = {
        "Stacking Ensemble"  : ("#e74c3c", 2.5, "-"),
        "Tuned CatBoost"     : ("#3498db", 1.8, "--"),
        "LightGBM (baseline)": ("#2ecc71", 1.4, ":"),
        "XGBoost (baseline)" : ("#9b59b6", 1.4, ":"),
    }
    for name, (y_prob, metrics) in results_dict.items():
        color, lw, ls = styles.get(name, ("#888888", 1.2, "-"))
        fpr, tpr, _   = roc_curve(y_test, y_prob)
        prec, rec, _  = precision_recall_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                 label=f"{name} (AUROC={metrics['auroc']:.4f})")
        ax2.plot(rec, prec, color=color, lw=lw, ls=ls,
                 label=f"{name} (AUPRC={metrics['auprc']:.4f})")

    ax1.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC — Stacking Ensemble vs All\nOption B", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.axhline(y_test.mean(), color="k", ls="--", lw=0.8, alpha=0.4,
                label=f"Prevalence ({y_test.mean():.2f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall — Stacking Ensemble vs All\nOption B", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_TUNING / "ensemble_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_cm(y_true, y_pred, metrics, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"])
    ax.set_yticklabels(["Non-sepsis","Sepsis"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\nSens={metrics['sensitivity']:.4f}  "
                 f"Spec={metrics['specificity']:.4f}  "
                 f"Threshold={metrics['threshold']:.4f}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = FIG_TUNING / "ensemble_cm.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_final_comparison(y_test, all_results):
    """Final bar chart comparing all approaches."""
    models  = list(all_results.keys())
    metrics = ["auroc", "auprc", "f1", "sensitivity", "specificity"]
    colors  = {"auroc":"#2ecc71","auprc":"#e74c3c","f1":"#3498db",
               "sensitivity":"#9b59b6","specificity":"#f39c12"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle("Final model comparison — all approaches\nOption B (Infection-only)",
                 fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals = [all_results[m][metric] for m in models]
        bar_colors = [colors[metric]] * len(models)
        # Highlight best
        best_idx = np.argmax(vals)
        bar_colors[best_idx] = "#c0392b" if metric in ("auroc","auprc") else "#1a5276"
        bars = ax.bar([m.replace(" ","\n") for m in models], vals,
                      color=bar_colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        ax.set_title(metric.upper(), fontweight="bold", fontsize=10)
        ax.set_ylim(min(vals)*0.97, 1.03)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_TUNING / "all_models_final_comparison.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("="*65)
    log.info("PHASE 2 — ENSEMBLE (Feature Interactions + Stacking)")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("="*65)

    # ── Load data ─────────────────────────────────────────────────────────────
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train.columns if c != TARGET_COL]
    y_train = train[TARGET_COL].values
    y_test  = test[TARGET_COL].values
    log.info(f"\nLoaded: train={train.shape}, test={test.shape}, features={len(feature_cols)}")

    # ── Load best params from Optuna ─────────────────────────────────────────
    best_params_path = RESULTS_DIR / "best_params.json"
    with open(best_params_path) as f:
        best_info = json.load(f)
    best_params = best_info["best_params"]
    log.info(f"  Loaded Optuna best params: {best_params}")

    # ── PART A: Feature interactions ──────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("PART A — Feature Interaction Engineering")
    log.info(f"{'='*55}")

    train_ix = create_interaction_features(train)
    test_ix  = create_interaction_features(test)

    ix_cols = [c for c in train_ix.columns if c.startswith("ix_")]
    log.info(f"  Created {len(ix_cols)} interaction features:")
    for col in ix_cols:
        log.info(f"    {col}")

    corr_ix = log_interaction_correlations(train_ix)

    # Save augmented datasets
    train_ix.to_csv(MODEL_DATA_DIR / "B_train_with_interactions.csv", index=False)
    test_ix.to_csv(MODEL_DATA_DIR / "B_test_with_interactions.csv", index=False)
    log.info(f"  Saved augmented datasets to model_datasets/")

    with open(ENSEMBLE_DIR / "interaction_feature_list.json", "w") as f:
        json.dump({
            "n_interaction_features": len(ix_cols),
            "features": ix_cols,
            "correlations_with_label": {c: round(float(corr_ix[c]), 4) for c in ix_cols},
        }, f, indent=2)

    # ── PART B: Stacking ensemble ─────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("PART B — Stacking Ensemble")
    log.info(f"{'='*55}")
    log.info("Base learners: Tuned CatBoost + XGBoost + LightGBM")
    log.info("Meta-learner:  Logistic Regression on OOF probabilities")
    log.info("Features:      Original 225 + 10 interaction features = 235")

    feat_cols_ix = [c for c in train_ix.columns if c != TARGET_COL]
    X_train_ix   = train_ix[feat_cols_ix]
    X_test_ix    = test_ix[feat_cols_ix]

    base_models = get_base_models(best_params)

    # Generate OOF predictions for meta-learner training
    oof_probs = generate_oof_predictions(base_models, X_train_ix, y_train)
    pd.DataFrame(oof_probs, columns=list(base_models.keys()))\
      .to_csv(RESULTS_DIR / "ensemble_oof_probs.csv", index=False)

    # Train meta-learner
    meta_learner = train_meta_learner(oof_probs, y_train)

    # Retrain all base models on FULL training set
    log.info(f"\n  Retraining base models on full training set...")
    import copy
    final_base_models = {}
    test_probs = np.zeros((len(y_test), len(base_models)))

    for i, (name, model) in enumerate(base_models.items()):
        m = copy.deepcopy(model)
        m.fit(X_train_ix, y_train)
        test_probs[:, i] = m.predict_proba(X_test_ix)[:, 1]
        final_base_models[name] = m
        auroc = roc_auc_score(y_test, test_probs[:, i])
        log.info(f"    {name:<20} test AUROC = {auroc:.4f}")

    # Meta-learner prediction on test
    y_prob_ensemble = meta_learner.predict_proba(test_probs)[:, 1]
    metrics_ensemble = compute_metrics_with_ci(y_test, y_prob_ensemble)

    log.info(f"\n  {'='*55}")
    log.info(f"  STACKING ENSEMBLE — TEST SET RESULTS")
    log.info(f"  {'='*55}")
    log.info(f"  AUROC : {metrics_ensemble['auroc']:.4f} "
             f"[{metrics_ensemble['auroc_ci_low']:.4f}–{metrics_ensemble['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_ensemble['auprc']:.4f} "
             f"[{metrics_ensemble['auprc_ci_low']:.4f}–{metrics_ensemble['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_ensemble['sensitivity']:.4f}  "
             f"Spec: {metrics_ensemble['specificity']:.4f}  F1: {metrics_ensemble['f1']:.4f}")
    log.info(f"  PPV   : {metrics_ensemble['ppv']:.4f}  NPV: {metrics_ensemble['npv']:.4f}  "
             f"Brier: {metrics_ensemble['brier_score']:.4f}")
    log.info(f"  TP={metrics_ensemble['tp']}  FP={metrics_ensemble['fp']}  "
             f"TN={metrics_ensemble['tn']}  FN={metrics_ensemble['fn']}")

    # ── Load tuned CatBoost for comparison ────────────────────────────────────
    from catboost import CatBoostClassifier
    tuned_path = MODELS_DIR / "run_tuned" / "catboost_tuned.cbm"
    tuned_model = CatBoostClassifier()
    tuned_model.load_model(str(tuned_path))
    # Tuned model uses original features (no interactions)
    X_test_orig = test[feature_cols]
    y_prob_tuned = tuned_model.predict_proba(X_test_orig)[:, 1]
    metrics_tuned = compute_metrics(y_test, y_prob_tuned)

    # DeLong test: ensemble vs tuned CatBoost
    auroc_e, auroc_t, z, p = delong_test(y_test, y_prob_ensemble, y_prob_tuned)
    log.info(f"\n  DeLong test (ensemble vs tuned CatBoost):")
    log.info(f"    Ensemble AUROC:     {auroc_e:.4f}")
    log.info(f"    Tuned CatBoost:     {auroc_t:.4f}")
    log.info(f"    Z-statistic:        {z:.4f}")
    log.info(f"    P-value:            {p:.4f}")
    log.info(f"    Significant:        {'YES (p<0.05)' if p < 0.05 else 'NO (p>=0.05)'}")

    # ── Save models ───────────────────────────────────────────────────────────
    with open(ENSEMBLE_DIR / "stacking_meta.pkl", "wb") as f:
        pickle.dump(meta_learner, f)
    for name, m in final_base_models.items():
        if "catboost" in name:
            m.save_model(str(ENSEMBLE_DIR / f"{name}.cbm"))
        else:
            with open(ENSEMBLE_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(m, f)
    log.info(f"\n  Models saved → {ENSEMBLE_DIR}")

    # ── Log to experiment tracker ─────────────────────────────────────────────
    metrics_ensemble["delong_z_vs_tuned"] = z
    metrics_ensemble["delong_p_vs_tuned"] = p
    log_experiment(
        experiment_id = "phase2_stacking_ensemble_B",
        dataset       = "B",
        model_name    = "stacking_ensemble",
        phase         = "ensemble",
        params        = {"base_models": list(base_models.keys()),
                         "meta_learner": "logistic_regression",
                         "n_interaction_features": len(ix_cols)},
        metrics       = metrics_ensemble,
        notes         = "Stacking: tuned CatBoost + XGBoost + LightGBM + LR meta. "
                        "Features: 225 original + 10 clinical interactions = 235",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    results_for_plot = {
        "Stacking Ensemble"  : (y_prob_ensemble, metrics_ensemble),
        "Tuned CatBoost"     : (y_prob_tuned,    metrics_tuned),
        "LightGBM (baseline)": (test_probs[:, list(base_models.keys()).index("lightgbm")],
                                compute_metrics(y_test,
                                test_probs[:, list(base_models.keys()).index("lightgbm")])),
        "XGBoost (baseline)" : (test_probs[:, list(base_models.keys()).index("xgboost")],
                                compute_metrics(y_test,
                                test_probs[:, list(base_models.keys()).index("xgboost")])),
    }
    plot_ensemble_vs_all(y_test, results_for_plot)

    y_pred_ensemble = (y_prob_ensemble >= metrics_ensemble["threshold"]).astype(int)
    plot_cm(y_test, y_pred_ensemble, metrics_ensemble, "Stacking Ensemble")

    # Final comparison: all approaches
    all_results_flat = {name: m for name, (_, m) in results_for_plot.items()}
    plot_final_comparison(y_test, all_results_flat)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("PHASE 2 ENSEMBLE COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  Interaction features added: {len(ix_cols)}")
    log.info(f"  Total features used:        {len(feat_cols_ix)}")
    log.info(f"  Ensemble AUROC: {metrics_ensemble['auroc']:.4f} "
             f"[{metrics_ensemble['auroc_ci_low']:.4f}–{metrics_ensemble['auroc_ci_high']:.4f}]")
    log.info(f"  Ensemble AUPRC: {metrics_ensemble['auprc']:.4f} "
             f"[{metrics_ensemble['auprc_ci_low']:.4f}–{metrics_ensemble['auprc_ci_high']:.4f}]")
    log.info(f"  vs Tuned CatBoost: AUROC={metrics_tuned['auroc']:.4f}  "
             f"DeLong p={p:.4f}")
    log.info(f"  Figures → {FIG_TUNING}/")
    log.info(f"  Log     → {log_path}")


if __name__ == "__main__":
    main()
