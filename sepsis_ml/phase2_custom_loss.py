"""
phase2_custom_loss.py
---------------------
Novel contribution: Phoenix-aware asymmetric loss function for CatBoost.

Standard cross-entropy treats every false negative equally.
This custom loss penalizes false negatives proportionally to the
patient's clinical severity — a sepsis patient with lactate=12 and
platelets=40 who gets missed should cost more than one with
lactate=2.5 and platelets=180.

Severity proxy used: normalised sum of key Phoenix variable z-scores
  severity = z(lactate_max) + z(1/platelets_min) + z(inr_max)
            + z(ddimer_max) + z(|base_excess_min|)

Loss formulation:
  For positive class (sepsis=1):
    weight_i = 1 + alpha * severity_i
    loss_i   = -weight_i * log(p_i)
  For negative class:
    loss_i   = -log(1 - p_i)  [standard]

Alpha controls how strongly severity modulates the penalty.
Optimised via Optuna alongside other hyperparameters.

Full Optuna tuning with the custom loss, 50 trials.
Evaluated against tuned CatBoost and ensemble.

Outputs:
  models/run_custom_loss/catboost_custom_loss.cbm
  results/custom_loss_best_params.json
  figures/phase2_tuning/custom_loss_roc_pr.png
  figures/phase2_tuning/custom_loss_cm.png
  figures/phase2_tuning/severity_distribution.png
  logs/phase2_custom_loss.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase2_custom_loss.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output
  - figures/phase2_tuning/custom_loss_roc_pr.png
  - figures/phase2_tuning/custom_loss_cm.png
  - figures/phase2_tuning/severity_distribution.png
  - results/custom_loss_best_params.json
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
from sklearn.preprocessing import StandardScaler
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
CUSTOM_DIR     = MODELS_DIR / "run_custom_loss"
CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_LOSS_TRIALS = 50   # fewer than full Optuna since search space is larger

log_path = LOGS_DIR / "phase2_custom_loss.log"
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

# ── Phoenix severity variables (available in our feature set) ─────────────────
SEVERITY_VARS = [
    ("lactate_max",      "high"),    # higher = worse
    ("platelets_min",    "low"),     # lower = worse (inverted)
    ("inr_max",          "high"),
    ("ddimer_max",       "high"),
    ("base_excess_min",  "extreme"), # more negative = worse (abs value)
    ("pt_mean",          "high"),
    ("ptt_max",          "high"),
]


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
        "sensitivity": float(tp/(tp+fn)) if (tp+fn)>0 else 0.0,
        "specificity": float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        "ppv"        : float(tp/(tp+fp)) if (tp+fp)>0 else 0.0,
        "npv"        : float(tn/(tn+fn)) if (tn+fn)>0 else 0.0,
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
    def compute_midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5*(i+j-1); i = j
        T2 = np.empty(N, dtype=float); T2[J] = T+1; return T2

    def fastDeLong(pst, m):
        n = pst.shape[1]-m
        pos=pst[:,:m]; neg=pst[:,m:]; k=pst.shape[0]
        tx=np.empty([k,m],dtype=float); ty=np.empty([k,n],dtype=float)
        tz=np.empty([k,m+n],dtype=float)
        for r in range(k):
            tx[r,:]=compute_midrank(pos[r,:]); ty[r,:]=compute_midrank(neg[r,:])
            tz[r,:]=compute_midrank(pst[r,:])
        aucs=(tz[:,:m].sum(1)-tx.sum(1))/(m*n)
        v01=(tz[:,:m]-tx[:,:])/n; v10=1.-(tz[:,m:]-ty[:,:])/m
        cov=np.cov(v01)/m+np.cov(v10)/n
        return aucs, cov

    y_true=np.array(y_true); order=np.argsort(-y_true)
    m=int(y_true.sum())
    pst=np.vstack([np.array(y_prob_1)[order],np.array(y_prob_2)[order]])
    aucs,cov=fastDeLong(pst,m)
    diff=aucs[0]-aucs[1]; se=np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1]); z=diff/se
    from scipy import stats
    p=2*stats.norm.sf(abs(z))
    return float(aucs[0]),float(aucs[1]),float(z),float(p)


# ══════════════════════════════════════════════════════════════════════════════
# Severity score computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_severity_scores(df: pd.DataFrame,
                             scaler: StandardScaler = None,
                             fit: bool = True) -> tuple:
    """
    Compute per-patient clinical severity score from Phoenix variables.
    Returns (severity_array, fitted_scaler).

    Each variable is z-scored then directionality-corrected so higher
    always means more severe. The sum is the severity index.
    """
    components = []
    available  = []

    for var, direction in SEVERITY_VARS:
        if var not in df.columns:
            continue
        vals = df[var].values.reshape(-1, 1)
        available.append((var, direction, vals))

    if not available:
        log.info("  WARNING: No severity variables found — using uniform weights")
        return np.ones(len(df)), scaler

    # Stack for scaling
    raw_matrix = np.hstack([v for _, _, v in available])

    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw_matrix)
    else:
        scaled = scaler.transform(raw_matrix)

    # Direction correction
    for i, (var, direction, _) in enumerate(available):
        if direction == "low":
            scaled[:, i] = -scaled[:, i]   # invert: lower original = higher severity
        elif direction == "extreme":
            scaled[:, i] = np.abs(scaled[:, i])  # deviation from normal in either direction

    severity = scaled.sum(axis=1)

    # Normalise to [0, 1] range
    smin, smax = severity.min(), severity.max()
    if smax > smin:
        severity = (severity - smin) / (smax - smin)
    else:
        severity = np.zeros_like(severity)

    return severity, scaler


# ══════════════════════════════════════════════════════════════════════════════
# Custom loss function
# ══════════════════════════════════════════════════════════════════════════════

class PhoenixAsymmetricLoss:
    """
    Custom CatBoost objective: asymmetric cross-entropy with
    severity-weighted false negative penalty.

    For a sepsis patient i:
      weight_i = base_weight * (1 + alpha * severity_i)
      grad_i   = -(weight_i) * (1 - p_i)
      hess_i   = weight_i * p_i * (1 - p_i)

    For a non-sepsis patient:
      grad_i   = p_i                (standard cross-entropy gradient)
      hess_i   = p_i * (1 - p_i)   (standard)

    alpha  : severity modulation strength (tuned by Optuna)
    base_weight: base positive class weight (replaces class_weights)
    """

    def __init__(self, severity_train: np.ndarray, alpha: float = 1.0,
                 base_weight: float = 2.7):
        self.severity    = severity_train
        self.alpha       = alpha
        self.base_weight = base_weight

    def calc_ders_range(self, approxes, targets, weights):
        """
        CatBoost calls this per batch.
        Must return a list of (grad, hess) tuples — one per sample.
        approxes: raw model output (log-odds)
        targets:  true labels
        weights:  sample weights (we override with severity weights)
        """
        assert len(approxes) == len(targets)
        result = []

        for i in range(len(targets)):
            p = 1.0 / (1.0 + np.exp(-approxes[i]))
            p = max(min(p, 1 - 1e-7), 1e-7)  # clip for numerical stability
            y = targets[i]

            if y == 1:
                # Severity-weighted false negative penalty
                sev_weight = self.base_weight * (1.0 + self.alpha * self.severity[i])
                g = -sev_weight * (1.0 - p)
                h = sev_weight * p * (1.0 - p)
            else:
                # Standard cross-entropy for negatives
                g = p
                h = p * (1.0 - p)

            result.append((g, h))

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Optuna objective with custom loss
# ══════════════════════════════════════════════════════════════════════════════

def objective_custom_loss(trial, X_df, y, severity_all):
    from catboost import CatBoostClassifier, Pool

    alpha       = trial.suggest_float("alpha", 0.1, 3.0)
    base_weight = trial.suggest_float("base_weight", 1.5, 5.0)
    iterations  = trial.suggest_int("iterations", 200, 1000)
    lr          = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    depth       = trial.suggest_int("depth", 4, 8)
    l2          = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    auprc_scores = []

    for train_idx, val_idx in skf.split(X_df, y):
        X_tr    = X_df.iloc[train_idx]
        X_val   = X_df.iloc[val_idx]
        y_tr    = y[train_idx]
        y_val   = y[val_idx]
        sev_tr  = severity_all[train_idx]

        loss_fn = PhoenixAsymmetricLoss(sev_tr, alpha=alpha, base_weight=base_weight)

        model = CatBoostClassifier(
            iterations     = iterations,
            learning_rate  = lr,
            depth          = depth,
            l2_leaf_reg    = l2,
            loss_function  = loss_fn,
            eval_metric    = "AUC",
            bootstrap_type = "Bernoulli",
            subsample      = 0.8,
            random_seed    = RANDOM_SEED,
            verbose        = 0,
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
        y_prob = model.predict_proba(X_val)[:, 1]
        auprc_scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(auprc_scores))


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_severity_distribution(severity_train, y_train):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(severity_train[y_train==0], bins=40, alpha=0.6,
            color="#3498db", label="Non-sepsis", density=True)
    ax.hist(severity_train[y_train==1], bins=40, alpha=0.6,
            color="#e74c3c", label="Sepsis", density=True)
    ax.set_xlabel("Clinical severity score (normalised)")
    ax.set_ylabel("Density")
    ax.set_title("Phoenix-derived severity score distribution\n"
                 "Sepsis vs non-sepsis (training set)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_TUNING / "severity_distribution.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_custom_loss_vs_tuned(y_test, y_prob_custom, y_prob_tuned,
                               metrics_custom, metrics_tuned):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for y_prob, label, color, lw in [
        (y_prob_custom, f"Custom loss (AUROC={metrics_custom['auroc']:.4f})", "#e74c3c", 2.5),
        (y_prob_tuned,  f"Tuned CatBoost (AUROC={metrics_tuned['auroc']:.4f})", "#3498db", 1.8),
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=lw, label=label)
        ax2.plot(rec, prec, color=color, lw=lw,
                 label=label.replace("AUROC","AUPRC").replace(
                     f"{metrics_custom['auroc']:.4f}", f"{metrics_custom['auprc']:.4f}").replace(
                     f"{metrics_tuned['auroc']:.4f}",  f"{metrics_tuned['auprc']:.4f}"))
    ax1.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.set_title("ROC — Custom Loss vs Tuned CatBoost\nOption B", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9); ax1.spines[["top","right"]].set_visible(False)
    ax2.axhline(y_test.mean(),color="k",ls="--",lw=0.8,alpha=0.4)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("PR — Custom Loss vs Tuned CatBoost\nOption B", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9); ax2.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_TUNING / "custom_loss_roc_pr.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


def plot_cm(y_true, y_pred, metrics, fname="custom_loss_cm.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"])
    ax.set_yticklabels(["Non-sepsis","Sepsis"])
    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]}",ha="center",va="center",
                    color="white" if cm[i,j]>thresh else "black",fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Custom Loss CatBoost\n"
                 f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
                 f"Threshold={metrics['threshold']:.4f}",
                 fontsize=9, fontweight="bold")
    plt.tight_layout()
    out = FIG_TUNING / fname
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png",".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    log.info("="*65)
    log.info("PHASE 2 — CUSTOM PHOENIX-AWARE LOSS (Option 2)")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"Trials: {CUSTOM_LOSS_TRIALS}  |  CV folds: {CV_FOLDS}  |  Metric: AUPRC")
    log.info("="*65)

    # ── Load data ─────────────────────────────────────────────────────────────
    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train.columns if c != TARGET_COL]
    X_train_df   = train[feature_cols]
    y_train      = train[TARGET_COL].values
    X_test_df    = test[feature_cols]
    y_test       = test[TARGET_COL].values
    log.info(f"\nLoaded: train={train.shape}, test={test.shape}")

    # ── Compute severity scores ───────────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info("Computing clinical severity scores...")
    log.info(f"{'='*55}")
    log.info(f"Severity variables used: {[v for v,_ in SEVERITY_VARS]}")

    severity_train, severity_scaler = compute_severity_scores(
        train, fit=True
    )
    severity_test, _ = compute_severity_scores(
        test, scaler=severity_scaler, fit=False
    )

    log.info(f"\n  Train severity stats:")
    log.info(f"    Mean (sepsis):     {severity_train[y_train==1].mean():.4f}")
    log.info(f"    Mean (non-sepsis): {severity_train[y_train==0].mean():.4f}")
    log.info(f"    Max severity:      {severity_train.max():.4f}")
    log.info(f"    Min severity:      {severity_train.min():.4f}")

    # Verify severity separates classes
    from scipy import stats as scipy_stats
    t_stat, t_p = scipy_stats.ttest_ind(
        severity_train[y_train==1],
        severity_train[y_train==0]
    )
    log.info(f"    t-test (sepsis vs non-sepsis): t={t_stat:.4f}, p={t_p:.2e}")
    log.info(f"    {'✓ Severity score separates classes' if t_p < 0.001 else 'WARNING: poor separation'}")

    plot_severity_distribution(severity_train, y_train)

    # Save severity scaler
    with open(CUSTOM_DIR / "severity_scaler.pkl", "wb") as f:
        pickle.dump(severity_scaler, f)

    # ── Optuna study with custom loss ─────────────────────────────────────────
    log.info(f"\n{'='*55}")
    log.info(f"Optuna study with custom loss ({CUSTOM_LOSS_TRIALS} trials)...")
    log.info(f"{'='*55}")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name = "catboost_custom_loss_optuna",
    )

    best_so_far = 0.0

    def callback(study, trial):
        nonlocal best_so_far
        if trial.value is not None and trial.value > best_so_far:
            best_so_far = trial.value
            log.info(f"  Trial {trial.number:>3} NEW BEST: AUPRC={trial.value:.4f}  "
                     f"alpha={trial.params.get('alpha',0):.3f}  "
                     f"base_weight={trial.params.get('base_weight',0):.3f}")
        elif trial.number % 10 == 0:
            log.info(f"  Trial {trial.number:>3}: AUPRC={trial.value:.4f}  "
                     f"(best: {best_so_far:.4f})")

    study.optimize(
        lambda trial: objective_custom_loss(
            trial, X_train_df, y_train, severity_train
        ),
        n_trials  = CUSTOM_LOSS_TRIALS,
        callbacks = [callback],
    )

    log.info(f"\n  Custom loss tuning complete.")
    log.info(f"  Best CV AUPRC: {study.best_value:.4f}")
    log.info(f"  Best params:   {study.best_params}")

    # Save params
    best_params = study.best_params
    custom_params_path = RESULTS_DIR / "custom_loss_best_params.json"
    with open(custom_params_path, "w") as f:
        json.dump({
            "best_cv_auprc": study.best_value,
            "best_params"  : best_params,
            "n_trials"     : CUSTOM_LOSS_TRIALS,
            "timestamp"    : datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)

    # ── Train final custom loss model on full training set ────────────────────
    log.info(f"\nTraining final custom loss model on full training set...")
    from catboost import CatBoostClassifier

    loss_fn = PhoenixAsymmetricLoss(
        severity_train,
        alpha       = best_params["alpha"],
        base_weight = best_params["base_weight"],
    )

    final_model = CatBoostClassifier(
        iterations            = best_params["iterations"],
        learning_rate         = best_params["learning_rate"],
        depth                 = best_params["depth"],
        l2_leaf_reg           = best_params["l2_leaf_reg"],
        loss_function         = loss_fn,
        eval_metric           = "AUC",
        bootstrap_type        = "Bernoulli",
        subsample             = 0.8,
        random_seed           = RANDOM_SEED,
        verbose               = 100,
        early_stopping_rounds = 50,
    )
    final_model.fit(
        X_train_df, y_train,
        eval_set=(X_test_df, y_test),
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_prob_custom = final_model.predict_proba(X_test_df)[:, 1]
    metrics_custom = compute_metrics_with_ci(y_test, y_prob_custom)

    log.info(f"\n  {'='*55}")
    log.info(f"  CUSTOM LOSS CATBOOST — TEST SET RESULTS")
    log.info(f"  {'='*55}")
    log.info(f"  AUROC : {metrics_custom['auroc']:.4f} "
             f"[{metrics_custom['auroc_ci_low']:.4f}–{metrics_custom['auroc_ci_high']:.4f}]")
    log.info(f"  AUPRC : {metrics_custom['auprc']:.4f} "
             f"[{metrics_custom['auprc_ci_low']:.4f}–{metrics_custom['auprc_ci_high']:.4f}]")
    log.info(f"  Sens  : {metrics_custom['sensitivity']:.4f}  "
             f"Spec: {metrics_custom['specificity']:.4f}  F1: {metrics_custom['f1']:.4f}")
    log.info(f"  PPV   : {metrics_custom['ppv']:.4f}  NPV: {metrics_custom['npv']:.4f}  "
             f"Brier: {metrics_custom['brier_score']:.4f}")
    log.info(f"  TP={metrics_custom['tp']}  FP={metrics_custom['fp']}  "
             f"TN={metrics_custom['tn']}  FN={metrics_custom['fn']}")

    # Load tuned model for comparison
    tuned_model = CatBoostClassifier()
    tuned_model.load_model(str(MODELS_DIR / "run_tuned" / "catboost_tuned.cbm"))
    y_prob_tuned  = tuned_model.predict_proba(X_test_df)[:, 1]
    metrics_tuned = compute_metrics(y_test, y_prob_tuned)

    log.info(f"\n  Comparison vs Tuned CatBoost:")
    log.info(f"    Custom loss AUROC: {metrics_custom['auroc']:.4f}  "
             f"vs Tuned: {metrics_tuned['auroc']:.4f}  "
             f"(Δ={metrics_custom['auroc']-metrics_tuned['auroc']:+.4f})")
    log.info(f"    Custom loss AUPRC: {metrics_custom['auprc']:.4f}  "
             f"vs Tuned: {metrics_tuned['auprc']:.4f}  "
             f"(Δ={metrics_custom['auprc']-metrics_tuned['auprc']:+.4f})")
    log.info(f"    Custom loss Sens:  {metrics_custom['sensitivity']:.4f}  "
             f"vs Tuned: {metrics_tuned['sensitivity']:.4f}  "
             f"(Δ={metrics_custom['sensitivity']-metrics_tuned['sensitivity']:+.4f})")

    # DeLong test
    auroc_c, auroc_t, z, p = delong_test(y_test, y_prob_custom, y_prob_tuned)
    log.info(f"\n  DeLong test (custom loss vs tuned CatBoost):")
    log.info(f"    Z={z:.4f}  p={p:.4f}  "
             f"{'Significant (p<0.05)' if p < 0.05 else 'Not significant'}")

    # ── Save model and results ────────────────────────────────────────────────
    final_model.save_model(str(CUSTOM_DIR / "catboost_custom_loss.cbm"))
    with open(CUSTOM_DIR / "severity_info.json", "w") as f:
        json.dump({
            "severity_variables"  : SEVERITY_VARS,
            "alpha_best"          : best_params["alpha"],
            "base_weight_best"    : best_params["base_weight"],
            "severity_mean_sepsis": float(severity_train[y_train==1].mean()),
            "severity_mean_nonsep": float(severity_train[y_train==0].mean()),
        }, f, indent=2)

    metrics_custom["delong_z_vs_tuned"] = z
    metrics_custom["delong_p_vs_tuned"] = p
    log_experiment(
        experiment_id = "phase2_custom_loss_B",
        dataset       = "B",
        model_name    = "catboost_custom_loss",
        phase         = "custom_loss",
        params        = best_params,
        metrics       = metrics_custom,
        notes         = f"Phoenix-aware asymmetric loss, alpha={best_params['alpha']:.3f}, "
                        f"base_weight={best_params['base_weight']:.3f}, "
                        f"severity from {len(SEVERITY_VARS)} Phoenix variables",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    log.info("\n--- Generating figures ---")
    plot_custom_loss_vs_tuned(y_test, y_prob_custom, y_prob_tuned,
                              metrics_custom, metrics_tuned)
    y_pred_custom = (y_prob_custom >= metrics_custom["threshold"]).astype(int)
    plot_cm(y_test, y_pred_custom, metrics_custom)

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'='*65}")
    log.info("PHASE 2 CUSTOM LOSS COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  Best alpha:      {best_params['alpha']:.4f}")
    log.info(f"  Best base_weight:{best_params['base_weight']:.4f}")
    log.info(f"  Test AUROC:      {metrics_custom['auroc']:.4f} "
             f"[{metrics_custom['auroc_ci_low']:.4f}–{metrics_custom['auroc_ci_high']:.4f}]")
    log.info(f"  Test AUPRC:      {metrics_custom['auprc']:.4f} "
             f"[{metrics_custom['auprc_ci_low']:.4f}–{metrics_custom['auprc_ci_high']:.4f}]")
    log.info(f"  Sensitivity:     {metrics_custom['sensitivity']:.4f}")
    log.info(f"  DeLong p:        {p:.4f}")
    log.info(f"\n  Now compare all three approaches:")
    log.info(f"    Tuned CatBoost:  AUROC={metrics_tuned['auroc']:.4f}")
    log.info(f"    Custom loss:     AUROC={metrics_custom['auroc']:.4f}")
    log.info(f"    Run ensemble script to get stacking result")
    log.info(f"\n  Log → {log_path}")


if __name__ == "__main__":
    main()