"""
=============================================================================
PHASE 3 — FULL EVALUATION SUITE
Early Prediction of Pediatric Sepsis Using Explainable AI (CatBoost + SHAP)
PIC Database v1.1.0 | Phoenix 2024 Criteria | Option B (Infection-Only Cohort)
=============================================================================

What this script does:
  1. Loads tuned CatBoost model + leakage-clean test set
  2. DeLong tests — tuned CatBoost vs all 5 baseline models
  3. Calibration analysis — before and after Platt scaling
  4. Decision Curve Analysis (DCA) — net clinical benefit across thresholds
  5. Subgroup analysis — by age band (neonate / infant / child / adolescent)
  6. Full confusion matrix + per-threshold metric table (already in Phase 2, reproduced here for completeness)

Outputs:
  Figures  → sepsis_ml/figures/phase3_evaluation/
  Results  → sepsis_ml/results/ (appended to experiment_log.json + results_summary.csv)
  Log      → sepsis_ml/logs/phase3_evaluation.log

Authors : Tanish Porwal & Uzair
Institute: NMIMS
Date     : April 2026
Dataset  : Option B — 633 test rows, 27.3% sepsis prevalence, 225 features
Model    : Tuned CatBoost (Optuna 100 trials) — sepsis_ml/models/run_tuned/catboost_tuned.cbm
=============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path

from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, f1_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pickle

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS — match your existing project structure exactly
# =============================================================================
# Adjust BASE_DIR if your repo root is elsewhere
BASE_DIR   = Path(__file__).resolve().parent.parent  # one level up from sepsis_ml/
DATA_DIR   = BASE_DIR / "model_datasets"
MODEL_DIR  = BASE_DIR / "sepsis_ml" / "models"
FIG_DIR    = BASE_DIR / "sepsis_ml" / "figures" / "phase3_evaluation"
RES_DIR    = BASE_DIR / "sepsis_ml" / "results"
LOG_DIR    = BASE_DIR / "sepsis_ml" / "logs"

for d in [FIG_DIR, RES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. LOGGING — append-only, same pattern as previous phases
# =============================================================================
log_path = LOG_DIR / "phase3_evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)
log.info("=" * 70)
log.info("PHASE 3 — FULL EVALUATION SUITE")
log.info(f"Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"Base dir      : {BASE_DIR}")
log.info("=" * 70)

# =============================================================================
# 3. CONSTANTS
# =============================================================================
RANDOM_SEED    = 42
TARGET_COL     = "sepsis_label"
BOOTSTRAP_ITER = 1000

# Age band definitions (years) — Fleming et al. 2011 / standard pediatric
# neonate: 0–28 days = 0–0.077y | infant: 29d–1y = 0.077–1y
# child: 1–12y | adolescent: 12–18y
AGE_BANDS = {
    "Neonate (0–28d)":    (0.0,   0.077),
    "Infant (29d–1y)":    (0.077, 1.0),
    "Child (1–12y)":      (1.0,   12.0),
    "Adolescent (12–18y)":(12.0,  18.0),
}

# Thresholds to evaluate for DCA and threshold table
THRESHOLDS = np.arange(0.01, 1.00, 0.01)

# Clinical thresholds of interest (from Phase 2A threshold analysis)
CLINICAL_THRESHOLDS = {
    "80% Sens": 0.577,
    "85% Sens": 0.447,
    "90% Sens (recommended)": 0.365,
    "92% Sens": 0.276,
    "95% Sens": 0.213,
}

np.random.seed(RANDOM_SEED)

# =============================================================================
# 4. LOAD DATA AND MODEL
# =============================================================================
log.info("Loading test set and model...")

test_path = DATA_DIR / "B_test_model_ready.csv"
if not test_path.exists():
    log.error(f"Test set not found: {test_path}")
    sys.exit(1)

df_test = pd.read_csv(test_path)
log.info(f"Test set loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")

# Separate features and target
if TARGET_COL not in df_test.columns:
    log.error(f"Target column '{TARGET_COL}' not found. Columns: {list(df_test.columns[:10])}")
    sys.exit(1)

y_test = df_test[TARGET_COL].values
X_test = df_test.drop(columns=[TARGET_COL])

# Drop any non-feature columns that may still be present (meta columns)
meta_cols = ["subject_id", "hadm_id", "icustay_id", "stay_id", "intime", "outtime",
             "phoenix_core_score", "phoenix_8_score"]
meta_present = [c for c in meta_cols if c in X_test.columns]
if meta_present:
    X_test = X_test.drop(columns=meta_present)
    log.info(f"Dropped meta columns from features: {meta_present}")

log.info(f"Feature matrix: {X_test.shape[0]} rows x {X_test.shape[1]} features")
log.info(f"Sepsis prevalence in test set: {y_test.mean():.1%} ({y_test.sum()} / {len(y_test)})")

# Load feature list if available (for SHAP phase later)
feature_list_path = DATA_DIR / "feature_list.json"
if feature_list_path.exists():
    with open(feature_list_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        expected_features = raw
    elif isinstance(raw, dict) and "feature_cols" in raw:
        expected_features = raw["feature_cols"]
    else:
        log.warning(f"feature_list.json format unrecognised — skipping alignment. Keys: {list(raw.keys())}")
        expected_features = None

    if expected_features is not None:
        missing = [c for c in expected_features if c not in X_test.columns]
        extra   = [c for c in X_test.columns if c not in expected_features]
        if missing:
            log.warning(f"{len(missing)} features missing from test set: {missing[:5]}")
        if extra:
            log.warning(f"{len(extra)} extra columns not in feature_list: {extra[:5]}")
        X_test = X_test[[c for c in expected_features if c in X_test.columns]]
        log.info(f"Feature matrix aligned to feature_list.json: {X_test.shape[1]} features")
    # Align columns to match training feature order
    missing = [c for c in expected_features if c not in X_test.columns]
    extra   = [c for c in X_test.columns if c not in expected_features]
    if missing:
        log.warning(f"{len(missing)} features in feature_list.json missing from test set: {missing[:5]}")
    if extra:
        log.warning(f"{len(extra)} extra columns in test set not in feature_list.json: {extra[:5]}")
    # Keep only expected features in correct order
    X_test = X_test[[c for c in expected_features if c in X_test.columns]]
    log.info(f"Feature matrix aligned to feature_list.json: {X_test.shape[1]} features")

# Load tuned CatBoost
tuned_model_path = MODEL_DIR / "run_tuned" / "catboost_tuned.cbm"
if not tuned_model_path.exists():
    log.error(f"Tuned model not found: {tuned_model_path}")
    sys.exit(1)

model_tuned = CatBoostClassifier()
model_tuned.load_model(str(tuned_model_path))
log.info(f"Tuned CatBoost loaded from {tuned_model_path}")

# Get predicted probabilities
prob_tuned = model_tuned.predict_proba(X_test)[:, 1]
log.info(f"Tuned CatBoost predictions: min={prob_tuned.min():.4f}, max={prob_tuned.max():.4f}, "
         f"mean={prob_tuned.mean():.4f}")

# =============================================================================
# 5. LOAD BASELINE MODELS (for DeLong tests)
# =============================================================================
log.info("Loading baseline models for DeLong tests...")

# Baseline models are in run_003 — load what exists
# CatBoost baseline uses .cbm; others use .pkl
baseline_model_files = {
    "CatBoost_baseline": MODEL_DIR / "run_003" / "catboost.cbm",
    "LightGBM":          MODEL_DIR / "run_003" / "lightgbm.pkl",
    "XGBoost":           MODEL_DIR / "run_003" / "xgboost.pkl",
    "RandomForest":      MODEL_DIR / "run_003" / "random_forest.pkl",
    "LogisticReg":       MODEL_DIR / "run_003" / "logistic_regression.pkl",
    "MLP":               MODEL_DIR / "run_003" / "mlp.pkl",
}

baseline_probs = {}
for name, path in baseline_model_files.items():
    if not path.exists():
        log.warning(f"Baseline model not found, skipping DeLong for {name}: {path}")
        continue
    try:
        if path.suffix == ".cbm":
            m = CatBoostClassifier()
            m.load_model(str(path))
            baseline_probs[name] = m.predict_proba(X_test)[:, 1]
        else:
            with open(path, "rb") as f:
                m = pickle.load(f)
            # sklearn pipeline or model — use predict_proba
            if hasattr(m, "predict_proba"):
                baseline_probs[name] = m.predict_proba(X_test)[:, 1]
            else:
                baseline_probs[name] = m.decision_function(X_test)
        log.info(f"  Loaded {name}")
    except Exception as e:
        log.warning(f"  Could not load {name}: {e}")

log.info(f"Loaded {len(baseline_probs)} baseline models for comparison")

# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def bootstrap_auroc(y_true, y_prob, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    """Bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.array(aucs)
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def bootstrap_auprc(y_true, y_prob, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    """Bootstrap confidence interval for AUPRC."""
    rng = np.random.RandomState(seed)
    auprcs = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auprcs.append(average_precision_score(y_true[idx], y_prob[idx]))
    auprcs = np.array(auprcs)
    return np.percentile(auprcs, 2.5), np.percentile(auprcs, 97.5)


def delong_test(y_true, prob_a, prob_b):
    """
    DeLong test for comparing two AUROCs.
    Returns (z_statistic, p_value).
    Implementation follows DeLong et al. Biometrics 1988.
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

    # Sort by label descending
    sorted_indices = np.argsort(y_true)[::-1]
    y_sorted = y_true[sorted_indices]
    m = int(y_sorted.sum())

    predictions_sorted = np.vstack([
        prob_a[sorted_indices],
        prob_b[sorted_indices]
    ])

    aucs, delongcov = fastDeLong(predictions_sorted, m)
    auc_diff = aucs[0] - aucs[1]
    var_diff  = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0
    z = auc_diff / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def metrics_at_threshold(y_true, y_prob, threshold):
    """Compute full metric set at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {"threshold": threshold, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv, "F1": f1}


# =============================================================================
# 7. SECTION A — DELONG TESTS: TUNED CATBOOST vs ALL BASELINES
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION A — DeLong Tests")
log.info("=" * 60)

auroc_tuned = roc_auc_score(y_test, prob_tuned)
ci_lo, ci_hi = bootstrap_auroc(y_test, prob_tuned)
log.info(f"Tuned CatBoost AUROC: {auroc_tuned:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]")

delong_results = []
for name, prob_baseline in baseline_probs.items():
    auroc_b = roc_auc_score(y_test, prob_baseline)
    z, p    = delong_test(y_test, prob_tuned, prob_baseline)
    ci_b_lo, ci_b_hi = bootstrap_auroc(y_test, prob_baseline)
    sig = "YES" if p < 0.05 else "NO"
    log.info(f"  vs {name:<20} | AUROC={auroc_b:.4f} [{ci_b_lo:.4f}–{ci_b_hi:.4f}] | "
             f"Z={z:+.4f} | p={p:.4f} | Significant: {sig}")
    delong_results.append({
        "comparison": f"Tuned CatBoost vs {name}",
        "auroc_tuned": auroc_tuned,
        "auroc_baseline": auroc_b,
        "z_statistic": z,
        "p_value": p,
        "significant_p05": sig
    })

delong_df = pd.DataFrame(delong_results)
delong_path = RES_DIR / "phase3_delong_results.csv"
delong_df.to_csv(delong_path, index=False)
log.info(f"DeLong results saved: {delong_path}")

# =============================================================================
# 8. SECTION B — CALIBRATION + PLATT SCALING
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION B — Calibration + Platt Scaling")
log.info("=" * 60)

# Need training set for Platt scaling — load it
train_path = DATA_DIR / "B_train_model_ready.csv"
if not train_path.exists():
    log.error(f"Training set not found for Platt scaling: {train_path}")
    sys.exit(1)

df_train    = pd.read_csv(train_path)
y_train     = df_train[TARGET_COL].values
X_train     = df_train.drop(columns=[TARGET_COL])
meta_present_train = [c for c in meta_cols if c in X_train.columns]
if meta_present_train:
    X_train = X_train.drop(columns=meta_present_train)
if feature_list_path.exists():
    X_train = X_train[[c for c in expected_features if c in X_train.columns]]

log.info(f"Training set loaded: {X_train.shape[0]} rows x {X_train.shape[1]} features")

# Platt scaling: fit logistic regression on train predictions → calibrated probabilities
prob_train_tuned = model_tuned.predict_proba(X_train)[:, 1]

# Fit Platt scaler on training set (sigmoid calibration)
platt_scaler = LogisticRegression(C=1.0, random_state=RANDOM_SEED)
platt_scaler.fit(prob_train_tuned.reshape(-1, 1), y_train)
prob_tuned_calibrated = platt_scaler.predict_proba(prob_tuned.reshape(-1, 1))[:, 1]

# Save scaler for deployment
scaler_path = MODEL_DIR / "run_tuned" / "platt_scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(platt_scaler, f)
log.info(f"Platt scaler saved: {scaler_path}")

# Calibration metrics
brier_before = brier_score_loss(y_test, prob_tuned)
brier_after  = brier_score_loss(y_test, prob_tuned_calibrated)
log.info(f"Brier score BEFORE Platt scaling: {brier_before:.4f}")
log.info(f"Brier score AFTER  Platt scaling: {brier_after:.4f}")

# Calibration curve data (10 bins)
frac_pos_before, mean_pred_before = calibration_curve(y_test, prob_tuned, n_bins=10, strategy="uniform")
frac_pos_after,  mean_pred_after  = calibration_curve(y_test, prob_tuned_calibrated, n_bins=10, strategy="uniform")

# Save calibration results
calib_results = {
    "brier_before": brier_before,
    "brier_after": brier_after,
    "brier_improvement": brier_before - brier_after,
    "calibration_before": {
        "mean_predicted": mean_pred_before.tolist(),
        "fraction_positive": frac_pos_before.tolist()
    },
    "calibration_after": {
        "mean_predicted": mean_pred_after.tolist(),
        "fraction_positive": frac_pos_after.tolist()
    }
}
with open(RES_DIR / "phase3_calibration_results.json", "w") as f:
    json.dump(calib_results, f, indent=2)
log.info("Calibration results saved")

# =============================================================================
# 9. SECTION C — DECISION CURVE ANALYSIS (DCA)
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION C — Decision Curve Analysis (DCA)")
log.info("=" * 60)

"""
DCA computes net benefit (NB) at each threshold probability pt:
  NB(pt) = (TP/N) - (FP/N) * (pt / (1 - pt))

where N = total patients, pt = threshold.

Comparators:
  - Treat all:  NB = prevalence - (1-prevalence) * pt/(1-pt)
  - Treat none: NB = 0 always
"""

n_test    = len(y_test)
prev      = y_test.mean()
dca_rows  = []

for pt in THRESHOLDS:
    if pt >= 1.0:
        continue
    y_pred = (prob_tuned >= pt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    nb_model    = (tp / n_test) - (fp / n_test) * (pt / (1 - pt))
    nb_treat_all = prev - (1 - prev) * (pt / (1 - pt))
    nb_treat_none = 0.0

    # Also compute for calibrated probabilities
    y_pred_cal = (prob_tuned_calibrated >= pt).astype(int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, y_pred_cal, labels=[0, 1]).ravel()
    nb_model_cal = (tp_c / n_test) - (fp_c / n_test) * (pt / (1 - pt))

    dca_rows.append({
        "threshold": pt,
        "nb_model": nb_model,
        "nb_model_calibrated": nb_model_cal,
        "nb_treat_all": nb_treat_all,
        "nb_treat_none": nb_treat_none
    })

dca_df = pd.DataFrame(dca_rows)
dca_path = RES_DIR / "phase3_dca_results.csv"
dca_df.to_csv(dca_path, index=False)
log.info(f"DCA results saved: {dca_path}")

# Log NB at clinical thresholds of interest
log.info("Net benefit at key clinical thresholds:")
for label, pt in CLINICAL_THRESHOLDS.items():
    row = dca_df[dca_df["threshold"] >= pt].iloc[0] if len(dca_df[dca_df["threshold"] >= pt]) > 0 else None
    if row is not None:
        log.info(f"  {label} (pt={pt:.3f}): NB_model={row['nb_model']:.4f}, "
                 f"NB_treat_all={row['nb_treat_all']:.4f}")

# =============================================================================
# 10. SECTION D — SUBGROUP ANALYSIS BY AGE BAND
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION D — Subgroup Analysis by Age Band")
log.info("=" * 60)

# age_years must be in the test set — check
age_col = "age_years"
if age_col not in df_test.columns:
    log.warning(f"'{age_col}' not found in test set — subgroup analysis skipped")
    subgroup_results = []
else:
    subgroup_results = []
    primary_threshold = 0.365  # recommended clinical threshold

    for band_name, (age_lo, age_hi) in AGE_BANDS.items():
        mask = (df_test[age_col] >= age_lo) & (df_test[age_col] < age_hi)
        n_band    = mask.sum()
        y_band    = y_test[mask]
        prob_band = prob_tuned[mask]

        if n_band < 10:
            log.warning(f"  {band_name}: only {n_band} patients — skipping (too small)")
            continue
        if len(np.unique(y_band)) < 2:
            log.warning(f"  {band_name}: only one class present (n={n_band}) — AUROC undefined")
            continue

        # Metrics
        auroc_band = roc_auc_score(y_band, prob_band)
        auprc_band = average_precision_score(y_band, prob_band)
        ci_lo_b, ci_hi_b = bootstrap_auroc(y_band, prob_band, n_iter=500)

        m = metrics_at_threshold(y_band, prob_band, primary_threshold)
        prev_band = y_band.mean()

        log.info(f"  {band_name}: n={n_band}, sepsis={y_band.sum()} ({prev_band:.1%}), "
                 f"AUROC={auroc_band:.4f} [{ci_lo_b:.4f}–{ci_hi_b:.4f}], "
                 f"AUPRC={auprc_band:.4f}, "
                 f"Sens={m['sensitivity']:.3f}, Spec={m['specificity']:.3f}")

        subgroup_results.append({
            "age_band": band_name,
            "n_patients": int(n_band),
            "n_sepsis": int(y_band.sum()),
            "prevalence": float(prev_band),
            "auroc": auroc_band,
            "auroc_ci_lo": ci_lo_b,
            "auroc_ci_hi": ci_hi_b,
            "auprc": auprc_band,
            "sensitivity_at_0365": m["sensitivity"],
            "specificity_at_0365": m["specificity"],
            "ppv_at_0365": m["PPV"],
            "npv_at_0365": m["NPV"],
            "f1_at_0365": m["F1"],
            "tp": m["TP"], "fp": m["FP"], "tn": m["TN"], "fn": m["FN"]
        })

    subgroup_df = pd.DataFrame(subgroup_results)
    subgroup_path = RES_DIR / "phase3_subgroup_results.csv"
    subgroup_df.to_csv(subgroup_path, index=False)
    log.info(f"Subgroup results saved: {subgroup_path}")

# =============================================================================
# 11. SECTION E — FULL THRESHOLD SENSITIVITY TABLE
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION E — Threshold Sensitivity Table")
log.info("=" * 60)

threshold_rows = []
for label, pt in CLINICAL_THRESHOLDS.items():
    m = metrics_at_threshold(y_test, prob_tuned, pt)
    auroc_v = roc_auc_score(y_test, prob_tuned)
    auprc_v = average_precision_score(y_test, prob_tuned)
    m["label"]  = label
    m["auroc"]  = auroc_v
    m["auprc"]  = auprc_v
    m["brier"]  = brier_score_loss(y_test, prob_tuned)
    threshold_rows.append(m)
    log.info(f"  {label} | thresh={pt:.3f} | Sens={m['sensitivity']:.3f} | "
             f"Spec={m['specificity']:.3f} | PPV={m['PPV']:.3f} | NPV={m['NPV']:.3f} | "
             f"F1={m['F1']:.3f} | TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

threshold_df = pd.DataFrame(threshold_rows)
threshold_path = RES_DIR / "phase3_threshold_table.csv"
threshold_df.to_csv(threshold_path, index=False)
log.info(f"Threshold table saved: {threshold_path}")

# =============================================================================
# 12. FIGURES
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Generating figures...")
log.info("=" * 60)

STYLE = {
    "tuned":       ("#1a56db", 2.5, "-",  "Tuned CatBoost (primary)"),
    "baseline_cb": ("#f97316", 1.5, "--", "CatBoost baseline"),
    "lgbm":        ("#10b981", 1.5, "--", "LightGBM"),
    "xgb":         ("#8b5cf6", 1.5, "--", "XGBoost"),
    "rf":          ("#f59e0b", 1.5, ":",  "Random Forest"),
    "lr":          ("#6b7280", 1.5, ":",  "Logistic Regression"),
    "mlp":         ("#ec4899", 1.5, ":",  "MLP"),
}

COLOR_MAP = {
    "CatBoost_baseline": STYLE["baseline_cb"],
    "LightGBM":          STYLE["lgbm"],
    "XGBoost":           STYLE["xgb"],
    "RandomForest":      STYLE["rf"],
    "LogisticReg":       STYLE["lr"],
    "MLP":               STYLE["mlp"],
}

# ------------------------------------------------------------------
# FIGURE 1 — ROC + PR curves: tuned CatBoost vs all baselines
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase 3 — ROC and PR Curves: Tuned CatBoost vs Baseline Models\n"
             "Option B (Infection-Only Cohort) | Test Set n=633",
             fontsize=12, fontweight="bold", y=1.01)

# ROC
ax = axes[0]
fpr_t, tpr_t, _ = roc_curve(y_test, prob_tuned)
ax.plot(fpr_t, tpr_t, color=STYLE["tuned"][0], lw=STYLE["tuned"][1],
        ls=STYLE["tuned"][2], label=f"{STYLE['tuned'][3]} (AUC={auroc_tuned:.4f})")

for name, prob_b in baseline_probs.items():
    s = COLOR_MAP.get(name, (STYLE["baseline_cb"][0], 1.5, "--", name))
    fpr_b, tpr_b, _ = roc_curve(y_test, prob_b)
    auroc_b = roc_auc_score(y_test, prob_b)
    ax.plot(fpr_b, tpr_b, color=s[0], lw=s[1], ls=s[2],
            label=f"{s[3]} (AUC={auroc_b:.4f})", alpha=0.8)

ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=10)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
ax.set_title("ROC Curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

# PR
ax = axes[1]
prec_t, rec_t, _ = precision_recall_curve(y_test, prob_tuned)
auprc_tuned = average_precision_score(y_test, prob_tuned)
ax.plot(rec_t, prec_t, color=STYLE["tuned"][0], lw=STYLE["tuned"][1],
        ls=STYLE["tuned"][2], label=f"{STYLE['tuned'][3]} (AP={auprc_tuned:.4f})")

for name, prob_b in baseline_probs.items():
    s = COLOR_MAP.get(name, (STYLE["baseline_cb"][0], 1.5, "--", name))
    prec_b, rec_b, _ = precision_recall_curve(y_test, prob_b)
    auprc_b = average_precision_score(y_test, prob_b)
    ax.plot(rec_b, prec_b, color=s[0], lw=s[1], ls=s[2],
            label=f"{s[3]} (AP={auprc_b:.4f})", alpha=0.8)

ax.axhline(y=y_test.mean(), color="k", ls="--", lw=0.8, alpha=0.5,
           label=f"Baseline prevalence ({y_test.mean():.3f})")
ax.set_xlabel("Recall (Sensitivity)", fontsize=10)
ax.set_ylabel("Precision (PPV)", fontsize=10)
ax.set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()
fig1_path = FIG_DIR / "phase3_fig1_roc_pr_all_models.png"
plt.savefig(fig1_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 1 saved: {fig1_path}")

# ------------------------------------------------------------------
# FIGURE 2 — DeLong test results (bar chart of AUROC differences)
# ------------------------------------------------------------------
if len(delong_results) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Phase 3 — DeLong Test: AUROC Difference vs Tuned CatBoost\n"
                 "(Positive = Tuned CatBoost better; * = p<0.05)",
                 fontsize=11, fontweight="bold")

    labels_d = [r["comparison"].replace("Tuned CatBoost vs ", "") for r in delong_results]
    diffs    = [r["auroc_tuned"] - r["auroc_baseline"] for r in delong_results]
    pvals    = [r["p_value"] for r in delong_results]
    colors_d = ["#1a56db" if d >= 0 else "#ef4444" for d in diffs]

    bars = ax.barh(labels_d, diffs, color=colors_d, alpha=0.75, edgecolor="white")
    for i, (bar, p) in enumerate(zip(bars, pvals)):
        annotation = f"p={p:.4f}" + (" *" if p < 0.05 else "")
        ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
                annotation, va="center", ha="left", fontsize=9)

    ax.axvline(x=0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("AUROC Difference (Tuned CatBoost − Baseline)", fontsize=10)
    ax.set_title("", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim([-0.03, max(diffs) + 0.015] if diffs else [-0.03, 0.05])
    plt.tight_layout()
    fig2_path = FIG_DIR / "phase3_fig2_delong_comparison.png"
    plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"Figure 2 saved: {fig2_path}")

# ------------------------------------------------------------------
# FIGURE 3 — Calibration: before and after Platt scaling
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase 3 — Calibration: Before and After Platt Scaling\n"
             "Tuned CatBoost | Option B Test Set n=633",
             fontsize=12, fontweight="bold", y=1.01)

for ax, mean_pred, frac_pos, prob_plot, title, brier_val, color in [
    (axes[0], mean_pred_before, frac_pos_before, prob_tuned,
     f"Before Platt Scaling\nBrier = {brier_before:.4f}", brier_before, "#1a56db"),
    (axes[1], mean_pred_after, frac_pos_after, prob_tuned_calibrated,
     f"After Platt Scaling\nBrier = {brier_after:.4f}", brier_after, "#10b981"),
]:
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=7,
            label=f"Model (Brier={brier_val:.4f})")

    # Histogram of predicted probabilities (secondary axis)
    ax2 = ax.twinx()
    ax2.hist(prob_plot[y_test == 0], bins=20, alpha=0.2, color="#6b7280", label="Non-sepsis")
    ax2.hist(prob_plot[y_test == 1], bins=20, alpha=0.3, color="#ef4444", label="Sepsis")
    ax2.set_ylabel("Count", fontsize=9, color="#6b7280")
    ax2.tick_params(axis="y", labelcolor="#6b7280")
    ax2.set_ylim(0, ax2.get_ylim()[1] * 5)  # compress histogram

    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
fig3_path = FIG_DIR / "phase3_fig3_calibration.png"
plt.savefig(fig3_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 3 saved: {fig3_path}")

# ------------------------------------------------------------------
# FIGURE 4 — Decision Curve Analysis
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Phase 3 — Decision Curve Analysis (DCA)\n"
             "Net Clinical Benefit Across Threshold Probabilities | Test Set n=633",
             fontsize=12, fontweight="bold")

# Clip NB to reasonable range for display
clip_lo, clip_hi = -0.05, prev + 0.05

ax.plot(dca_df["threshold"], dca_df["nb_model"].clip(clip_lo, clip_hi),
        color="#1a56db", lw=2.5, label="Tuned CatBoost (uncalibrated)")
ax.plot(dca_df["threshold"], dca_df["nb_model_calibrated"].clip(clip_lo, clip_hi),
        color="#10b981", lw=2.0, ls="--", label="Tuned CatBoost (Platt-calibrated)")
ax.plot(dca_df["threshold"], dca_df["nb_treat_all"].clip(clip_lo, clip_hi),
        color="#f97316", lw=1.5, ls=":", label="Treat all")
ax.axhline(y=0, color="black", lw=1.2, ls="-", label="Treat none (NB=0)")

# Mark clinical thresholds
for label, pt in CLINICAL_THRESHOLDS.items():
    ax.axvline(x=pt, color="#6b7280", lw=0.8, ls="--", alpha=0.6)
    ax.text(pt + 0.005, clip_hi * 0.85, label.split(" ")[0],
            fontsize=7, color="#6b7280", rotation=90, va="top")

ax.set_xlabel("Threshold Probability", fontsize=11)
ax.set_ylabel("Net Benefit", fontsize=11)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([clip_lo, clip_hi])
plt.tight_layout()
fig4_path = FIG_DIR / "phase3_fig4_dca.png"
plt.savefig(fig4_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 4 saved: {fig4_path}")

# ------------------------------------------------------------------
# FIGURE 5 — Subgroup analysis by age band
# ------------------------------------------------------------------
if subgroup_results:
    subgroup_df_plot = pd.DataFrame(subgroup_results)
    n_bands = len(subgroup_df_plot)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, n_bands * 1.2 + 2)))
    fig.suptitle("Phase 3 — Subgroup Analysis by Age Band\n"
                 "Tuned CatBoost | Option B Test Set | Threshold=0.365",
                 fontsize=12, fontweight="bold", y=1.02)

    band_colors = ["#1a56db", "#f97316", "#10b981", "#8b5cf6"][:n_bands]

    # AUROC by band
    ax = axes[0]
    y_pos = range(n_bands)
    bars = ax.barh(subgroup_df_plot["age_band"], subgroup_df_plot["auroc"],
                   xerr=[
                       subgroup_df_plot["auroc"] - subgroup_df_plot["auroc_ci_lo"],
                       subgroup_df_plot["auroc_ci_hi"] - subgroup_df_plot["auroc"]
                   ],
                   color=band_colors, alpha=0.8, capsize=4, edgecolor="white")
    ax.axvline(x=auroc_tuned, color="black", lw=1.2, ls="--",
               label=f"Overall AUROC={auroc_tuned:.4f}")
    for bar, row in zip(bars, subgroup_df_plot.itertuples()):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={row.n_patients} ({row.prevalence:.0%} sepsis)",
                va="center", fontsize=8.5)
    ax.set_xlabel("AUROC (95% bootstrap CI)", fontsize=10)
    ax.set_title("AUROC by Age Band", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim([0.5, 1.05])

    # Sens / Spec by band
    ax = axes[1]
    x = np.arange(n_bands)
    width = 0.35
    bars1 = ax.bar(x - width / 2, subgroup_df_plot["sensitivity_at_0365"],
                   width, label="Sensitivity", color="#1a56db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, subgroup_df_plot["specificity_at_0365"],
                   width, label="Specificity", color="#10b981", alpha=0.8)
    ax.axhline(y=0.9, color="#1a56db", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(y=0.963, color="#10b981", lw=0.8, ls="--", alpha=0.5)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_df_plot["age_band"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Sensitivity & Specificity by Age Band\n(threshold=0.365)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.12])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig5_path = FIG_DIR / "phase3_fig5_subgroup_analysis.png"
    plt.savefig(fig5_path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"Figure 5 saved: {fig5_path}")

# ------------------------------------------------------------------
# FIGURE 6 — Summary dashboard (4-panel)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Phase 3 — Evaluation Summary Dashboard\n"
             "Tuned CatBoost | Option B (Infection-Only) | Test Set n=633",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# Panel A: ROC
ax_roc = fig.add_subplot(gs[0, 0])
ax_roc.plot(fpr_t, tpr_t, color="#1a56db", lw=2.5,
            label=f"Tuned CatBoost (AUC={auroc_tuned:.4f})")
ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
ax_roc.set_xlabel("FPR", fontsize=9)
ax_roc.set_ylabel("TPR", fontsize=9)
ax_roc.set_title("ROC Curve", fontsize=10, fontweight="bold")
ax_roc.legend(fontsize=8)
ax_roc.grid(alpha=0.3)

# Panel B: PR
ax_pr = fig.add_subplot(gs[0, 1])
ax_pr.plot(rec_t, prec_t, color="#f97316", lw=2.5,
           label=f"Tuned CatBoost (AP={auprc_tuned:.4f})")
ax_pr.axhline(y=y_test.mean(), color="k", ls="--", lw=0.8, alpha=0.5,
              label=f"Baseline ({y_test.mean():.3f})")
ax_pr.set_xlabel("Recall", fontsize=9)
ax_pr.set_ylabel("Precision", fontsize=9)
ax_pr.set_title("Precision-Recall Curve", fontsize=10, fontweight="bold")
ax_pr.legend(fontsize=8)
ax_pr.grid(alpha=0.3)

# Panel C: Calibration after Platt
ax_cal = fig.add_subplot(gs[1, 0])
ax_cal.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect")
ax_cal.plot(mean_pred_before, frac_pos_before, "o--", color="#6b7280", lw=1.5,
            ms=5, label=f"Before (Brier={brier_before:.4f})", alpha=0.7)
ax_cal.plot(mean_pred_after, frac_pos_after, "o-", color="#10b981", lw=2,
            ms=7, label=f"After Platt (Brier={brier_after:.4f})")
ax_cal.set_xlabel("Mean Predicted Probability", fontsize=9)
ax_cal.set_ylabel("Fraction of Positives", fontsize=9)
ax_cal.set_title("Calibration Curve", fontsize=10, fontweight="bold")
ax_cal.legend(fontsize=8)
ax_cal.grid(alpha=0.3)

# Panel D: DCA
ax_dca = fig.add_subplot(gs[1, 1])
ax_dca.plot(dca_df["threshold"], dca_df["nb_model"].clip(clip_lo, clip_hi),
            color="#1a56db", lw=2.5, label="Model")
ax_dca.plot(dca_df["threshold"], dca_df["nb_treat_all"].clip(clip_lo, clip_hi),
            color="#f97316", lw=1.5, ls=":", label="Treat all")
ax_dca.axhline(y=0, color="black", lw=1.2, label="Treat none")
ax_dca.axvline(x=0.365, color="#6b7280", lw=0.8, ls="--", alpha=0.7,
               label="Recommended threshold (0.365)")
ax_dca.set_xlabel("Threshold Probability", fontsize=9)
ax_dca.set_ylabel("Net Benefit", fontsize=9)
ax_dca.set_title("Decision Curve Analysis", fontsize=10, fontweight="bold")
ax_dca.legend(fontsize=8)
ax_dca.grid(alpha=0.3)
ax_dca.set_xlim([0, 1])
ax_dca.set_ylim([clip_lo, clip_hi])

dashboard_path = FIG_DIR / "phase3_fig6_summary_dashboard.png"
plt.savefig(dashboard_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 6 (dashboard) saved: {dashboard_path}")

# =============================================================================
# 13. APPEND TO EXPERIMENT LOG
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Appending results to experiment log...")
log.info("=" * 60)

exp_entry = {
    "phase": "Phase3_Evaluation",
    "timestamp": datetime.now().isoformat(),
    "dataset": "Option_B",
    "model": "Tuned_CatBoost",
    "n_test": int(len(y_test)),
    "n_sepsis_test": int(y_test.sum()),
    "auroc": float(auroc_tuned),
    "auroc_ci": [float(ci_lo), float(ci_hi)],
    "auprc": float(auprc_tuned),
    "brier_before_platt": float(brier_before),
    "brier_after_platt": float(brier_after),
    "delong_n_comparisons": len(delong_results),
    "delong_n_significant": sum(1 for r in delong_results if r["significant_p05"] == "YES"),
    "subgroup_n_bands": len(subgroup_results),
    "metrics_at_recommended_threshold_0365": {
        k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
        for k, v in metrics_at_threshold(y_test, prob_tuned, 0.365).items()
    },
    "artifacts": {
        "delong_csv": str(delong_path),
        "calibration_json": str(RES_DIR / "phase3_calibration_results.json"),
        "dca_csv": str(dca_path),
        "subgroup_csv": str(subgroup_path) if subgroup_results else None,
        "threshold_table_csv": str(threshold_path),
        "figures": [str(FIG_DIR / f"phase3_fig{i}_{n}.png")
                    for i, n in enumerate(["roc_pr_all_models", "delong_comparison",
                                           "calibration", "dca", "subgroup_analysis",
                                           "summary_dashboard"], 1)]
    }
}

exp_log_path = RES_DIR / "experiment_log.json"
if exp_log_path.exists():
    try:
        with open(exp_log_path, "r") as f:
            exp_log = json.load(f)
        if not isinstance(exp_log, list):
            exp_log = [exp_log]
    except json.JSONDecodeError:
        log.warning("experiment_log.json is corrupted — backing up and starting fresh")
        import shutil
        shutil.copy(exp_log_path, str(exp_log_path) + ".corrupted_backup")
        exp_log = []
else:
    exp_log = []
exp_log.append(exp_entry)
with open(exp_log_path, "w") as f:
    json.dump(exp_log, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
log.info(f"Experiment log updated: {exp_log_path}")

# Append to results_summary CSV
summary_row = pd.DataFrame([{
    "phase": "Phase3_Evaluation",
    "timestamp": datetime.now().isoformat(),
    "model": "Tuned_CatBoost",
    "dataset": "Option_B",
    "auroc": auroc_tuned,
    "auprc": auprc_tuned,
    "brier_before_platt": brier_before,
    "brier_after_platt": brier_after,
    "sensitivity_0365": metrics_at_threshold(y_test, prob_tuned, 0.365)["sensitivity"],
    "specificity_0365": metrics_at_threshold(y_test, prob_tuned, 0.365)["specificity"],
    "notes": "Full evaluation: DeLong, calibration, DCA, subgroup"
}])
summary_path = RES_DIR / "results_summary.csv"
if summary_path.exists():
    existing = pd.read_csv(summary_path)
    summary_row = pd.concat([existing, summary_row], ignore_index=True)
summary_row.to_csv(summary_path, index=False)
log.info(f"Results summary updated: {summary_path}")

# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================
log.info("")
log.info("=" * 70)
log.info("PHASE 3 COMPLETE — SUMMARY")
log.info("=" * 70)
log.info(f"Tuned CatBoost AUROC        : {auroc_tuned:.4f} [{ci_lo:.4f}–{ci_hi:.4f}]")
log.info(f"Tuned CatBoost AUPRC        : {auprc_tuned:.4f}")
log.info(f"Brier score (before Platt)  : {brier_before:.4f}")
log.info(f"Brier score (after Platt)   : {brier_after:.4f}")
log.info(f"DeLong comparisons done     : {len(delong_results)}")
sig_count = sum(1 for r in delong_results if r["significant_p05"] == "YES")
log.info(f"Significant improvements    : {sig_count} / {len(delong_results)}")
log.info(f"Age bands analysed          : {len(subgroup_results)}")
log.info(f"Figures saved to            : {FIG_DIR}")
log.info(f"Results saved to            : {RES_DIR}")
log.info(f"Log saved to                : {log_path}")
log.info("")
log.info("Next step: Phase 4 — SHAP Explainability")
log.info("  → Global beeswarm (top 20 features)")
log.info("  → Dependence plots: lactate_max, ddimer_max, pt_mean, platelets_min, base_excess_min")
log.info("  → Waterfall plots: TP, FN, FP patient examples")
log.info("  → SHAP interaction: lactate_max x platelets_min")
log.info("=" * 70)
