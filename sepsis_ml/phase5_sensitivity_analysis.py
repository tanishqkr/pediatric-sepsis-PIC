"""
=============================================================================
PHASE 5 — SENSITIVITY ANALYSIS (Option A — Full Cohort)
Early Prediction of Pediatric Sepsis Using Explainable AI (CatBoost + SHAP)
PIC Database v1.1.0 | Phoenix 2024 Criteria
=============================================================================

Purpose:
  Validate that primary findings (Option B) generalise to the full cohort.
  Answers the question: "Does our model story hold when we change the cohort
  definition from infection-only to all ICU patients?"

What this script does:
  1. Feature preparation — applies identical leakage-clean prep to Option A
     (same 80-column drops, same encoding, same feature alignment as Option B)
  2. Trains CatBoost using FIXED tuned hyperparameters from Phase 2A
     (no Optuna retuning — deliberately held constant to isolate cohort effect)
  3. Full evaluation — same metrics as Phase 3
  4. Head-to-head comparison table: Option A vs Option B
  5. SHAP top-20 feature importance (quick, no interaction matrix)

Key differences from Option B:
  - n_train: 9,957 (vs 2,530)
  - n_test:  2,490 (vs 633)
  - Sepsis prevalence: 6.9% (vs 27.3%)
  - Class weights: {0:1, 1:13.4} (vs {0:1, 1:2.7})
  - Clinical framing: sepsis vs ALL non-sepsis (vs sepsis vs infected-non-sepsis)

Outputs (all inside sepsis_ml/sensitivity_analysis/):
  models/        → catboost_A_tuned.cbm
  figures/       → all evaluation + SHAP figures
  results/       → metrics JSON, comparison CSV, SHAP importance CSV
  logs/          → phase5_sensitivity_analysis.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase5_sensitivity_analysis.py

Authors : Tanish Porwal & Uzair
Institute: NMIMS
Date     : April 2026
=============================================================================
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pathlib import Path
from scipy import stats

import shap
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, f1_score,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
BASE_DIR   = Path(__file__).resolve().parent.parent
DB_DIR     = BASE_DIR / "paediatric-intensive-care-database-1.1.0"
OUTPUT2    = DB_DIR / "output2"

# All sensitivity analysis outputs go here
SA_DIR     = BASE_DIR / "sepsis_ml" / "sensitivity_analysis"
SA_MODELS  = SA_DIR / "models"
SA_FIGS    = SA_DIR / "figures"
SA_RESULTS = SA_DIR / "results"
SA_LOGS    = SA_DIR / "logs"

for d in [SA_MODELS, SA_FIGS, SA_RESULTS, SA_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# Option B model-ready files (for feature alignment reference)
MODEL_DATA_DIR = BASE_DIR / "model_datasets"

# =============================================================================
# 2. LOGGING
# =============================================================================
log_path = SA_LOGS / "phase5_sensitivity_analysis.log"
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
log.info("PHASE 5 — SENSITIVITY ANALYSIS (Option A — Full Cohort)")
log.info(f"Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"Base dir      : {BASE_DIR}")
log.info("=" * 70)

# =============================================================================
# 3. CONSTANTS
# =============================================================================
RANDOM_SEED        = 42
TARGET_COL         = "sepsis_label"
BOOTSTRAP_ITER     = 1000
CV_FOLDS           = 5

# Option A class weights — inverse frequency (6.9% sepsis → ratio 13.4:1)
CLASS_WEIGHT_A     = {0: 1.0, 1: 13.4}

# Fixed tuned hyperparameters from Phase 2A (Option B)
# These are held CONSTANT — sensitivity analysis isolates cohort effect only
TUNED_PARAMS = {
    "iterations"          : 510,
    "learning_rate"       : 0.0342,
    "depth"               : 7,
    "l2_leaf_reg"         : 3.754,
    "bagging_temperature" : 0.975,
    "random_strength"     : 0.050,
    "border_count"        : 233,
    "class_weights"       : [1.0, 13.4],   # updated for Option A
    "eval_metric"         : "AUC",
    "random_seed"         : RANDOM_SEED,
    "verbose"             : 100,
    "early_stopping_rounds": 50,
}

# Clinical thresholds — same as Phase 3 for direct comparison
CLINICAL_THRESHOLDS = {
    "80% Sens": 0.577,
    "85% Sens": 0.447,
    "90% Sens (recommended)": 0.365,
    "92% Sens": 0.276,
    "95% Sens": 0.213,
}

# Option B reference results (from Phase 3) for comparison table
OPTION_B_RESULTS = {
    "auroc":        0.9868,
    "auroc_ci":     [0.9794, 0.9931],
    "auprc":        0.9691,
    "sensitivity":  0.896,
    "specificity":  0.963,
    "ppv":          0.901,
    "npv":          0.961,
    "f1":           0.899,
    "brier":        0.0385,
    "n_train":      2530,
    "n_test":       633,
    "prevalence":   0.273,
    "threshold":    0.365,
}

np.random.seed(RANDOM_SEED)

# =============================================================================
# 4. DROP DECISIONS — identical to phase0_1_feature_prep.py
# =============================================================================
DROP_LEAKAGE       = ["expire_flag", "hosp_expire", "los_hours"]
DROP_ZERO_VARIANCE = ["vaso_vasopressin", "suspected_infection",
                      "symptom_anhelation_and_cyanosis", "symptom_infection"]
DROP_FREE_TEXT     = ["diagnosis", "icd10"]
DROP_NO_VARIANCE   = ["ethnicity"]
DROP_DUPLICATES    = ["age_months"]
DROP_REFERENCE     = ["phoenix_core_score", "phoenix_8_score"]
DROP_VASO          = ["n_vasoactives_24h", "vaso_dopamine", "vaso_epinephrine",
                      "vaso_norepinephrine", "vaso_dobutamine", "vaso_milrinone"]
DROP_META          = ["subject_id", "hadm_id", "icustay_id", "stay_id",
                      "intime", "outtime"]

GENDER_MAP    = {"M": 1, "F": 0}
CARE_UNIT_COL = "care_unit"

# =============================================================================
# 5. FEATURE PREPARATION — same logic as phase0_1_feature_prep.py
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION 1 — Feature Preparation (Option A)")
log.info("=" * 60)

# Load raw Option A files from output2
train_path_raw = OUTPUT2 / "option_A_train.csv"
test_path_raw  = OUTPUT2 / "option_A_test.csv"

if not train_path_raw.exists():
    log.error(f"Option A train not found: {train_path_raw}")
    log.error("Expected location: paediatric-intensive-care-database-1.1.0/output2/option_A_train.csv")
    sys.exit(1)

train_raw = pd.read_csv(train_path_raw)
test_raw  = pd.read_csv(test_path_raw)
log.info(f"Raw Option A train: {train_raw.shape}")
log.info(f"Raw Option A test:  {test_raw.shape}")
log.info(f"Sepsis prevalence  — train: {train_raw[TARGET_COL].mean():.1%}, "
         f"test: {test_raw[TARGET_COL].mean():.1%}")


def apply_feature_prep(df, split_name, fit_encoder=True, care_unit_dummies=None):
    """
    Apply identical feature preparation as phase0_1_feature_prep.py.
    Returns (prepared_df, care_unit_dummies)
    """
    log.info(f"  [{split_name}] Input shape: {df.shape}")
    y = df[TARGET_COL].copy()

    # Drop meta columns
    to_drop = [c for c in DROP_META if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)

    # Drop leakage
    to_drop = [c for c in DROP_LEAKAGE if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info(f"  [{split_name}] Dropped leakage: {to_drop}")

    # Drop zero variance
    to_drop = [c for c in DROP_ZERO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop free text
    to_drop = [c for c in DROP_FREE_TEXT if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop no-variance categorical
    to_drop = [c for c in DROP_NO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop duplicates
    to_drop = [c for c in DROP_DUPLICATES if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop reference (score columns)
    to_drop = [c for c in DROP_REFERENCE if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop vasoactive columns
    to_drop = [c for c in DROP_VASO if c in df.columns]
    df = df.drop(columns=to_drop)
    log.info(f"  [{split_name}] Dropped vaso: {len(to_drop)} columns")

    # Drop _count columns
    count_drop = [c for c in df.columns if c.endswith("_count")]
    df = df.drop(columns=count_drop)
    log.info(f"  [{split_name}] Dropped _count: {len(count_drop)} columns")

    # Drop _was_measured columns
    was_drop = [c for c in df.columns if c.endswith("_was_measured")]
    df = df.drop(columns=was_drop)
    log.info(f"  [{split_name}] Dropped _was_measured: {len(was_drop)} columns")

    # Encode gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP).fillna(0)

    # One-hot encode care_unit
    if CARE_UNIT_COL in df.columns:
        if fit_encoder:
            dummies = pd.get_dummies(df[CARE_UNIT_COL], prefix="care_unit",
                                     drop_first=False)
            care_unit_dummies = list(dummies.columns)
        else:
            dummies = pd.get_dummies(df[CARE_UNIT_COL], prefix="care_unit",
                                     drop_first=False)
            for col in care_unit_dummies:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[care_unit_dummies]
        df = df.drop(columns=[CARE_UNIT_COL])
        df = pd.concat([df, dummies], axis=1)
        log.info(f"  [{split_name}] care_unit one-hot: {care_unit_dummies}")

    # Drop target, handle remaining objects
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.info(f"  [{split_name}] Dropping remaining object cols: {obj_cols}")
        df = df.drop(columns=obj_cols)

    # Handle missing values
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        log.info(f"  [{split_name}] {n_missing} missing values — filling with 0")
        df = df.fillna(0)

    # Re-attach target
    df[TARGET_COL] = y.values
    log.info(f"  [{split_name}] Final shape: {df.shape} "
             f"(features: {df.shape[1]-1}, missing: {n_missing})")

    return df, care_unit_dummies


train_prep, care_unit_dummies = apply_feature_prep(
    train_raw.copy(), "A_train", fit_encoder=True
)
test_prep, _ = apply_feature_prep(
    test_raw.copy(), "A_test", fit_encoder=False,
    care_unit_dummies=care_unit_dummies
)

# Align column order: train drives order, test matches
col_order = list(train_prep.columns)
# Add any missing cols in test as 0
for col in col_order:
    if col not in test_prep.columns:
        test_prep[col] = 0
test_prep = test_prep[col_order]

# Column consistency check
train_cols = set(train_prep.columns)
test_cols  = set(test_prep.columns)
only_train = train_cols - test_cols
only_test  = test_cols - train_cols
if only_train or only_test:
    log.warning(f"Column mismatch — only_train: {only_train}, only_test: {only_test}")
else:
    log.info(f"Column consistency: PASS — {len(train_cols)} columns match exactly")

# Now align to Option B feature list for cross-cohort comparability
feature_list_path = MODEL_DATA_DIR / "feature_list.json"
if feature_list_path.exists():
    with open(feature_list_path) as f:
        raw_fl = json.load(f)
    expected_features = raw_fl["feature_cols"] if isinstance(raw_fl, dict) else raw_fl

    # Find features present in both
    common_features = [f for f in expected_features if f in train_prep.columns]
    extra_in_A      = [f for f in train_prep.columns
                       if f not in expected_features and f != TARGET_COL]
    missing_from_A  = [f for f in expected_features if f not in train_prep.columns]

    log.info(f"Feature alignment vs Option B:")
    log.info(f"  Common features:       {len(common_features)}")
    log.info(f"  Extra in Option A:     {len(extra_in_A)} {extra_in_A[:5] if extra_in_A else ''}")
    log.info(f"  Missing from Option A: {len(missing_from_A)} {missing_from_A[:5] if missing_from_A else ''}")

    # Keep common features + target (add missing as 0 for full alignment)
    for f in missing_from_A:
        train_prep[f] = 0
        test_prep[f]  = 0
        log.info(f"  Added missing feature as 0: {f}")

    feature_cols = expected_features
    train_prep = train_prep[feature_cols + [TARGET_COL]]
    test_prep  = test_prep[feature_cols + [TARGET_COL]]
    log.info(f"  Aligned to Option B feature set: {len(feature_cols)} features")
else:
    feature_cols = [c for c in train_prep.columns if c != TARGET_COL]
    log.info(f"  feature_list.json not found — using {len(feature_cols)} features")

# Save prepared datasets
train_prep.to_csv(SA_RESULTS / "A_train_model_ready.csv", index=False)
test_prep.to_csv(SA_RESULTS  / "A_test_model_ready.csv", index=False)
log.info(f"Prepared datasets saved to {SA_RESULTS}")

# Final shapes
y_train = train_prep[TARGET_COL].values
y_test  = test_prep[TARGET_COL].values
X_train = train_prep[feature_cols]
X_test  = test_prep[feature_cols]

log.info(f"Final — Train: {X_train.shape}, Test: {X_test.shape}")
log.info(f"Sepsis — Train: {y_train.sum()} ({y_train.mean():.1%}), "
         f"Test: {y_test.sum()} ({y_test.mean():.1%})")

# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def bootstrap_auroc(y_true, y_prob, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_iter):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def bootstrap_auprc(y_true, y_prob, n_iter=BOOTSTRAP_ITER, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_iter):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(average_precision_score(y_true[idx], y_prob[idx]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    return {"threshold": threshold, "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "sensitivity": sens, "specificity": spec,
            "PPV": ppv, "NPV": npv, "F1": f1}


# =============================================================================
# 7. TRAIN CATBOOST WITH FIXED TUNED HYPERPARAMETERS
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION 2 — Training CatBoost (fixed tuned params, Option A)")
log.info("=" * 60)
log.info("NOTE: Hyperparameters held FIXED from Phase 2A (Option B tuning).")
log.info("Only class_weights updated for Option A prevalence (1:13.4).")
log.info(f"Tuned params: {TUNED_PARAMS}")

model_A = CatBoostClassifier(**TUNED_PARAMS)
model_A.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
)

model_path = SA_MODELS / "catboost_A_tuned.cbm"
model_A.save_model(str(model_path))
log.info(f"Model saved: {model_path}")

# =============================================================================
# 8. EVALUATION
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION 3 — Evaluation")
log.info("=" * 60)

prob_A = model_A.predict_proba(X_test)[:, 1]

auroc_A = roc_auc_score(y_test, prob_A)
auprc_A = average_precision_score(y_test, prob_A)
brier_A = brier_score_loss(y_test, prob_A)
ci_lo_auroc, ci_hi_auroc = bootstrap_auroc(y_test, prob_A)
ci_lo_auprc, ci_hi_auprc = bootstrap_auprc(y_test, prob_A)

log.info(f"AUROC : {auroc_A:.4f} [{ci_lo_auroc:.4f}–{ci_hi_auroc:.4f}]")
log.info(f"AUPRC : {auprc_A:.4f} [{ci_lo_auprc:.4f}–{ci_hi_auprc:.4f}]")
log.info(f"Brier : {brier_A:.4f}")

# Metrics at all clinical thresholds
log.info("\nMetrics at clinical thresholds:")
threshold_rows = []
for label, pt in CLINICAL_THRESHOLDS.items():
    m = metrics_at_threshold(y_test, prob_A, pt)
    m["label"] = label
    m["auroc"] = auroc_A
    m["auprc"] = auprc_A
    threshold_rows.append(m)
    log.info(f"  {label} | thresh={pt:.3f} | Sens={m['sensitivity']:.3f} | "
             f"Spec={m['specificity']:.3f} | PPV={m['PPV']:.3f} | "
             f"NPV={m['NPV']:.3f} | F1={m['F1']:.3f} | "
             f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}")

threshold_df = pd.DataFrame(threshold_rows)
threshold_df.to_csv(SA_RESULTS / "phase5_threshold_table.csv", index=False)

# Get metrics at recommended threshold
m_recommended = metrics_at_threshold(y_test, prob_A, 0.365)

# =============================================================================
# 9. HEAD-TO-HEAD COMPARISON TABLE: Option A vs Option B
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION 4 — Option A vs Option B Comparison")
log.info("=" * 60)

comparison = {
    "metric":           ["AUROC", "AUPRC", "Sensitivity (0.365)",
                         "Specificity (0.365)", "PPV (0.365)", "NPV (0.365)",
                         "F1 (0.365)", "Brier Score",
                         "N Train", "N Test", "Sepsis Prevalence"],
    "option_B":         [OPTION_B_RESULTS["auroc"],
                         OPTION_B_RESULTS["auprc"],
                         OPTION_B_RESULTS["sensitivity"],
                         OPTION_B_RESULTS["specificity"],
                         OPTION_B_RESULTS["ppv"],
                         OPTION_B_RESULTS["npv"],
                         OPTION_B_RESULTS["f1"],
                         OPTION_B_RESULTS["brier"],
                         OPTION_B_RESULTS["n_train"],
                         OPTION_B_RESULTS["n_test"],
                         f"{OPTION_B_RESULTS['prevalence']:.1%}"],
    "option_A":         [auroc_A,
                         auprc_A,
                         m_recommended["sensitivity"],
                         m_recommended["specificity"],
                         m_recommended["PPV"],
                         m_recommended["NPV"],
                         m_recommended["F1"],
                         brier_A,
                         int(len(y_train)),
                         int(len(y_test)),
                         f"{y_test.mean():.1%}"],
}

comparison_df = pd.DataFrame(comparison)
comparison_path = SA_RESULTS / "phase5_comparison_A_vs_B.csv"
comparison_df.to_csv(comparison_path, index=False)

log.info("Option A vs Option B:")
log.info(f"  {'Metric':<28} {'Option B':>12} {'Option A':>12} {'Delta':>10}")
log.info(f"  {'-'*64}")
numeric_metrics = ["AUROC", "AUPRC", "Sensitivity (0.365)",
                   "Specificity (0.365)", "PPV (0.365)", "NPV (0.365)",
                   "F1 (0.365)", "Brier Score"]
for _, row in comparison_df.iterrows():
    if row["metric"] in numeric_metrics:
        try:
            b_val = float(row["option_B"])
            a_val = float(row["option_A"])
            delta = a_val - b_val
            log.info(f"  {row['metric']:<28} {b_val:>12.4f} {a_val:>12.4f} {delta:>+10.4f}")
        except (ValueError, TypeError):
            log.info(f"  {row['metric']:<28} {str(row['option_B']):>12} "
                     f"{str(row['option_A']):>12}")
    else:
        log.info(f"  {row['metric']:<28} {str(row['option_B']):>12} "
                 f"{str(row['option_A']):>12}")

# =============================================================================
# 10. SHAP — TOP 20 FEATURES (no interaction matrix)
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("SECTION 5 — SHAP Feature Importance (Option A)")
log.info("=" * 60)

explainer_A   = shap.TreeExplainer(model_A)
shap_values_A = explainer_A.shap_values(X_test)

if isinstance(shap_values_A, list):
    shap_values_A = shap_values_A[1]

mean_abs_shap_A = np.abs(shap_values_A).mean(axis=0)
importance_A = pd.DataFrame({
    "feature":       feature_cols,
    "mean_abs_shap": mean_abs_shap_A
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance_A["rank"] = importance_A.index + 1

importance_A.to_csv(SA_RESULTS / "phase5_shap_importance_A.csv", index=False)

log.info("Top 20 features by mean |SHAP| — Option A:")
for _, row in importance_A.head(20).iterrows():
    log.info(f"  {row['rank']:>2}. {row['feature']:<35} {row['mean_abs_shap']:.5f}")

# =============================================================================
# 11. FIGURES
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Generating figures...")
log.info("=" * 60)

# ------------------------------------------------------------------
# FIGURE 1 — ROC + PR curves (Option A)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Phase 5 — Sensitivity Analysis: Option A (Full Cohort)\n"
             "ROC and PR Curves | Tuned CatBoost (fixed params) | Test n=2,490",
             fontsize=11, fontweight="bold", y=1.02)

fpr_A, tpr_A, _ = roc_curve(y_test, prob_A)
axes[0].plot(fpr_A, tpr_A, color="#1a56db", lw=2.5,
             label=f"Option A (AUC={auroc_A:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
axes[0].set_xlabel("FPR", fontsize=10)
axes[0].set_ylabel("TPR", fontsize=10)
axes[0].set_title("ROC Curve", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

prec_A, rec_A, _ = precision_recall_curve(y_test, prob_A)
axes[1].plot(rec_A, prec_A, color="#f97316", lw=2.5,
             label=f"Option A (AP={auprc_A:.4f})")
axes[1].axhline(y=y_test.mean(), color="k", ls="--", lw=0.8, alpha=0.5,
                label=f"Prevalence ({y_test.mean():.3f})")
axes[1].set_xlabel("Recall", fontsize=10)
axes[1].set_ylabel("Precision", fontsize=10)
axes[1].set_title("Precision-Recall Curve", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(SA_FIGS / "phase5_fig1_roc_pr_A.png", dpi=180, bbox_inches="tight")
plt.close()
log.info("Figure 1 saved: phase5_fig1_roc_pr_A.png")

# ------------------------------------------------------------------
# FIGURE 2 — Side-by-side ROC comparison: Option A vs Option B
# ------------------------------------------------------------------
# Load Option B probabilities for overlay
B_test_path = MODEL_DATA_DIR / "B_test_model_ready.csv"
B_model_path = BASE_DIR / "sepsis_ml" / "models" / "run_tuned" / "catboost_tuned.cbm"

if B_test_path.exists() and B_model_path.exists():
    df_B_test = pd.read_csv(B_test_path)
    y_test_B  = df_B_test[TARGET_COL].values
    X_test_B  = df_B_test[[c for c in feature_cols if c in df_B_test.columns]]

    model_B = CatBoostClassifier()
    model_B.load_model(str(B_model_path))
    prob_B  = model_B.predict_proba(X_test_B)[:, 1]

    auroc_B = roc_auc_score(y_test_B, prob_B)
    auprc_B = average_precision_score(y_test_B, prob_B)
    fpr_B, tpr_B, _ = roc_curve(y_test_B, prob_B)
    prec_B, rec_B, _ = precision_recall_curve(y_test_B, prob_B)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Phase 5 — Sensitivity Analysis: Option A vs Option B\n"
                 "Tuned CatBoost | Same hyperparameters, different cohort definitions",
                 fontsize=11, fontweight="bold", y=1.02)

    # ROC overlay
    axes[0].plot(fpr_B, tpr_B, color="#1a56db", lw=2.5,
                 label=f"Option B — Infection-only (AUC={auroc_B:.4f})")
    axes[0].plot(fpr_A, tpr_A, color="#f97316", lw=2.0, ls="--",
                 label=f"Option A — Full cohort (AUC={auroc_A:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    axes[0].set_xlabel("FPR", fontsize=10)
    axes[0].set_ylabel("TPR", fontsize=10)
    axes[0].set_title("ROC Curve Comparison", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # PR overlay
    axes[1].plot(rec_B, prec_B, color="#1a56db", lw=2.5,
                 label=f"Option B (AP={auprc_B:.4f}, prev={y_test_B.mean():.3f})")
    axes[1].plot(rec_A, prec_A, color="#f97316", lw=2.0, ls="--",
                 label=f"Option A (AP={auprc_A:.4f}, prev={y_test.mean():.3f})")
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].set_title("PR Curve Comparison", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(SA_FIGS / "phase5_fig2_A_vs_B_comparison.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    log.info("Figure 2 saved: phase5_fig2_A_vs_B_comparison.png")

# ------------------------------------------------------------------
# FIGURE 3 — SHAP bar comparison: Option A vs Option B top 20
# ------------------------------------------------------------------
# Load Option B SHAP importance
shap_B_path = BASE_DIR / "sepsis_ml" / "results" / "phase4_feature_importance.csv"
if shap_B_path.exists():
    importance_B = pd.read_csv(shap_B_path)

    top20_A = importance_A.head(20)[["feature", "mean_abs_shap"]].copy()
    top20_A.columns = ["feature", "shap_A"]
    top20_B = importance_B.head(20)[["feature", "mean_abs_shap"]].copy()
    top20_B.columns = ["feature", "shap_B"]

    # Merge on feature — show all features that appear in either top 20
    merged = pd.merge(top20_A, top20_B, on="feature", how="outer").fillna(0)
    merged["max_shap"] = merged[["shap_A", "shap_B"]].max(axis=1)
    merged = merged.sort_values("max_shap", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.suptitle("SHAP Feature Importance: Option A vs Option B\n"
                 "Mean |SHAP| — Top 20 Features",
                 fontsize=12, fontweight="bold")

    y_pos = np.arange(len(merged))
    width = 0.38
    ax.barh(y_pos + width / 2, merged["shap_B"][::-1].values,
            width, label="Option B (infection-only)", color="#1a56db", alpha=0.8)
    ax.barh(y_pos - width / 2, merged["shap_A"][::-1].values,
            width, label="Option A (full cohort)", color="#f97316", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged["feature"][::-1].values, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("", fontsize=10)
    plt.tight_layout()
    plt.savefig(SA_FIGS / "phase5_fig3_shap_A_vs_B.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    log.info("Figure 3 saved: phase5_fig3_shap_A_vs_B.png")

# ------------------------------------------------------------------
# FIGURE 4 — Summary comparison dashboard
# ------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
fig.suptitle("Phase 5 — Sensitivity Analysis Summary\n"
             "Option A (Full Cohort) vs Option B (Infection-Only) | Tuned CatBoost",
             fontsize=13, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Panel A: AUROC comparison bar
ax1 = fig.add_subplot(gs[0, 0])
models_label = ["Option B\n(infection-only)\nn=633", "Option A\n(full cohort)\nn=2,490"]
aurocs       = [OPTION_B_RESULTS["auroc"], auroc_A]
colors_bar   = ["#1a56db", "#f97316"]
bars         = ax1.bar(models_label, aurocs, color=colors_bar, alpha=0.85,
                        edgecolor="white", width=0.5)
for bar, val in zip(bars, aurocs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylim([0.85, 1.01])
ax1.set_ylabel("AUROC", fontsize=10)
ax1.set_title("AUROC Comparison", fontsize=11, fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

# Panel B: AUPRC comparison bar
ax2 = fig.add_subplot(gs[0, 1])
auprcs = [OPTION_B_RESULTS["auprc"], auprc_A]
bars   = ax2.bar(models_label, auprcs, color=colors_bar, alpha=0.85,
                  edgecolor="white", width=0.5)
for bar, val in zip(bars, auprcs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax2.set_ylim([0.5, 1.05])
ax2.set_ylabel("AUPRC", fontsize=10)
ax2.set_title("AUPRC Comparison", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# Panel C: Sens/Spec comparison
ax3 = fig.add_subplot(gs[1, 0])
metrics_names = ["Sensitivity", "Specificity", "PPV", "NPV", "F1"]
vals_B = [OPTION_B_RESULTS["sensitivity"], OPTION_B_RESULTS["specificity"],
          OPTION_B_RESULTS["ppv"], OPTION_B_RESULTS["npv"], OPTION_B_RESULTS["f1"]]
vals_A = [m_recommended["sensitivity"], m_recommended["specificity"],
          m_recommended["PPV"], m_recommended["NPV"], m_recommended["F1"]]
x = np.arange(len(metrics_names))
w = 0.35
ax3.bar(x - w/2, vals_B, w, label="Option B", color="#1a56db", alpha=0.8)
ax3.bar(x + w/2, vals_A, w, label="Option A", color="#f97316", alpha=0.8)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics_names, fontsize=9)
ax3.set_ylim([0, 1.12])
ax3.set_ylabel("Score", fontsize=10)
ax3.set_title(f"Metrics at Threshold=0.365", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
for bar in ax3.patches:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)

# Panel D: ROC overlay
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(fpr_A, tpr_A, color="#f97316", lw=2.0,
         label=f"Option A (AUC={auroc_A:.4f})")
if B_test_path.exists() and B_model_path.exists():
    ax4.plot(fpr_B, tpr_B, color="#1a56db", lw=2.0, ls="--",
             label=f"Option B (AUC={auroc_B:.4f})")
ax4.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
ax4.set_xlabel("FPR", fontsize=9)
ax4.set_ylabel("TPR", fontsize=9)
ax4.set_title("ROC Overlay", fontsize=11, fontweight="bold")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.savefig(SA_FIGS / "phase5_fig4_summary_dashboard.png",
            dpi=180, bbox_inches="tight")
plt.close()
log.info("Figure 4 saved: phase5_fig4_summary_dashboard.png")

# =============================================================================
# 12. SAVE RESULTS
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Saving results...")
log.info("=" * 60)

results = {
    "phase": "Phase5_SensitivityAnalysis",
    "timestamp": datetime.now().isoformat(),
    "dataset": "Option_A_full_cohort",
    "model": "CatBoost_fixed_tuned_params_from_PhaseB",
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "n_sepsis_train": int(y_train.sum()),
    "n_sepsis_test": int(y_test.sum()),
    "prevalence_train": float(y_train.mean()),
    "prevalence_test": float(y_test.mean()),
    "auroc": float(auroc_A),
    "auroc_ci": [float(ci_lo_auroc), float(ci_hi_auroc)],
    "auprc": float(auprc_A),
    "auprc_ci": [float(ci_lo_auprc), float(ci_hi_auprc)],
    "brier": float(brier_A),
    "metrics_at_0365": {
        k: float(v) if isinstance(v, (np.floating, float)) else int(v)
        for k, v in m_recommended.items()
    },
    "option_B_reference": OPTION_B_RESULTS,
    "delta_auroc_A_minus_B": float(auroc_A - OPTION_B_RESULTS["auroc"]),
    "delta_auprc_A_minus_B": float(auprc_A - OPTION_B_RESULTS["auprc"]),
    "top_10_features_shap": importance_A.head(10)[["feature", "mean_abs_shap"]].to_dict("records"),
    "hyperparameters_used": TUNED_PARAMS,
    "note": "Hyperparameters fixed from Phase 2A Option B tuning. "
            "Only class_weights updated for Option A prevalence (1:13.4). "
            "This isolates the effect of cohort definition on performance."
}

with open(SA_RESULTS / "phase5_results.json", "w") as f:
    json.dump(results, f, indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer)
                               else float(o) if isinstance(o, np.floating) else str(o))
log.info(f"Results saved: {SA_RESULTS / 'phase5_results.json'}")

# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================
log.info("")
log.info("=" * 70)
log.info("PHASE 5 COMPLETE — SENSITIVITY ANALYSIS SUMMARY")
log.info("=" * 70)
log.info(f"{'Metric':<30} {'Option B':>12} {'Option A':>12} {'Delta':>10}")
log.info(f"{'-'*66}")
log.info(f"{'AUROC':<30} {OPTION_B_RESULTS['auroc']:>12.4f} {auroc_A:>12.4f} "
         f"{auroc_A - OPTION_B_RESULTS['auroc']:>+10.4f}")
log.info(f"{'AUPRC':<30} {OPTION_B_RESULTS['auprc']:>12.4f} {auprc_A:>12.4f} "
         f"{auprc_A - OPTION_B_RESULTS['auprc']:>+10.4f}")
log.info(f"{'Sensitivity (0.365)':<30} {OPTION_B_RESULTS['sensitivity']:>12.3f} "
         f"{m_recommended['sensitivity']:>12.3f} "
         f"{m_recommended['sensitivity'] - OPTION_B_RESULTS['sensitivity']:>+10.3f}")
log.info(f"{'Specificity (0.365)':<30} {OPTION_B_RESULTS['specificity']:>12.3f} "
         f"{m_recommended['specificity']:>12.3f} "
         f"{m_recommended['specificity'] - OPTION_B_RESULTS['specificity']:>+10.3f}")
log.info(f"{'Brier Score':<30} {OPTION_B_RESULTS['brier']:>12.4f} {brier_A:>12.4f} "
         f"{brier_A - OPTION_B_RESULTS['brier']:>+10.4f}")
log.info(f"{'N Train':<30} {OPTION_B_RESULTS['n_train']:>12} {len(y_train):>12}")
log.info(f"{'N Test':<30} {OPTION_B_RESULTS['n_test']:>12} {len(y_test):>12}")
log.info(f"{'Prevalence':<30} {OPTION_B_RESULTS['prevalence']:>12.1%} "
         f"{y_test.mean():>12.1%}")
log.info("")
log.info(f"All outputs saved to: {SA_DIR}")
log.info(f"  Models   : {SA_MODELS}")
log.info(f"  Figures  : {SA_FIGS}")
log.info(f"  Results  : {SA_RESULTS}")
log.info(f"  Logs     : {SA_LOGS}")
log.info("")
log.info("Interpretation guidance:")
log.info("  AUROC within 0.02 of Option B → results generalise ✓")
log.info("  AUPRC drop expected (6.9% vs 27.3% prevalence) → use AUROC for comparison")
log.info("  Top SHAP features should remain lactate, D-dimer, platelets → validates signal")
log.info("")
log.info("Next steps: Phase 6 — TabNet + improved ANN")
log.info("=" * 70)
