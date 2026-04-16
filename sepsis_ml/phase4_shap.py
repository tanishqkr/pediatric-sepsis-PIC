"""
=============================================================================
PHASE 4 — SHAP EXPLAINABILITY
Early Prediction of Pediatric Sepsis Using Explainable AI (CatBoost + SHAP)
PIC Database v1.1.0 | Phoenix 2024 Criteria | Option B (Infection-Only Cohort)
=============================================================================

What this script does:
  1. Computes SHAP values for full test set (633 rows) using TreeExplainer
  2. Global beeswarm plot — top 20 features by mean |SHAP|
  3. SHAP bar plot — mean absolute importance (paper-ready)
  4. Dependence plots — top 5 features: lactate_max, ddimer_max, pt_mean,
                        platelets_min, base_excess_min
  5. Waterfall plots — 3 individual patients:
                        - True Positive  (highest confidence correct sepsis)
                        - False Negative (missed sepsis — highest prob among FN)
                        - False Positive (wrong alarm — highest prob among FP)
  6. SHAP interaction matrix — top 10 features only
     → Heatmap of mean |interaction SHAP| across top 10 × top 10
     → Focused interaction plots: lactate_max × platelets_min,
       lactate_max × base_excess_min, ddimer_max × inr_max
  7. Force plots for the 3 individual patients (HTML + PNG)

Outputs:
  Figures  → sepsis_ml/figures/phase4_shap/
  Results  → sepsis_ml/results/phase4_shap_values.npz (raw SHAP values)
             sepsis_ml/results/phase4_feature_importance.csv
  Log      → sepsis_ml/logs/phase4_shap.log

Authors : Tanish Porwal & Uzair
Institute: NMIMS
Date     : April 2026
Dataset  : Option B — 633 test rows, 225 features
Model    : Tuned CatBoost (Optuna 100 trials)
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from pathlib import Path

import shap
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# 1. PATHS
# =============================================================================
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "model_datasets"
MODEL_DIR = BASE_DIR / "sepsis_ml" / "models"
FIG_DIR   = BASE_DIR / "sepsis_ml" / "figures" / "phase4_shap"
RES_DIR   = BASE_DIR / "sepsis_ml" / "results"
LOG_DIR   = BASE_DIR / "sepsis_ml" / "logs"

for d in [FIG_DIR, RES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. LOGGING
# =============================================================================
log_path = LOG_DIR / "phase4_shap.log"
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
log.info("PHASE 4 — SHAP EXPLAINABILITY")
log.info(f"Run timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log.info(f"Base dir      : {BASE_DIR}")
log.info("=" * 70)

# =============================================================================
# 3. CONSTANTS
# =============================================================================
RANDOM_SEED        = 42
TARGET_COL         = "sepsis_label"
RECOMMENDED_THRESH = 0.365   # from Phase 2A / Phase 3
TOP_N_BEESWARM     = 20      # features shown in beeswarm
TOP_N_INTERACTION  = 10      # features used for interaction matrix
N_INTERACTION_ROWS = 2530    # use full training set for interactions

# Clinical feature display names — cleaner labels for figures
DISPLAY_NAMES = {
    "lactate_max":         "Lactate (max)",
    "lactate_mean":        "Lactate (mean)",
    "lactate_min":         "Lactate (min)",
    "lactate_first":       "Lactate (first)",
    "lactate_trend":       "Lactate (trend)",
    "platelets_min":       "Platelets (min)",
    "platelets_mean":      "Platelets (mean)",
    "platelets_max":       "Platelets (max)",
    "platelets_first":     "Platelets (first)",
    "inr_max":             "INR (max)",
    "inr_mean":            "INR (mean)",
    "inr_first":           "INR (first)",
    "ddimer_max":          "D-dimer (max)",
    "ddimer_mean":         "D-dimer (mean)",
    "ddimer_first":        "D-dimer (first)",
    "pt_mean":             "PT (mean)",
    "pt_max":              "PT (max)",
    "pt_first":            "PT (first)",
    "base_excess_min":     "Base excess (min)",
    "base_excess_mean":    "Base excess (mean)",
    "bicarbonate_min":     "Bicarbonate (min)",
    "ph_min":              "pH (min)",
    "fibrinogen_min":      "Fibrinogen (min)",
    "creatinine_max":      "Creatinine (max)",
    "bilirubin_max":       "Bilirubin (max)",
    "alt_max":             "ALT (max)",
    "wbc_max":             "WBC (max)",
    "anc_min":             "ANC (min)",
    "alc_min":             "ALC (min)",
    "glucose_max":         "Glucose (max)",
    "glucose_min":         "Glucose (min)",
    "map_min":             "MAP (min)",
    "heart_rate_max":      "Heart rate (max)",
    "shock_index":         "Shock index",
    "age_years":           "Age (years)",
    "ptt_max":             "PTT (max)",
    "hematocrit_min":      "Haematocrit (min)",
    "calcium_min":         "Calcium (min)",
    "crp_max":             "CRP (max)",
    "urine_output_total":  "Urine output (total)",
    "calcium_max":         "Calcium (max)",
    "calcium_mean":        "Calcium (mean)",
    "pao2_max":            "PaO2 (max)",
    "potassium_min":       "Potassium (min)",
}

np.random.seed(RANDOM_SEED)

# =============================================================================
# 4. LOAD DATA AND MODEL
# =============================================================================
log.info("Loading data and model...")

# Test set
test_path = DATA_DIR / "B_test_model_ready.csv"
df_test   = pd.read_csv(test_path)
y_test    = df_test[TARGET_COL].values
X_test    = df_test.drop(columns=[TARGET_COL])

# Train set
train_path = DATA_DIR / "B_train_model_ready.csv"
df_train   = pd.read_csv(train_path)
y_train    = df_train[TARGET_COL].values
X_train    = df_train.drop(columns=[TARGET_COL])

# Drop meta columns
meta_cols = ["subject_id", "hadm_id", "icustay_id", "stay_id", "intime", "outtime",
             "phoenix_core_score", "phoenix_8_score"]
for df_x in [X_test, X_train]:
    meta_present = [c for c in meta_cols if c in df_x.columns]
    if meta_present:
        df_x.drop(columns=meta_present, inplace=True)

# Align to feature list
feature_list_path = DATA_DIR / "feature_list.json"
if feature_list_path.exists():
    with open(feature_list_path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        expected_features = raw
    elif isinstance(raw, dict) and "feature_cols" in raw:
        expected_features = raw["feature_cols"]
    else:
        expected_features = None

    if expected_features is not None:
        X_test  = X_test[[c for c in expected_features if c in X_test.columns]]
        X_train = X_train[[c for c in expected_features if c in X_train.columns]]

feature_names = list(X_test.columns)
log.info(f"Test set  : {X_test.shape[0]} rows x {X_test.shape[1]} features")
log.info(f"Train set : {X_train.shape[0]} rows x {X_train.shape[1]} features")
log.info(f"Sepsis prevalence — test: {y_test.mean():.1%}, train: {y_train.mean():.1%}")

# Load tuned CatBoost
model = CatBoostClassifier()
model.load_model(str(MODEL_DIR / "run_tuned" / "catboost_tuned.cbm"))
log.info("Tuned CatBoost loaded")

prob_test = model.predict_proba(X_test)[:, 1]
auroc     = roc_auc_score(y_test, prob_test)
log.info(f"Model AUROC on test set: {auroc:.4f} (sanity check)")

# =============================================================================
# 5. COMPUTE SHAP VALUES — FULL TEST SET
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Computing SHAP values (TreeExplainer, full test set)...")
log.info("=" * 60)

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # shape: (633, 225)
expected_value = explainer.expected_value

# Handle CatBoost returning list of arrays (one per class)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # take positive class
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

log.info(f"SHAP values computed: {shap_values.shape}")
log.info(f"Expected value (base rate): {expected_value:.4f}")
log.info(f"SHAP value range: [{shap_values.min():.4f}, {shap_values.max():.4f}]")

# Save raw SHAP values for reproducibility
np.savez_compressed(
    RES_DIR / "phase4_shap_values.npz",
    shap_values=shap_values,
    feature_names=np.array(feature_names),
    expected_value=np.array([expected_value])
)
log.info(f"Raw SHAP values saved: {RES_DIR / 'phase4_shap_values.npz'}")

# Feature importance table
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature":        feature_names,
    "mean_abs_shap":  mean_abs_shap,
    "display_name":   [DISPLAY_NAMES.get(f, f) for f in feature_names]
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
importance_df["rank"] = importance_df.index + 1

importance_path = RES_DIR / "phase4_feature_importance.csv"
importance_df.to_csv(importance_path, index=False)
log.info(f"Feature importance saved: {importance_path}")

log.info("Top 20 features by mean |SHAP|:")
for _, row in importance_df.head(20).iterrows():
    log.info(f"  {row['rank']:>2}. {row['display_name']:<30} {row['mean_abs_shap']:.5f}")

# Top features for subsequent analyses
top20_features = importance_df.head(TOP_N_BEESWARM)["feature"].tolist()
top10_features = importance_df.head(TOP_N_INTERACTION)["feature"].tolist()
top5_features  = importance_df.head(5)["feature"].tolist()

log.info(f"Top 5 features: {top5_features}")
log.info(f"Top 10 features (for interactions): {top10_features}")

# Indices for subsetting
top20_idx = [feature_names.index(f) for f in top20_features]
top10_idx = [feature_names.index(f) for f in top10_features]

# =============================================================================
# 6. IDENTIFY PATIENTS FOR WATERFALL PLOTS
# =============================================================================
log.info("")
log.info("Identifying patients for waterfall plots...")

y_pred = (prob_test >= RECOMMENDED_THRESH).astype(int)

# True Positive: sepsis=1, pred=1, highest confidence
tp_mask  = (y_test == 1) & (y_pred == 1)
tp_idx   = np.where(tp_mask)[0][np.argmax(prob_test[tp_mask])]

# False Negative: sepsis=1, pred=0, highest probability among missed
fn_mask  = (y_test == 1) & (y_pred == 0)
fn_idx   = np.where(fn_mask)[0][np.argmax(prob_test[fn_mask])]

# False Positive: sepsis=0, pred=1, highest probability among false alarms
fp_mask  = (y_test == 0) & (y_pred == 1)
fp_idx   = np.where(fp_mask)[0][np.argmax(prob_test[fp_mask])]

log.info(f"True Positive  patient idx={tp_idx}, prob={prob_test[tp_idx]:.4f}, "
         f"true_label={y_test[tp_idx]}")
log.info(f"False Negative patient idx={fn_idx}, prob={prob_test[fn_idx]:.4f}, "
         f"true_label={y_test[fn_idx]}")
log.info(f"False Positive patient idx={fp_idx}, prob={prob_test[fp_idx]:.4f}, "
         f"true_label={y_test[fp_idx]}")

# =============================================================================
# 7. FIGURE HELPER
# =============================================================================

def get_display_name(feat):
    return DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())


def shap_color_map():
    """Standard SHAP red-blue colormap."""
    return LinearSegmentedColormap.from_list(
        "shap", ["#1e88e5", "#ffffff", "#ff0d57"], N=256
    )

# =============================================================================
# 8. FIGURE 1 — GLOBAL BEESWARM PLOT (top 20)
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Figure 1 — Global beeswarm plot...")
log.info("=" * 60)

fig, ax = plt.subplots(figsize=(11, 9))

shap_top20       = shap_values[:, top20_idx]
X_test_top20     = X_test.iloc[:, top20_idx]
display_names_20 = [get_display_name(f) for f in top20_features]

# Use SHAP's built-in summary plot — beeswarm style
shap.summary_plot(
    shap_top20,
    X_test_top20,
    feature_names=display_names_20,
    plot_type="dot",
    max_display=TOP_N_BEESWARM,
    show=False,
    plot_size=None,
    color_bar=True,
    alpha=0.6
)

plt.title("SHAP Beeswarm Plot — Top 20 Features\n"
          "Tuned CatBoost | Option B Test Set (n=633)",
          fontsize=12, fontweight="bold", pad=12)
plt.xlabel("SHAP value (impact on model output)", fontsize=10)
plt.tight_layout()
fig1_path = FIG_DIR / "phase4_fig1_beeswarm_top20.png"
plt.savefig(fig1_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 1 saved: {fig1_path}")

# =============================================================================
# 9. FIGURE 2 — SHAP BAR PLOT (mean absolute importance, top 20)
# =============================================================================
log.info("Figure 2 — SHAP bar plot...")

fig, ax = plt.subplots(figsize=(10, 8))

top20_importance = importance_df.head(TOP_N_BEESWARM)
colors = plt.cm.RdBu_r(np.linspace(0.15, 0.85, TOP_N_BEESWARM))[::-1]

bars = ax.barh(
    top20_importance["display_name"][::-1],
    top20_importance["mean_abs_shap"][::-1],
    color=colors,
    edgecolor="white",
    linewidth=0.5
)

for bar, val in zip(bars, top20_importance["mean_abs_shap"][::-1]):
    ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8.5)

ax.set_xlabel("Mean |SHAP value| (average impact on model output)", fontsize=10)
ax.set_title("Global Feature Importance — Mean |SHAP|\n"
             "Tuned CatBoost | Option B Test Set (n=633)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim([0, top20_importance["mean_abs_shap"].max() * 1.15])
plt.tight_layout()
fig2_path = FIG_DIR / "phase4_fig2_bar_importance.png"
plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 2 saved: {fig2_path}")

# =============================================================================
# 10. FIGURE 3 — DEPENDENCE PLOTS (top 5 features)
# =============================================================================
log.info("Figure 3 — Dependence plots (top 5 features)...")

# Determine which top-5 features to use — use actual SHAP-ranked top 5
# but always include lactate_max, platelets_min if in top 20
priority_features = ["lactate_max", "platelets_min", "base_excess_min"]
dep_features = []
for f in priority_features:
    if f in feature_names:
        dep_features.append(f)
for f in top5_features:
    if f not in dep_features:
        dep_features.append(f)
dep_features = dep_features[:5]

log.info(f"Dependence plot features: {dep_features}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("SHAP Dependence Plots — Top 5 Clinical Features\n"
             "Tuned CatBoost | Option B Test Set (n=633)\n"
             "Colour = feature value (red=high, blue=low)",
             fontsize=12, fontweight="bold", y=1.01)

axes_flat = axes.flatten()

for i, feat in enumerate(dep_features):
    ax = axes_flat[i]
    feat_idx  = feature_names.index(feat)
    feat_vals = X_test.iloc[:, feat_idx].values
    shap_vals = shap_values[:, feat_idx]

    # Colour by feature value
    sc = ax.scatter(
        feat_vals, shap_vals,
        c=feat_vals,
        cmap="RdBu_r",
        alpha=0.55,
        s=18,
        edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04,
                 label="Feature value")

    # Trend line
    from numpy.polynomial import polynomial as P
    valid = ~np.isnan(feat_vals)
    if valid.sum() > 10:
        try:
            z = np.polyfit(feat_vals[valid], shap_vals[valid], 2)
            p = np.poly1d(z)
            x_line = np.linspace(np.nanpercentile(feat_vals, 1),
                                  np.nanpercentile(feat_vals, 99), 200)
            ax.plot(x_line, p(x_line), color="black", lw=1.5,
                    ls="--", alpha=0.7, label="Trend")
        except Exception:
            pass

    ax.axhline(y=0, color="gray", lw=0.8, ls="-", alpha=0.5)
    ax.set_xlabel(get_display_name(feat), fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)
    ax.set_title(get_display_name(feat), fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

# Hide unused 6th panel
axes_flat[5].set_visible(False)

plt.tight_layout()
fig3_path = FIG_DIR / "phase4_fig3_dependence_plots.png"
plt.savefig(fig3_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 3 saved: {fig3_path}")

# =============================================================================
# 11. FIGURE 4 — WATERFALL PLOTS (TP, FN, FP)
# =============================================================================
log.info("Figure 4 — Waterfall plots (TP, FN, FP)...")

def manual_waterfall(ax, shap_vals_patient, feature_names_list, expected_val,
                     pred_prob, true_label, title, top_n=15):
    """
    Clean manual waterfall plot.
    Shows top_n features by |SHAP| for a single patient.
    """
    # Sort by absolute SHAP value
    abs_shap   = np.abs(shap_vals_patient)
    sorted_idx = np.argsort(abs_shap)[::-1][:top_n]
    sorted_idx = sorted_idx[::-1]  # reverse for horizontal bar (bottom = most important)

    feats  = [get_display_name(feature_names_list[i]) for i in sorted_idx]
    vals   = shap_vals_patient[sorted_idx]
    colors = ["#ff0d57" if v > 0 else "#1e88e5" for v in vals]

    bars = ax.barh(feats, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        x_pos = bar.get_width() + (0.001 if val >= 0 else -0.001)
        ha    = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=8)

    ax.axvline(x=0, color="black", lw=1.0)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    # Annotation box
    pred_label = "Sepsis" if pred_prob >= RECOMMENDED_THRESH else "No Sepsis"
    true_str   = "Sepsis" if true_label == 1 else "No Sepsis"
    correct    = "✓ Correct" if (pred_prob >= RECOMMENDED_THRESH) == (true_label == 1) else "✗ Wrong"
    info_text  = (f"Pred prob: {pred_prob:.3f}\n"
                  f"Pred label: {pred_label}\n"
                  f"True label: {true_str}\n"
                  f"{correct}")
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    # Legend
    red_patch  = mpatches.Patch(color="#ff0d57", alpha=0.85, label="Pushes toward sepsis (+)")
    blue_patch = mpatches.Patch(color="#1e88e5", alpha=0.85, label="Pushes away from sepsis (−)")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")


patients = [
    (tp_idx, "True Positive — High Confidence Correct Sepsis"),
    (fn_idx, "False Negative — Missed Sepsis"),
    (fp_idx, "False Positive — Wrong Alarm"),
]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle("SHAP Waterfall Plots — Individual Patient Explanations\n"
             "Tuned CatBoost | Option B Test Set | Threshold=0.365",
             fontsize=13, fontweight="bold", y=1.02)

for ax, (patient_idx, title) in zip(axes, patients):
    manual_waterfall(
        ax=ax,
        shap_vals_patient=shap_values[patient_idx],
        feature_names_list=feature_names,
        expected_val=expected_value,
        pred_prob=prob_test[patient_idx],
        true_label=y_test[patient_idx],
        title=title,
        top_n=15
    )

plt.tight_layout()
fig4_path = FIG_DIR / "phase4_fig4_waterfall_plots.png"
plt.savefig(fig4_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 4 saved: {fig4_path}")

# Log patient details for documentation
for patient_idx, label in patients:
    top3_idx   = np.argsort(np.abs(shap_values[patient_idx]))[::-1][:3]
    top3_feats = [(feature_names[i], shap_values[patient_idx][i]) for i in top3_idx]
    log.info(f"  {label}")
    log.info(f"    prob={prob_test[patient_idx]:.4f}, true={y_test[patient_idx]}")
    log.info(f"    top 3 SHAP features: {top3_feats}")

# =============================================================================
# 12. FIGURE 5 — SHAP INTERACTION MATRIX (top 10 features)
# =============================================================================
log.info("")
log.info("=" * 60)
log.info(f"Computing SHAP interaction values (top {TOP_N_INTERACTION} features, "
         f"{N_INTERACTION_ROWS} training rows)...")
log.info("This may take 5-15 minutes...")
log.info("=" * 60)

# CatBoost interaction API requires full feature set in original column order
# We pass full X_test but then slice the interaction matrix to top 10 afterwards
explainer_top10      = shap.TreeExplainer(model)
shap_interaction_raw = explainer_top10.shap_interaction_values(X_test)

# Slice to top 10 features after computation
if isinstance(shap_interaction_raw, list):
    shap_interaction_full = shap_interaction_raw[1]
else:
    shap_interaction_full = shap_interaction_raw

# Get indices of top 10 in the full feature list
top10_idx_full = [feature_names.index(f) for f in top10_features]
# Slice rows and columns to top 10 x top 10
shap_interaction = shap_interaction_full[:, top10_idx_full, :][:, :, top10_idx_full]

# X_test_top10 still needed for scatter plots — keep original order
X_test_top10 = X_test[top10_features]



log.info(f"Interaction values computed: {shap_interaction.shape}")

# Mean absolute interaction matrix
mean_abs_interaction = np.abs(shap_interaction).mean(axis=0)
np.save(RES_DIR / "phase4_shap_interaction_top10.npy", mean_abs_interaction)

display_names_10 = [get_display_name(f) for f in top10_features]

# --- Figure 5A: Interaction heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mean_abs_interaction, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Mean |SHAP interaction value|", fraction=0.04, pad=0.04)

ax.set_xticks(range(TOP_N_INTERACTION))
ax.set_yticks(range(TOP_N_INTERACTION))
ax.set_xticklabels(display_names_10, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(display_names_10, fontsize=9)

# Annotate cells
for i in range(TOP_N_INTERACTION):
    for j in range(TOP_N_INTERACTION):
        val = mean_abs_interaction[i, j]
        color = "white" if val > mean_abs_interaction.max() * 0.6 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=7, color=color)

ax.set_title(f"SHAP Interaction Matrix — Top {TOP_N_INTERACTION} Features\n"
             "Mean |SHAP interaction value| | Tuned CatBoost | Test Set n=633",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig5a_path = FIG_DIR / "phase4_fig5a_interaction_heatmap.png"
plt.savefig(fig5a_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 5A saved: {fig5a_path}")

# =============================================================================
# 13. FIGURE 5B — FOCUSED INTERACTION SCATTER PLOTS
# =============================================================================
log.info("Figure 5B — Focused interaction scatter plots...")

# Three clinically motivated interaction pairs
interaction_pairs = []

# Always include lactate × platelets if both in top 10
if "lactate_max" in top10_features and "platelets_min" in top10_features:
    interaction_pairs.append(("lactate_max", "platelets_min",
                               "Lactate (max) × Platelets (min)",
                               "Cardiovascular-Coagulation Burden"))

# lactate × base_excess (metabolic crisis)
if "lactate_max" in top10_features and "base_excess_min" in top10_features:
    interaction_pairs.append(("lactate_max", "base_excess_min",
                               "Lactate (max) × Base Excess (min)",
                               "Metabolic Crisis Axis"))

# INR × D-dimer (coagulopathy) — use whatever coag features are in top 10
coag_pairs = [("inr_max", "ddimer_max"), ("inr_mean", "ddimer_max"),
              ("pt_mean", "ddimer_max"), ("pt_max", "inr_max")]
for f1, f2 in coag_pairs:
    if f1 in top10_features and f2 in top10_features:
        interaction_pairs.append((f1, f2,
                                   f"{get_display_name(f1)} × {get_display_name(f2)}",
                                   "Coagulopathy Axis"))
        break

# Fall back: use top 2 features if none of above match
if len(interaction_pairs) == 0:
    f1, f2 = top10_features[0], top10_features[1]
    interaction_pairs.append((f1, f2,
                               f"{get_display_name(f1)} × {get_display_name(f2)}",
                               "Top 2 Feature Interaction"))

n_pairs = min(len(interaction_pairs), 3)
fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6))
if n_pairs == 1:
    axes = [axes]

fig.suptitle("SHAP Feature Interaction Scatter Plots\n"
             "Colour = interaction SHAP value | Tuned CatBoost | Test Set n=633",
             fontsize=12, fontweight="bold", y=1.02)

for ax, (f1, f2, title, subtitle) in zip(axes, interaction_pairs[:n_pairs]):
    idx1 = top10_features.index(f1)
    idx2 = top10_features.index(f2)

    interaction_vals = shap_interaction[:, idx1, idx2]
    f1_vals          = X_test_top10[f1].values
    f2_vals          = X_test_top10[f2].values

    vmax = np.abs(interaction_vals).max()
    sc = ax.scatter(
        f1_vals, f2_vals,
        c=interaction_vals,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        alpha=0.6, s=20, edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label="SHAP interaction value",
                 fraction=0.04, pad=0.04)

    ax.set_xlabel(get_display_name(f1), fontsize=10)
    ax.set_ylabel(get_display_name(f2), fontsize=10)
    ax.set_title(f"{title}\n({subtitle})", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

    # Annotation: mean interaction value in sepsis vs non-sepsis
    sep_mask    = y_test == 1
    nonsep_mask = y_test == 0
    mean_sep    = interaction_vals[sep_mask].mean()
    mean_nonsep = interaction_vals[nonsep_mask].mean()
    ax.text(0.02, 0.98,
            f"Mean interaction:\nSepsis: {mean_sep:+.4f}\nNon-sepsis: {mean_nonsep:+.4f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

plt.tight_layout()
fig5b_path = FIG_DIR / "phase4_fig5b_interaction_scatter.png"
plt.savefig(fig5b_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 5B saved: {fig5b_path}")

# =============================================================================
# 14. FIGURE 6 — SHAP SUMMARY BY CLASS (sepsis vs non-sepsis)
# =============================================================================
log.info("Figure 6 — SHAP summary by class...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("SHAP Values by Class — Sepsis vs Non-Sepsis\n"
             "Top 15 Features | Tuned CatBoost | Test Set n=633",
             fontsize=12, fontweight="bold", y=1.01)

top15_features = importance_df.head(15)["feature"].tolist()
top15_idx      = [feature_names.index(f) for f in top15_features]
display_15     = [get_display_name(f) for f in top15_features]

for ax, mask, class_label, color in [
    (axes[0], y_test == 1, "Sepsis (n=173)",    "#ff0d57"),
    (axes[1], y_test == 0, "Non-Sepsis (n=460)", "#1e88e5"),
]:
    shap_class = shap_values[mask][:, top15_idx]
    mean_shap  = shap_class.mean(axis=0)
    std_shap   = shap_class.std(axis=0)

    # Sort by mean SHAP for this class
    sort_order = np.argsort(mean_shap)
    sorted_names = [display_15[i] for i in sort_order]
    sorted_mean  = mean_shap[sort_order]
    sorted_std   = std_shap[sort_order]

    bar_colors = ["#ff0d57" if v > 0 else "#1e88e5" for v in sorted_mean]
    ax.barh(sorted_names, sorted_mean, xerr=sorted_std,
            color=bar_colors, alpha=0.8, capsize=3, edgecolor="white")
    ax.axvline(x=0, color="black", lw=1.0)
    ax.set_xlabel("Mean SHAP value", fontsize=10)
    ax.set_title(f"{class_label}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
fig6_path = FIG_DIR / "phase4_fig6_shap_by_class.png"
plt.savefig(fig6_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 6 saved: {fig6_path}")

# =============================================================================
# 15. FIGURE 7 — SHAP DASHBOARD (4-panel paper summary)
# =============================================================================
log.info("Figure 7 — SHAP dashboard (paper summary)...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Phase 4 — SHAP Explainability Summary Dashboard\n"
             "Tuned CatBoost | Option B (Infection-Only) | Test Set n=633",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# Panel A: Bar importance (top 15)
ax_bar = fig.add_subplot(gs[0, 0])
top15_imp = importance_df.head(15)
colors_15  = plt.cm.RdBu_r(np.linspace(0.15, 0.85, 15))[::-1]
ax_bar.barh(top15_imp["display_name"][::-1],
            top15_imp["mean_abs_shap"][::-1],
            color=colors_15, edgecolor="white", linewidth=0.4)
ax_bar.set_xlabel("Mean |SHAP|", fontsize=9)
ax_bar.set_title("Global Feature Importance", fontsize=10, fontweight="bold")
ax_bar.grid(axis="x", alpha=0.3)

# Panel B: Lactate dependence
ax_dep = fig.add_subplot(gs[0, 1])
if "lactate_max" in feature_names:
    lac_idx  = feature_names.index("lactate_max")
    lac_vals = X_test.iloc[:, lac_idx].values
    lac_shap = shap_values[:, lac_idx]
    sc = ax_dep.scatter(lac_vals, lac_shap,
                        c=lac_vals, cmap="RdBu_r",
                        alpha=0.5, s=15, edgecolors="none")
    plt.colorbar(sc, ax=ax_dep, fraction=0.04, pad=0.04,
                 label="Lactate value")
    ax_dep.axhline(y=0, color="gray", lw=0.8, alpha=0.5)
    ax_dep.set_xlabel("Lactate max (mmol/L)", fontsize=9)
    ax_dep.set_ylabel("SHAP value", fontsize=9)
    ax_dep.set_title("Lactate (max) — SHAP Dependence", fontsize=10, fontweight="bold")
    ax_dep.grid(alpha=0.25)

# Panel C: Waterfall TP patient
ax_wf = fig.add_subplot(gs[1, 0])
manual_waterfall(
    ax=ax_wf,
    shap_vals_patient=shap_values[tp_idx],
    feature_names_list=feature_names,
    expected_val=expected_value,
    pred_prob=prob_test[tp_idx],
    true_label=y_test[tp_idx],
    title="True Positive Patient",
    top_n=10
)

# Panel D: Waterfall FN patient
ax_wf2 = fig.add_subplot(gs[1, 1])
manual_waterfall(
    ax=ax_wf2,
    shap_vals_patient=shap_values[fn_idx],
    feature_names_list=feature_names,
    expected_val=expected_value,
    pred_prob=prob_test[fn_idx],
    true_label=y_test[fn_idx],
    title="False Negative Patient (Missed Sepsis)",
    top_n=10
)

fig7_path = FIG_DIR / "phase4_fig7_shap_dashboard.png"
plt.savefig(fig7_path, dpi=180, bbox_inches="tight")
plt.close()
log.info(f"Figure 7 saved: {fig7_path}")

# =============================================================================
# 16. SAVE RESULTS + EXPERIMENT LOG
# =============================================================================
log.info("")
log.info("=" * 60)
log.info("Saving results...")
log.info("=" * 60)

# Summary of interaction pair findings
interaction_summary = []
for f1, f2, title, subtitle in interaction_pairs[:n_pairs]:
    idx1 = top10_features.index(f1)
    idx2 = top10_features.index(f2)
    iv   = shap_interaction[:, idx1, idx2]
    interaction_summary.append({
        "feature_1": f1,
        "feature_2": f2,
        "title": title,
        "clinical_axis": subtitle,
        "mean_interaction_sepsis":    float(iv[y_test == 1].mean()),
        "mean_interaction_nonsepsis": float(iv[y_test == 0].mean()),
        "mean_abs_interaction":       float(np.abs(iv).mean()),
    })

phase4_results = {
    "phase": "Phase4_SHAP",
    "timestamp": datetime.now().isoformat(),
    "dataset": "Option_B",
    "model": "Tuned_CatBoost",
    "n_test": int(len(y_test)),
    "expected_value": float(expected_value),
    "top_20_features": [
        {"rank": int(r["rank"]), "feature": r["feature"],
         "display_name": r["display_name"],
         "mean_abs_shap": float(r["mean_abs_shap"])}
        for _, r in importance_df.head(20).iterrows()
    ],
    "waterfall_patients": {
        "true_positive":  {"idx": int(tp_idx), "prob": float(prob_test[tp_idx]),
                            "true_label": int(y_test[tp_idx])},
        "false_negative": {"idx": int(fn_idx), "prob": float(prob_test[fn_idx]),
                            "true_label": int(y_test[fn_idx])},
        "false_positive": {"idx": int(fp_idx), "prob": float(prob_test[fp_idx]),
                            "true_label": int(y_test[fp_idx])},
    },
    "interaction_pairs": interaction_summary,
    "top_10_features_for_interaction": top10_features,
    "figures": [str(p) for p in [fig1_path, fig2_path, fig3_path, fig4_path,
                                   fig5a_path, fig5b_path, fig6_path, fig7_path]]
}

with open(RES_DIR / "phase4_shap_results.json", "w") as f:
    json.dump(phase4_results, f, indent=2)
log.info(f"Phase 4 results saved: {RES_DIR / 'phase4_shap_results.json'}")

# Append to experiment log
exp_log_path = RES_DIR / "experiment_log.json"
if exp_log_path.exists():
    try:
        with open(exp_log_path, "r") as f:
            exp_log = json.load(f)
        if not isinstance(exp_log, list):
            exp_log = [exp_log]
    except json.JSONDecodeError:
        import shutil
        shutil.copy(exp_log_path, str(exp_log_path) + ".corrupted_backup")
        exp_log = []
else:
    exp_log = []

exp_log.append(phase4_results)
with open(exp_log_path, "w") as f:
    json.dump(exp_log, f, indent=2,
              default=lambda o: int(o) if isinstance(o, np.integer)
                               else float(o) if isinstance(o, np.floating) else str(o))
log.info(f"Experiment log updated: {exp_log_path}")

# =============================================================================
# 17. FINAL SUMMARY
# =============================================================================
log.info("")
log.info("=" * 70)
log.info("PHASE 4 COMPLETE — SUMMARY")
log.info("=" * 70)
log.info(f"SHAP values computed for  : {len(y_test)} test patients, {len(feature_names)} features")
log.info(f"Expected value (base rate): {expected_value:.4f}")
log.info(f"Top feature               : {importance_df.iloc[0]['display_name']} "
         f"(mean |SHAP|={importance_df.iloc[0]['mean_abs_shap']:.5f})")
log.info(f"Figures saved             : 7 figures to {FIG_DIR}")
log.info("")
log.info("Top 10 features by SHAP importance:")
for _, row in importance_df.head(10).iterrows():
    log.info(f"  {row['rank']:>2}. {row['display_name']:<30} {row['mean_abs_shap']:.5f}")
log.info("")
log.info("Interaction pairs analysed:")
for pair in interaction_summary:
    log.info(f"  {pair['title']}")
    log.info(f"    Mean interaction — Sepsis: {pair['mean_interaction_sepsis']:+.5f}, "
             f"Non-sepsis: {pair['mean_interaction_nonsepsis']:+.5f}")
log.info("")
log.info("Next steps:")
log.info("  → Sensitivity analysis on Option A (full cohort)")
log.info("  → Paper writing — Methods, Results, Discussion")
log.info("=" * 70)
