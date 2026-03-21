"""
phase0_2_leakage_investigation.py
----------------------------------
Systematic investigation of all data leakage suspects.
CatBoost AUROC=1.000 on test set is impossible in real clinical data.
This script finds the root cause before any further modelling.

Investigates 6 suspects:
  1. _count columns (measurement frequency — circular with Phoenix labeling)
  2. n_vasoactives_24h + vaso flags (part of Phoenix cardiovascular score)
  3. _trend features (trajectory may encode severity = outcome)
  4. fluid_input_total_ml / urine_output (resuscitation proxy)
  5. Train/test patient overlap (split at row vs patient level)
  6. Imputation leakage (medians from full dataset vs train-only)

Outputs:
  diagnostics/phase0_leakage/leakage_report.json
  diagnostics/phase0_leakage/feature_correlations.csv
  figures/phase0_leakage/  — isolation test plots
  logs/phase0_2_leakage_investigation.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase0_2_leakage_investigation.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output (paste here)
  - figures/phase0_leakage/isolation_test_aurocs.png
  - diagnostics/phase0_leakage/leakage_report.json
  - diagnostics/phase0_leakage/feature_correlations.csv
"""

import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FILES, DIAG_DIR, LOGS_DIR, TARGET_COL, RANDOM_SEED

# ── Output dirs ───────────────────────────────────────────────────────────────
DIAG_LEAKAGE = DIAG_DIR / "phase0_leakage"
FIG_LEAKAGE  = Path(__file__).resolve().parent.parent / "sepsis_ml" / "figures" / "phase0_leakage"
DIAG_LEAKAGE.mkdir(parents=True, exist_ok=True)
FIG_LEAKAGE.mkdir(parents=True, exist_ok=True)

REPO_ROOT      = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase0_2_leakage_investigation.log"
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


def quick_auroc(X, y, label=""):
    """Train a quick CatBoost (50 trees) on X, return 5-fold CV AUROC."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=50, learning_rate=0.1, depth=4,
        random_seed=RANDOM_SEED, verbose=0,
        class_weights=[1.0, 2.7],
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    mean_auroc = float(np.mean(scores))
    log.info(f"    {label}: CV AUROC = {mean_auroc:.4f} ± {np.std(scores):.4f}")
    return mean_auroc


def quick_auroc_train_test(X_train, y_train, X_test, y_test, label=""):
    """Train on full train, evaluate on test."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=100, learning_rate=0.1, depth=4,
        random_seed=RANDOM_SEED, verbose=0,
        class_weights=[1.0, 2.7],
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, y_prob))
    auprc = float(average_precision_score(y_test, y_prob))
    log.info(f"    {label}: Test AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    return auroc, auprc


# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("PHASE 0.2 — LEAKAGE INVESTIGATION")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("CatBoost AUROC=1.000 is impossible — finding the cause.")
    log.info("=" * 65)

    train = pd.read_csv(TRAIN_FILE)
    test  = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train.columns if c != TARGET_COL]
    X_train = train[feature_cols]
    y_train = train[TARGET_COL].values
    X_test  = test[feature_cols]
    y_test  = test[TARGET_COL].values

    log.info(f"\nLoaded: train={train.shape}, test={test.shape}")
    log.info(f"Features: {len(feature_cols)}")

    report   = {}
    iso_results = {}   # isolation test results for plotting

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 1 — _count columns
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 1: _count columns")
    log.info("Hypothesis: measurement frequency is circular with Phoenix labeling.")
    log.info(f"{'='*55}")

    count_cols = [c for c in feature_cols if c.endswith("_count")]
    log.info(f"  Found {len(count_cols)} _count columns")

    corr_counts = train[count_cols + [TARGET_COL]].corr()[TARGET_COL]\
                      .drop(TARGET_COL).abs().sort_values(ascending=False)
    log.info(f"  Top 10 _count correlations with label:")
    for col, r in corr_counts.head(10).items():
        log.info(f"    {col:<45} r={r:.4f}")

    log.info(f"\n  Isolation test — CatBoost trained on COUNT COLUMNS ONLY:")
    auroc_counts, auprc_counts = quick_auroc_train_test(
        X_train[count_cols].values, y_train,
        X_test[count_cols].values, y_test,
        "Count cols only"
    )
    iso_results["count_cols_only"] = auroc_counts

    report["suspect_1_count_cols"] = {
        "n_count_cols"   : len(count_cols),
        "top_correlation": float(corr_counts.iloc[0]),
        "top_col"        : corr_counts.index[0],
        "isolation_auroc": auroc_counts,
        "isolation_auprc": auprc_counts,
        "verdict"        : "LEAKAGE" if auroc_counts > 0.85 else "ACCEPTABLE",
    }
    log.info(f"  VERDICT: {'LEAKAGE CONFIRMED' if auroc_counts > 0.85 else 'ACCEPTABLE'} "
             f"(isolation AUROC={auroc_counts:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 2 — vasoactive columns
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 2: n_vasoactives_24h + individual vaso flags")
    log.info("Hypothesis: vasoactives are part of Phoenix cardiovascular score.")
    log.info(f"{'='*55}")

    vaso_cols = [c for c in feature_cols if "vaso" in c.lower() or "vasoactive" in c.lower()]
    log.info(f"  Found {len(vaso_cols)} vasoactive columns: {vaso_cols}")

    corr_vaso = train[vaso_cols + [TARGET_COL]].corr()[TARGET_COL]\
                    .drop(TARGET_COL).abs().sort_values(ascending=False)
    for col, r in corr_vaso.items():
        log.info(f"    {col:<45} r={r:.4f}")

    # Check: among sepsis patients, what % have n_vasoactives_24h > 0?
    sep_vaso_pct = (train.loc[train[TARGET_COL]==1, "n_vasoactives_24h"] > 0).mean() * 100
    nonsep_vaso_pct = (train.loc[train[TARGET_COL]==0, "n_vasoactives_24h"] > 0).mean() * 100
    log.info(f"\n  % with n_vasoactives_24h > 0:")
    log.info(f"    Sepsis:     {sep_vaso_pct:.1f}%")
    log.info(f"    Non-sepsis: {nonsep_vaso_pct:.1f}%")

    log.info(f"\n  Isolation test — CatBoost trained on VASO COLUMNS ONLY:")
    auroc_vaso, auprc_vaso = quick_auroc_train_test(
        X_train[vaso_cols].values, y_train,
        X_test[vaso_cols].values, y_test,
        "Vaso cols only"
    )
    iso_results["vaso_cols_only"] = auroc_vaso

    report["suspect_2_vaso_cols"] = {
        "n_vaso_cols"         : len(vaso_cols),
        "n_vasoactives_corr"  : float(corr_vaso.get("n_vasoactives_24h", 0)),
        "sepsis_vaso_pct"     : float(sep_vaso_pct),
        "nonsepsis_vaso_pct"  : float(nonsep_vaso_pct),
        "isolation_auroc"     : auroc_vaso,
        "isolation_auprc"     : auprc_vaso,
        "verdict"             : "LEAKAGE" if auroc_vaso > 0.85 else "ACCEPTABLE",
    }
    log.info(f"  VERDICT: {'LEAKAGE CONFIRMED' if auroc_vaso > 0.85 else 'ACCEPTABLE'} "
             f"(isolation AUROC={auroc_vaso:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 3 — _trend columns
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 3: _trend columns (last - first delta)")
    log.info("Hypothesis: deterioration trajectory encodes severity = outcome.")
    log.info(f"{'='*55}")

    trend_cols = [c for c in feature_cols if c.endswith("_trend")]
    log.info(f"  Found {len(trend_cols)} _trend columns")

    corr_trend = train[trend_cols + [TARGET_COL]].corr()[TARGET_COL]\
                     .drop(TARGET_COL).abs().sort_values(ascending=False)
    log.info(f"  Top 10 _trend correlations with label:")
    for col, r in corr_trend.head(10).items():
        log.info(f"    {col:<45} r={r:.4f}")

    if len(trend_cols) > 0:
        log.info(f"\n  Isolation test — CatBoost trained on TREND COLUMNS ONLY:")
        auroc_trend, auprc_trend = quick_auroc_train_test(
            X_train[trend_cols].values, y_train,
            X_test[trend_cols].values, y_test,
            "Trend cols only"
        )
        iso_results["trend_cols_only"] = auroc_trend
        verdict_trend = "LEAKAGE" if auroc_trend > 0.85 else "ACCEPTABLE"
    else:
        auroc_trend, auprc_trend = 0.5, 0.5
        iso_results["trend_cols_only"] = 0.5
        verdict_trend = "NOT PRESENT"

    report["suspect_3_trend_cols"] = {
        "n_trend_cols"   : len(trend_cols),
        "top_correlation": float(corr_trend.iloc[0]) if len(trend_cols) > 0 else 0,
        "isolation_auroc": auroc_trend,
        "verdict"        : verdict_trend,
    }
    log.info(f"  VERDICT: {verdict_trend} (isolation AUROC={auroc_trend:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 4 — fluid + urine columns
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 4: fluid_input_total_ml / urine_output_total_ml")
    log.info("Hypothesis: large resuscitation volumes = treatment for septic shock.")
    log.info(f"{'='*55}")

    fluid_cols = [c for c in feature_cols if any(k in c.lower() for k in
                  ["fluid", "urine", "output_total", "input_total", "balance"])]
    log.info(f"  Found {len(fluid_cols)} fluid/urine columns: {fluid_cols}")

    if fluid_cols:
        corr_fluid = train[fluid_cols + [TARGET_COL]].corr()[TARGET_COL]\
                         .drop(TARGET_COL).abs().sort_values(ascending=False)
        for col, r in corr_fluid.items():
            log.info(f"    {col:<45} r={r:.4f}")

        auroc_fluid, auprc_fluid = quick_auroc_train_test(
            X_train[fluid_cols].values, y_train,
            X_test[fluid_cols].values, y_test,
            "Fluid cols only"
        )
        iso_results["fluid_cols_only"] = auroc_fluid
        verdict_fluid = "LEAKAGE" if auroc_fluid > 0.85 else "ACCEPTABLE"
    else:
        auroc_fluid, auprc_fluid = 0.5, 0.5
        iso_results["fluid_cols_only"] = 0.5
        verdict_fluid = "NOT PRESENT"

    report["suspect_4_fluid_cols"] = {
        "n_fluid_cols"   : len(fluid_cols),
        "isolation_auroc": auroc_fluid,
        "verdict"        : verdict_fluid,
    }
    log.info(f"  VERDICT: {verdict_fluid} (isolation AUROC={auroc_fluid:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 5 — train/test patient overlap
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 5: Train/test patient overlap")
    log.info("Hypothesis: split was row-level not patient-level — same patient in both.")
    log.info(f"{'='*55}")

    # Check using meta files which contain subject_id
    meta_train_path = FILES.get("B_meta")
    overlap_verdict = "CANNOT VERIFY"
    n_overlap = -1

    if meta_train_path and Path(meta_train_path).exists():
        meta_train = pd.read_csv(meta_train_path)
        log.info(f"  Meta train columns: {list(meta_train.columns)}")
        if "subject_id" in meta_train.columns:
            train_ids = set(meta_train["subject_id"].values)
            # We don't have meta_test but can check raw B_test
            raw_test = pd.read_csv(FILES["B_test"])
            if "subject_id" in raw_test.columns:
                test_ids = set(raw_test["subject_id"].values)
                n_overlap = len(train_ids & test_ids)
                overlap_verdict = "CLEAN" if n_overlap == 0 else "LEAKAGE"
                log.info(f"  Patient overlap between train and test: {n_overlap}")
                log.info(f"  VERDICT: {overlap_verdict}")
            else:
                log.info("  subject_id not in test file — checking raw option_B files")
    else:
        # Try raw files directly
        raw_train = pd.read_csv(FILES["B_train"])
        raw_test  = pd.read_csv(FILES["B_test"])
        if "subject_id" in raw_train.columns and "subject_id" in raw_test.columns:
            train_ids = set(raw_train["subject_id"].values)
            test_ids  = set(raw_test["subject_id"].values)
            n_overlap = len(train_ids & test_ids)
            overlap_verdict = "CLEAN" if n_overlap == 0 else "LEAKAGE"
            log.info(f"  Patient IDs in train: {len(train_ids)}")
            log.info(f"  Patient IDs in test:  {len(test_ids)}")
            log.info(f"  Overlap: {n_overlap}")
        else:
            log.info("  subject_id column not found in raw files — cannot verify")
            log.info("  NOTE: step1_cohort_filter.py filtered to first ICU stay per patient")
            log.info("  This makes patient-level leakage very unlikely but unverified here")

    report["suspect_5_patient_overlap"] = {
        "n_overlap"     : n_overlap,
        "verdict"       : overlap_verdict,
    }
    log.info(f"  VERDICT: {overlap_verdict}")

    # ══════════════════════════════════════════════════════════════════════════
    # SUSPECT 6 — imputation leakage
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("SUSPECT 6: Imputation leakage")
    log.info("Hypothesis: medians computed on full dataset, not train-only.")
    log.info(f"{'='*55}")

    # Compare medians of numeric columns between train and full (train+test)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    full = pd.concat([train, test], ignore_index=True)

    train_medians = train[numeric_cols].median()
    full_medians  = full[numeric_cols].median()
    median_diff   = (train_medians - full_medians).abs()
    max_diff      = float(median_diff.max())
    mean_diff     = float(median_diff.mean())
    n_different   = int((median_diff > 0.001).sum())

    log.info(f"  Max median difference (train vs full): {max_diff:.6f}")
    log.info(f"  Mean median difference:                {mean_diff:.6f}")
    log.info(f"  Columns with diff > 0.001:             {n_different}")

    if max_diff < 0.01:
        log.info("  VERDICT: NEGLIGIBLE — medians nearly identical (expected for 80/20 split)")
        imputation_verdict = "NEGLIGIBLE"
    else:
        log.info(f"  VERDICT: INVESTIGATE — some medians differ significantly")
        imputation_verdict = "INVESTIGATE"

    report["suspect_6_imputation"] = {
        "max_median_diff" : max_diff,
        "mean_median_diff": mean_diff,
        "n_cols_different": n_different,
        "verdict"         : imputation_verdict,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # All features correlation ranking
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("ALL FEATURES — Correlation with label (top 40)")
    log.info(f"{'='*55}")

    all_corr = train[feature_cols + [TARGET_COL]].corr()[TARGET_COL]\
                   .drop(TARGET_COL).abs().sort_values(ascending=False)
    log.info(f"\n  {'Feature':<45} {'|r|':>6}  Type")
    log.info(f"  {'-'*60}")
    for col, r in all_corr.head(40).items():
        col_type = "count" if col.endswith("_count") else \
                   "vaso"  if "vaso" in col.lower() else \
                   "trend" if col.endswith("_trend") else \
                   "fluid" if any(k in col for k in ["fluid","urine","balance"]) else \
                   "clinical"
        flag = " *** SUSPECT" if col_type in ("count","vaso") and r > 0.3 else ""
        log.info(f"  {col:<45} {r:>6.4f}  {col_type}{flag}")

    # Save full correlation CSV
    corr_df = pd.DataFrame({"feature": all_corr.index, "abs_corr_with_label": all_corr.values})
    corr_df["type"] = corr_df["feature"].apply(lambda c:
        "count" if c.endswith("_count") else
        "vaso"  if "vaso" in c.lower() else
        "trend" if c.endswith("_trend") else
        "fluid" if any(k in c for k in ["fluid","urine","balance"]) else
        "clinical")
    corr_df.to_csv(DIAG_LEAKAGE / "feature_correlations.csv", index=False)
    log.info(f"\n  Full correlation table → {DIAG_LEAKAGE}/feature_correlations.csv")

    # ══════════════════════════════════════════════════════════════════════════
    # Isolation test — what AUROC do clinical-only features get?
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*55}")
    log.info("ISOLATION TEST — Clinical features only (no count/vaso/trend/fluid)")
    log.info(f"{'='*55}")

    clinical_cols = [c for c in feature_cols
                     if not c.endswith("_count")
                     and "vaso" not in c.lower()
                     and not c.endswith("_trend")
                     and not any(k in c for k in ["fluid","urine","balance","n_antibiotics",
                                                  "n_cultures","n_positive_cultures"])]
    log.info(f"  Clinical-only features: {len(clinical_cols)}")
    log.info(f"  Removed: {len(feature_cols) - len(clinical_cols)} suspect columns")

    auroc_clinical, auprc_clinical = quick_auroc_train_test(
        X_train[clinical_cols].values, y_train,
        X_test[clinical_cols].values, y_test,
        "Clinical cols only"
    )
    iso_results["clinical_only"]  = auroc_clinical
    iso_results["all_features"]   = None  # placeholder, we know it's ~1.0

    log.info(f"\n  Clinical-only AUROC={auroc_clinical:.4f}")
    if auroc_clinical > 0.85:
        log.info("  Still high — leakage may exist in clinical columns too")
    elif 0.75 <= auroc_clinical <= 0.92:
        log.info("  This is the realistic range for clinical sepsis prediction")
        log.info("  The suspect columns are inflating full-feature performance")
    else:
        log.info("  Lower than expected — may have removed too much or data is genuinely hard")

    report["clinical_only_test"] = {
        "n_clinical_cols"  : len(clinical_cols),
        "n_removed_cols"   : len(feature_cols) - len(clinical_cols),
        "isolation_auroc"  : auroc_clinical,
        "isolation_auprc"  : auprc_clinical,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Plot isolation test results
    # ══════════════════════════════════════════════════════════════════════════
    plot_data = {
        "All features\n(305 cols)": 1.000,
        "Count cols\nonly": iso_results["count_cols_only"],
        "Vaso cols\nonly": iso_results["vaso_cols_only"],
        "Trend cols\nonly": iso_results["trend_cols_only"],
        "Fluid cols\nonly": iso_results["fluid_cols_only"],
        "Clinical\nonly": iso_results["clinical_only"],
    }

    colors = []
    for k, v in plot_data.items():
        if v >= 0.95:
            colors.append("#C0392B")
        elif v >= 0.85:
            colors.append("#E67E22")
        else:
            colors.append("#27AE60")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(list(plot_data.keys()), list(plot_data.values()),
                  color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, plot_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0.85, color="#E67E22", linestyle="--", lw=1, alpha=0.7,
               label="Leakage threshold (0.85)")
    ax.axhline(0.92, color="#C0392B", linestyle="--", lw=1, alpha=0.7,
               label="Strong leakage (0.92)")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Test AUROC (CatBoost 100 trees)")
    ax.set_title("Isolation Tests — Which feature groups are driving perfect performance?\n"
                 "Red = confirmed leakage, Orange = suspect, Green = realistic",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_LEAKAGE / "isolation_test_aurocs.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"\n  Saved isolation test plot → {out}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final verdict
    # ══════════════════════════════════════════════════════════════════════════
    log.info(f"\n{'='*65}")
    log.info("FINAL VERDICT SUMMARY")
    log.info(f"{'='*65}")
    for suspect, data in report.items():
        verdict = data.get("verdict", "?")
        auroc   = data.get("isolation_auroc", data.get("n_overlap", "?"))
        log.info(f"  {suspect:<35} → {verdict}  (auroc/overlap={auroc})")

    log.info(f"\n  Clinical-only AUROC: {auroc_clinical:.4f}")
    log.info(f"  This is the realistic model performance after removing suspect columns.")

    log.info(f"\n  RECOMMENDED ACTION:")
    if iso_results["count_cols_only"] > 0.85:
        log.info(f"  1. DROP all _count columns ({len(count_cols)} columns)")
    if iso_results["vaso_cols_only"] > 0.85:
        log.info(f"  2. DROP n_vasoactives_24h + individual vaso flags ({len(vaso_cols)} columns)")
    if iso_results["trend_cols_only"] > 0.85:
        log.info(f"  3. DROP _trend columns ({len(trend_cols)} columns)")
    if iso_results["fluid_cols_only"] > 0.85:
        log.info(f"  4. REVIEW fluid columns ({len(fluid_cols)} columns)")
    log.info(f"  5. Retrain all models on cleaned features")
    log.info(f"  6. Expected realistic AUROC range: {auroc_clinical:.3f} (clinical-only test)")

    # Save report
    report["clinical_isolation_auroc"] = auroc_clinical
    report["timestamp"] = datetime.now().isoformat(timespec="seconds")
    with open(DIAG_LEAKAGE / "leakage_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\n  Report saved → {DIAG_LEAKAGE}/leakage_report.json")
    log.info(f"  Log saved    → {log_path}")
    log.info(f"\nPhase 0.2 complete.")


if __name__ == "__main__":
    main()
