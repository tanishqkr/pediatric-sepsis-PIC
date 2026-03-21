"""
phase0_1_feature_prep.py
------------------------
Produces the final, frozen model-ready datasets for Option B (infection-only).
These are the ONLY files used for all model training. Do not modify after running.

Inputs:
  output2/option_B_train.csv
  output2/option_B_test.csv

Outputs:
  model_datasets/B_train_model_ready.csv   ← everyone trains from this
  model_datasets/B_test_model_ready.csv    ← everyone evaluates on this
  model_datasets/feature_list.json         ← exact feature names in order
  model_datasets/prep_report.json          ← full audit of what was dropped/encoded
  diagnostics/phase0_prep/prep_log.txt     ← detailed log
  logs/phase0_1_feature_prep.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase0_1_feature_prep.py
"""

import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FILES, DIAG_DIR, LOGS_DIR, TARGET_COL, RANDOM_SEED
)

# ── Model datasets folder — at repo root, easy for all collaborators ──────────
REPO_ROOT       = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR  = REPO_ROOT / "model_datasets"
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Phase-specific diagnostics subfolder ─────────────────────────────────────
DIAG_PREP = DIAG_DIR / "phase0_prep"
DIAG_PREP.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase0_1_feature_prep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

# ══════════════════════════════════════════════════════════════════════════════
# Drop / encode decisions — single source of truth
# All decisions documented with reason for the methods section
# ══════════════════════════════════════════════════════════════════════════════

DROP_LEAKAGE = [
    "expire_flag",           # outcome recorded after ICU stay — label leakage
    "hosp_expire",           # same, r=0.995 with expire_flag
]

DROP_ZERO_VARIANCE = [
    "vaso_vasopressin",               # constant 0 in entire Option B
    "suspected_infection",            # constant 1 in Option B (everyone is infected by definition)
    "symptom_anhelation_and_cyanosis",# constant 0
]

DROP_FREE_TEXT = [
    "diagnosis",   # 488 unique Chinese text values — unusable as feature
    "icd10",       # 487 unique ICD codes — too high cardinality, text
]

DROP_NO_VARIANCE = [
    "ethnicity",   # 96.8% Han ethnic — essentially constant, no clinical signal
]

DROP_DUPLICATES = [
    "age_months",  # perfect duplicate of age_years (r=1.0), age_years kept
]

# _was_measured deduplication:
# All 4 stats (max/min/mean/first) for the same variable share one indicator.
# Keep only _max_was_measured, drop min/mean/first variants.
# This removes 3 redundant columns per variable with was_measured indicators.
WAS_MEASURED_KEEP_SUFFIX  = "_max_was_measured"
WAS_MEASURED_DROP_SUFFIXES = ["_min_was_measured", "_mean_was_measured", "_first_was_measured"]

# Categorical encodings
GENDER_MAP    = {"M": 1, "F": 0}
CARE_UNIT_COL = "care_unit"
CARE_UNIT_CATEGORIES = ["NICU", "PICU", "SICU", "CICU", "General ICU"]

# Columns to keep as reference (not features, not dropped — stored in prep report)
REFERENCE_COLS = ["phoenix_core_score", "phoenix_8_score"]


# ══════════════════════════════════════════════════════════════════════════════
# Main preparation function
# ══════════════════════════════════════════════════════════════════════════════

def prepare(df: pd.DataFrame, split_name: str, fit_encoder: bool = True,
            care_unit_dummies: list = None) -> tuple:
    """
    Apply all feature prep steps to a dataframe.
    fit_encoder=True for train (derives dummy columns).
    fit_encoder=False for test (applies same columns as train).
    Returns (prepared_df, care_unit_dummies_list)
    """
    report = {}
    original_cols = df.shape[1]
    original_rows = df.shape[0]
    log.info(f"\n  [{split_name}] Starting shape: {df.shape}")

    # ── 1. Separate target and reference cols ─────────────────────────────────
    y = df[TARGET_COL].copy()
    ref_cols_present = [c for c in REFERENCE_COLS if c in df.columns]
    ref_data = df[ref_cols_present].copy()

    # ── 2. Drop leakage columns ───────────────────────────────────────────────
    to_drop = [c for c in DROP_LEAKAGE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_leakage"] = to_drop
    log.info(f"  Dropped leakage:        {to_drop}")

    # ── 3. Drop zero-variance columns ─────────────────────────────────────────
    to_drop = [c for c in DROP_ZERO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_zero_variance"] = to_drop
    log.info(f"  Dropped zero-variance:  {to_drop}")

    # ── 4. Drop free text columns ─────────────────────────────────────────────
    to_drop = [c for c in DROP_FREE_TEXT if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_free_text"] = to_drop
    log.info(f"  Dropped free text:      {to_drop}")

    # ── 5. Drop no-variance categorical columns ───────────────────────────────
    to_drop = [c for c in DROP_NO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_no_variance"] = to_drop
    log.info(f"  Dropped no-variance:    {to_drop}")

    # ── 6. Drop duplicate columns ─────────────────────────────────────────────
    to_drop = [c for c in DROP_DUPLICATES if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_duplicates"] = to_drop
    log.info(f"  Dropped duplicates:     {to_drop}")

    # ── 7. Drop reference columns from feature set ────────────────────────────
    to_drop = [c for c in REFERENCE_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_reference"] = to_drop
    log.info(f"  Dropped reference cols: {to_drop}")

    # ── 8. Deduplicate _was_measured indicators ───────────────────────────────
    was_drop = []
    for col in list(df.columns):
        for suffix in WAS_MEASURED_DROP_SUFFIXES:
            if col.endswith(suffix):
                was_drop.append(col)
    df = df.drop(columns=was_drop)
    report["dropped_was_measured_duplicates"] = len(was_drop)
    log.info(f"  Dropped redundant _was_measured duplicates: {len(was_drop)} columns")

    # ── 9. Encode gender ──────────────────────────────────────────────────────
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP)
        unmapped = df["gender"].isnull().sum()
        if unmapped > 0:
            log.info(f"  WARNING: {unmapped} gender values could not be mapped")
            df["gender"] = df["gender"].fillna(0)
        report["gender_encoded"] = "M=1, F=0"
        log.info(f"  Encoded gender: M=1, F=0")

    # ── 10. One-hot encode care_unit ──────────────────────────────────────────
    if CARE_UNIT_COL in df.columns:
        if fit_encoder:
            dummies = pd.get_dummies(
                df[CARE_UNIT_COL], prefix="care_unit", drop_first=False
            )
            care_unit_dummies = list(dummies.columns)
        else:
            # Apply same columns as train — fill missing with 0
            dummies = pd.get_dummies(
                df[CARE_UNIT_COL], prefix="care_unit", drop_first=False
            )
            for col in care_unit_dummies:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[care_unit_dummies]

        df = df.drop(columns=[CARE_UNIT_COL])
        df = pd.concat([df, dummies], axis=1)
        report["care_unit_encoded"] = care_unit_dummies
        log.info(f"  One-hot encoded care_unit: {care_unit_dummies}")

    # ── 11. Drop TARGET_COL from features (re-attach at end) ──────────────────
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # ── 12. Final checks ──────────────────────────────────────────────────────
    # Verify no remaining object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.info(f"  WARNING: object columns still present — dropping: {obj_cols}")
        df = df.drop(columns=obj_cols)
        report["dropped_remaining_object"] = obj_cols

    # Verify no missing values
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        missing_cols = df.columns[df.isnull().any()].tolist()
        log.info(f"  WARNING: {n_missing} missing values remain in: {missing_cols}")
        log.info(f"  Filling with 0 (should not happen — check preprocessing)")
        df = df.fillna(0)

    # Re-attach target
    df[TARGET_COL] = y.values

    report["original_cols"]  = int(original_cols)
    report["final_cols"]     = int(df.shape[1])
    report["feature_cols"]   = int(df.shape[1] - 1)  # excluding target
    report["rows"]           = int(original_rows)
    report["missing_after"]  = int(n_missing)

    log.info(f"  [{split_name}] Final shape: {df.shape}  "
             f"(features: {df.shape[1]-1}, target: 1)")
    log.info(f"  Missing values after prep: {n_missing} ✓" if n_missing == 0
             else f"  Missing values after prep: {n_missing} — filled with 0")

    return df, care_unit_dummies, report


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("PHASE 0.1 — FEATURE PREPARATION (Option B)")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("Producing final frozen model-ready datasets.")
    log.info("=" * 60)

    # ── Load Option B ─────────────────────────────────────────────────────────
    log.info("\nLoading Option B train and test...")
    train = pd.read_csv(FILES["B_train"])
    test  = pd.read_csv(FILES["B_test"])
    log.info(f"  Raw train: {train.shape}")
    log.info(f"  Raw test:  {test.shape}")

    # ── Prepare train (fit encoders) ──────────────────────────────────────────
    log.info("\n--- Preparing TRAIN ---")
    train_prep, care_unit_dummies, report_train = prepare(
        train, "B_train", fit_encoder=True
    )

    # ── Prepare test (apply same encoders) ────────────────────────────────────
    log.info("\n--- Preparing TEST ---")
    test_prep, _, report_test = prepare(
        test, "B_test", fit_encoder=False,
        care_unit_dummies=care_unit_dummies
    )

    # ── Verify train/test columns match exactly ───────────────────────────────
    log.info("\n--- Column consistency check ---")
    train_cols = set(train_prep.columns)
    test_cols  = set(test_prep.columns)
    only_train = train_cols - test_cols
    only_test  = test_cols  - train_cols
    if only_train:
        log.info(f"  WARNING: columns only in train: {only_train}")
    if only_test:
        log.info(f"  WARNING: columns only in test: {only_test}")
    if not only_train and not only_test:
        log.info(f"  Train/test columns match exactly ✓  ({len(train_cols)} columns)")

    # Enforce same column order
    col_order = list(train_prep.columns)
    test_prep = test_prep[col_order]

    # ── Class distribution check ──────────────────────────────────────────────
    log.info("\n--- Class distribution ---")
    for name, df in [("Train", train_prep), ("Test", test_prep)]:
        vc = df[TARGET_COL].value_counts().sort_index()
        total = len(df)
        log.info(f"  {name}: non-sepsis={vc[0]:,} ({100*vc[0]/total:.1f}%)  "
                 f"sepsis={vc[1]:,} ({100*vc[1]/total:.1f}%)  "
                 f"ratio={vc[0]/vc[1]:.1f}:1")

    # ── Save model-ready CSVs ─────────────────────────────────────────────────
    log.info("\n--- Saving outputs ---")
    train_out = MODEL_DATA_DIR / "B_train_model_ready.csv"
    test_out  = MODEL_DATA_DIR / "B_test_model_ready.csv"
    train_prep.to_csv(train_out, index=False)
    test_prep.to_csv(test_out,  index=False)
    log.info(f"  Train → {train_out}")
    log.info(f"  Test  → {test_out}")

    # ── Save feature list ─────────────────────────────────────────────────────
    feature_cols = [c for c in train_prep.columns if c != TARGET_COL]
    feature_list_path = MODEL_DATA_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump({
            "n_features"  : len(feature_cols),
            "target_col"  : TARGET_COL,
            "feature_cols": feature_cols,
            "generated"   : datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)
    log.info(f"  Feature list ({len(feature_cols)} features) → {feature_list_path}")

    # ── Save prep report ──────────────────────────────────────────────────────
    prep_report = {
        "timestamp"     : datetime.now().isoformat(timespec="seconds"),
        "dataset"       : "Option B — Infection-only",
        "train_report"  : report_train,
        "test_report"   : report_test,
        "drop_decisions": {
            "leakage"              : DROP_LEAKAGE,
            "zero_variance"        : DROP_ZERO_VARIANCE,
            "free_text"            : DROP_FREE_TEXT,
            "no_variance"          : DROP_NO_VARIANCE,
            "duplicates"           : DROP_DUPLICATES,
            "was_measured_policy"  : "keep _max_was_measured only per variable",
        },
        "encode_decisions": {
            "gender"    : "M=1, F=0",
            "care_unit" : f"one-hot → {care_unit_dummies}",
            "ethnicity" : "dropped (96.8% Han, no variance)",
        },
        "files": {
            "train": str(train_out),
            "test" : str(test_out),
            "features": str(feature_list_path),
        }
    }
    report_path = MODEL_DATA_DIR / "prep_report.json"
    with open(report_path, "w") as f:
        json.dump(prep_report, f, indent=2, default=str)
    log.info(f"  Prep report → {report_path}")

    # ── README for collaborators ──────────────────────────────────────────────
    readme = f"""# model_datasets/

Generated: {datetime.now().isoformat(timespec='seconds')}
Dataset: PIC Sepsis — Option B (Infection-only cohort)

## Files

| File | Description |
|------|-------------|
| B_train_model_ready.csv | Training set — {train_prep.shape[0]} rows, {len(feature_cols)} features + target |
| B_test_model_ready.csv  | Test set — {test_prep.shape[0]} rows, {len(feature_cols)} features + target |
| feature_list.json       | Exact feature names in column order |
| prep_report.json        | Full audit of all drop/encode decisions |

## Target column
`{TARGET_COL}` — 1=sepsis, 0=infected-non-sepsis

## Class distribution
- Train: {train_prep[TARGET_COL].sum()} sepsis / {(train_prep[TARGET_COL]==0).sum()} non-sepsis ({100*train_prep[TARGET_COL].mean():.1f}% prevalence)
- Test:  {test_prep[TARGET_COL].sum()} sepsis / {(test_prep[TARGET_COL]==0).sum()} non-sepsis ({100*test_prep[TARGET_COL].mean():.1f}% prevalence)

## IMPORTANT
- Do NOT modify these files. They are the frozen final datasets.
- Do NOT use option_A or option_B raw files for modelling.
- class_weights = {{0: 1, 1: 2.7}} for all models (NO SMOTE).
- Random seed = 42 for all experiments.
- See prep_report.json for full documentation of all preprocessing decisions.
"""
    readme_path = MODEL_DATA_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    log.info(f"  README → {readme_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 0.1 COMPLETE")
    log.info("=" * 60)
    log.info(f"  Features:       {len(feature_cols)}")
    log.info(f"  Train rows:     {train_prep.shape[0]}")
    log.info(f"  Test rows:      {test_prep.shape[0]}")
    log.info(f"  Missing values: 0 ✓")
    log.info(f"  Output folder:  {MODEL_DATA_DIR}")
    log.info(f"\n  Ready for modelling. Load with:")
    log.info(f"    import pandas as pd")
    log.info(f"    train = pd.read_csv('model_datasets/B_train_model_ready.csv')")
    log.info(f"    test  = pd.read_csv('model_datasets/B_test_model_ready.csv')")


if __name__ == "__main__":
    main()
