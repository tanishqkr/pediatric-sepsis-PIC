"""
phase0_1_feature_prep.py
------------------------
Produces the final, frozen model-ready datasets for Option B (infection-only).
These are the ONLY files used for all model training.

Drop decisions informed by phase0_2_leakage_investigation.py:
  - _count columns (40): measurement frequency circular with Phoenix labeling
  - vaso columns (6): part of Phoenix cardiovascular score that created the label
  - _was_measured columns (33): binary physician-ordering signal, same circularity as _count
  - los_hours (1): length of stay is an outcome variable, unknown at prediction time
  - Standard drops: leakage, zero-variance, free text, duplicates

Inputs:
  output2/option_B_train.csv
  output2/option_B_test.csv

Outputs:
  model_datasets/B_train_model_ready.csv
  model_datasets/B_test_model_ready.csv
  model_datasets/feature_list.json
  model_datasets/prep_report.json
  model_datasets/README.md
  diagnostics/phase0_prep/prep_log.txt
  logs/phase0_1_feature_prep.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase0_1_feature_prep.py

OUTPUTS NEEDED AFTER RUNNING:
  - Full terminal output (paste)
  - Quick check: python -c "import pandas as pd; df=pd.read_csv('model_datasets/B_train_model_ready.csv'); print(df.shape)"
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
from config import FILES, DIAG_DIR, LOGS_DIR, TARGET_COL, RANDOM_SEED

REPO_ROOT      = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "model_datasets"
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

DIAG_PREP = DIAG_DIR / "phase0_prep"
DIAG_PREP.mkdir(parents=True, exist_ok=True)

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
# Drop decisions — single source of truth, all with documented reasons
# ══════════════════════════════════════════════════════════════════════════════

DROP_LEAKAGE = [
    "expire_flag",    # mortality outcome — recorded after ICU stay
    "hosp_expire",    # same, r=0.995 with expire_flag
    "los_hours",      # length of stay — outcome variable, unknown at prediction time
                      # also showed imputation leak (median diff=4.165 train vs full)
]

DROP_ZERO_VARIANCE = [
    "vaso_vasopressin",
    "suspected_infection",
    "symptom_anhelation_and_cyanosis",
    "symptom_infection",
]

DROP_FREE_TEXT = [
    "diagnosis",
    "icd10",
]

DROP_NO_VARIANCE = [
    "ethnicity",
]

DROP_DUPLICATES = [
    "age_months",
]

DROP_REFERENCE = [
    "phoenix_core_score",
    "phoenix_8_score",
]

# Vasoactive columns — part of Phoenix cardiovascular score (label construction)
# Isolation AUROC=0.863 on vaso columns alone confirms circularity
DROP_VASO = [
    "n_vasoactives_24h",
    "vaso_dopamine",
    "vaso_epinephrine",
    "vaso_norepinephrine",
    "vaso_dobutamine",
    "vaso_milrinone",
]

# Suffix-based drops — applied to all columns matching these patterns
DROP_SUFFIX_COUNT        = "_count"        # 40 cols — measurement frequency proxy
DROP_SUFFIX_WAS_MEASURED = "_was_measured" # 33 cols — binary ordering signal

# Categorical encodings
GENDER_MAP    = {"M": 1, "F": 0}
CARE_UNIT_COL = "care_unit"


# ══════════════════════════════════════════════════════════════════════════════
def prepare(df: pd.DataFrame, split_name: str, fit_encoder: bool = True,
            care_unit_dummies: list = None) -> tuple:

    report = {}
    log.info(f"\n  [{split_name}] Starting shape: {df.shape}")

    y   = df[TARGET_COL].copy()
    ref = [c for c in DROP_REFERENCE if c in df.columns]

    # 1. Drop leakage
    to_drop = [c for c in DROP_LEAKAGE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_leakage"] = to_drop
    log.info(f"  Dropped leakage:          {to_drop}")

    # 2. Drop zero-variance
    to_drop = [c for c in DROP_ZERO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_zero_variance"] = to_drop
    log.info(f"  Dropped zero-variance:    {to_drop}")

    # 3. Drop free text
    to_drop = [c for c in DROP_FREE_TEXT if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_free_text"] = to_drop
    log.info(f"  Dropped free text:        {to_drop}")

    # 4. Drop no-variance categorical
    to_drop = [c for c in DROP_NO_VARIANCE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_no_variance"] = to_drop
    log.info(f"  Dropped no-variance:      {to_drop}")

    # 5. Drop duplicates
    to_drop = [c for c in DROP_DUPLICATES if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_duplicates"] = to_drop
    log.info(f"  Dropped duplicates:       {to_drop}")

    # 6. Drop reference cols
    to_drop = [c for c in DROP_REFERENCE if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_reference"] = to_drop
    log.info(f"  Dropped reference:        {to_drop}")

    # 7. Drop vasoactive columns (Phoenix cardiovascular score proxy)
    to_drop = [c for c in DROP_VASO if c in df.columns]
    df = df.drop(columns=to_drop)
    report["dropped_vaso"] = to_drop
    log.info(f"  Dropped vaso (leakage):   {to_drop}")

    # 8. Drop all _count columns (measurement frequency)
    count_drop = [c for c in df.columns if c.endswith(DROP_SUFFIX_COUNT)]
    df = df.drop(columns=count_drop)
    report["dropped_count_cols"] = len(count_drop)
    log.info(f"  Dropped _count columns:   {len(count_drop)} columns")

    # 9. Drop all _was_measured columns (binary ordering signal)
    was_drop = [c for c in df.columns if c.endswith(DROP_SUFFIX_WAS_MEASURED)]
    df = df.drop(columns=was_drop)
    report["dropped_was_measured_cols"] = len(was_drop)
    log.info(f"  Dropped _was_measured:    {len(was_drop)} columns")

    # 10. Encode gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAP).fillna(0)
        report["gender_encoded"] = "M=1, F=0"
        log.info(f"  Encoded gender: M=1, F=0")

    # 11. One-hot encode care_unit
    if CARE_UNIT_COL in df.columns:
        if fit_encoder:
            dummies = pd.get_dummies(df[CARE_UNIT_COL], prefix="care_unit", drop_first=False)
            care_unit_dummies = list(dummies.columns)
        else:
            dummies = pd.get_dummies(df[CARE_UNIT_COL], prefix="care_unit", drop_first=False)
            for col in care_unit_dummies:
                if col not in dummies.columns:
                    dummies[col] = 0
            dummies = dummies[care_unit_dummies]
        df = df.drop(columns=[CARE_UNIT_COL])
        df = pd.concat([df, dummies], axis=1)
        report["care_unit_encoded"] = care_unit_dummies
        log.info(f"  One-hot encoded care_unit: {care_unit_dummies}")

    # 12. Drop target, verify, re-attach
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        log.info(f"  WARNING: dropping remaining object cols: {obj_cols}")
        df = df.drop(columns=obj_cols)

    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        log.info(f"  WARNING: {n_missing} missing values — filling with 0")
        df = df.fillna(0)

    df[TARGET_COL] = y.values

    report["final_cols"]    = int(df.shape[1])
    report["feature_cols"]  = int(df.shape[1] - 1)
    report["rows"]          = int(df.shape[0])
    report["missing_after"] = int(n_missing)

    log.info(f"  [{split_name}] Final shape: {df.shape}  "
             f"(features: {df.shape[1]-1}, target: 1)")
    log.info(f"  Missing values: {n_missing} {'✓' if n_missing == 0 else '— filled'}")

    return df, care_unit_dummies, report


# ══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 65)
    log.info("PHASE 0.1 — FEATURE PREPARATION (Option B, v2 — leakage-clean)")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("Dropping: _count, _was_measured, vaso, los_hours + standard drops")
    log.info("=" * 65)

    log.info("\nLoading Option B train and test...")
    train = pd.read_csv(FILES["B_train"])
    test  = pd.read_csv(FILES["B_test"])
    log.info(f"  Raw train: {train.shape}")
    log.info(f"  Raw test:  {test.shape}")

    log.info("\n--- Preparing TRAIN ---")
    train_prep, care_unit_dummies, report_train = prepare(
        train, "B_train", fit_encoder=True
    )

    log.info("\n--- Preparing TEST ---")
    test_prep, _, report_test = prepare(
        test, "B_test", fit_encoder=False,
        care_unit_dummies=care_unit_dummies
    )

    log.info("\n--- Column consistency check ---")
    train_cols = set(train_prep.columns)
    test_cols  = set(test_prep.columns)
    only_train = train_cols - test_cols
    only_test  = test_cols  - train_cols
    if only_train: log.info(f"  WARNING: only in train: {only_train}")
    if only_test:  log.info(f"  WARNING: only in test:  {only_test}")
    if not only_train and not only_test:
        log.info(f"  Train/test columns match exactly ✓  ({len(train_cols)} columns)")

    col_order = list(train_prep.columns)
    test_prep = test_prep[col_order]

    log.info("\n--- Class distribution ---")
    for name, df in [("Train", train_prep), ("Test", test_prep)]:
        vc    = df[TARGET_COL].value_counts().sort_index()
        total = len(df)
        log.info(f"  {name}: non-sepsis={vc[0]:,} ({100*vc[0]/total:.1f}%)  "
                 f"sepsis={vc[1]:,} ({100*vc[1]/total:.1f}%)  "
                 f"ratio={vc[0]/vc[1]:.1f}:1")

    log.info("\n--- Saving outputs ---")
    train_out = MODEL_DATA_DIR / "B_train_model_ready.csv"
    test_out  = MODEL_DATA_DIR / "B_test_model_ready.csv"
    train_prep.to_csv(train_out, index=False)
    test_prep.to_csv(test_out,  index=False)
    log.info(f"  Train → {train_out}")
    log.info(f"  Test  → {test_out}")

    feature_cols = [c for c in train_prep.columns if c != TARGET_COL]
    feature_list_path = MODEL_DATA_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump({
            "n_features"  : len(feature_cols),
            "target_col"  : TARGET_COL,
            "feature_cols": feature_cols,
            "generated"   : datetime.now().isoformat(timespec="seconds"),
            "version"     : "v2-leakage-clean",
            "dropped_groups": {
                "count_cols"       : "_count suffix (40) — measurement frequency proxy",
                "was_measured_cols": "_was_measured suffix (33) — binary ordering signal",
                "vaso_cols"        : "6 vasoactive cols — Phoenix cardiovascular score proxy",
                "los_hours"        : "outcome variable unknown at prediction time",
            }
        }, f, indent=2)
    log.info(f"  Feature list ({len(feature_cols)} features) → {feature_list_path}")

    prep_report = {
        "timestamp"     : datetime.now().isoformat(timespec="seconds"),
        "version"       : "v2-leakage-clean",
        "dataset"       : "Option B — Infection-only",
        "train_report"  : report_train,
        "test_report"   : report_test,
    }
    report_path = MODEL_DATA_DIR / "prep_report.json"
    with open(report_path, "w") as f:
        json.dump(prep_report, f, indent=2, default=str)
    log.info(f"  Prep report → {report_path}")

    readme = f"""# model_datasets/ — v2 leakage-clean

Generated: {datetime.now().isoformat(timespec='seconds')}
Dataset: PIC Sepsis — Option B (Infection-only cohort)
Version: v2-leakage-clean (drops _count, _was_measured, vaso, los_hours)

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
- Train: {train_prep[TARGET_COL].sum()} sepsis / {(train_prep[TARGET_COL]==0).sum()} non-sepsis
- Test:  {test_prep[TARGET_COL].sum()} sepsis / {(test_prep[TARGET_COL]==0).sum()} non-sepsis

## What was dropped and why
- `_count` columns (40): measurement frequency — circular with Phoenix labeling
- `_was_measured` columns (33): binary physician-ordering signal — same circularity
- vaso columns (6): part of Phoenix cardiovascular score that defines the label
- `los_hours`: outcome variable — unknown at early prediction time
- Standard: leakage (expire_flag, hosp_expire), zero-variance, free text, duplicates

## IMPORTANT
- class_weights = {{0: 1, 1: 2.7}} for all models
- Random seed = 42
- Expected realistic AUROC: 0.82–0.90
"""
    with open(MODEL_DATA_DIR / "README.md", "w") as f:
        f.write(readme)

    log.info("\n" + "=" * 65)
    log.info("PHASE 0.1 COMPLETE (v2 leakage-clean)")
    log.info("=" * 65)
    log.info(f"  Features:       {len(feature_cols)}")
    log.info(f"  Train rows:     {train_prep.shape[0]}")
    log.info(f"  Test rows:      {test_prep.shape[0]}")
    log.info(f"  Missing values: 0 ✓")
    log.info(f"  Output folder:  {MODEL_DATA_DIR}")
    log.info(f"\n  Dropped vs previous version:")
    log.info(f"    Previous: 305 features (leakage confirmed)")
    log.info(f"    Current:  {len(feature_cols)} features (leakage-clean)")
    log.info(f"    Removed:  {305 - len(feature_cols)} columns")


if __name__ == "__main__":
    main()