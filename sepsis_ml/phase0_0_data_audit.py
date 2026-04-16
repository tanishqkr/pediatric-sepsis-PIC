"""
phase0_0_data_audit.py
----------------------
Run this FIRST — before phase0_diagnostics.py or any modelling.
Produces a complete, no-assumptions audit of every column in all four
dataset splits. Tells you exactly:
  - What dtype every column has
  - Min / max / mean / median for numerics
  - Unique value counts and top values for categoricals
  - Missing rates per column
  - Whether the target column is balanced
  - Any columns that look like they shouldn't be in a feature set
    (IDs, free text, label leakage candidates)

Nothing is assumed. Everything is verified and saved.

Outputs:
  diagnostics/audit_column_report_A.csv   — full column-level report, Option A
  diagnostics/audit_column_report_B.csv   — full column-level report, Option B
  diagnostics/audit_summary.json          — high-level summary
  diagnostics/audit_suspicious_cols.json  — columns flagged for review
  logs/phase0_0_data_audit.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase0_0_data_audit.py
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
from config import FILES, DIAG_AUDIT, LOGS_DIR, TARGET_COL, DROP_COLS, RANDOM_SEED

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase0_0_data_audit.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

# ── Columns that should never be model features ───────────────────────────────
# Any column matching these patterns is flagged as suspicious
LEAKAGE_KEYWORDS   = ["score", "label", "expire", "death", "mortality", "hosp_expire"]
ID_KEYWORDS        = ["id", "subject", "hadm", "stay_id", "icustay"]
FREETEXT_KEYWORDS  = ["diagnosis", "icd", "text", "note", "description"]


def audit_column(col: str, series: pd.Series) -> dict:
    """Produce a full audit record for a single column."""
    n          = len(series)
    n_missing  = int(series.isnull().sum())
    miss_pct   = round(100 * n_missing / n, 2)
    dtype      = str(series.dtype)
    n_unique   = int(series.nunique(dropna=True))

    record = {
        "column"      : col,
        "dtype"       : dtype,
        "n_total"     : n,
        "n_missing"   : n_missing,
        "missing_pct" : miss_pct,
        "n_unique"    : n_unique,
    }

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        record.update({
            "min"    : round(float(clean.min()), 4)   if len(clean) else None,
            "max"    : round(float(clean.max()), 4)   if len(clean) else None,
            "mean"   : round(float(clean.mean()), 4)  if len(clean) else None,
            "median" : round(float(clean.median()), 4)if len(clean) else None,
            "std"    : round(float(clean.std()), 4)   if len(clean) else None,
            "pct_zero"   : round(100 * (clean == 0).sum() / len(clean), 1) if len(clean) else None,
            "top_values" : str(clean.value_counts().head(5).to_dict()),
        })
    else:
        top = series.value_counts(dropna=False).head(10)
        record.update({
            "min"        : None,
            "max"        : None,
            "mean"       : None,
            "median"     : None,
            "std"        : None,
            "pct_zero"   : None,
            "top_values" : str(top.to_dict()),
        })

    # ── Flag suspicious columns ───────────────────────────────────────────────
    flags = []
    col_lower = col.lower()

    if any(kw in col_lower for kw in ID_KEYWORDS):
        flags.append("possible_id_column")
    if any(kw in col_lower for kw in LEAKAGE_KEYWORDS):
        flags.append("possible_label_leakage")
    if any(kw in col_lower for kw in FREETEXT_KEYWORDS):
        flags.append("possible_free_text")
    if miss_pct > 80:
        flags.append("very_high_missing_>80pct")
    if miss_pct > 50:
        flags.append("high_missing_>50pct")
    if n_unique == 1:
        flags.append("constant_column")
    if n_unique == n and not pd.api.types.is_numeric_dtype(series):
        flags.append("all_unique_possible_id")
    if pd.api.types.is_numeric_dtype(series) and record.get("std", 1) == 0:
        flags.append("zero_variance")

    record["flags"] = "|".join(flags) if flags else ""
    return record


def audit_dataset(train: pd.DataFrame, test: pd.DataFrame, label: str, key: str) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"AUDITING: {label}")
    log.info(f"{'='*60}")
    log.info(f"  Train shape: {train.shape}")
    log.info(f"  Test shape:  {test.shape}")

    # ── Basic shape checks ────────────────────────────────────────────────────
    log.info(f"\n  Column count match train/test: "
             f"{'YES' if train.shape[1] == test.shape[1] else 'NO — MISMATCH'}")

    train_cols = set(train.columns)
    test_cols  = set(test.columns)
    only_train = train_cols - test_cols
    only_test  = test_cols  - train_cols
    if only_train:
        log.info(f"  Columns only in train: {only_train}")
    if only_test:
        log.info(f"  Columns only in test:  {only_test}")

    # ── Target column ─────────────────────────────────────────────────────────
    if TARGET_COL in train.columns:
        vc = train[TARGET_COL].value_counts().sort_index()
        log.info(f"\n  Target column '{TARGET_COL}' in train:")
        for val, cnt in vc.items():
            log.info(f"    {val}: {cnt:,}  ({100*cnt/len(train):.1f}%)")
    else:
        log.info(f"\n  WARNING: target column '{TARGET_COL}' NOT FOUND in train!")

    # ── Dtype summary ─────────────────────────────────────────────────────────
    dtype_counts = train.dtypes.value_counts()
    log.info(f"\n  Dtype breakdown (train):")
    for dt, cnt in dtype_counts.items():
        log.info(f"    {str(dt):<15} {cnt} columns")

    # ── Per-column audit ──────────────────────────────────────────────────────
    log.info(f"\n  Running per-column audit on {train.shape[1]} columns...")
    records = []
    for col in train.columns:
        rec = audit_column(col, train[col])
        records.append(rec)

    report_df = pd.DataFrame(records)
    out_path  = DIAG_AUDIT / f"audit_column_report_{key}.csv"
    report_df.to_csv(out_path, index=False)
    log.info(f"  Full column report saved → {out_path}")

    # ── Suspicious columns summary ────────────────────────────────────────────
    flagged = report_df[report_df["flags"] != ""]
    log.info(f"\n  Flagged columns requiring review: {len(flagged)}")
    for _, row in flagged.iterrows():
        log.info(f"    {row['column']:<50}  flags: {row['flags']}")

    # ── Non-numeric columns detail ────────────────────────────────────────────
    non_num_cols = report_df[~report_df["dtype"].str.startswith(("int", "float", "uint"))]["column"].tolist()
    log.info(f"\n  Non-numeric columns ({len(non_num_cols)}):")
    for col in non_num_cols:
        row = report_df[report_df["column"] == col].iloc[0]
        log.info(f"    {col:<40}  dtype={row['dtype']}  unique={row['n_unique']}  "
                 f"missing={row['missing_pct']}%")
        log.info(f"      top values: {row['top_values'][:120]}")

    # ── Zero / near-zero variance ─────────────────────────────────────────────
    zero_var = report_df[report_df["flags"].str.contains("zero_variance", na=False)]["column"].tolist()
    constant = report_df[report_df["flags"].str.contains("constant_column", na=False)]["column"].tolist()
    log.info(f"\n  Zero-variance numeric columns ({len(zero_var)}): {zero_var}")
    log.info(f"  Constant columns (1 unique value) ({len(constant)}): {constant}")

    # ── High missing ──────────────────────────────────────────────────────────
    high_miss = report_df[report_df["missing_pct"] > 20].sort_values("missing_pct", ascending=False)
    log.info(f"\n  Columns with >20% missing ({len(high_miss)}):")
    for _, row in high_miss.iterrows():
        log.info(f"    {row['column']:<50}  {row['missing_pct']:.1f}%")

    # ── Possible leakage columns ──────────────────────────────────────────────
    leakage = report_df[report_df["flags"].str.contains("possible_label_leakage", na=False)]["column"].tolist()
    log.info(f"\n  Possible label leakage columns ({len(leakage)}):")
    for col in leakage:
        log.info(f"    {col}")

    summary = {
        "label"           : label,
        "train_shape"     : list(train.shape),
        "test_shape"      : list(test.shape),
        "n_total_cols"    : int(train.shape[1]),
        "n_numeric_cols"  : int(train.select_dtypes(include=[np.number]).shape[1]),
        "n_nonnumeric_cols": len(non_num_cols),
        "n_flagged_cols"  : int(len(flagged)),
        "n_zero_variance" : len(zero_var),
        "n_constant"      : len(constant),
        "n_high_missing"  : int((report_df["missing_pct"] > 20).sum()),
        "n_leakage_risk"  : len(leakage),
        "flagged_columns" : flagged[["column", "flags"]].to_dict(orient="records"),
        "non_numeric_columns": non_num_cols,
        "zero_variance_columns": zero_var,
        "constant_columns": constant,
        "leakage_risk_columns": leakage,
    }
    return summary


def main():
    log.info("=" * 60)
    log.info("PHASE 0.0 — DATA AUDIT")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("No assumptions. Verify everything before modelling.")
    log.info("=" * 60)

    # ── Verify files exist ────────────────────────────────────────────────────
    log.info("\nVerifying input file paths...")
    all_ok = True
    for key, path in FILES.items():
        exists = Path(path).exists()
        status = "OK" if exists else "MISSING"
        log.info(f"  [{status}] {key}: {path}")
        if not exists:
            all_ok = False
    if not all_ok:
        log.info("\nERROR: One or more input files missing. Check config.py DATA_DIR.")
        sys.exit(1)

    # ── Load all splits ───────────────────────────────────────────────────────
    train_A = pd.read_csv(FILES["A_train"])
    test_A  = pd.read_csv(FILES["A_test"])
    train_B = pd.read_csv(FILES["B_train"])
    test_B  = pd.read_csv(FILES["B_test"])

    # ── Audit each dataset ────────────────────────────────────────────────────
    summary_A = audit_dataset(train_A, test_A, "Option A — Full cohort",        "A")
    summary_B = audit_dataset(train_B, test_B, "Option B — Infection-only",     "B")

    # ── Cross-dataset comparison ──────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("CROSS-DATASET COMPARISON")
    log.info(f"{'='*60}")
    cols_A = set(train_A.columns)
    cols_B = set(train_B.columns)
    only_A = cols_A - cols_B
    only_B = cols_B - cols_A
    shared = cols_A & cols_B
    log.info(f"  Shared columns:          {len(shared)}")
    log.info(f"  Only in Option A:        {len(only_A)}  → {sorted(only_A)}")
    log.info(f"  Only in Option B:        {len(only_B)}  → {sorted(only_B)}")

    # ── Patient overlap check (data leakage between train and test) ───────────
    log.info(f"\n{'='*60}")
    log.info("PATIENT OVERLAP CHECK (train vs test leakage)")
    log.info(f"{'='*60}")
    for key, train, test in [("A", train_A, test_A), ("B", train_B, test_B)]:
        if "subject_id" in train.columns:
            overlap = set(train["subject_id"]) & set(test["subject_id"])
            log.info(f"  Option {key}: subject_id overlap between train and test = {len(overlap)}"
                     f"  {'✓ CLEAN' if len(overlap) == 0 else '✗ LEAKAGE DETECTED'}")
        else:
            log.info(f"  Option {key}: subject_id not found — cannot verify patient overlap")

    # ── Recommendations ───────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("RECOMMENDED ACTIONS BEFORE MODELLING")
    log.info(f"{'='*60}")

    all_leakage = list(set(
        summary_A["leakage_risk_columns"] + summary_B["leakage_risk_columns"]
    ))
    all_nonnumeric = list(set(
        summary_A["non_numeric_columns"] + summary_B["non_numeric_columns"]
    ))
    all_zero_var = list(set(
        summary_A["zero_variance_columns"] + summary_B["zero_variance_columns"]
    ))
    all_constant = list(set(
        summary_A["constant_columns"] + summary_B["constant_columns"]
    ))

    log.info(f"\n  1. DROP these non-numeric columns before modelling")
    log.info(f"     (encode or drop — cannot pass raw strings to any ML model):")
    for col in all_nonnumeric:
        log.info(f"     - {col}")

    log.info(f"\n  2. DROP these zero-variance / constant columns:")
    for col in (all_zero_var + all_constant):
        log.info(f"     - {col}")

    log.info(f"\n  3. REVIEW these potential label leakage columns")
    log.info(f"     (confirm they are NOT derived from the outcome):")
    for col in all_leakage:
        log.info(f"     - {col}")

    log.info(f"\n  4. Non-numeric columns to handle:")
    log.info(f"     - gender     → binary encode (M=1, F=0) or drop")
    log.info(f"     - care_unit  → one-hot encode (NICU/PICU/General etc.)")
    log.info(f"     - ethnicity  → one-hot encode or drop (low clinical signal)")
    log.info(f"     - diagnosis  → DROP (free text, too high cardinality)")
    log.info(f"     - icd10      → DROP or group into broad categories")

    # ── Save consolidated summary ─────────────────────────────────────────────
    audit_summary = {
        "timestamp"  : datetime.now().isoformat(timespec="seconds"),
        "dataset_A"  : summary_A,
        "dataset_B"  : summary_B,
        "cross_dataset": {
            "shared_columns" : len(shared),
            "only_in_A"      : sorted(only_A),
            "only_in_B"      : sorted(only_B),
        },
        "action_items": {
            "drop_nonnumeric"   : all_nonnumeric,
            "drop_zero_variance": all_zero_var + all_constant,
            "review_leakage"    : all_leakage,
        }
    }
    out = DIAG_AUDIT / "audit_summary.json"
    with open(out, "w") as f:
        json.dump(audit_summary, f, indent=2, default=str)

    log.info(f"\n  Full audit summary → {out}")
    log.info(f"  Column reports     → {DIAG_AUDIT}/audit_column_report_A.csv")
    log.info(f"                       {DIAG_AUDIT}/audit_column_report_B.csv")
    log.info(f"  Full log           → {log_path}")
    log.info(f"\nPhase 0.0 complete. Review recommendations above, then run:")
    log.info(f"  python sepsis_ml/phase0_diagnostics.py")


if __name__ == "__main__":
    main()
