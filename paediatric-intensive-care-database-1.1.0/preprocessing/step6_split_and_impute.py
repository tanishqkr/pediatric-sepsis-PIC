"""
STEP 6 — Train/Test Split and Imputation
==========================================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Takes Option A and Option B from Step 5 and produces
final model-ready datasets with:

  1. Stratified train/test split (80/20)
     - Stratified by sepsis_label to preserve class balance
     - Patient-level split already guaranteed (Step 1 — first stay only)

  2. Median imputation — FITTED ON TRAIN ONLY, applied to both
     This prevents data leakage. The test set never influences
     imputation values.

  3. Missingness indicator columns
     For columns with >20% missing in training set:
     Add binary _was_measured column (1=real value, 0=imputed)
     Captures clinical signal: 'test not ordered = patient less sick'

  4. Column cleanup
     - Drop score columns from features (they contain label info)
       Keep only: sepsis_label, phoenix_core_score, phoenix_8_score
       Drop individual component scores (score_cardiovascular etc.)
       These scores ARE the label — including them as features = leakage
     - Drop blood_glucose_chart (97.6% missing — pure noise)
     - Identifier columns (subject_id, hadm_id, icustay_id) removed
       from model features but kept in a separate metadata file

Output files (per dataset A and B)
-----------------------------------
  option_A_train.csv / option_B_train.csv   — training set, imputed
  option_A_test.csv  / option_B_test.csv    — test set, imputed
  option_A_meta_train.csv / ...             — identifiers for audit
  option_A_imputation_stats.json            — medians used (for reproducibility)
  logs/step6_log.txt                        — full split and imputation summary

Why 80/20 split
---------------
Standard split for datasets of this size (3,000-12,000 rows).
Test set of 20% gives:
  Option A: ~2,490 test rows, ~173 sepsis in test
  Option B: ~633  test rows, ~173 sepsis in test
Sufficient for reliable AUROC estimation.

Why stratified
--------------
Without stratification, random splits can produce test sets with
very different class balance by chance, especially for Option A
where sepsis is only 6.9%. Stratification guarantees the same
6.9% in both train and test.
"""

import os
import csv
import json
import math
import random
import statistics
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

DATASETS = {
    'A': os.path.join("output2", "option_A_full.csv"),
    'B': os.path.join("output2", "option_B_infection_only.csv"),
}

OUTPUT_DIR        = "output2"
LOG_DIR           = os.path.join(OUTPUT_DIR, "logs")
LOG_FILE          = os.path.join(LOG_DIR, "step6_log.txt")

TRAIN_RATIO       = 0.80
RANDOM_SEED       = 42        # fixed seed for reproducibility
INDICATOR_THRESH  = 0.20      # columns with >20% missing get indicator column

# Columns to drop from model features
# Score columns contain label information — including = leakage
SCORE_COLS_DROP   = [
    'score_respiratory', 'score_cardiovascular', 'score_coagulation',
    'score_neurological', 'score_endocrine', 'score_immunologic',
    'score_renal', 'score_hepatic',
]

# Identifier columns — kept in metadata, removed from features
IDENTIFIER_COLS   = ['subject_id', 'hadm_id', 'icustay_id']

# Columns to exclude entirely (too sparse or redundant)
EXCLUDE_COLS      = [
    # blood_glucose_chart — 97.6% missing, no signal
    'blood_glucose_chart_max', 'blood_glucose_chart_min',
    'blood_glucose_chart_mean', 'blood_glucose_chart_first',
    'blood_glucose_chart_count',
]

# Columns that are non-numeric (categorical or text) — no imputation
NON_NUMERIC_COLS  = [
    'gender', 'care_unit', 'diagnosis', 'icd10', 'ethnicity',
    'sepsis_label', 'suspected_infection',
    'phoenix_core_score', 'phoenix_8_score',
    'expire_flag', 'hosp_expire',
    'had_surgery_flag', 'had_cardiac_surgery_flag',
]
# Also all vaso_ and symptom_ columns are binary 0/1 — no imputation needed
# Detected automatically


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(s):
    if s is None or str(s).strip() in ('', 'None'): return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except: return None


def is_binary_col(values):
    """True if column only contains 0, 1, or missing."""
    non_empty = [v for v in values if v not in ('', 'None', None)]
    if not non_empty: return False
    unique = set(str(v) for v in non_empty)
    return unique.issubset({'0', '1', '0.0', '1.0'})


def stratified_split(rows, label_col, train_ratio, seed):
    """
    Stratified train/test split.
    Ensures same class balance in train and test.
    Returns (train_rows, test_rows).
    """
    random.seed(seed)

    # Group by label
    by_label = defaultdict(list)
    for r in rows:
        by_label[r.get(label_col, '0')].append(r)

    train, test = [], []
    for label, group in by_label.items():
        random.shuffle(group)
        n_train = int(len(group) * train_ratio)
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    # Shuffle final sets so labels are not sorted
    random.shuffle(train)
    random.shuffle(test)

    return train, test


def compute_imputation_stats(train_rows, numeric_cols):
    """
    Compute median for each numeric column from training rows only.
    Returns dict: col -> {median, missing_rate, n_present, add_indicator}
    """
    stats = {}
    n = len(train_rows)

    for col in numeric_cols:
        vals = []
        for r in train_rows:
            v = safe_float(r.get(col))
            if v is not None:
                vals.append(v)

        missing_n    = n - len(vals)
        missing_rate = missing_n / n if n > 0 else 0
        median_val   = statistics.median(vals) if vals else 0.0

        stats[col] = {
            'median':        round(median_val, 6),
            'missing_rate':  round(missing_rate, 4),
            'n_present':     len(vals),
            'n_missing':     missing_n,
            'add_indicator': missing_rate > INDICATOR_THRESH,
        }

    return stats


def apply_imputation(rows, impute_stats, numeric_cols):
    """
    Apply median imputation to rows.
    For columns with add_indicator=True, also add _was_measured column.
    Returns (imputed_rows, new_fieldnames_additions).
    """
    indicator_cols = [c for c in numeric_cols
                      if impute_stats[c]['add_indicator']]

    imputed = []
    for row in rows:
        new_row = dict(row)

        for col in numeric_cols:
            val = safe_float(row.get(col))
            is_missing = val is None

            # Add indicator if needed
            if impute_stats[col]['add_indicator']:
                new_row[f'{col}_was_measured'] = 0 if is_missing else 1

            # Impute if missing
            if is_missing:
                new_row[col] = impute_stats[col]['median']

        imputed.append(new_row)

    return imputed, indicator_cols


def write_csv(filepath, rows, fieldnames):
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN PER-DATASET PROCESSING
# =============================================================================

def process_dataset(name, input_csv, log_lines):

    print(f"\n--- Processing Option {name} ---")
    log_lines.append(f"\n{'='*65}")
    log_lines.append(f"OPTION {name}: {input_csv}")
    log_lines.append(f"{'='*65}")

    # Load
    rows = []
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_cols = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    n = len(rows)
    n_pos = sum(1 for r in rows if r.get('sepsis_label') == '1')
    print(f"  Loaded {n:,} rows ({n_pos:,} sepsis, {n-n_pos:,} non-sepsis)")

    # ── Column classification ─────────────────────────────────────────────────
    # Determine which columns get imputation, which are metadata, which are dropped

    drop_cols    = set(SCORE_COLS_DROP + EXCLUDE_COLS + IDENTIFIER_COLS)
    meta_cols    = IDENTIFIER_COLS + ['sepsis_label']

    # Detect binary columns automatically
    binary_cols  = set()
    for col in all_cols:
        if col in drop_cols or col in NON_NUMERIC_COLS:
            continue
        vals = [r.get(col, '') for r in rows[:500]]  # sample first 500
        if is_binary_col(vals):
            binary_cols.add(col)

    # Numeric columns = all columns that are not identifiers, not dropped,
    # not non-numeric, not binary
    numeric_cols = [
        col for col in all_cols
        if col not in drop_cols
        and col not in NON_NUMERIC_COLS
        and col not in binary_cols
        and col not in IDENTIFIER_COLS
    ]

    log_lines += [
        f"Column breakdown:",
        f"  Total columns in input:     {len(all_cols)}",
        f"  Dropped (score leakage):    {len(SCORE_COLS_DROP)}",
        f"  Dropped (too sparse):       {len(EXCLUDE_COLS)}",
        f"  Identifier (metadata only): {len(IDENTIFIER_COLS)}",
        f"  Binary (no imputation):     {len(binary_cols)}",
        f"  Non-numeric categorical:    {len(NON_NUMERIC_COLS)}",
        f"  Numeric (will impute):      {len(numeric_cols)}",
        "",
    ]

    # ── Stratified split ──────────────────────────────────────────────────────
    train_rows, test_rows = stratified_split(
        rows, 'sepsis_label', TRAIN_RATIO, RANDOM_SEED
    )

    train_pos = sum(1 for r in train_rows if r.get('sepsis_label') == '1')
    test_pos  = sum(1 for r in test_rows  if r.get('sepsis_label') == '1')

    log_lines += [
        f"Train/Test Split (stratified, seed={RANDOM_SEED}):",
        f"  Train: {len(train_rows):,} rows  "
        f"({train_pos:,} sepsis = {train_pos/len(train_rows)*100:.1f}%)",
        f"  Test:  {len(test_rows):,} rows  "
        f"({test_pos:,} sepsis = {test_pos/len(test_rows)*100:.1f}%)",
        "",
    ]

    # ── Compute imputation stats on TRAIN only ────────────────────────────────
    print(f"  Computing imputation medians from training set...")
    impute_stats = compute_imputation_stats(train_rows, numeric_cols)

    n_with_indicator = sum(
        1 for s in impute_stats.values() if s['add_indicator']
    )
    log_lines += [
        f"Imputation (fitted on train only):",
        f"  Numeric columns imputed:          {len(numeric_cols)}",
        f"  Columns with >20% missing         {n_with_indicator}",
        f"  (these get _was_measured indicator)",
        "",
    ]

    # Log high-missing columns
    high_missing = [
        (col, s['missing_rate'], s['median'])
        for col, s in impute_stats.items()
        if s['missing_rate'] > 0.20
    ]
    high_missing.sort(key=lambda x: -x[1])
    log_lines.append(
        f"  {'Column':<35} {'Missing%':>9} {'Median':>10}"
    )
    log_lines.append("  " + "-"*56)
    for col, rate, med in high_missing[:20]:
        log_lines.append(
            f"  {col:<35} {rate*100:>8.1f}% {med:>10.4f}"
        )
    if len(high_missing) > 20:
        log_lines.append(f"  ... and {len(high_missing)-20} more")
    log_lines.append("")

    # ── Apply imputation ──────────────────────────────────────────────────────
    print(f"  Applying imputation...")
    train_imp, indicator_cols = apply_imputation(
        train_rows, impute_stats, numeric_cols
    )
    test_imp, _               = apply_imputation(
        test_rows,  impute_stats, numeric_cols
    )

    # ── Build final fieldnames ────────────────────────────────────────────────
    # Start with all columns except dropped ones and identifiers
    feature_cols = [
        col for col in all_cols
        if col not in drop_cols
        and col not in IDENTIFIER_COLS
    ]

    # Insert indicator columns right after their source column
    final_cols = []
    for col in feature_cols:
        final_cols.append(col)
        if col in numeric_cols and impute_stats[col]['add_indicator']:
            final_cols.append(f'{col}_was_measured')

    # Metadata cols for audit file
    meta_fieldnames = IDENTIFIER_COLS + ['sepsis_label',
                                          'phoenix_core_score',
                                          'phoenix_8_score']

    # ── Write files ───────────────────────────────────────────────────────────
    prefix = f"option_{name}"

    train_path = os.path.join(OUTPUT_DIR, f"{prefix}_train.csv")
    test_path  = os.path.join(OUTPUT_DIR, f"{prefix}_test.csv")
    meta_train = os.path.join(OUTPUT_DIR, f"{prefix}_meta_train.csv")
    meta_test  = os.path.join(OUTPUT_DIR, f"{prefix}_meta_test.csv")
    stats_path = os.path.join(LOG_DIR,    f"{prefix}_imputation_stats.json")

    write_csv(train_path, train_imp, final_cols)
    write_csv(test_path,  test_imp,  final_cols)
    write_csv(meta_train, train_rows, meta_fieldnames)
    write_csv(meta_test,  test_rows,  meta_fieldnames)

    # Save imputation stats as JSON for reproducibility
    with open(stats_path, 'w') as f:
        json.dump(impute_stats, f, indent=2)

    total_cols = len(final_cols)
    log_lines += [
        f"Output files:",
        f"  Train: {train_path}",
        f"    Rows: {len(train_imp):,}   Columns: {total_cols}",
        f"  Test:  {test_path}",
        f"    Rows: {len(test_imp):,}   Columns: {total_cols}",
        f"  Metadata train: {meta_train}",
        f"  Metadata test:  {meta_test}",
        f"  Imputation stats: {stats_path}",
        "",
    ]

    print(f"  Saved train ({len(train_imp):,} rows) and "
          f"test ({len(test_imp):,} rows), {total_cols} columns each")

    return {
        'n_train':      len(train_imp),
        'n_test':       len(test_imp),
        'n_cols':       total_cols,
        'train_pos':    train_pos,
        'test_pos':     test_pos,
    }


# =============================================================================
# MAIN
# =============================================================================

def run_step6():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    print("\n" + "="*60)
    print("STEP 6: Train/Test Split and Imputation")
    print("="*60)

    log_lines = [
        "=" * 65,
        "STEP 6 — Train/Test Split and Imputation",
        "=" * 65,
        "",
        f"Split ratio:       {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}",
        f"Random seed:       {RANDOM_SEED}",
        f"Indicator thresh:  >{INDICATOR_THRESH*100:.0f}% missing",
        f"Score cols dropped: {len(SCORE_COLS_DROP)} "
        f"(prevent label leakage into features)",
        "",
    ]

    results = {}
    for name, path in DATASETS.items():
        results[name] = process_dataset(name, path, log_lines)

    # Final summary
    log_lines += [
        "=" * 65,
        "STEP 6 SUMMARY",
        "=" * 65,
        "",
        f"{'Dataset':<12} {'Train':>8} {'Test':>8} {'Cols':>6} "
        f"{'Train+pos':>10} {'Test+pos':>9}",
        "-" * 57,
    ]
    for name, r in results.items():
        log_lines.append(
            f"Option {name:<6} {r['n_train']:>8,} {r['n_test']:>8,} "
            f"{r['n_cols']:>6}  {r['train_pos']:>8,}     {r['test_pos']:>7,}"
        )

    log_lines += [
        "",
        "IMPORTANT NOTES",
        "  1. Imputation medians computed on TRAIN set only — no leakage",
        "  2. Same medians applied to TEST set",
        "  3. _was_measured indicator columns added for >20% missing cols",
        "  4. Score component columns dropped — they encode the label",
        "  5. Identifier columns in meta files only — not in model features",
        "  6. Random seed fixed at 42 — fully reproducible",
        "",
        "READY FOR MODELLING",
        "  Primary model: CatBoost (handles categoricals natively)",
        "  Comparison:    XGBoost, LightGBM, Random Forest,",
        "                 Logistic Regression, MLP",
        "  Explainability: SHAP TreeExplainer on best model",
        "",
        "Next: Step 7 — Model training and comparison",
    ]

    log_str = '\n'.join(log_lines)
    print('\n' + log_str)

    with open(LOG_FILE, 'w') as f:
        f.write(log_str)
    print(f"\nLog saved -> {LOG_FILE}")
    print("\nStep 6 complete. Preprocessing pipeline finished.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step6()
