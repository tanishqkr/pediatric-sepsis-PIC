"""
PIC Sepsis Dataset Preparation
================================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Takes the full cohort CSV (pic_sepsis_cohort.csv) and produces:

1. option_A_full.csv
   Positive  : 985  sepsis patients        (label=1)
   Negative  : 12,914 all non-sepsis       (label=0)
   Use case  : Standard approach, larger dataset, easier task

2. option_B_infection_only.csv
   Positive  : 985  sepsis patients        (label=1)
   Negative  : 2,526 suspected infection   (label=0)
              but Phoenix score < 2
   Use case  : Clinically meaningful, harder task,
               real ICU decision scenario

Missing Value Strategy
----------------------
For columns with <= 20% missing : Median imputation
For columns with >  20% missing : Median imputation
                                  + add binary indicator column
                                    (variable_name_was_measured: 1/0)
                                  This captures the fact that
                                  "not tested = probably not suspected sick"
                                  which is itself a clinical signal.

Identifier columns (subject_id, hadm_id etc.) and binary columns
(symptom flags, vasoactive flags) are NOT imputed.

How to run
----------
    python prepare_datasets.py --input output/pic_sepsis_cohort.csv
                               --output_dir output

Output
------
    output/option_A_full.csv
    output/option_B_infection_only.csv
    output/dataset_summary.txt
"""

import os
import csv
import argparse
import statistics


# =============================================================================
# CONFIGURATION
# =============================================================================

# These columns are identifiers or categorical — never impute
SKIP_IMPUTE = {
    'subject_id', 'hadm_id', 'icustay_id',
    'gender', 'admission_diagnosis', 'icd10_code',
    'sepsis_label', 'suspected_infection',
}

# These columns are already binary (0/1) — no imputation needed
# We detect these automatically by checking if all non-empty values are 0 or 1

# Threshold for adding missingness indicator column
INDICATOR_THRESHOLD = 0.20   # 20% missing -> add _was_measured column


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(s):
    """String to float, None if not possible"""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def is_binary_column(values):
    """
    Check if a column is binary (only contains 0, 1, or empty).
    These don't need imputation.
    """
    non_empty = [v for v in values if v not in ('', 'None', None)]
    if not non_empty:
        return False
    unique = set(non_empty)
    return unique.issubset({'0', '1', '0.0', '1.0'})


def compute_column_stats(rows, col):
    """
    For a given column across all rows, compute:
    - missing count and rate
    - median of non-missing values
    - whether it's binary
    """
    all_vals     = [r.get(col, '') for r in rows]
    numeric_vals = [safe_float(v) for v in all_vals]
    non_null     = [v for v in numeric_vals if v is not None]
    missing_n    = len(all_vals) - len(non_null)
    missing_rate = missing_n / len(all_vals) if all_vals else 0
    median_val   = statistics.median(non_null) if non_null else 0.0
    binary       = is_binary_column(all_vals)

    return {
        'missing_n':    missing_n,
        'missing_rate': missing_rate,
        'median':       median_val,
        'is_binary':    binary,
        'total':        len(all_vals),
    }


# =============================================================================
# MAIN
# =============================================================================

def prepare_datasets(input_csv, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load full cohort
    # ------------------------------------------------------------------
    print("Loading cohort CSV...")
    rows = []
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    print(f"  Loaded {len(rows):,} rows, {len(fieldnames)} columns")

    # ------------------------------------------------------------------
    # Split into groups
    # ------------------------------------------------------------------
    sepsis          = [r for r in rows if r['sepsis_label'] == '1']
    no_infection    = [r for r in rows if r['sepsis_label'] == '0'
                       and r['suspected_infection'] == '0']
    inf_no_sepsis   = [r for r in rows if r['sepsis_label'] == '0'
                       and r['suspected_infection'] == '1']

    print(f"\n  Group breakdown:")
    print(f"    Sepsis (label=1):                    {len(sepsis):>6,}")
    print(f"    Suspected infection, no sepsis:      {len(inf_no_sepsis):>6,}")
    print(f"    No infection suspected:              {len(no_infection):>6,}")

    # Build both datasets BEFORE imputation
    option_a_raw = sepsis + no_infection + inf_no_sepsis  # all rows
    option_b_raw = sepsis + inf_no_sepsis                 # infection-only

    # ------------------------------------------------------------------
    # Compute imputation statistics on FULL cohort
    # (Always fit imputation on full data to avoid data leakage from
    #  using only training data at this stage)
    # ------------------------------------------------------------------
    print("\nComputing column statistics for imputation...")

    col_stats = {}
    numeric_cols = []     # columns that need imputation
    indicator_cols = []   # columns that also need a _was_measured flag

    for col in fieldnames:
        if col in SKIP_IMPUTE:
            continue
        stats = compute_column_stats(rows, col)
        col_stats[col] = stats

        if not stats['is_binary'] and stats['total'] > 0:
            numeric_cols.append(col)
            if stats['missing_rate'] > INDICATOR_THRESHOLD:
                indicator_cols.append(col)

    print(f"  Numeric columns to impute: {len(numeric_cols)}")
    print(f"  Columns with >20% missing (get indicator): {len(indicator_cols)}")

    # Print missing data summary for key clinical columns
    key_cols = [
        'lactate_max', 'platelets_min', 'inr_max', 'ddimer_max',
        'fibrinogen_min', 'map_min', 'creatinine_mgdl_max',
        'bilirubin_mgdl_max', 'spo2_min', 'heart_rate_max',
        'resp_rate_max', 'temperature_max', 'ph_min', 'base_excess_min'
    ]
    print(f"\n  Key column missing rates:")
    print(f"  {'Column':<30} {'Missing':>8} {'Rate':>8} {'Median':>10} {'Indicator':>10}")
    print(f"  {'-'*68}")
    for col in key_cols:
        if col in col_stats:
            s = col_stats[col]
            ind = 'YES' if col in indicator_cols else 'no'
            print(f"  {col:<30} {s['missing_n']:>8,} {s['missing_rate']*100:>7.1f}% "
                  f"{s['median']:>10.3f} {ind:>10}")

    # ------------------------------------------------------------------
    # Apply imputation and build new fieldnames
    # ------------------------------------------------------------------
    print("\nApplying imputation...")

    # New fieldnames: original + indicator columns inserted after their source
    new_fieldnames = []
    for col in fieldnames:
        new_fieldnames.append(col)
        if col in indicator_cols:
            new_fieldnames.append(f'{col}_was_measured')

    def impute_row(row):
        """
        Take one row dict, return new row dict with:
        - Missing numeric values filled with median
        - Indicator columns added for high-missing columns
        """
        new_row = {}
        for col in fieldnames:
            val = row.get(col, '')

            # Skip non-numeric / identifier columns
            if col in SKIP_IMPUTE or col_stats.get(col, {}).get('is_binary', True):
                new_row[col] = val
                # Add indicator if needed
                if col in indicator_cols:
                    new_row[f'{col}_was_measured'] = 0  # binary cols shouldn't be here
                continue

            # Check if missing
            is_missing = (val == '' or val == 'None')

            # Add indicator column if this is a high-missing column
            if col in indicator_cols:
                new_row[f'{col}_was_measured'] = 0 if is_missing else 1

            # Impute if missing
            if is_missing:
                new_row[col] = col_stats[col]['median']
            else:
                new_row[col] = val

        return new_row

    # ------------------------------------------------------------------
    # Write Option A
    # ------------------------------------------------------------------
    print("\nWriting Option A (full dataset)...")
    option_a_path = os.path.join(output_dir, 'option_A_full.csv')
    option_a_imputed = [impute_row(r) for r in option_a_raw]

    with open(option_a_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(option_a_imputed)

    a_sepsis  = sum(1 for r in option_a_imputed if r['sepsis_label'] == '1')
    a_total   = len(option_a_imputed)
    print(f"  Saved {a_total:,} rows -> {option_a_path}")
    print(f"  Sepsis: {a_sepsis:,} ({a_sepsis/a_total*100:.1f}%)")
    print(f"  Non-sepsis: {a_total-a_sepsis:,} ({(a_total-a_sepsis)/a_total*100:.1f}%)")

    # ------------------------------------------------------------------
    # Write Option B
    # ------------------------------------------------------------------
    print("\nWriting Option B (infection-only dataset)...")
    option_b_path = os.path.join(output_dir, 'option_B_infection_only.csv')
    option_b_imputed = [impute_row(r) for r in option_b_raw]

    with open(option_b_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(option_b_imputed)

    b_sepsis  = sum(1 for r in option_b_imputed if r['sepsis_label'] == '1')
    b_total   = len(option_b_imputed)
    print(f"  Saved {b_total:,} rows -> {option_b_path}")
    print(f"  Sepsis: {b_sepsis:,} ({b_sepsis/b_total*100:.1f}%)")
    print(f"  Non-sepsis: {b_total-b_sepsis:,} ({(b_total-b_sepsis)/b_total*100:.1f}%)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = [
        "=" * 60,
        "DATASET PREPARATION SUMMARY",
        "=" * 60,
        "",
        "OPTION A — Full Dataset",
        "-" * 40,
        f"  Total rows:       {a_total:,}",
        f"  Sepsis (pos):     {a_sepsis:,}  ({a_sepsis/a_total*100:.1f}%)",
        f"  Non-sepsis (neg): {a_total-a_sepsis:,}  ({(a_total-a_sepsis)/a_total*100:.1f}%)",
        f"  Class ratio:      1 : {(a_total-a_sepsis)/a_sepsis:.1f}",
        f"  Columns:          {len(new_fieldnames)}",
        f"  Use case:         Baseline modelling, larger dataset",
        "",
        "OPTION B — Infection-Only Dataset",
        "-" * 40,
        f"  Total rows:       {b_total:,}",
        f"  Sepsis (pos):     {b_sepsis:,}  ({b_sepsis/b_total*100:.1f}%)",
        f"  Non-sepsis (neg): {b_total-b_sepsis:,}  ({(b_total-b_sepsis)/b_total*100:.1f}%)",
        f"  Class ratio:      1 : {(b_total-b_sepsis)/b_sepsis:.1f}",
        f"  Columns:          {len(new_fieldnames)}",
        f"  Use case:         Clinically meaningful, real ICU scenario",
        "",
        "=" * 60,
        "MISSING VALUE HANDLING",
        "=" * 60,
        "",
        f"  Strategy: Median imputation for all numeric columns",
        f"  Additional: Columns with >20% missing also get a",
        f"  '_was_measured' binary indicator column",
        f"  Rationale: Missingness is informative in clinical data",
        f"  (sicker patients get more tests)",
        "",
        f"  Columns imputed:                  {len(numeric_cols)}",
        f"  Columns with indicator added:      {len(indicator_cols)}",
        f"  Total columns after preparation:   {len(new_fieldnames)}",
        "",
        "COLUMNS WITH INDICATOR FLAGS ADDED (>20% missing):",
        "-" * 40,
    ]

    for col in sorted(indicator_cols):
        s = col_stats[col]
        summary.append(f"  {col:<40} {s['missing_rate']*100:.1f}% missing")

    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))

    print('\n' + '\n'.join(summary))
    print(f"\nSummary saved -> {summary_path}")
    print("\nDone. Discuss with your team and pick Option A or B for modelling.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare Option A and Option B datasets with imputation'
    )
    parser.add_argument(
        '--input',
        default='output/pic_sepsis_cohort.csv',
        help='Path to full cohort CSV (default: output/pic_sepsis_cohort.csv)'
    )
    parser.add_argument(
        '--output_dir',
        default='output',
        help='Output folder (default: output/)'
    )
    args = parser.parse_args()
    prepare_datasets(args.input, args.output_dir)
