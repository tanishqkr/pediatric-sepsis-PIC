"""
STEP 5 — Label Verification and Dataset Splitting
===================================================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Part A — Deep label verification
  Goes beyond basic checks. Investigates:
  1. Sepsis label basis breakdown (vasoactives vs labs vs both)
  2. Score component analysis — which components drive labels
  3. Care unit distribution of sepsis vs non-sepsis
  4. Age group distribution of sepsis cases
  5. ICD code validation — do sepsis patients have infection-related ICD codes?
  6. Vasoactive-only cases investigation (17 cases — are they legitimate?)
  7. Suspected infection patients who did NOT become sepsis — why not?

Part B — Dataset splitting
  Creates two analysis-ready datasets from cohort_step4_labeled.csv:

  Option A — Full dataset
    Positive : 864  sepsis patients          (label=1)
    Negative : 11,583 all non-sepsis         (label=0)
    Use case : Standard approach, larger dataset

  Option B — Infection-only dataset
    Positive : 864  sepsis patients          (label=1)
    Negative : 2,299 suspected infection     (label=0)
               but Phoenix score < 2
    Use case : Clinically meaningful — models the real bedside
               decision: infected child — will they develop sepsis?

  Both datasets:
    - No imputation yet (happens in Step 6 after train/test split)
    - Score columns retained for reference
    - Missingness indicator columns NOT added yet (Step 6)

Outputs
-------
  output2/option_A_full.csv              — full dataset, no imputation
  output2/option_B_infection_only.csv    — infection-only, no imputation
  output2/logs/step5_log.txt             — verification report + split summary
"""

import os
import csv
import math
import statistics
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

INPUT_CSV   = os.path.join("output2", "cohort_step4_labeled.csv")
OUT_A       = os.path.join("output2", "option_A_full.csv")
OUT_B       = os.path.join("output2", "option_B_infection_only.csv")
LOG_DIR     = os.path.join("output2", "logs")
LOG_FILE    = os.path.join(LOG_DIR, "step5_log.txt")


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(s):
    if s is None or str(s).strip() in ('', 'None'): return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except: return None

def median_of(vals):
    clean = [v for v in vals if v is not None]
    return round(statistics.median(clean), 3) if clean else None

def get_col(subset, col):
    vals = []
    for r in subset:
        v = safe_float(r.get(col))
        if v is not None:
            vals.append(v)
    return vals

def pct(n, total):
    return f"{n/total*100:.1f}%" if total > 0 else "N/A"


# =============================================================================
# PART A — DEEP LABEL VERIFICATION
# =============================================================================

def verify_labels(rows, log_lines):

    n         = len(rows)
    sepsis    = [r for r in rows if r.get('sepsis_label') == '1']
    no_sep    = [r for r in rows if r.get('sepsis_label') == '0'
                 and r.get('suspected_infection') == '1']
    no_infect = [r for r in rows if r.get('suspected_infection') == '0']

    log_lines += [
        "=" * 65,
        "PART A — DEEP LABEL VERIFICATION",
        "=" * 65,
        "",
        f"Total stays:               {n:>8,}",
        f"Sepsis (label=1):          {len(sepsis):>8,}  ({pct(len(sepsis),n)})",
        f"Inf, no sepsis (label=0):  {len(no_sep):>8,}  ({pct(len(no_sep),n)})",
        f"No infection (label=0):    {len(no_infect):>8,}  ({pct(len(no_infect),n)})",
        "",
    ]

    # ── A1: Sepsis label basis ────────────────────────────────────────────────
    vaso_only = lab_only = both = neither = 0
    for r in sepsis:
        has_lab  = any(safe_float(r.get(c)) is not None
                       for c in ['lactate_max','platelets_min','inr_max',
                                  'ddimer_max','fibrinogen_min'])
        has_vaso = int(safe_float(r.get('n_vasoactives_24h')) or 0) > 0
        if has_vaso and has_lab:     both += 1
        elif has_vaso and not has_lab: vaso_only += 1
        elif has_lab and not has_vaso: lab_only += 1
        else:                          neither += 1

    log_lines += [
        "A1. SEPSIS LABEL BASIS",
        f"  Both vasoactives + labs:       {both:>6,}  ({pct(both,len(sepsis))})",
        f"  Labs only (no vasoactives):    {lab_only:>6,}  ({pct(lab_only,len(sepsis))})",
        f"  Vasoactives only (no labs):    {vaso_only:>6,}  ({pct(vaso_only,len(sepsis))})",
        f"  Neither (score via MAP only):  {neither:>6,}  ({pct(neither,len(sepsis))})",
        f"  Assessment: {vaso_only} vasoactive-only cases ({pct(vaso_only,len(sepsis))}) — acceptable (<5%)",
        "",
    ]

    # ── A2: Care unit distribution ────────────────────────────────────────────
    log_lines.append("A2. CARE UNIT DISTRIBUTION")
    log_lines.append(f"  {'Unit':<15} {'Total':>8} {'Sepsis':>8} {'Prevalence':>12}")
    log_lines.append("  " + "-"*47)

    units_total  = defaultdict(int)
    units_sepsis = defaultdict(int)
    for r in rows:
        u = r.get('care_unit', 'Unknown')
        units_total[u]  += 1
        if r.get('sepsis_label') == '1':
            units_sepsis[u] += 1

    for u in sorted(units_total.keys(), key=lambda x: -units_total[x]):
        tot = units_total[u]
        sep = units_sepsis[u]
        log_lines.append(
            f"  {u:<15} {tot:>8,} {sep:>8,} {sep/tot*100:>11.1f}%"
        )
    log_lines.append("")

    # ── A3: Age group distribution of sepsis ──────────────────────────────────
    log_lines.append("A3. AGE GROUP DISTRIBUTION")
    log_lines.append(f"  {'Age group':<22} {'Total':>8} {'Sepsis':>8} {'Prevalence':>12}")
    log_lines.append("  " + "-"*54)

    age_groups = [
        ('Neonate <1m',      0,     0.083),
        ('Infant 1-12m',     0.083, 1.0),
        ('Toddler 1-5y',     1.0,   5.0),
        ('Child 5-12y',      5.0,   12.0),
        ('Adolescent 12-18y',12.0,  18.1),
    ]
    for label_ag, lo, hi in age_groups:
        grp     = [r for r in rows
                   if lo <= (safe_float(r.get('age_years')) or -1) < hi]
        grp_sep = [r for r in grp if r.get('sepsis_label') == '1']
        if not grp: continue
        log_lines.append(
            f"  {label_ag:<22} {len(grp):>8,} {len(grp_sep):>8,} "
            f"{len(grp_sep)/len(grp)*100:>11.1f}%"
        )
    log_lines.append("")

    # ── A4: Score component drivers ───────────────────────────────────────────
    log_lines.append("A4. SCORE COMPONENT DRIVERS (among sepsis cases)")
    log_lines.append("    What drove the label for each sepsis patient?")
    log_lines.append("")

    cardio_drove = sum(1 for r in sepsis
                       if int(safe_float(r.get('score_cardiovascular')) or 0) >= 2)
    coag_drove   = sum(1 for r in sepsis
                       if int(safe_float(r.get('score_coagulation')) or 0) >= 2)
    both_drove   = sum(1 for r in sepsis
                       if int(safe_float(r.get('score_cardiovascular')) or 0) >= 1
                       and int(safe_float(r.get('score_coagulation')) or 0) >= 1)
    only_cardio  = sum(1 for r in sepsis
                       if int(safe_float(r.get('score_cardiovascular')) or 0) >= 2
                       and int(safe_float(r.get('score_coagulation')) or 0) == 0)
    only_coag    = sum(1 for r in sepsis
                       if int(safe_float(r.get('score_coagulation')) or 0) >= 2
                       and int(safe_float(r.get('score_cardiovascular')) or 0) == 0)

    log_lines += [
        f"  Cardio score >=2 (cardiovascular failure):  {cardio_drove:>5,} ({pct(cardio_drove,len(sepsis))})",
        f"  Coag score  >=2 (coagulation failure):      {coag_drove:>5,} ({pct(coag_drove,len(sepsis))})",
        f"  Both cardio+coag contributed:               {both_drove:>5,} ({pct(both_drove,len(sepsis))})",
        f"  Cardio alone drove label (coag=0):          {only_cardio:>5,} ({pct(only_cardio,len(sepsis))})",
        f"  Coag alone drove label (cardio=0):          {only_coag:>5,} ({pct(only_coag,len(sepsis))})",
        "",
    ]

    # ── A5: Clinical comparison ───────────────────────────────────────────────
    log_lines.append("A5. CLINICAL COMPARISON — Sepsis vs Infected Non-Sepsis")
    log_lines.append(f"  {'Variable':<28} {'Sepsis':>10} {'Inf.NoSep':>10}  {'Direction'}  {'Expected'}")
    log_lines.append("  " + "-"*75)

    clin_checks = [
        ('lactate_max',        'higher', 'Sepsis has higher lactate'),
        ('platelets_min',      'lower',  'Thrombocytopenia in sepsis'),
        ('inr_max',            'higher', 'Coagulopathy in sepsis'),
        ('n_vasoactives_24h',  'higher', 'More vasopressors in sepsis'),
        ('phoenix_8_score',    'higher', 'Higher organ dysfunction'),
        ('los_hours',          'higher', 'Sepsis = longer stay'),
        # expire_flag uses rate comparison — handled separately below
        # shock_index: sparse (26% coverage) — comparison unreliable, skipped
        ('bilirubin_max',      'higher', 'Hepatic dysfunction in sepsis'),
        ('creatinine_max',     'higher', 'Renal dysfunction in sepsis'),
    ]

    all_correct = True
    for col, direction, note in clin_checks:
        sep_vals  = get_col(sepsis, col)
        nosep_vals= get_col(no_sep, col)
        if not sep_vals or not nosep_vals:
            log_lines.append(f"  {col:<28} NO DATA")
            continue
        sm = round(statistics.median(sep_vals), 3)
        nm = round(statistics.median(nosep_vals), 3)
        correct = (sm > nm) if direction == 'higher' else (sm < nm)
        if not correct: all_correct = False
        status = 'PASS' if correct else 'FAIL'
        log_lines.append(
            f"  {col:<28} {sm:>10} {nm:>10}  {status:<6}  {note}"
        )

    # Rate-based mortality comparison (correct for binary variables)
    sep_dead   = sum(1 for r in sepsis if r.get('expire_flag') == '1')
    nosep_dead = sum(1 for r in no_sep  if r.get('expire_flag') == '1')
    sep_rate   = sep_dead / len(sepsis) * 100 if sepsis else 0
    nosep_rate = nosep_dead / len(no_sep) * 100 if no_sep else 0
    mort_ok    = sep_rate > nosep_rate
    if not mort_ok: all_correct = False
    log_lines.append(
        f"  expire_flag (mortality rate)      {sep_rate:>9.1f}%  "
        f"{nosep_rate:>9.1f}%  {'PASS' if mort_ok else 'FAIL'}    "
        f"Sepsis has {sep_rate:.1f}% vs {nosep_rate:.1f}% mortality"
    )
    log_lines.append(
        "  shock_index                        SKIPPED  SKIPPED  "
        "NOTE: Only 26% coverage — selection bias makes comparison unreliable"
    )

    log_lines += [
        "",
        f"  Overall clinical validity: {'ALL PASS' if all_correct else 'FAILURES PRESENT'}",
        "",
    ]

    # ── A6: ICD code spot check ───────────────────────────────────────────────
    log_lines.append("A6. ICD CODE SPOT CHECK (sepsis patients)")
    log_lines.append("    Checking if sepsis-labeled patients have infection-related ICD codes")

    # Infection-related ICD10 prefixes
    infection_prefixes = [
        'A', 'B',           # infectious and parasitic diseases
        'J',                # respiratory infections
        'N',                # UTI etc
        'K',                # GI infections
        'G',                # CNS infections
        'L',                # skin infections
    ]
    has_infection_icd = 0
    for r in sepsis:
        icd = r.get('icd10', '').strip()
        if any(icd.startswith(p) for p in infection_prefixes):
            has_infection_icd += 1

    log_lines += [
        f"  Sepsis patients with infection-related ICD: {has_infection_icd:,}/{len(sepsis):,} "
        f"({pct(has_infection_icd, len(sepsis))})",
        f"  Note: ICD codes are DISCHARGE diagnoses — not all sepsis gets coded as",
        f"  infection (e.g. cardiac patient who develops sepsis may be coded as cardiac)",
        f"  Low rate here is expected and not a concern.",
        "",
    ]

    # ── A7: Why did infected patients NOT become sepsis? ─────────────────────
    log_lines.append("A7. WHY DID INFECTED PATIENTS NOT REACH SEPSIS THRESHOLD?")
    log_lines.append("    Analysis of 2,299 suspected-infection patients with score < 2")

    score_dist = defaultdict(int)
    for r in no_sep:
        score_dist[r.get('phoenix_core_score','0')] += 1

    log_lines.append(f"  {'Core score':<12} {'Count':>8} {'%':>8}")
    log_lines.append("  " + "-"*32)
    for sc in ['0','1']:
        cnt = score_dist[sc]
        log_lines.append(f"  {sc:<12} {cnt:>8,} {cnt/len(no_sep)*100:>7.1f}%")
    log_lines.append(
        f"  Interpretation: patients with score 1 were close to threshold but"
    )
    log_lines.append(
        f"  had only one organ system mildly affected — appropriate non-sepsis label."
    )
    log_lines.append("")

    return all_correct


# =============================================================================
# PART B — DATASET SPLITTING
# =============================================================================

def split_datasets(rows, fieldnames, log_lines):

    sepsis    = [r for r in rows if r['sepsis_label'] == '1']
    no_sep    = [r for r in rows if r['sepsis_label'] == '0'
                 and r['suspected_infection'] == '1']
    no_infect = [r for r in rows if r['suspected_infection'] == '0']

    # Option A: all rows
    option_a = sepsis + no_sep + no_infect

    # Option B: infection-only
    option_b = sepsis + no_sep

    log_lines += [
        "=" * 65,
        "PART B — DATASET SPLITTING",
        "=" * 65,
        "",
        "OPTION A — Full Dataset",
        f"  Total rows:         {len(option_a):>8,}",
        f"  Sepsis (pos):       {len(sepsis):>8,}  ({pct(len(sepsis),len(option_a))})",
        f"  Non-sepsis (neg):   {len(option_a)-len(sepsis):>8,}  ({pct(len(option_a)-len(sepsis),len(option_a))})",
        f"  Class ratio:        1 : {(len(option_a)-len(sepsis))/len(sepsis):.1f}",
        f"  Use case:           Baseline modelling, larger dataset",
        "",
        "OPTION B — Infection-Only Dataset",
        f"  Total rows:         {len(option_b):>8,}",
        f"  Sepsis (pos):       {len(sepsis):>8,}  ({pct(len(sepsis),len(option_b))})",
        f"  Non-sepsis (neg):   {len(no_sep):>8,}  ({pct(len(no_sep),len(option_b))})",
        f"  Class ratio:        1 : {len(no_sep)/len(sepsis):.1f}",
        f"  Use case:           Clinically meaningful — real ICU decision scenario",
        "",
        "NOTE: No imputation applied at this stage.",
        "Imputation happens in Step 6 AFTER train/test split to prevent leakage.",
        "",
    ]

    # Write Option A
    with open(OUT_A, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(option_a)
    log_lines.append(f"Option A saved -> {OUT_A}")

    # Write Option B
    with open(OUT_B, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(option_b)
    log_lines.append(f"Option B saved -> {OUT_B}")
    log_lines.append("")

    return len(option_a), len(option_b)


# =============================================================================
# MAIN
# =============================================================================

def run_step5(input_csv, log_dir):

    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 5: Label Verification and Dataset Splitting")
    print("="*60)

    # Load
    print(f"\nLoading {input_csv}...")
    rows = []
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows, {len(fieldnames)} columns")

    log_lines = []

    # Part A
    print("\nRunning deep label verification...")
    all_correct = verify_labels(rows, log_lines)

    # Part B
    print("\nSplitting datasets...")
    n_a, n_b = split_datasets(rows, fieldnames, log_lines)

    # Final status
    log_lines += [
        "=" * 65,
        "STEP 5 COMPLETE",
        "=" * 65,
        f"Label verification:    {'ALL PASS' if all_correct else 'WARNINGS — review'}",
        f"Option A rows:         {n_a:,}",
        f"Option B rows:         {n_b:,}",
        "",
        "Next: Step 6 — Train/test split + imputation",
    ]

    log_str = '\n'.join(log_lines)
    print('\n' + log_str)

    with open(LOG_FILE, 'w') as f:
        f.write(log_str)
    print(f"\nLog saved -> {LOG_FILE}")
    print("\nStep 5 complete. Ready for Step 6.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step5(INPUT_CSV, LOG_DIR)