"""
STEP 4 — Phoenix Sepsis Labeling
==================================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Reads cohort_step3.2_features.csv (feature matrix without labels).
Computes the Phoenix Sepsis Score for each ICU stay using the worst
values already extracted in Step 3.
Assigns sepsis label: 1 if suspected infection AND Phoenix core score >= 2.

Phoenix Criteria References
----------------------------
  Sanchez-Pinto LN et al. Development and Validation of the Phoenix
  Criteria for Pediatric Sepsis and Septic Shock. JAMA. 2024;331(8):675-686.
  https://doi.org/10.1001/jama.2024.0196

  Schlapbach LJ et al. International Consensus Criteria for Pediatric
  Sepsis and Septic Shock. JAMA. 2024;331(8):665-674.
  https://doi.org/10.1001/jama.2024.0179

Phoenix core score = sum of 4 components (max possible = 13):
  1. Respiratory    (0-3 pts) — SET TO 0: FiO2/ventilation absent in PIC
  2. Cardiovascular (0-6 pts) — lactate + MAP + vasoactives
  3. Coagulation    (0-2 pts) — platelets + INR + D-Dimer + fibrinogen
  4. Neurological   (0-2 pts) — SET TO 0: GCS absent in PIC

Phoenix-8 score adds 4 more:
  5. Endocrine      (0-1 pt)  — glucose
  6. Immunologic    (0-1 pt)  — ANC + ALC
  7. Renal          (0-1 pt)  — creatinine (age-adjusted)
  8. Hepatic        (0-1 pt)  — bilirubin + ALT

Missing data convention (per Phoenix paper)
-------------------------------------------
  Missing values = 0 points. Do NOT impute before scoring.
  Rationale: if a test was not ordered, clinical team likely had
  no concern about that organ system. Absence of testing is
  informative and should not artificially inflate the score.

Which values are used for scoring
-----------------------------------
  Scoring uses the WORST value in the 24h window per Phoenix paper.
  These are already available in Step 3.2 features:
    high=bad variables: use _max  (lactate, INR, D-dimer, bilirubin, ALT, glucose)
    low=bad variables:  use _min  (platelets, fibrinogen, MAP, ANC, ALC, creatinine)
  n_vasoactives_24h is already the count of distinct vasoactive drugs given.

Outputs
-------
  output2/cohort_step4_labeled.csv   — full feature matrix WITH label columns
  output2/logs/step4_log.txt         — scoring summary and label breakdown
"""

import os
import csv
import math
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

INPUT_CSV  = os.path.join("output2", "cohort_step3.2_features.csv")
OUTPUT_CSV = os.path.join("output2", "cohort_step4_labeled.csv")
LOG_DIR    = os.path.join("output2", "logs")
LOG_FILE   = os.path.join(LOG_DIR, "step4_log.txt")


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(s):
    """String to float, None if missing or invalid."""
    if s is None or str(s).strip() in ('', 'None'):
        return None
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def safe_int(s):
    """String to int, 0 if missing."""
    v = safe_float(s)
    return int(v) if v is not None else 0


# =============================================================================
# PHOENIX SCORING FUNCTIONS
# =============================================================================

def score_respiratory():
    """
    Respiratory score (0-3 pts).
    NOT COMPUTABLE in PIC:
      - FiO2 not recorded
      - PaO2/FiO2 ratio not computable
      - SpO2/FiO2 ratio not computable
      - Mechanical ventilation flag absent
    Returns 0 per Phoenix missing-data convention.
    """
    return 0


def score_cardiovascular(lactate_max, map_min, n_vasoactives, age_months):
    """
    Cardiovascular score (0-6 pts).
    Sum of three sub-components:

    Vasoactives (0-2 pts):
      0 drugs  = 0 pts
      1 drug   = 1 pt
      2+ drugs = 2 pts

    Lactate mmol/L (0-2 pts):
      < 5      = 0 pts
      5-10.9   = 1 pt
      >= 11    = 2 pts

    MAP mmHg age-adjusted (0-2 pts):
      Source: Sanchez-Pinto et al. 2024 Table 1
    """
    score = 0

    # Sub-component 1: Vasoactives
    n = n_vasoactives if n_vasoactives is not None else 0
    if n >= 2:
        score += 2
    elif n == 1:
        score += 1

    # Sub-component 2: Lactate (mmol/L)
    if lactate_max is not None:
        if lactate_max >= 11.0:
            score += 2
        elif lactate_max >= 5.0:
            score += 1

    # Sub-component 3: MAP (mmHg) — age-adjusted thresholds
    # Age in months. Thresholds: (normal_floor, critical_floor)
    # Below normal_floor = 1 pt, below critical_floor = 2 pts
    if map_min is not None and age_months is not None:
        if age_months < 1:
            normal_floor, critical_floor = 31, 17
        elif age_months < 12:
            normal_floor, critical_floor = 39, 25
        elif age_months < 24:
            normal_floor, critical_floor = 44, 31
        elif age_months < 60:
            normal_floor, critical_floor = 45, 32
        elif age_months < 144:
            normal_floor, critical_floor = 49, 36
        else:
            normal_floor, critical_floor = 52, 38

        if map_min < critical_floor:
            score += 2
        elif map_min < normal_floor:
            score += 1

    return score


def score_coagulation(platelets_min, inr_max, ddimer_max, fibrinogen_min):
    """
    Coagulation score (0-2 pts max).
    1 point per abnormal lab, maximum 2 points total:
      Platelets  < 100 x10^9/L  -> 1 pt
      INR        > 1.3           -> 1 pt
      D-Dimer    > 2 mg/L FEU   -> 1 pt
      Fibrinogen < 1.0 g/L      -> 1 pt
    """
    score = 0
    if platelets_min is not None and platelets_min < 100:
        score += 1
    if inr_max is not None and inr_max > 1.3:
        score += 1
    if ddimer_max is not None and ddimer_max > 2.0:
        score += 1
    if fibrinogen_min is not None and fibrinogen_min < 1.0:
        score += 1
    return min(score, 2)


def score_neurological():
    """
    Neurological score (0-2 pts).
    NOT COMPUTABLE in PIC:
      - GCS not recorded in CHARTEVENTS
      - Bilateral fixed pupils not recorded
    Returns 0 per Phoenix missing-data convention.
    """
    return 0


def score_endocrine(glucose_max, glucose_min):
    """
    Endocrine score (0-1 pt).
    Blood glucose < 50 mg/dL OR > 150 mg/dL -> 1 pt.
    We check both max (hyperglycaemia) and min (hypoglycaemia).
    """
    if glucose_min is not None and glucose_min < 50:
        return 1
    if glucose_max is not None and glucose_max > 150:
        return 1
    return 0


def score_immunologic(anc_min, alc_min):
    """
    Immunologic score (0-1 pt).
    ANC < 0.5 x10^9/L  -> 1 pt
    ALC < 1.0 x10^9/L  -> 1 pt
    Either condition sufficient for 1 pt.
    """
    if anc_min is not None and anc_min < 0.5:
        return 1
    if alc_min is not None and alc_min < 1.0:
        return 1
    return 0


def score_renal(creatinine_max, age_months):
    """
    Renal score (0-1 pt).
    Age-adjusted creatinine threshold (mg/dL):
      Source: Sanchez-Pinto et al. 2024 Table 1
    """
    if creatinine_max is None or age_months is None:
        return 0
    if age_months < 1:
        threshold = 0.8
    elif age_months < 12:
        threshold = 0.3
    elif age_months < 24:
        threshold = 0.4
    elif age_months < 60:
        threshold = 0.6
    elif age_months < 144:
        threshold = 0.7
    else:
        threshold = 1.0
    return 1 if creatinine_max >= threshold else 0


def score_hepatic(bilirubin_max, alt_max):
    """
    Hepatic score (0-1 pt).
    Total bilirubin >= 4 mg/dL  -> 1 pt
    ALT > 102 IU/L              -> 1 pt
    Either condition sufficient.
    """
    if bilirubin_max is not None and bilirubin_max >= 4.0:
        return 1
    if alt_max is not None and alt_max > 102.0:
        return 1
    return 0


# =============================================================================
# MAIN LABELING PIPELINE
# =============================================================================

def run_step4(input_csv, output_csv, log_dir):

    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 4: Phoenix Sepsis Labeling")
    print("="*60)

    # Load feature matrix
    print(f"\nLoading {input_csv}...")
    rows = []
    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)
    print(f"  Loaded {len(rows):,} rows, {len(fieldnames)} columns")

    # Score tracking
    score_counts = defaultdict(int)
    component_counts = defaultdict(lambda: defaultdict(int))
    sepsis_count   = 0
    no_sep_count   = 0
    no_infect_count= 0

    # Label basis tracking
    vaso_only      = 0
    lab_only       = 0
    both           = 0

    labeled_rows = []

    print("\nComputing Phoenix scores...")

    for row in rows:

        # ── Extract values needed for scoring ─────────────────────────────────
        age_months     = safe_float(row.get('age_months'))

        # Cardiovascular
        lactate_max    = safe_float(row.get('lactate_max'))
        map_min        = safe_float(row.get('map_min'))
        n_vasoactives  = safe_int(row.get('n_vasoactives_24h'))

        # Coagulation
        platelets_min  = safe_float(row.get('platelets_min'))
        inr_max        = safe_float(row.get('inr_max'))
        ddimer_max     = safe_float(row.get('ddimer_max'))
        fibrinogen_min = safe_float(row.get('fibrinogen_min'))

        # Endocrine
        glucose_max    = safe_float(row.get('glucose_max'))
        glucose_min    = safe_float(row.get('glucose_min'))

        # Immunologic
        anc_min        = safe_float(row.get('anc_min'))
        alc_min        = safe_float(row.get('alc_min'))

        # Renal
        creatinine_max = safe_float(row.get('creatinine_max'))

        # Hepatic
        bilirubin_max  = safe_float(row.get('bilirubin_max'))
        alt_max        = safe_float(row.get('alt_max'))

        # Infection criterion (already computed in Step 3)
        suspected_inf  = safe_int(row.get('suspected_infection'))

        # ── Compute Phoenix score components ──────────────────────────────────
        s_resp    = score_respiratory()
        s_cardio  = score_cardiovascular(
                        lactate_max, map_min, n_vasoactives, age_months)
        s_coag    = score_coagulation(
                        platelets_min, inr_max, ddimer_max, fibrinogen_min)
        s_neuro   = score_neurological()
        s_endo    = score_endocrine(glucose_max, glucose_min)
        s_immuno  = score_immunologic(anc_min, alc_min)
        s_renal   = score_renal(creatinine_max, age_months)
        s_hepatic = score_hepatic(bilirubin_max, alt_max)

        # ── Phoenix core and Phoenix-8 scores ─────────────────────────────────
        phoenix_core = s_resp + s_cardio + s_coag + s_neuro
        phoenix_8    = phoenix_core + s_endo + s_immuno + s_renal + s_hepatic

        # ── Assign label ──────────────────────────────────────────────────────
        # Sepsis = suspected infection AND Phoenix core score >= 2
        sepsis_label = 1 if (suspected_inf == 1 and phoenix_core >= 2) else 0

        # ── Track statistics ──────────────────────────────────────────────────
        score_counts[phoenix_core] += 1
        component_counts['cardio'][s_cardio] += 1
        component_counts['coag'][s_coag] += 1
        component_counts['endo'][s_endo] += 1
        component_counts['immuno'][s_immuno] += 1
        component_counts['renal'][s_renal] += 1
        component_counts['hepatic'][s_hepatic] += 1

        if suspected_inf == 0:
            no_infect_count += 1
        elif sepsis_label == 1:
            sepsis_count += 1
            # Track what drove the label
            has_lab = any(v is not None for v in [lactate_max, platelets_min,
                          inr_max, ddimer_max, fibrinogen_min])
            has_vaso = n_vasoactives > 0
            if has_vaso and not has_lab:
                vaso_only += 1
            elif has_lab and not has_vaso:
                lab_only += 1
            else:
                both += 1
        else:
            no_sep_count += 1

        # ── Build output row ──────────────────────────────────────────────────
        new_row = dict(row)
        new_row['score_respiratory']    = s_resp
        new_row['score_cardiovascular'] = s_cardio
        new_row['score_coagulation']    = s_coag
        new_row['score_neurological']   = s_neuro
        new_row['score_endocrine']      = s_endo
        new_row['score_immunologic']    = s_immuno
        new_row['score_renal']          = s_renal
        new_row['score_hepatic']        = s_hepatic
        new_row['phoenix_core_score']   = phoenix_core
        new_row['phoenix_8_score']      = phoenix_8
        new_row['sepsis_label']         = sepsis_label

        labeled_rows.append(new_row)

    # ── Write output CSV ──────────────────────────────────────────────────────
    print(f"\nWriting {output_csv}...")
    new_fieldnames = fieldnames + [
        'score_respiratory', 'score_cardiovascular', 'score_coagulation',
        'score_neurological', 'score_endocrine', 'score_immunologic',
        'score_renal', 'score_hepatic', 'phoenix_core_score',
        'phoenix_8_score', 'sepsis_label'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(labeled_rows)
    print(f"  Saved {len(labeled_rows):,} rows, {len(new_fieldnames)} columns")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(labeled_rows)
    prev = f"{sepsis_count/n*100:.1f}%" if n > 0 else "N/A"

    summary = [
        "=" * 65,
        "STEP 4 SUMMARY — Phoenix Sepsis Labeling",
        "=" * 65,
        "",
        "COHORT BREAKDOWN",
        f"  Total ICU stays:                        {n:>8,}",
        f"  No suspected infection:                 {no_infect_count:>8,}  ({no_infect_count/n*100:.1f}%)",
        f"  Suspected infection, no sepsis:         {no_sep_count:>8,}  ({no_sep_count/n*100:.1f}%)",
        f"  SEPSIS (label=1):                       {sepsis_count:>8,}  ({sepsis_count/n*100:.1f}%)",
        f"  Sepsis prevalence overall:              {prev}",
        "",
        f"SEPSIS LABEL BASIS (among {sepsis_count} sepsis cases)",
        f"  Vasoactives only (no lab data):         {vaso_only:>8,}  ({vaso_only/max(sepsis_count,1)*100:.1f}%)",
        f"  Labs only (no vasoactives):             {lab_only:>8,}  ({lab_only/max(sepsis_count,1)*100:.1f}%)",
        f"  Both vasoactives + labs:                {both:>8,}  ({both/max(sepsis_count,1)*100:.1f}%)",
        "",
        "PHOENIX CORE SCORE DISTRIBUTION",
        f"  {'Score':<8} {'Count':>8} {'%':>8}",
        "  " + "-"*26,
    ]
    for sc in sorted(score_counts.keys()):
        cnt = score_counts[sc]
        marker = " <- SEPSIS THRESHOLD" if sc == 2 else ""
        summary.append(
            f"  {sc:<8} {cnt:>8,} {cnt/n*100:>7.1f}%{marker}"
        )

    summary += [
        "",
        "COMPONENT SCORE DISTRIBUTIONS",
        f"  {'Component':<15} " + "  ".join(f"  [{i}pts]:{component_counts['cardio'].get(i,0):,}" for i in range(7)),
    ]

    for comp_name in ['cardio','coag','endo','immuno','renal','hepatic']:
        max_pts = {'cardio':6,'coag':2,'endo':1,'immuno':1,'renal':1,'hepatic':1}[comp_name]
        dist = "  ".join(
            f"[{i}]:{component_counts[comp_name].get(i,0):,}"
            for i in range(max_pts+1)
        )
        summary.append(f"  {comp_name:<15} {dist}")

    summary += [
        "",
        "KNOWN LIMITATIONS (documented)",
        "  Respiratory score: 0 for all — FiO2/ventilation unavailable in PIC",
        "  Neurological score: 0 for all — GCS unavailable in PIC",
        "  Label is conservative — patients septic via respiratory/neuro",
        "  dysfunction only will be missed (mislabeled as non-sepsis)",
        "",
        f"OUTPUT: {output_csv}",
        "=" * 65,
    ]

    summary_str = '\n'.join(summary)
    print('\n' + summary_str)

    with open(LOG_FILE, 'w') as f:
        f.write(summary_str)
    print(f"\nLog saved -> {LOG_FILE}")
    print("\nStep 4 complete. Ready for Step 5 (label verification).")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step4(INPUT_CSV, OUTPUT_CSV, LOG_DIR)