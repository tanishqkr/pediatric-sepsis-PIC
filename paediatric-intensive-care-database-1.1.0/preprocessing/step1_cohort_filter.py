"""
STEP 1 — Cohort Filtering
=========================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Reads PATIENTS, ADMISSIONS, and ICUSTAYS from V1.1.0/
Applies the following filters in order:

  Filter 1 — Age 0-18 years at ICU admission
  Filter 2 — First ICU stay per patient only (prevents data leakage)
  Filter 3 — Minimum ICU stay of 6 hours (ensures enough data exists)

Outputs
-------
  output2/cohort_step1.csv        — filtered cohort, one row per ICU stay
  output2/logs/step1_log.txt      — detailed log of every filter decision

Why these filters
-----------------
  Age 0-18   : Phoenix criteria validated for this range only
  First stay : Multiple stays from same patient cause data leakage in ML
               (model memorises patient rather than learning patterns)
               Standard practice in clinical ML — Fleuren et al. 2020,
               Le et al. 2019, Chanci et al. 2025
  Min 6h     : Patients discharged in <6h have insufficient measurements
               for meaningful feature extraction
"""

import os
import csv
import math
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = "V1.1.0"
OUTPUT_DIR = "output2"
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
MIN_STAY_HOURS = 6      # minimum ICU stay to include
MAX_AGE_YEARS  = 18     # maximum age at ICU admission
MIN_AGE_YEARS  = 0      # minimum age (exclude birth hospitalisations edge cases)


# =============================================================================
# HELPERS
# =============================================================================

def parse_dt(s):
    if not s or not s.strip():
        return None
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None

def safe_float(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# =============================================================================
# LOAD TABLES
# =============================================================================

def load_patients(data_dir):
    print("  Loading PATIENTS...")
    out = {}
    with open(os.path.join(data_dir, 'PATIENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            dob = parse_dt(row['DOB'])
            if dob is None:
                continue
            out[row['SUBJECT_ID']] = {
                'gender':      row['GENDER'],
                'dob':         dob,
                'dod':         parse_dt(row['DOD']),
                'expire_flag': int(row['EXPIRE_FLAG']),
            }
    print(f"    -> {len(out):,} patients loaded")
    return out


def load_admissions(data_dir):
    print("  Loading ADMISSIONS...")
    out = {}
    with open(os.path.join(data_dir, 'ADMISSIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['HADM_ID']] = {
                'subject_id':  row['SUBJECT_ID'],
                'admittime':   parse_dt(row['ADMITTIME']),
                'dischtime':   parse_dt(row['DISCHTIME']),
                'diagnosis':   row['DIAGNOSIS'],
                'icd10':       row['ICD10_CODE_CN'],
                'hosp_expire': int(row['HOSPITAL_EXPIRE_FLAG']),
                'ethnicity':   row['ETHNICITY'],
                'insurance':   row['INSURANCE'],
            }
    print(f"    -> {len(out):,} admissions loaded")
    return out


def load_icustays(data_dir):
    print("  Loading ICUSTAYS...")
    out = []
    with open(os.path.join(data_dir, 'ICUSTAYS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            intime  = parse_dt(row['INTIME'])
            outtime = parse_dt(row['OUTTIME'])
            los     = safe_float(row['LOS'])
            if intime is None:
                continue
            out.append({
                'subject_id':  row['SUBJECT_ID'],
                'hadm_id':     row['HADM_ID'],
                'icustay_id':  row['ICUSTAY_ID'],
                'intime':      intime,
                'outtime':     outtime,
                'los_days':    los,
                'care_unit':   row['FIRST_CAREUNIT'],
            })
    print(f"    -> {len(out):,} ICU stays loaded")
    return out


# =============================================================================
# MAIN FILTER PIPELINE
# =============================================================================

def run_step1(data_dir, output_dir, log_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Cohort Filtering")
    print("="*60)
    print("\nLoading tables...")

    patients   = load_patients(data_dir)
    admissions = load_admissions(data_dir)
    icustays   = load_icustays(data_dir)

    # Tracking counters for log
    counts = {
        'total_stays':          len(icustays),
        'no_patient_record':    0,
        'no_dob':               0,
        'age_out_of_range':     0,
        'not_first_stay':       0,
        'stay_too_short':       0,
        'passed':               0,
    }

    # ── Filter 1 + 2 prep: Sort stays per patient by intime ──────────────────
    # This lets us identify the FIRST stay per patient reliably
    print("\nSorting stays by admission time per patient...")
    stays_by_patient = defaultdict(list)
    for stay in icustays:
        stays_by_patient[stay['subject_id']].append(stay)

    # Sort each patient's stays chronologically
    for sid in stays_by_patient:
        stays_by_patient[sid].sort(key=lambda x: x['intime'])

    # Mark first stay for each patient
    first_stay_ids = set()
    for sid, stays in stays_by_patient.items():
        first_stay_ids.add(stays[0]['icustay_id'])

    print(f"  Patients with multiple ICU stays: "
          f"{sum(1 for s in stays_by_patient.values() if len(s) > 1):,}")
    print(f"  Total stays marked as first stay: {len(first_stay_ids):,}")

    # ── Apply all filters ─────────────────────────────────────────────────────
    print("\nApplying filters...")

    eligible = []
    log_rows = []  # detailed log of excluded patients

    for stay in icustays:
        sid       = stay['subject_id']
        hadm_id   = stay['hadm_id']
        icustay_id= stay['icustay_id']
        intime    = stay['intime']

        # Get patient record
        p = patients.get(sid)
        if p is None:
            counts['no_patient_record'] += 1
            log_rows.append({'icustay_id': icustay_id, 'subject_id': sid,
                             'reason': 'no_patient_record', 'detail': ''})
            continue

        # Compute age at ICU admission
        dob = p['dob']
        age_days  = (intime - dob).days
        age_years = age_days / 365.25
        age_months= age_days / 30.4375

        # ── Filter 1: Age 0-18 ────────────────────────────────────────────────
        if not (MIN_AGE_YEARS <= age_years <= MAX_AGE_YEARS):
            counts['age_out_of_range'] += 1
            log_rows.append({'icustay_id': icustay_id, 'subject_id': sid,
                             'reason': 'age_out_of_range',
                             'detail': f'age={age_years:.1f}y'})
            continue

        # ── Filter 2: First ICU stay only ─────────────────────────────────────
        if icustay_id not in first_stay_ids:
            counts['not_first_stay'] += 1
            log_rows.append({'icustay_id': icustay_id, 'subject_id': sid,
                             'reason': 'not_first_stay',
                             'detail': f'intime={intime}'})
            continue

        # ── Filter 3: Minimum stay 6 hours ────────────────────────────────────
        los_hours = (stay['los_days'] or 0) * 24
        # If LOS missing, compute from outtime if available
        if stay['outtime'] is not None:
            los_hours = (stay['outtime'] - intime).total_seconds() / 3600
        
        if los_hours < MIN_STAY_HOURS:
            counts['stay_too_short'] += 1
            log_rows.append({'icustay_id': icustay_id, 'subject_id': sid,
                             'reason': 'stay_too_short',
                             'detail': f'los_hours={los_hours:.1f}'})
            continue

        # ── Passed all filters ────────────────────────────────────────────────
        counts['passed'] += 1

        # Get admission info
        adm = admissions.get(hadm_id, {})

        eligible.append({
            'subject_id':          sid,
            'hadm_id':             hadm_id,
            'icustay_id':          icustay_id,
            'intime':              intime.strftime('%Y-%m-%d %H:%M:%S'),
            'outtime':             stay['outtime'].strftime('%Y-%m-%d %H:%M:%S')
                                   if stay['outtime'] else '',
            'los_days':            round(stay['los_days'], 4)
                                   if stay['los_days'] else '',
            'los_hours':           round(los_hours, 2),
            'age_years':           round(age_years, 3),
            'age_months':          round(age_months, 1),
            'gender':              p['gender'],
            'expire_flag':         p['expire_flag'],
            'hospital_expire_flag':adm.get('hosp_expire', ''),
            'admission_diagnosis': adm.get('diagnosis', ''),
            'icd10_code':          adm.get('icd10', ''),
            'ethnicity':           adm.get('ethnicity', ''),
            'care_unit':           stay['care_unit'],
        })

    # ── Write output ──────────────────────────────────────────────────────────
    print("\nWriting outputs...")

    out_csv = os.path.join(output_dir, 'cohort_step1.csv')
    if eligible:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(eligible[0].keys()))
            writer.writeheader()
            writer.writerows(eligible)
    print(f"  Saved {len(eligible):,} eligible stays -> {out_csv}")

    # ── Write exclusion log ───────────────────────────────────────────────────
    log_csv = os.path.join(log_dir, 'step1_exclusions.csv')
    if log_rows:
        with open(log_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f,
                fieldnames=['icustay_id','subject_id','reason','detail'])
            writer.writeheader()
            writer.writerows(log_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = [
        "=" * 60,
        "STEP 1 SUMMARY — Cohort Filtering",
        "=" * 60,
        f"",
        f"INPUT",
        f"  Total ICU stays in PIC V1.1.0:      {counts['total_stays']:>8,}",
        f"  Total unique patients:               {len(patients):>8,}",
        f"",
        f"EXCLUSIONS",
        f"  No patient record:                  {counts['no_patient_record']:>8,}",
        f"  Age out of range (0-18y):            {counts['age_out_of_range']:>8,}",
        f"  Not first ICU stay for patient:      {counts['not_first_stay']:>8,}",
        f"  Stay too short (<6 hours):           {counts['stay_too_short']:>8,}",
        f"",
        f"OUTPUT",
        f"  Eligible stays (passed all filters): {counts['passed']:>8,}",
        f"",
        f"FILTER PARAMETERS",
        f"  Age range:        {MIN_AGE_YEARS} – {MAX_AGE_YEARS} years",
        f"  Stay type:        First ICU stay per patient only",
        f"  Minimum stay:     {MIN_STAY_HOURS} hours",
        f"",
        f"OUTPUT FILE: {out_csv}",
        f"EXCLUSION LOG: {log_csv}",
        "=" * 60,
    ]

    summary_str = '\n'.join(summary)
    print('\n' + summary_str)

    log_txt = os.path.join(log_dir, 'step1_log.txt')
    with open(log_txt, 'w') as f:
        f.write(summary_str)
    print(f"\nLog saved -> {log_txt}")
    print("\nStep 1 complete. Ready for Step 2.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step1(DATA_DIR, OUTPUT_DIR, LOG_DIR)
