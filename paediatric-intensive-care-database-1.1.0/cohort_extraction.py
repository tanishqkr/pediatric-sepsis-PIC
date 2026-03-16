"""
PIC Database — Phoenix Sepsis Cohort Extraction Pipeline
=========================================================
Authors : Uzair & Tanish | NMIMS | 2026
Based on : Sanchez-Pinto et al. (2024) & Schlapbach et al. (2024) — JAMA Phoenix Criteria

What this script does
---------------------
1. Loads all PIC tables from V1.1.0/
2. Computes age at ICU admission, filters to 0-18 years
3. Identifies suspected infection (antibiotic + microbiology culture within 24h of ICU admit)
4. Extracts worst values in the first 24h window for every Phoenix variable
5. Converts units from Chinese lab standards to Phoenix-expected units
6. Computes partial Phoenix score (cardiovascular + coagulation + renal + hepatic + endocrine + immunologic)
   NOTE: Respiratory and Neurological scores are dropped — FiO2, PaO2, GCS not available in PIC
7. Labels each ICU stay: sepsis=1 if score>=2 AND suspected infection, else 0
8. Saves one CSV with all records to output/pic_sepsis_cohort.csv

How to run
----------
    python cohort_extraction.py --data_dir /path/to/paediatric-intensive-care-database-1.1.0/V1.1.0

Output
------
    output/pic_sepsis_cohort.csv   — one row per ICU stay, all features + label
    output/extraction_summary.txt  — counts and basic stats
"""

import os
import csv
import argparse
import math
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# CONFIGURATION — ITEMIDs and drug keywords
# =============================================================================

# Lab ITEMIDs from D_LABITEMS that map to Phoenix variables
LAB_ITEMIDS = {
    # Cardiovascular
    'lactate':       ['5227'],
    # Coagulation
    'platelets':     ['5129'],
    'inr':           ['5174'],
    'ddimer':        ['5163'],
    'fibrinogen':    ['5164'],
    # Renal — creatinine in umol/L in PIC, convert to mg/dL (/88.4)
    'creatinine':    ['5032', '5041', '6954'],
    # Hepatic — bilirubin in umol/L in PIC, convert to mg/dL (/17.1)
    'bilirubin':     ['5075', '5255'],
    'alt':           ['5026', '5195'],
    # Endocrine — glucose in mmol/L in PIC, convert to mg/dL (*18)
    'glucose':       ['5047', '5223'],
    # Immunologic
    'anc':           ['5094'],   # absolute neutrophil count (cells/mm3 * 1000 from 10^9/L)
    'alc':           ['5110'],   # absolute lymphocyte count
    'wbc':           ['5141'],
}

# Chart ITEMIDs from D_ITEMS for vitals
CHART_ITEMIDS = {
    'systolic_bp':   '1016',
    'diastolic_bp':  '1015',
    'spo2':          '1006',
    'heart_rate':    '1003',
    'resp_rate':     '1004',
    'temperature':   '1001',
}

# Vasoactive drug keywords (Phoenix cardiovascular score)
VASOACTIVE_KEYWORDS = [
    'dopamine', 'epinephrine', 'adrenaline', 'norepinephrine', 'noradrenaline',
    'vasopressin', 'dobutamine', 'milrinone'
]

# Antibiotic keywords for suspected infection criterion
ANTIBIOTIC_KEYWORDS = [
    'penicillin', 'ampicillin', 'amoxicillin', 'cefazolin', 'cefotaxime',
    'ceftriaxone', 'ceftazidime', 'cefepime', 'cefuroxime', 'meropenem',
    'imipenem', 'vancomycin', 'piperacillin', 'levofloxacin', 'ciprofloxacin',
    'azithromycin', 'moxifloxacin', 'aztreonam', 'clindamycin', 'linezolid',
    'metronidazole', 'fluconazole', 'voriconazole', 'acyclovir', 'ganciclovir',
    'gentamicin', 'tobramycin', 'amikacin', 'rifampin', 'doxycycline',
    'tetracycline', 'chloramphenicol', 'erythromycin', 'roxithromycin',
    'clarithromycin', 'sulfamethoxazole', 'trimethoprim', 'nitrofurantoin',
    'cefoperazone', 'tigecycline', 'colistin', 'polymyxin'
]


# =============================================================================
# UNIT CONVERSIONS
# =============================================================================

def convert_bilirubin(val_umol):
    """umol/L → mg/dL  (divide by 17.1)"""
    return val_umol / 17.1

def convert_creatinine(val_umol):
    """umol/L → mg/dL  (divide by 88.4)"""
    return val_umol / 88.4

def convert_glucose(val_mmol):
    """mmol/L → mg/dL  (multiply by 18)"""
    return val_mmol * 18


# =============================================================================
# PHOENIX SCORING FUNCTIONS
# Each function takes extracted worst values and returns a score component
# =============================================================================

def score_cardiovascular(lactate, map_mmhg, n_vasoactives, age_months):
    """
    Cardiovascular score (0-6 points, sum of 3 sub-components):
      Vasoactives : 0 = none, 1 = one drug, 2 = two or more drugs
      Lactate     : 0 = <5, 1 = 5-10.9, 2 = >=11  (mmol/L)
      MAP         : 0/1/2 depending on age-adjusted thresholds
    """
    score = 0

    # Vasoactive medications
    if n_vasoactives == 1:
        score += 1
    elif n_vasoactives >= 2:
        score += 2

    # Lactate
    if lactate is not None:
        if lactate >= 11:
            score += 2
        elif lactate >= 5:
            score += 1

    # MAP — age-adjusted thresholds (age in months)
    # Phoenix paper Table: MAP thresholds for 0 points (normal)
    if map_mmhg is not None and age_months is not None:
        # Get the threshold for 1 point (below normal) and 2 points (critically low)
        # Format: (lower_bound_1pt, lower_bound_2pt)
        # i.e. MAP < lower_bound_1pt = 2 pts, between = 1 pt, >= normal = 0 pts
        if age_months < 1:
            normal, critical = 31, 17
        elif age_months < 12:
            normal, critical = 39, 25
        elif age_months < 24:
            normal, critical = 44, 31
        elif age_months < 60:
            normal, critical = 45, 32
        elif age_months < 144:
            normal, critical = 49, 36
        else:
            normal, critical = 52, 38

        if map_mmhg < critical:
            score += 2
        elif map_mmhg < normal:
            score += 1

    return score


def score_coagulation(platelets, inr, ddimer, fibrinogen):
    """
    Coagulation score (0-2 points, 1 point per abnormal lab, max 2):
      Platelets  : <100 = 1 point
      INR        : >1.3 = 1 point
      D-Dimer    : >2 mg/L = 1 point
      Fibrinogen : <100 mg/dL = 1 point
    Note: fibrinogen in PIC is in g/L. Phoenix threshold is 100 mg/dL = 1 g/L.
    """
    score = 0
    if platelets is not None and platelets < 100:
        score += 1
    if inr is not None and inr > 1.3:
        score += 1
    if ddimer is not None and ddimer > 2:
        score += 1
    if fibrinogen is not None and fibrinogen < 1.0:  # PIC unit is g/L, threshold 1 g/L
        score += 1
    return min(score, 2)  # max 2 points


def score_neurological():
    """
    Neurological score — NOT COMPUTABLE in PIC (no GCS, no pupil data).
    Returns 0 always. Documented limitation.
    """
    return 0


def score_endocrine(glucose_mgdl):
    """
    Endocrine score (0-1 point):
      Blood glucose <50 mg/dL OR >150 mg/dL = 1 point
    """
    if glucose_mgdl is None:
        return 0
    return 1 if (glucose_mgdl < 50 or glucose_mgdl > 150) else 0


def score_immunologic(anc, alc):
    """
    Immunologic score (0-1 point):
      ANC <500 cells/mm3 = 1 point
      ALC <1000 cells/mm3 = 1 point
    Note: PIC stores these as 10^9/L. 1 x10^9/L = 1000 cells/mm3
    So ANC < 0.5 x10^9/L = 1 pt, ALC < 1.0 x10^9/L = 1 pt
    """
    score = 0
    if anc is not None and anc < 0.5:
        score = 1
    if alc is not None and alc < 1.0:
        score = 1
    return score


def score_renal(creatinine_mgdl, age_months):
    """
    Renal score (0-1 point): age-adjusted creatinine threshold
    """
    if creatinine_mgdl is None or age_months is None:
        return 0

    # Age-adjusted creatinine thresholds (mg/dL) for 1 point
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

    return 1 if creatinine_mgdl >= threshold else 0


def score_hepatic(bilirubin_mgdl, alt):
    """
    Hepatic score (0-1 point):
      Total bilirubin >= 4 mg/dL = 1 point
      ALT > 102 IU/L = 1 point
    """
    if bilirubin_mgdl is not None and bilirubin_mgdl >= 4:
        return 1
    if alt is not None and alt > 102:
        return 1
    return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_datetime(s):
    """Parse datetime string, return None if empty or invalid."""
    if not s or s.strip() == '':
        return None
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def safe_float(s):
    """Convert string to float, return None if not possible."""
    try:
        v = float(s)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def get_worst(values, mode='min'):
    """
    Get the worst value from a list of floats.
    mode='min' returns minimum (e.g. platelets — lower is worse)
    mode='max' returns maximum (e.g. lactate — higher is worse)
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return min(clean) if mode == 'min' else max(clean)


def compute_map(systolic, diastolic):
    """MAP = diastolic + (systolic - diastolic) / 3"""
    if systolic is None or diastolic is None:
        return None
    return diastolic + (systolic - diastolic) / 3.0


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_patients(data_dir):
    """
    Load PATIENTS.csv.
    Returns dict: subject_id -> {gender, dob, dod, expire_flag}
    """
    print("Loading PATIENTS...")
    patients = {}
    with open(os.path.join(data_dir, 'PATIENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            patients[row['SUBJECT_ID']] = {
                'gender':      row['GENDER'],
                'dob':         parse_datetime(row['DOB']),
                'dod':         parse_datetime(row['DOD']),
                'expire_flag': int(row['EXPIRE_FLAG'])
            }
    print(f"  Loaded {len(patients)} patients")
    return patients


def load_admissions(data_dir):
    """
    Load ADMISSIONS.csv.
    Returns dict: hadm_id -> {subject_id, admittime, dischtime, diagnosis, icd10}
    """
    print("Loading ADMISSIONS...")
    admissions = {}
    with open(os.path.join(data_dir, 'ADMISSIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            admissions[row['HADM_ID']] = {
                'subject_id':    row['SUBJECT_ID'],
                'admittime':     parse_datetime(row['ADMITTIME']),
                'dischtime':     parse_datetime(row['DISCHTIME']),
                'diagnosis':     row['DIAGNOSIS'],
                'icd10':         row['ICD10_CODE_CN'],
                'expire_flag':   int(row['HOSPITAL_EXPIRE_FLAG'])
            }
    print(f"  Loaded {len(admissions)} admissions")
    return admissions


def load_icustays(data_dir):
    """
    Load ICUSTAYS.csv.
    Returns list of dicts, each representing one ICU stay.
    """
    print("Loading ICUSTAYS...")
    icustays = []
    with open(os.path.join(data_dir, 'ICUSTAYS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            intime  = parse_datetime(row['INTIME'])
            outtime = parse_datetime(row['OUTTIME'])
            if intime is None:
                continue  # skip stays with no admission time
            icustays.append({
                'subject_id':  row['SUBJECT_ID'],
                'hadm_id':     row['HADM_ID'],
                'icustay_id':  row['ICUSTAY_ID'],
                'intime':      intime,
                'outtime':     outtime,
                'los':         safe_float(row['LOS'])
            })
    print(f"  Loaded {len(icustays)} ICU stays")
    return icustays


def load_labevents(data_dir):
    """
    Load LABEVENTS.csv — only the ITEMIDs we need.
    Returns dict: subject_id -> list of {itemid, charttime, valuenum}
    """
    print("Loading LABEVENTS (filtering relevant ITEMIDs)...")

    # Flatten all itemids we care about into one set for fast lookup
    all_itemids = set()
    for ids in LAB_ITEMIDS.values():
        all_itemids.update(ids)

    labs = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'LABEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ITEMID'] in all_itemids:
                val = safe_float(row['VALUENUM'])
                t   = parse_datetime(row['CHARTTIME'])
                if val is not None and t is not None:
                    labs[row['SUBJECT_ID']].append({
                        'itemid':    row['ITEMID'],
                        'charttime': t,
                        'value':     val
                    })
                    count += 1
    print(f"  Loaded {count} relevant lab events across {len(labs)} patients")
    return labs


def load_chartevents(data_dir):
    """
    Load CHARTEVENTS.csv — only the ITEMIDs we need.
    Returns dict: subject_id -> list of {itemid, charttime, valuenum}
    """
    print("Loading CHARTEVENTS (filtering relevant ITEMIDs)...")

    all_itemids = set(CHART_ITEMIDS.values())

    charts = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'CHARTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ITEMID'] in all_itemids:
                val = safe_float(row['VALUENUM'])
                t   = parse_datetime(row['CHARTTIME'])
                if val is not None and t is not None:
                    charts[row['SUBJECT_ID']].append({
                        'itemid':    row['ITEMID'],
                        'charttime': t,
                        'value':     val
                    })
                    count += 1
    print(f"  Loaded {count} relevant chart events across {len(charts)} patients")
    return charts


def load_prescriptions(data_dir):
    """
    Load PRESCRIPTIONS.csv — only antibiotics and vasoactives.
    Returns dict: subject_id -> list of {drug_name, startdate, hadm_id, is_vasoactive, is_antibiotic}
    """
    print("Loading PRESCRIPTIONS (filtering antibiotics and vasoactives)...")

    prescriptions = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            drug = row['DRUG_NAME_EN'].lower()
            is_vaso = any(k in drug for k in VASOACTIVE_KEYWORDS)
            is_abx  = any(k in drug for k in ANTIBIOTIC_KEYWORDS)

            if is_vaso or is_abx:
                startdate = parse_datetime(row['STARTDATE'])
                if startdate is None:
                    continue
                prescriptions[row['SUBJECT_ID']].append({
                    'hadm_id':       row['HADM_ID'],
                    'drug_name':     row['DRUG_NAME_EN'],
                    'startdate':     startdate,
                    'is_vasoactive': is_vaso,
                    'is_antibiotic': is_abx
                })
                count += 1
    print(f"  Loaded {count} relevant prescriptions across {len(prescriptions)} patients")
    return prescriptions


def load_microbiologyevents(data_dir):
    """
    Load MICROBIOLOGYEVENTS.csv.
    We only need to know: did this patient have a culture ordered?
    Returns dict: subject_id -> list of {charttime, hadm_id}
    """
    print("Loading MICROBIOLOGYEVENTS...")

    micro = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'MICROBIOLOGYEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = parse_datetime(row['CHARTTIME'])
            if t is not None:
                micro[row['SUBJECT_ID']].append({
                    'hadm_id':   row['HADM_ID'],
                    'charttime': t
                })
                count += 1
    print(f"  Loaded {count} microbiology events across {len(micro)} patients")
    return micro


# =============================================================================
# CORE EXTRACTION — per ICU stay
# =============================================================================

def extract_window(events_list, intime, hours=24):
    """
    Filter events to only those within the first N hours of ICU admission.
    events_list: list of dicts with 'charttime' key
    Returns filtered list.
    """
    window_end = intime + timedelta(hours=hours)
    return [e for e in events_list if intime <= e['charttime'] <= window_end]


def get_lab_values(lab_events_window, itemid_list):
    """
    From a windowed lab event list, extract all values for given itemid_list.
    Returns list of floats.
    """
    return [e['value'] for e in lab_events_window if e['itemid'] in itemid_list]


def check_suspected_infection(sid, hadm_id, intime, prescriptions, micro, hours=24):
    """
    Suspected infection = antibiotic prescription AND microbiology culture
    both within 24 hours of ICU admission.

    Returns (has_suspected_infection: bool, n_antibiotics: int)
    """
    window_end = intime + timedelta(hours=hours)

    # Check for antibiotic in window
    patient_rx = prescriptions.get(sid, [])
    abx_in_window = [
        rx for rx in patient_rx
        if rx['is_antibiotic']
        and intime <= rx['startdate'] <= window_end
    ]

    # Check for microbiology culture in window
    patient_micro = micro.get(sid, [])
    micro_in_window = [
        m for m in patient_micro
        if intime <= m['charttime'] <= window_end
    ]

    has_infection = len(abx_in_window) > 0 and len(micro_in_window) > 0
    return has_infection, len(abx_in_window)


def count_vasoactives(sid, intime, prescriptions, hours=24):
    """
    Count number of DISTINCT vasoactive drug types given within 24h of ICU admit.
    Phoenix: 0=none, 1=one drug, 2=two or more
    """
    window_end = intime + timedelta(hours=hours)
    patient_rx = prescriptions.get(sid, [])

    vaso_drugs = set()
    for rx in patient_rx:
        if rx['is_vasoactive'] and intime <= rx['startdate'] <= window_end:
            # Normalize drug name to count distinct types
            drug = rx['drug_name'].lower()
            for keyword in VASOACTIVE_KEYWORDS:
                if keyword in drug:
                    vaso_drugs.add(keyword)
                    break

    return len(vaso_drugs)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(data_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STAGE 1 — Load all tables
    # ------------------------------------------------------------------
    print("\n=== STAGE 1: Loading Data ===")
    patients      = load_patients(data_dir)
    admissions    = load_admissions(data_dir)
    icustays      = load_icustays(data_dir)
    labs          = load_labevents(data_dir)
    charts        = load_chartevents(data_dir)
    prescriptions = load_prescriptions(data_dir)
    micro         = load_microbiologyevents(data_dir)

    # ------------------------------------------------------------------
    # STAGE 2 — Filter ICU stays by age 0-18
    # ------------------------------------------------------------------
    print("\n=== STAGE 2: Filtering Age 0-18 years ===")
    eligible_stays = []
    age_excluded   = 0

    for stay in icustays:
        sid = stay['subject_id']
        p   = patients.get(sid)
        if p is None or p['dob'] is None:
            continue

        age_days   = (stay['intime'] - p['dob']).days
        age_years  = age_days / 365.25
        age_months = age_days / 30.4375

        # Filter: 0 to 18 years, exclude obvious de-identification anomalies
        if 0 <= age_years <= 18:
            stay['age_years']  = round(age_years, 2)
            stay['age_months'] = round(age_months, 1)
            stay['gender']     = p['gender']
            stay['expire_flag']= p['expire_flag']
            eligible_stays.append(stay)
        else:
            age_excluded += 1

    print(f"  Eligible stays (age 0-18): {len(eligible_stays)}")
    print(f"  Excluded (age out of range): {age_excluded}")

    # ------------------------------------------------------------------
    # STAGE 3-7 — For each eligible stay, extract features + label
    # ------------------------------------------------------------------
    print("\n=== STAGE 3-7: Feature Extraction + Phoenix Scoring ===")

    records     = []
    n_sepsis    = 0
    n_no_sepsis = 0
    n_no_infection = 0

    for i, stay in enumerate(eligible_stays):
        if i % 1000 == 0:
            print(f"  Processing stay {i}/{len(eligible_stays)}...")

        sid        = stay['subject_id']
        hadm_id    = stay['hadm_id']
        intime     = stay['intime']
        age_months = stay['age_months']

        # ---- STAGE 3: Suspected Infection ----
        has_infection, n_abx = check_suspected_infection(
            sid, hadm_id, intime, prescriptions, micro
        )

        # ---- STAGE 4: Extract values in first 24h window ----
        # Labs
        patient_labs   = extract_window(labs.get(sid, []), intime, 24)
        # Chart vitals
        patient_charts = extract_window(charts.get(sid, []), intime, 24)

        def get_lab(var_name):
            return get_lab_values(patient_labs, LAB_ITEMIDS[var_name])

        def get_chart(itemid):
            return [e['value'] for e in patient_charts if e['itemid'] == itemid]

        # Raw value lists
        lactate_vals    = get_lab('lactate')
        platelet_vals   = get_lab('platelets')
        inr_vals        = get_lab('inr')
        ddimer_vals     = get_lab('ddimer')
        fibrinogen_vals = get_lab('fibrinogen')
        creat_vals      = get_lab('creatinine')
        bili_vals       = get_lab('bilirubin')
        alt_vals        = get_lab('alt')
        glucose_vals    = get_lab('glucose')
        anc_vals        = get_lab('anc')
        alc_vals        = get_lab('alc')
        wbc_vals        = get_lab('wbc')

        sys_vals        = get_chart(CHART_ITEMIDS['systolic_bp'])
        dia_vals        = get_chart(CHART_ITEMIDS['diastolic_bp'])
        spo2_vals       = get_chart(CHART_ITEMIDS['spo2'])
        hr_vals         = get_chart(CHART_ITEMIDS['heart_rate'])
        rr_vals         = get_chart(CHART_ITEMIDS['resp_rate'])
        temp_vals       = get_chart(CHART_ITEMIDS['temperature'])

        # ---- STAGE 5: Unit Conversions ----
        # Bilirubin: umol/L -> mg/dL
        bili_mgdl   = [convert_bilirubin(v) for v in bili_vals]
        # Creatinine: umol/L -> mg/dL
        creat_mgdl  = [convert_creatinine(v) for v in creat_vals]
        # Glucose: mmol/L -> mg/dL
        glucose_mgdl = [convert_glucose(v) for v in glucose_vals]

        # ---- Worst values (most abnormal in 24h window) ----
        # For each variable, pick the value that would give the HIGHEST Phoenix score
        lactate_worst   = get_worst(lactate_vals,    'max')
        platelet_worst  = get_worst(platelet_vals,   'min')
        inr_worst       = get_worst(inr_vals,        'max')
        ddimer_worst    = get_worst(ddimer_vals,     'max')
        fibrinogen_worst= get_worst(fibrinogen_vals, 'min')
        creat_worst     = get_worst(creat_mgdl,      'max')
        bili_worst      = get_worst(bili_mgdl,       'max')
        alt_worst       = get_worst(alt_vals,        'max')
        glucose_worst   = get_worst(glucose_mgdl,    'max')  # hyperglycemia
        glucose_min     = get_worst(glucose_mgdl,    'min')  # hypoglycemia
        anc_worst       = get_worst(anc_vals,        'min')
        alc_worst       = get_worst(alc_vals,        'min')
        wbc_worst       = get_worst(wbc_vals,        'max')

        sys_worst       = get_worst(sys_vals,        'min')
        dia_worst       = get_worst(dia_vals,        'min')
        spo2_worst      = get_worst(spo2_vals,       'min')
        hr_max          = get_worst(hr_vals,         'max')
        rr_max          = get_worst(rr_vals,         'max')
        temp_max        = get_worst(temp_vals,       'max')
        temp_min        = get_worst(temp_vals,       'min')

        # Compute MAP from worst systolic + worst diastolic
        map_worst = compute_map(sys_worst, dia_worst)

        # For endocrine: worst glucose = whichever is further from normal (50-150)
        # Pick hypoglycemia if <50, hyperglycemia if >150
        glucose_for_endocrine = None
        if glucose_min is not None and glucose_min < 50:
            glucose_for_endocrine = glucose_min
        elif glucose_worst is not None and glucose_worst > 150:
            glucose_for_endocrine = glucose_worst
        elif glucose_worst is not None:
            glucose_for_endocrine = glucose_worst

        # Vasoactive count
        n_vasoactives = count_vasoactives(sid, intime, prescriptions)

        # ---- STAGE 6: Compute Phoenix Score Components ----
        s_cardio   = score_cardiovascular(lactate_worst, map_worst, n_vasoactives, age_months)
        s_coag     = score_coagulation(platelet_worst, inr_worst, ddimer_worst, fibrinogen_worst)
        s_neuro    = score_neurological()   # always 0 — no GCS in PIC
        s_endo     = score_endocrine(glucose_for_endocrine)
        s_immuno   = score_immunologic(anc_worst, alc_worst)
        s_renal    = score_renal(creat_worst, age_months)
        s_hepatic  = score_hepatic(bili_worst, alt_worst)

        # Core Phoenix score (4 components used in official definition)
        # We can only compute cardiovascular + coagulation from these 4
        # Respiratory = 0 (unavailable), Neurological = 0 (unavailable)
        phoenix_core    = s_cardio + s_coag + s_neuro   # respiratory always 0
        # Full Phoenix-8 score (all 8 components)
        phoenix_8       = phoenix_core + s_endo + s_immuno + s_renal + s_hepatic

        # ---- STAGE 7: Label ----
        # Sepsis = suspected infection AND Phoenix core score >= 2
        sepsis_label = 1 if (has_infection and phoenix_core >= 2) else 0

        if not has_infection:
            n_no_infection += 1
        elif sepsis_label == 1:
            n_sepsis += 1
        else:
            n_no_sepsis += 1

        # ---- Build record row ----
        adm = admissions.get(hadm_id, {})

        record = {
            # Identifiers
            'subject_id':           sid,
            'hadm_id':              hadm_id,
            'icustay_id':           stay['icustay_id'],

            # Demographics
            'age_years':            stay['age_years'],
            'age_months':           stay['age_months'],
            'gender':               stay['gender'],
            'los_days':             stay['los'],
            'hospital_expire_flag': stay['expire_flag'],

            # Admission info
            'admission_diagnosis':  adm.get('diagnosis', ''),
            'icd10_code':           adm.get('icd10', ''),

            # Infection flags
            'suspected_infection':  int(has_infection),
            'n_antibiotics_24h':    n_abx,
            'has_microbiology':     int(len(micro.get(sid, [])) > 0),

            # Vasoactives
            'n_vasoactives_24h':    n_vasoactives,

            # Raw worst lab values (original units)
            'lactate_max':          lactate_worst,
            'platelets_min':        platelet_worst,
            'inr_max':              inr_worst,
            'ddimer_max':           ddimer_worst,
            'fibrinogen_min':       fibrinogen_worst,
            'creatinine_umol_max':  get_worst(creat_vals, 'max'),
            'bilirubin_umol_max':   get_worst(bili_vals, 'max'),
            'alt_max':              alt_worst,
            'glucose_mmol_max':     get_worst(glucose_vals, 'max'),
            'glucose_mmol_min':     get_worst(glucose_vals, 'min'),
            'anc_min':              anc_worst,
            'alc_min':              alc_worst,
            'wbc_max':              wbc_worst,

            # Converted units (for Phoenix scoring)
            'creatinine_mgdl_max':  creat_worst,
            'bilirubin_mgdl_max':   bili_worst,
            'glucose_mgdl_max':     glucose_worst,
            'glucose_mgdl_min':     glucose_min,

            # Vitals
            'systolic_bp_min':      sys_worst,
            'diastolic_bp_min':     dia_worst,
            'map_min':              round(map_worst, 1) if map_worst else None,
            'spo2_min':             spo2_worst,
            'heart_rate_max':       hr_max,
            'resp_rate_max':        rr_max,
            'temp_max':             temp_max,
            'temp_min':             temp_min,

            # Phoenix score components
            'score_cardiovascular': s_cardio,
            'score_coagulation':    s_coag,
            'score_neurological':   s_neuro,   # always 0
            'score_endocrine':      s_endo,
            'score_immunologic':    s_immuno,
            'score_renal':          s_renal,
            'score_hepatic':        s_hepatic,
            'phoenix_core_score':   phoenix_core,
            'phoenix_8_score':      phoenix_8,

            # LABEL — this is what the model predicts
            'sepsis_label':         sepsis_label,
        }

        records.append(record)

    # ------------------------------------------------------------------
    # STAGE 8 — Write CSV
    # ------------------------------------------------------------------
    print(f"\n=== STAGE 8: Writing Output ===")

    output_csv = os.path.join(output_dir, 'pic_sepsis_cohort.csv')

    if records:
        fieldnames = list(records[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved {len(records)} records to: {output_csv}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary_lines = [
        "=== EXTRACTION SUMMARY ===",
        f"Total patients in PIC:              {len(patients)}",
        f"Total ICU stays:                    {len(icustays)}",
        f"ICU stays age 0-18 (eligible):      {len(eligible_stays)}",
        f"  -> No suspected infection:        {n_no_infection}",
        f"  -> Suspected infection, no sepsis:{n_no_sepsis}",
        f"  -> SEPSIS (label=1):              {n_sepsis}",
        f"  -> Total in output CSV:           {len(records)}",
        f"",
        f"Sepsis prevalence:                  {n_sepsis/len(records)*100:.1f}%" if records else "",
        f"",
        "=== LIMITATIONS (documented) ===",
        "Respiratory score: NOT computed — FiO2, PaO2, mechanical ventilation unavailable in PIC",
        "Neurological score: NOT computed — GCS unavailable in PIC",
        "Both components set to 0 per Phoenix missing data convention",
        "",
        "=== UNIT CONVERSIONS APPLIED ===",
        "Bilirubin:  umol/L -> mg/dL  (divide by 17.1)",
        "Creatinine: umol/L -> mg/dL  (divide by 88.4)",
        "Glucose:    mmol/L -> mg/dL  (multiply by 18)",
    ]

    summary_path = os.path.join(output_dir, 'extraction_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print('\n' + '\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")
    print("\nDone.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PIC Phoenix Sepsis Cohort Extraction Pipeline'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to V1.1.0 folder, e.g. /path/to/paediatric-intensive-care-database-1.1.0/V1.1.0'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output folder (default: output/)'
    )
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir)
