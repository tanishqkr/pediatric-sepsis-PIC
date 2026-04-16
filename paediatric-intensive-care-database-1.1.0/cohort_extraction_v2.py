"""
PIC Database — Phoenix Sepsis Cohort Extraction Pipeline v2
============================================================
Authors : Uzair & Tanish | NMIMS | 2026
Based on : Sanchez-Pinto et al. (2024) & Schlapbach et al. (2024) — JAMA Phoenix Criteria

WHAT THIS SCRIPT DOES
---------------------
Reads all 9 relevant PIC tables from V1.1.0/ and produces ONE CSV file where:
  - Every row   = one ICU stay
  - Every column = one feature (with 5 statistics per continuous variable)
  - Last column  = sepsis_label (0 or 1)

TABLES USED
-----------
  PATIENTS              -> age, gender, mortality
  ADMISSIONS            -> admission diagnosis, ICD code, timing
  ICUSTAYS              -> ICU in/out time, length of stay
  LABEVENTS             -> all lab values (Phoenix + extended)
  CHARTEVENTS           -> vitals (BP, HR, RR, SpO2, Temp)
  PRESCRIPTIONS         -> vasoactives + antibiotics
  MICROBIOLOGYEVENTS    -> suspected infection criterion
  OUTPUTEVENTS          -> urine output (renal proxy)
  INPUTEVENTS           -> fluid input (resuscitation proxy)
  EMR_SYMPTOMS          -> clinical symptoms (binary flags)

FEATURE ENGINEERING APPROACH
-----------------------------
For every continuous variable: max, min, mean, first value, count of readings
  - "first" captures the state on admission
  - "max/min" capture the worst extreme
  - "mean" captures the average burden
  - "count" captures how sick the patient was (more tests = more concern)
This is Approach 2 — richer than just worst value, avoids throwing away data.

PHOENIX LABELING
----------------
Label uses WORST value in 24h per Phoenix paper (not the 5-stat approach).
Features use the 5-stat approach.
These are intentionally separate design decisions.

KNOWN LIMITATIONS (documented)
-------------------------------
  Respiratory score: NOT computed — FiO2, mechanical ventilation absent from PIC
  Neurological score: NOT computed — GCS absent from PIC
  Both set to 0 per Phoenix missing-data convention (missing = 0 points)
  Procalcitonin: Only 18 readings in full database — excluded as too sparse
  BNP: Only 9 readings — excluded
  EMR_SYMPTOMS: Only ~883/12881 patients have symptoms documented

UNIT CONVERSIONS
----------------
  Bilirubin:  umol/L  -> mg/dL  (÷ 17.1)
  Creatinine: umol/L  -> mg/dL  (÷ 88.4)
  Glucose:    mmol/L  -> mg/dL  (× 18)
  Fibrinogen: g/L     -> Phoenix threshold is 1 g/L (no conversion, just threshold change)
  All others already in Phoenix-compatible units

HOW TO RUN
----------
  python cohort_extraction_v2.py --data_dir /path/to/V1.1.0 --output_dir output

OUTPUT
------
  output/pic_sepsis_cohort.csv      — main feature matrix, one row per ICU stay
  output/extraction_summary.txt     — counts, prevalence, column list
"""

import os
import csv
import math
import argparse
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# SECTION 1 — CONFIGURATION
# All ITEMIDs and keyword lists live here. Change nothing else if you want
# to add/remove features — just edit this section.
# =============================================================================

# ---- Lab ITEMIDs ----
# Format: 'feature_name': ['itemid1', 'itemid2', ...]
# Multiple ITEMIDs = same measurement done by different methods in PIC
# We pool them and take stats across all readings regardless of method

LAB_ITEMIDS = {

    # --- Phoenix Core: Cardiovascular ---
    'lactate':          ['5227'],

    # --- Phoenix Core: Coagulation ---
    'platelets':        ['5129'],
    'inr':              ['5174'],
    'ddimer':           ['5163'],
    'fibrinogen':       ['5164'],

    # --- Phoenix Extended: Renal ---
    # PIC unit: umol/L — convert to mg/dL by dividing by 88.4
    'creatinine':       ['5032', '5041', '6954'],

    # --- Phoenix Extended: Hepatic ---
    # PIC unit: umol/L — convert to mg/dL by dividing by 17.1
    'bilirubin':        ['5075', '5255'],
    'alt':              ['5026', '5195'],
    'ast':              ['5031'],

    # --- Phoenix Extended: Endocrine ---
    # PIC unit: mmol/L — convert to mg/dL by multiplying by 18
    'glucose':          ['5047', '5223'],

    # --- Phoenix Extended: Immunologic ---
    # PIC unit: 10^9/L. Phoenix thresholds: ANC<0.5, ALC<1.0 (in 10^9/L)
    'anc':              ['5094'],
    'alc':              ['5110'],
    'wbc':              ['5141'],

    # --- Blood Gas Panel ---
    # All measured together — if patient has blood gas, all present
    'ph':               ['5237'],
    'pco2':             ['5235'],
    'pao2':             ['5239'],       # We have pO2 but not FiO2 — cannot compute ratio
    'base_excess':      ['5211'],
    'bicarbonate':      ['5248'],
    'anion_gap':        ['5212'],
    'spo2_lab':         ['5252'],       # SpO2 from blood gas (different from chart SpO2)

    # --- Hematology ---
    'hemoglobin':       ['5099', '5257'],
    'hematocrit':       ['5097', '5225'],
    'pt':               ['5186'],
    'ptt':              ['5161'],

    # --- Chemistry ---
    'sodium':           ['5062', '5230'],
    'potassium':        ['5056', '5226'],
    'calcium':          ['5034', '5215'],
    'albumin':          ['5024'],
    'crp':              ['5626', '5821'],
    'ldh':              ['5057'],
    'urea':             ['5033'],
    'uric_acid':        ['5083'],
    'ck':               ['5038'],
}

# ---- Chart ITEMIDs (vitals from CHARTEVENTS) ----
CHART_ITEMIDS = {
    'spo2':             '1006',
    'systolic_bp':      '1016',
    'diastolic_bp':     '1015',
    'heart_rate':       '1003',
    'resp_rate':        '1004',
    'temperature':      '1001',
    'blood_glucose_chart': '1011',  # Bedside glucose — only 918 readings but include
}

# ---- Vasoactive drug keywords (for Phoenix cardiovascular score) ----
VASOACTIVE_KEYWORDS = [
    'dopamine', 'epinephrine', 'adrenaline',
    'norepinephrine', 'noradrenaline',
    'vasopressin', 'dobutamine', 'milrinone'
]

# ---- Individual vasoactive flags (for granular features) ----
VASOACTIVE_INDIVIDUAL = {
    'dopamine':        ['dopamine'],
    'epinephrine':     ['epinephrine', 'adrenaline'],
    'norepinephrine':  ['norepinephrine', 'noradrenaline'],
    'vasopressin':     ['vasopressin'],
    'dobutamine':      ['dobutamine'],
    'milrinone':       ['milrinone'],
}

# ---- Antibiotic keywords (for suspected infection criterion) ----
ANTIBIOTIC_KEYWORDS = [
    'penicillin', 'ampicillin', 'amoxicillin',
    'cefazolin', 'cefotaxime', 'ceftriaxone', 'ceftazidime',
    'cefepime', 'cefuroxime', 'cefoperazone',
    'meropenem', 'imipenem', 'ertapenem',
    'vancomycin', 'linezolid', 'teicoplanin',
    'piperacillin', 'aztreonam',
    'levofloxacin', 'ciprofloxacin', 'moxifloxacin', 'ofloxacin',
    'azithromycin', 'erythromycin', 'clarithromycin', 'roxithromycin',
    'clindamycin', 'metronidazole',
    'gentamicin', 'tobramycin', 'amikacin',
    'rifampin', 'doxycycline', 'tetracycline', 'minocycline',
    'chloramphenicol', 'sulfamethoxazole', 'trimethoprim',
    'nitrofurantoin', 'tigecycline', 'colistin', 'polymyxin',
    'fluconazole', 'voriconazole', 'itraconazole', 'amphotericin',
    'acyclovir', 'ganciclovir',
]

# ---- EMR Symptoms to use as binary features ----
# Selected based on: (1) clinical relevance to sepsis, (2) count > 500 in database
SEPSIS_RELEVANT_SYMPTOMS = [
    'fever',
    'cough',
    'rale',
    'twitching',
    'cyanosis',
    'anhelation',
    'listless',
    'cool extremities',
    'warm extremities',
    'edema',
    'hemorrhage',
    'dysphoria',
    'diarrhea',
    'skin jaundice',
    'arrhythmia',
    'vomiting',
    'swelling',
    'tenderness',
    'phlegm sound',
    'three depressions sign',
    'abdominal distension',
    'erythra',
    'anhelation and cyanosis',
    'moist rale',
    'dry and moist rales',
    'infection',
    'pharyngeal red',
    'headache',
    'chest tightness',
    'rebound tenderness',
    'abdominal tenderness',
]


# =============================================================================
# SECTION 2 — UNIT CONVERSIONS
# =============================================================================

def bili_to_mgdl(v):
    """Bilirubin: umol/L -> mg/dL"""
    return v / 17.1

def creat_to_mgdl(v):
    """Creatinine: umol/L -> mg/dL"""
    return v / 88.4

def glucose_to_mgdl(v):
    """Glucose: mmol/L -> mg/dL"""
    return v * 18.0


# =============================================================================
# SECTION 3 — PHOENIX SCORING FUNCTIONS
# These take WORST values and return score components.
# Used only for labeling, not for features.
# =============================================================================

def phoenix_cardiovascular(lactate, map_mmhg, n_vasoactives, age_months):
    """
    Cardiovascular score (0-6 points, sum of 3 sub-components):
    Vasoactives: 0=none, 1=one drug, 2=two or more
    Lactate:     0=<5, 1=5-10.9, 2=>=11 (mmol/L)
    MAP:         0/1/2 by age-adjusted threshold
    """
    score = 0

    # Vasoactives
    if n_vasoactives == 1:
        score += 1
    elif n_vasoactives >= 2:
        score += 2

    # Lactate (mmol/L)
    if lactate is not None:
        if lactate >= 11:
            score += 2
        elif lactate >= 5:
            score += 1

    # MAP — age-adjusted (age in months)
    if map_mmhg is not None and age_months is not None:
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


def phoenix_coagulation(platelets, inr, ddimer, fibrinogen_gl):
    """
    Coagulation score (0-2 points max, 1 point per abnormal lab):
    Platelets  < 100 (1000/uL)  -> 1 point
    INR        > 1.3            -> 1 point
    D-Dimer    > 2 (mg/L FEU)   -> 1 point
    Fibrinogen < 1.0 (g/L)      -> 1 point  [PIC unit is g/L, threshold = 100mg/dL = 1g/L]
    """
    score = 0
    if platelets is not None and platelets < 100:
        score += 1
    if inr is not None and inr > 1.3:
        score += 1
    if ddimer is not None and ddimer > 2:
        score += 1
    if fibrinogen_gl is not None and fibrinogen_gl < 1.0:
        score += 1
    return min(score, 2)


def phoenix_endocrine(glucose_mgdl):
    """
    Endocrine score (0-1 point):
    Glucose < 50 mg/dL OR > 150 mg/dL -> 1 point
    """
    if glucose_mgdl is None:
        return 0
    return 1 if (glucose_mgdl < 50 or glucose_mgdl > 150) else 0


def phoenix_immunologic(anc_10_9, alc_10_9):
    """
    Immunologic score (0-1 point):
    ANC < 0.5 x10^9/L  -> 1 point
    ALC < 1.0 x10^9/L  -> 1 point
    """
    if anc_10_9 is not None and anc_10_9 < 0.5:
        return 1
    if alc_10_9 is not None and alc_10_9 < 1.0:
        return 1
    return 0


def phoenix_renal(creat_mgdl, age_months):
    """
    Renal score (0-1 point): age-adjusted creatinine threshold (mg/dL)
    """
    if creat_mgdl is None or age_months is None:
        return 0
    if age_months < 1:      threshold = 0.8
    elif age_months < 12:   threshold = 0.3
    elif age_months < 24:   threshold = 0.4
    elif age_months < 60:   threshold = 0.6
    elif age_months < 144:  threshold = 0.7
    else:                   threshold = 1.0
    return 1 if creat_mgdl >= threshold else 0


def phoenix_hepatic(bilirubin_mgdl, alt_iul):
    """
    Hepatic score (0-1 point):
    Bilirubin >= 4 mg/dL -> 1 point
    ALT > 102 IU/L       -> 1 point
    """
    if bilirubin_mgdl is not None and bilirubin_mgdl >= 4:
        return 1
    if alt_iul is not None and alt_iul > 102:
        return 1
    return 0


# =============================================================================
# SECTION 4 — HELPER UTILITIES
# =============================================================================

def parse_dt(s):
    """Parse datetime string -> datetime object or None"""
    if not s or not s.strip():
        return None
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def safe_float(s):
    """String -> float or None"""
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def window_filter(events, intime, hours=24):
    """
    Keep only events within [intime, intime + hours].
    events: list of dicts with 'charttime' key (datetime).
    """
    end = intime + timedelta(hours=hours)
    return [e for e in events if intime <= e['charttime'] <= end]


def compute_stats(values):
    """
    Given a list of floats, return dict of stats.
    Returns None for everything if list is empty.
    Stats: max, min, mean, first (chronologically first), count
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return {'max': None, 'min': None, 'mean': None,
                'first': None, 'count': 0}
    return {
        'max':   max(clean),
        'min':   min(clean),
        'mean':  round(sum(clean) / len(clean), 4),
        'first': clean[0],       # already sorted by charttime when loaded
        'count': len(clean),
    }


def worst(values, direction='max'):
    """Single worst value — used only for Phoenix label calculation"""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(clean) if direction == 'max' else min(clean)


def compute_map(sys_val, dia_val):
    """MAP = diastolic + (systolic - diastolic) / 3"""
    if sys_val is None or dia_val is None:
        return None
    return round(dia_val + (sys_val - dia_val) / 3.0, 1)


def clean_col(name):
    """Make column name safe: lowercase, spaces to underscores"""
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')


# =============================================================================
# SECTION 5 — DATA LOADING FUNCTIONS
# One function per table. Each prints progress and returns clean data structure.
# =============================================================================

def load_patients(data_dir):
    print("  Loading PATIENTS...")
    out = {}
    with open(os.path.join(data_dir, 'PATIENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['SUBJECT_ID']] = {
                'gender':       row['GENDER'],
                'dob':          parse_dt(row['DOB']),
                'dod':          parse_dt(row['DOD']),
                'expire_flag':  int(row['EXPIRE_FLAG']),
            }
    print(f"    -> {len(out):,} patients")
    return out


def load_admissions(data_dir):
    print("  Loading ADMISSIONS...")
    out = {}
    with open(os.path.join(data_dir, 'ADMISSIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out[row['HADM_ID']] = {
                'subject_id':   row['SUBJECT_ID'],
                'admittime':    parse_dt(row['ADMITTIME']),
                'dischtime':    parse_dt(row['DISCHTIME']),
                'diagnosis':    row['DIAGNOSIS'],
                'icd10':        row['ICD10_CODE_CN'],
                'hosp_expire':  int(row['HOSPITAL_EXPIRE_FLAG']),
            }
    print(f"    -> {len(out):,} admissions")
    return out


def load_icustays(data_dir):
    print("  Loading ICUSTAYS...")
    out = []
    with open(os.path.join(data_dir, 'ICUSTAYS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            intime = parse_dt(row['INTIME'])
            if intime is None:
                continue
            out.append({
                'subject_id':   row['SUBJECT_ID'],
                'hadm_id':      row['HADM_ID'],
                'icustay_id':   row['ICUSTAY_ID'],
                'intime':       intime,
                'outtime':      parse_dt(row['OUTTIME']),
                'los':          safe_float(row['LOS']),
            })
    print(f"    -> {len(out):,} ICU stays")
    return out


def load_labevents(data_dir):
    """
    Load only the ITEMIDs we care about.
    Returns dict: subject_id -> list of {itemid, charttime, value}
    List is sorted by charttime so 'first' stat is correct.
    """
    print("  Loading LABEVENTS (large file, please wait)...")
    all_ids = set()
    for ids in LAB_ITEMIDS.values():
        all_ids.update(ids)

    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'LABEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ITEMID'] not in all_ids:
                continue
            val = safe_float(row['VALUENUM'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'itemid':    row['ITEMID'],
                'charttime': t,
                'value':     val,
            })
            count += 1

    # Sort each patient's labs by time so 'first' is truly first
    for sid in out:
        out[sid].sort(key=lambda x: x['charttime'])

    print(f"    -> {count:,} relevant lab events across {len(out):,} patients")
    return out


def load_chartevents(data_dir):
    """
    Load only the ITEMIDs we care about.
    Returns dict: subject_id -> list of {itemid, charttime, value}
    """
    print("  Loading CHARTEVENTS (large file, please wait)...")
    all_ids = set(CHART_ITEMIDS.values())

    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'CHARTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ITEMID'] not in all_ids:
                continue
            val = safe_float(row['VALUENUM'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'itemid':    row['ITEMID'],
                'charttime': t,
                'value':     val,
            })
            count += 1

    for sid in out:
        out[sid].sort(key=lambda x: x['charttime'])

    print(f"    -> {count:,} relevant chart events across {len(out):,} patients")
    return out


def load_prescriptions(data_dir):
    """
    Load only antibiotics and vasoactives.
    Returns dict: subject_id -> list of {drug_name, startdate, is_vasoactive, is_antibiotic}
    """
    print("  Loading PRESCRIPTIONS (large file, please wait)...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            drug    = row['DRUG_NAME_EN'].lower()
            is_vaso = any(k in drug for k in VASOACTIVE_KEYWORDS)
            is_abx  = any(k in drug for k in ANTIBIOTIC_KEYWORDS)
            if not (is_vaso or is_abx):
                continue
            t = parse_dt(row['STARTDATE'])
            if t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'hadm_id':       row['HADM_ID'],
                'drug_name':     row['DRUG_NAME_EN'],
                'drug_lower':    drug,
                'startdate':     t,
                'is_vasoactive': is_vaso,
                'is_antibiotic': is_abx,
            })
            count += 1
    print(f"    -> {count:,} relevant prescriptions across {len(out):,} patients")
    return out


def load_microbiologyevents(data_dir):
    """
    Returns dict: subject_id -> list of {charttime, hadm_id, has_organism}
    has_organism = True if an actual pathogen was identified (not 'no growth')
    """
    print("  Loading MICROBIOLOGYEVENTS...")
    # Keywords that indicate no pathogen found
    negative_keywords = [
        'no bacterial', 'no bacteria', 'no growth', 'no fungus',
        'normal flora', 'uncultured', 'not detected', 'not found',
        'no pathogenic', '无细菌', '无生长', '正常菌群',
    ]
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'MICROBIOLOGYEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = parse_dt(row['CHARTTIME'])
            if t is None:
                continue
            org = row['ORG_NAME'].lower()
            has_organism = (
                len(org) > 0 and
                not any(k in org for k in negative_keywords)
            )
            out[row['SUBJECT_ID']].append({
                'hadm_id':      row['HADM_ID'],
                'charttime':    t,
                'has_organism': has_organism,
            })
            count += 1
    print(f"    -> {count:,} microbiology events across {len(out):,} patients")
    return out


def load_outputevents(data_dir):
    """
    Urine output only (ITEMID 1034).
    Returns dict: subject_id -> list of {charttime, value_ml}
    """
    print("  Loading OUTPUTEVENTS...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'OUTPUTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['ITEMID'] != '1034':
                continue
            val = safe_float(row['VALUE'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'charttime': t,
                'value':     val,
            })
            count += 1
    print(f"    -> {count:,} urine output events across {len(out):,} patients")
    return out


def load_inputevents(data_dir):
    """
    Total fluid input.
    Returns dict: subject_id -> list of {charttime, amount_ml}
    """
    print("  Loading INPUTEVENTS...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'INPUTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            val = safe_float(row['AMOUNT'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'charttime': t,
                'value':     val,
            })
            count += 1
    print(f"    -> {count:,} fluid input events across {len(out):,} patients")
    return out


def load_emr_symptoms(data_dir):
    """
    Returns dict: subject_id -> dict of {symptom_name: attribute}
    attribute: '+' = present, '-' = absent
    We only keep the FIRST recording of each symptom per patient (admission snapshot).
    """
    print("  Loading EMR_SYMPTOMS...")
    # subject -> {symptom -> attribute at first recordtime}
    out = defaultdict(dict)
    # Track first recordtime per subject to get admission snapshot
    first_time = {}
    count = 0

    rows_by_subject = defaultdict(list)
    with open(os.path.join(data_dir, 'EMR_SYMPTOMS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            t = parse_dt(row['RECORDTIME'])
            if t is None:
                continue
            rows_by_subject[row['SUBJECT_ID']].append({
                'recordtime':   t,
                'symptom':      row['SYMPTOM_NAME'].strip().lower(),
                'attribute':    row['SYMPTOM_ATTRIBUTE'].strip(),
            })
            count += 1

    # For each patient, take first EMR record (admission assessment)
    for sid, rows in rows_by_subject.items():
        rows.sort(key=lambda x: x['recordtime'])
        earliest_time = rows[0]['recordtime']
        # Take all symptoms from the first EMR entry (same timestamp)
        for r in rows:
            if r['recordtime'] == earliest_time:
                out[sid][r['symptom']] = r['attribute']

    print(f"    -> {count:,} symptom records, {len(out):,} patients with symptoms")
    return out


# =============================================================================
# SECTION 6 — PER-STAY FEATURE EXTRACTION HELPERS
# =============================================================================

def get_lab_stats(patient_labs_window, itemid_list, converter=None):
    """
    Extract values for given itemids from windowed lab events.
    Apply converter function if provided (unit conversion).
    Returns stats dict from compute_stats().
    """
    vals = []
    for e in patient_labs_window:
        if e['itemid'] in itemid_list:
            v = converter(e['value']) if converter else e['value']
            vals.append(v)
    return compute_stats(vals)


def get_chart_stats(patient_charts_window, itemid):
    """
    Extract values for a single itemid from windowed chart events.
    Returns stats dict from compute_stats().
    """
    vals = [e['value'] for e in patient_charts_window if e['itemid'] == itemid]
    return compute_stats(vals)


def get_worst_lab(patient_labs_window, itemid_list, direction, converter=None):
    """Get single worst value — used only for Phoenix label calculation"""
    vals = []
    for e in patient_labs_window:
        if e['itemid'] in itemid_list:
            v = converter(e['value']) if converter else e['value']
            vals.append(v)
    return worst(vals, direction)


def get_worst_chart(patient_charts_window, itemid, direction):
    vals = [e['value'] for e in patient_charts_window if e['itemid'] == itemid]
    return worst(vals, direction)


def check_suspected_infection(sid, intime, prescriptions, micro, hours=24):
    """
    Phoenix infection criterion:
    Antibiotic prescribed AND microbiology culture ordered
    both within 24h of ICU admission.
    Returns: (bool, n_antibiotics, n_cultures, n_positive_cultures)
    """
    end = intime + timedelta(hours=hours)

    abx = [
        rx for rx in prescriptions.get(sid, [])
        if rx['is_antibiotic'] and intime <= rx['startdate'] <= end
    ]
    cultures = [
        m for m in micro.get(sid, [])
        if intime <= m['charttime'] <= end
    ]
    positive_cultures = [m for m in cultures if m['has_organism']]

    has_infection = len(abx) > 0 and len(cultures) > 0
    return has_infection, len(abx), len(cultures), len(positive_cultures)


def count_vasoactives(sid, intime, prescriptions, hours=24):
    """
    Count distinct vasoactive drug types within 24h.
    Returns: (total_count, dict of individual drug flags)
    """
    end = intime + timedelta(hours=hours)
    found = set()
    for rx in prescriptions.get(sid, []):
        if not rx['is_vasoactive']:
            continue
        if not (intime <= rx['startdate'] <= end):
            continue
        drug = rx['drug_lower']
        for keyword in VASOACTIVE_KEYWORDS:
            if keyword in drug:
                found.add(keyword)
                break

    individual = {}
    for name, keywords in VASOACTIVE_INDIVIDUAL.items():
        individual[f'vaso_{name}'] = int(any(k in found for k in keywords))

    return len(found), individual


def get_symptom_flags(sid, symptoms_data):
    """
    Return binary flags for each sepsis-relevant symptom.
    1 = present (+), 0 = absent (-) or not documented.
    """
    patient_symptoms = symptoms_data.get(sid, {})
    flags = {}
    for symptom in SEPSIS_RELEVANT_SYMPTOMS:
        col = f'symptom_{clean_col(symptom)}'
        attr = patient_symptoms.get(symptom.lower(), None)
        # 1 if documented as present, 0 if absent or not documented
        flags[col] = 1 if attr == '+' else 0
    return flags


# =============================================================================
# SECTION 7 — MAIN PIPELINE
# =============================================================================

def run_pipeline(data_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # STAGE 1 — Load all tables
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 1: Loading all tables")
    print("="*60)

    patients      = load_patients(data_dir)
    admissions    = load_admissions(data_dir)
    icustays      = load_icustays(data_dir)
    labs          = load_labevents(data_dir)
    charts        = load_chartevents(data_dir)
    prescriptions = load_prescriptions(data_dir)
    micro         = load_microbiologyevents(data_dir)
    output_events = load_outputevents(data_dir)
    input_events  = load_inputevents(data_dir)
    symptoms      = load_emr_symptoms(data_dir)

    # ------------------------------------------------------------------
    # STAGE 2 — Filter ICU stays by age 0-18
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 2: Filtering to age 0-18 years")
    print("="*60)

    eligible = []
    excluded_age = 0

    for stay in icustays:
        sid = stay['subject_id']
        p   = patients.get(sid)
        if p is None or p['dob'] is None:
            continue

        age_days   = (stay['intime'] - p['dob']).days
        age_years  = age_days / 365.25
        age_months = age_days / 30.4375

        # Valid pediatric range: 0 to 18 years
        # Exclude negatives (data error) and >18 (adult)
        # Exclude >300 years (de-identification artifact for old patients)
        if 0 <= age_years <= 18:
            stay['age_years']   = round(age_years, 3)
            stay['age_months']  = round(age_months, 1)
            stay['gender']      = p['gender']
            stay['expire_flag'] = p['expire_flag']
            eligible.append(stay)
        else:
            excluded_age += 1

    print(f"  Eligible stays (age 0-18): {len(eligible):,}")
    print(f"  Excluded (age out of range or error): {excluded_age:,}")

    # ------------------------------------------------------------------
    # STAGE 3-7 — Per-stay extraction
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 3-7: Extracting features and computing Phoenix scores")
    print("="*60)

    records        = []
    n_sepsis       = 0
    n_no_sepsis    = 0
    n_no_infection = 0

    for i, stay in enumerate(eligible):

        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(eligible):,} stays | "
                  f"sepsis so far: {n_sepsis:,}")

        sid        = stay['subject_id']
        hadm_id    = stay['hadm_id']
        intime     = stay['intime']
        age_months = stay['age_months']

        # ---- STAGE 3: Suspected Infection ----
        has_infection, n_abx, n_cultures, n_pos_cultures = \
            check_suspected_infection(sid, intime, prescriptions, micro)

        # ---- STAGE 4: Filter events to first 24h window ----
        labs_w   = window_filter(labs.get(sid, []),          intime, 24)
        charts_w = window_filter(charts.get(sid, []),        intime, 24)
        output_w = window_filter(output_events.get(sid, []), intime, 24)
        input_w  = window_filter(input_events.get(sid, []),  intime, 24)

        # ---- STAGE 5: Compute 5-stat features (Approach 2) ----
        # Each call returns {'max', 'min', 'mean', 'first', 'count'}

        # Labs — raw units
        s_lactate     = get_lab_stats(labs_w, LAB_ITEMIDS['lactate'])
        s_platelets   = get_lab_stats(labs_w, LAB_ITEMIDS['platelets'])
        s_inr         = get_lab_stats(labs_w, LAB_ITEMIDS['inr'])
        s_ddimer      = get_lab_stats(labs_w, LAB_ITEMIDS['ddimer'])
        s_fibrinogen  = get_lab_stats(labs_w, LAB_ITEMIDS['fibrinogen'])
        s_alt         = get_lab_stats(labs_w, LAB_ITEMIDS['alt'])
        s_ast         = get_lab_stats(labs_w, LAB_ITEMIDS['ast'])
        s_anc         = get_lab_stats(labs_w, LAB_ITEMIDS['anc'])
        s_alc         = get_lab_stats(labs_w, LAB_ITEMIDS['alc'])
        s_wbc         = get_lab_stats(labs_w, LAB_ITEMIDS['wbc'])
        s_ph          = get_lab_stats(labs_w, LAB_ITEMIDS['ph'])
        s_pco2        = get_lab_stats(labs_w, LAB_ITEMIDS['pco2'])
        s_pao2        = get_lab_stats(labs_w, LAB_ITEMIDS['pao2'])
        s_be          = get_lab_stats(labs_w, LAB_ITEMIDS['base_excess'])
        s_hco3        = get_lab_stats(labs_w, LAB_ITEMIDS['bicarbonate'])
        s_ag          = get_lab_stats(labs_w, LAB_ITEMIDS['anion_gap'])
        s_spo2_lab    = get_lab_stats(labs_w, LAB_ITEMIDS['spo2_lab'])
        s_hemoglobin  = get_lab_stats(labs_w, LAB_ITEMIDS['hemoglobin'])
        s_hematocrit  = get_lab_stats(labs_w, LAB_ITEMIDS['hematocrit'])
        s_pt          = get_lab_stats(labs_w, LAB_ITEMIDS['pt'])
        s_ptt         = get_lab_stats(labs_w, LAB_ITEMIDS['ptt'])
        s_sodium      = get_lab_stats(labs_w, LAB_ITEMIDS['sodium'])
        s_potassium   = get_lab_stats(labs_w, LAB_ITEMIDS['potassium'])
        s_calcium     = get_lab_stats(labs_w, LAB_ITEMIDS['calcium'])
        s_albumin     = get_lab_stats(labs_w, LAB_ITEMIDS['albumin'])
        s_crp         = get_lab_stats(labs_w, LAB_ITEMIDS['crp'])
        s_ldh         = get_lab_stats(labs_w, LAB_ITEMIDS['ldh'])
        s_urea        = get_lab_stats(labs_w, LAB_ITEMIDS['urea'])
        s_uric_acid   = get_lab_stats(labs_w, LAB_ITEMIDS['uric_acid'])
        s_ck          = get_lab_stats(labs_w, LAB_ITEMIDS['ck'])

        # Labs — converted units (for Phoenix scoring and as features)
        s_creatinine_mgdl  = get_lab_stats(labs_w, LAB_ITEMIDS['creatinine'],
                                            converter=creat_to_mgdl)
        s_bilirubin_mgdl   = get_lab_stats(labs_w, LAB_ITEMIDS['bilirubin'],
                                            converter=bili_to_mgdl)
        s_glucose_mgdl     = get_lab_stats(labs_w, LAB_ITEMIDS['glucose'],
                                            converter=glucose_to_mgdl)

        # Also keep original units as features
        s_creatinine_umol = get_lab_stats(labs_w, LAB_ITEMIDS['creatinine'])
        s_bilirubin_umol  = get_lab_stats(labs_w, LAB_ITEMIDS['bilirubin'])
        s_glucose_mmol    = get_lab_stats(labs_w, LAB_ITEMIDS['glucose'])

        # Chart vitals
        s_spo2      = get_chart_stats(charts_w, CHART_ITEMIDS['spo2'])
        s_sys_bp    = get_chart_stats(charts_w, CHART_ITEMIDS['systolic_bp'])
        s_dia_bp    = get_chart_stats(charts_w, CHART_ITEMIDS['diastolic_bp'])
        s_hr        = get_chart_stats(charts_w, CHART_ITEMIDS['heart_rate'])
        s_rr        = get_chart_stats(charts_w, CHART_ITEMIDS['resp_rate'])
        s_temp      = get_chart_stats(charts_w, CHART_ITEMIDS['temperature'])
        s_bg_chart  = get_chart_stats(charts_w, CHART_ITEMIDS['blood_glucose_chart'])

        # MAP — computed from systolic and diastolic
        # We pair each reading by time and compute MAP for each pair
        sys_by_time = {e['charttime']: e['value']
                       for e in charts_w if e['itemid'] == CHART_ITEMIDS['systolic_bp']}
        dia_by_time = {e['charttime']: e['value']
                       for e in charts_w if e['itemid'] == CHART_ITEMIDS['diastolic_bp']}
        map_vals = []
        for t in sorted(sys_by_time):
            if t in dia_by_time:
                m = compute_map(sys_by_time[t], dia_by_time[t])
                if m is not None:
                    map_vals.append(m)
        s_map = compute_stats(map_vals)

        # Fluid balance
        urine_total_24h = sum(e['value'] for e in output_w)
        fluid_total_24h = sum(e['value'] for e in input_w)
        icu_hours = max(
            stay['los'] * 24 if stay['los'] else 24,
            1  # avoid division by zero
        )
        urine_per_hour = round(urine_total_24h / min(icu_hours, 24), 2) \
                         if urine_total_24h > 0 else 0

        # Vasoactives
        n_vasoactives, vaso_flags = count_vasoactives(sid, intime, prescriptions)

        # Symptoms
        symptom_flags = get_symptom_flags(sid, symptoms)

        # ---- STAGE 6: Phoenix scoring (uses WORST values for label) ----
        # Get worst values specifically for label calculation
        w_lactate    = get_worst_lab(labs_w, LAB_ITEMIDS['lactate'],     'max')
        w_platelets  = get_worst_lab(labs_w, LAB_ITEMIDS['platelets'],   'min')
        w_inr        = get_worst_lab(labs_w, LAB_ITEMIDS['inr'],         'max')
        w_ddimer     = get_worst_lab(labs_w, LAB_ITEMIDS['ddimer'],      'max')
        w_fibrinogen = get_worst_lab(labs_w, LAB_ITEMIDS['fibrinogen'],  'min')
        w_creat_mgdl = get_worst_lab(labs_w, LAB_ITEMIDS['creatinine'],  'max',
                                     converter=creat_to_mgdl)
        w_bili_mgdl  = get_worst_lab(labs_w, LAB_ITEMIDS['bilirubin'],   'max',
                                     converter=bili_to_mgdl)
        w_alt        = get_worst_lab(labs_w, LAB_ITEMIDS['alt'],         'max')
        w_anc        = get_worst_lab(labs_w, LAB_ITEMIDS['anc'],         'min')
        w_alc        = get_worst_lab(labs_w, LAB_ITEMIDS['alc'],         'min')

        # Worst MAP
        w_sys_bp     = get_worst_chart(charts_w, CHART_ITEMIDS['systolic_bp'],  'min')
        w_dia_bp     = get_worst_chart(charts_w, CHART_ITEMIDS['diastolic_bp'], 'min')
        w_map        = compute_map(w_sys_bp, w_dia_bp)

        # Worst glucose for endocrine: check both extremes
        g_max = s_glucose_mgdl['max']
        g_min = s_glucose_mgdl['min']
        if g_min is not None and g_min < 50:
            w_glucose_endo = g_min
        elif g_max is not None and g_max > 150:
            w_glucose_endo = g_max
        else:
            w_glucose_endo = g_max

        # Compute Phoenix score components
        s_cardio  = phoenix_cardiovascular(w_lactate, w_map, n_vasoactives, age_months)
        s_coag    = phoenix_coagulation(w_platelets, w_inr, w_ddimer, w_fibrinogen)
        s_neuro   = 0   # GCS unavailable in PIC
        s_resp    = 0   # FiO2/ventilation unavailable in PIC
        s_endo    = phoenix_endocrine(w_glucose_endo)
        s_immuno  = phoenix_immunologic(w_anc, w_alc)
        s_renal   = phoenix_renal(w_creat_mgdl, age_months)
        s_hepatic = phoenix_hepatic(w_bili_mgdl, w_alt)

        phoenix_core = s_resp + s_cardio + s_coag + s_neuro
        phoenix_8    = phoenix_core + s_endo + s_immuno + s_renal + s_hepatic

        # ---- STAGE 7: Label ----
        sepsis_label = 1 if (has_infection and phoenix_core >= 2) else 0

        if not has_infection:
            n_no_infection += 1
        elif sepsis_label == 1:
            n_sepsis += 1
        else:
            n_no_sepsis += 1

        # ---- Build record ----
        adm = admissions.get(hadm_id, {})

        def flat(prefix, stats_dict):
            """Flatten stats dict into prefixed columns"""
            return {
                f'{prefix}_max':   stats_dict['max'],
                f'{prefix}_min':   stats_dict['min'],
                f'{prefix}_mean':  stats_dict['mean'],
                f'{prefix}_first': stats_dict['first'],
                f'{prefix}_count': stats_dict['count'],
            }

        record = {}

        # Identifiers
        record['subject_id']    = sid
        record['hadm_id']       = hadm_id
        record['icustay_id']    = stay['icustay_id']

        # Demographics
        record['age_years']     = stay['age_years']
        record['age_months']    = stay['age_months']
        record['gender']        = stay['gender']
        record['los_days']      = stay['los']
        record['hospital_expire_flag'] = stay['expire_flag']

        # Admission info
        record['admission_diagnosis'] = adm.get('diagnosis', '')
        record['icd10_code']          = adm.get('icd10', '')

        # Infection
        record['suspected_infection']   = int(has_infection)
        record['n_antibiotics_24h']     = n_abx
        record['n_cultures_24h']        = n_cultures
        record['n_positive_cultures']   = n_pos_cultures

        # Vasoactives
        record['n_vasoactives_24h'] = n_vasoactives
        record.update(vaso_flags)

        # Fluid balance
        record['urine_output_total_ml']  = urine_total_24h
        record['urine_output_per_hour']  = urine_per_hour
        record['fluid_input_total_ml']   = fluid_total_24h

        # Lab features — 5 stats each
        record.update(flat('lactate',           s_lactate))
        record.update(flat('platelets',         s_platelets))
        record.update(flat('inr',               s_inr))
        record.update(flat('ddimer',            s_ddimer))
        record.update(flat('fibrinogen',        s_fibrinogen))
        record.update(flat('alt',               s_alt))
        record.update(flat('ast',               s_ast))
        record.update(flat('anc',               s_anc))
        record.update(flat('alc',               s_alc))
        record.update(flat('wbc',               s_wbc))
        record.update(flat('ph',                s_ph))
        record.update(flat('pco2',              s_pco2))
        record.update(flat('pao2',              s_pao2))
        record.update(flat('base_excess',       s_be))
        record.update(flat('bicarbonate',       s_hco3))
        record.update(flat('anion_gap',         s_ag))
        record.update(flat('spo2_lab',          s_spo2_lab))
        record.update(flat('hemoglobin',        s_hemoglobin))
        record.update(flat('hematocrit',        s_hematocrit))
        record.update(flat('pt',                s_pt))
        record.update(flat('ptt',               s_ptt))
        record.update(flat('sodium',            s_sodium))
        record.update(flat('potassium',         s_potassium))
        record.update(flat('calcium',           s_calcium))
        record.update(flat('albumin',           s_albumin))
        record.update(flat('crp',               s_crp))
        record.update(flat('ldh',               s_ldh))
        record.update(flat('urea',              s_urea))
        record.update(flat('uric_acid',         s_uric_acid))
        record.update(flat('ck',                s_ck))

        # Converted unit features
        record.update(flat('creatinine_mgdl',   s_creatinine_mgdl))
        record.update(flat('bilirubin_mgdl',    s_bilirubin_mgdl))
        record.update(flat('glucose_mgdl',      s_glucose_mgdl))
        record.update(flat('creatinine_umol',   s_creatinine_umol))
        record.update(flat('bilirubin_umol',    s_bilirubin_umol))
        record.update(flat('glucose_mmol',      s_glucose_mmol))

        # Vital features — 5 stats each
        record.update(flat('spo2',              s_spo2))
        record.update(flat('systolic_bp',       s_sys_bp))
        record.update(flat('diastolic_bp',      s_dia_bp))
        record.update(flat('map',               s_map))
        record.update(flat('heart_rate',        s_hr))
        record.update(flat('resp_rate',         s_rr))
        record.update(flat('temperature',       s_temp))
        record.update(flat('blood_glucose_chart', s_bg_chart))

        # Phoenix score components (for reference/analysis)
        record['score_respiratory']   = s_resp    # always 0
        record['score_cardiovascular']= s_cardio
        record['score_coagulation']   = s_coag
        record['score_neurological']  = s_neuro   # always 0
        record['score_endocrine']     = s_endo
        record['score_immunologic']   = s_immuno
        record['score_renal']         = s_renal
        record['score_hepatic']       = s_hepatic
        record['phoenix_core_score']  = phoenix_core
        record['phoenix_8_score']     = phoenix_8

        # Symptom flags
        record.update(symptom_flags)

        # LABEL — what the model predicts
        record['sepsis_label'] = sepsis_label

        records.append(record)

    # ------------------------------------------------------------------
    # STAGE 8 — Write output CSV
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STAGE 8: Writing output")
    print("="*60)

    output_csv = os.path.join(output_dir, 'pic_sepsis_cohort.csv')

    if records:
        fieldnames = list(records[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved {len(records):,} records -> {output_csv}")
        print(f"  Columns: {len(fieldnames)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total     = len(records)
    prev      = f"{n_sepsis/total*100:.1f}%" if total else "N/A"
    pos_neg   = f"{n_no_sepsis/(n_sepsis if n_sepsis else 1):.1f}:1"

    summary = [
        "=" * 60,
        "EXTRACTION SUMMARY",
        "=" * 60,
        f"Total patients in PIC:                {len(patients):>8,}",
        f"Total ICU stays:                      {len(icustays):>8,}",
        f"ICU stays age 0-18 (eligible):        {len(eligible):>8,}",
        f"  -> No suspected infection:          {n_no_infection:>8,}",
        f"  -> Suspected infection, no sepsis:  {n_no_sepsis:>8,}",
        f"  -> SEPSIS (label=1):                {n_sepsis:>8,}",
        f"  -> Total records in output CSV:     {total:>8,}",
        f"",
        f"Sepsis prevalence:                    {prev}",
        f"Negative:Positive ratio:              {pos_neg}",
        f"Total feature columns:                {len(fieldnames) if records else 0}",
        f"",
        "=" * 60,
        "KNOWN LIMITATIONS",
        "=" * 60,
        "Respiratory score: NOT computed — FiO2, PaO2/FiO2, SpO2/FiO2,",
        "  and mechanical ventilation flag unavailable in PIC database.",
        "  Score set to 0 per Phoenix missing-data convention.",
        "Neurological score: NOT computed — GCS and pupil data",
        "  unavailable in PIC database. Score set to 0.",
        "Consequence: Label is conservative. Patients septic only via",
        "  respiratory/neuro dysfunction will be mislabeled as non-sepsis.",
        "  This is consistent with all prior PIC-based ML studies.",
        "",
        "=" * 60,
        "UNIT CONVERSIONS APPLIED",
        "=" * 60,
        "Bilirubin:  umol/L -> mg/dL  (divide by 17.1)",
        "Creatinine: umol/L -> mg/dL  (divide by 88.4)",
        "Glucose:    mmol/L -> mg/dL  (multiply by 18)",
        "Fibrinogen: g/L, threshold 1.0 g/L (= 100 mg/dL in Phoenix)",
        "ANC/ALC:    10^9/L, threshold 0.5 and 1.0 respectively",
        "",
        "=" * 60,
        "FEATURE ENGINEERING",
        "=" * 60,
        "Each continuous variable has 5 columns:",
        "  _max   : maximum value in first 24h (worst for high-bad variables)",
        "  _min   : minimum value in first 24h (worst for low-bad variables)",
        "  _mean  : average value in first 24h",
        "  _first : first recorded value (admission state)",
        "  _count : number of measurements (proxy for clinical concern)",
        "",
        "Phoenix label uses WORST values per paper specification.",
        "Model features use all 5 statistics (Approach 2).",
    ]

    summary_path = os.path.join(output_dir, 'extraction_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))

    print('\n' + '\n'.join(summary))
    print(f"\nSummary saved -> {summary_path}")
    print("\nDone. Ready for modelling.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PIC Phoenix Sepsis Cohort Extraction v2'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Path to V1.1.0 folder'
    )
    parser.add_argument(
        '--output_dir',
        default='output',
        help='Output folder (default: output/)'
    )
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir)
