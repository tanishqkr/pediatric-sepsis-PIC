"""
STEP 3.1 — Feature Extraction (Fixed)
=============================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
Reads the Step 1 cohort (output2/cohort_step1.csv) and all raw V1.1.0
tables. For each eligible ICU stay extracts features from the first 24
hours of ICU admission.

Caps from Step 2.1 (output2/logs/step2.1_caps.json) are applied on the
fly — values outside physiological range are set to None before any
statistics are computed.

Feature groups extracted
------------------------
  A. Lab values (LABEVENTS)          — 5 stats each: max, min, mean, first, count
  B. Vital signs (CHARTEVENTS)       — 5 stats each
  C. Derived vitals                  — MAP (computed), shock index (HR/SBP)
  D. Trend features                  — change = last - first for key variables
  E. Age-adjusted vital flags        — binary: is HR/RR/BP abnormal for age?
  F. Fluid balance (OUTPUT+INPUT)    — total urine output, urine/hr, fluid input
  G. Vasoactive drugs (PRESCRIPTIONS)— count + individual drug flags
  H. Infection markers (MICRO+RX)    — antibiotic count, culture count, positive culture
  I. Clinical symptoms (EMR)         — 31 binary flags
  J. Surgical context (SURGERY_INFO) — had surgery flag, cardiac surgery flag

Feature engineering notes
--------------------------
  5-statistic method per variable:
    _max   : maximum in 24h window (worst for high=bad variables)
    _min   : minimum in 24h window (worst for low=bad variables)
    _mean  : mean in 24h window (average burden)
    _first : first recorded value (admission state, before intervention)
    _count : number of readings (proxy for clinical concern)

  Trend features (new vs previous pipeline):
    _trend = last_value - first_value
    Positive = getting worse for high=bad variables (e.g. rising lactate)
    Negative = improving

  Shock index (new):
    shock_index = heart_rate_mean / systolic_bp_mean
    Classic emergency medicine severity score. >1.0 indicates shock.

  Age-adjusted vital flags (new):
    Using published pediatric reference ranges (Flemmer & Hill 2007,
    Fleming et al. 2011). Binary 1=abnormal for age, 0=normal.

Unit conversions applied during extraction
------------------------------------------
  Haemoglobin : raw g/L  × 0.1  = g/dL   (PIC stores in g/L)
  Bilirubin   : umol/L   ÷ 17.1 = mg/dL
  Creatinine  : umol/L   ÷ 88.4 = mg/dL
  Glucose     : mmol/L   × 18   = mg/dL

Collinearity fix
----------------
  Previous pipeline kept both raw and converted units as features.
  This version keeps ONLY the clinically standard unit per variable.
  Raw unit versions are dropped from features (but Phoenix scoring
  still uses the correct units internally).

  blood_glucose_chart (97.6% missing) excluded entirely.

Outputs
-------
  output2/cohort_step3.1_features.csv   — full feature matrix, no imputation yet
  output2/logs/step3.1_log.txt          — feature counts, missing rates
"""

import os
import csv
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = "V1.1.0"
OUTPUT_DIR  = "output2"
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
COHORT_FILE = os.path.join(OUTPUT_DIR, "cohort_step1.csv")
CAPS_FILE   = os.path.join(LOG_DIR, "step2.1_caps.json")
OUT_CSV     = os.path.join(OUTPUT_DIR, "cohort_step3.1_features.csv")
WINDOW_HOURS = 24


# =============================================================================
# LAB ITEMID CONFIGURATION
# =============================================================================
# Format: 'feature_name': {'itemids': [...], 'unit': '...'}
# Unit is what the feature will be stored as after conversion

LAB_FEATURES = {
    # Phoenix core
    'lactate':          {'itemids': ['5227'],                   'unit': 'mmol/L'},
    'platelets':        {'itemids': ['5129'],                   'unit': '10^9/L'},
    'inr':              {'itemids': ['5174'],                   'unit': 'ratio'},
    'ddimer':           {'itemids': ['5163'],                   'unit': 'mg/L'},
    'fibrinogen':       {'itemids': ['5164'],                   'unit': 'g/L'},
    # Renal — converted umol/L -> mg/dL
    'creatinine':       {'itemids': ['5032','5041','6954'],     'unit': 'mg/dL',
                         'convert': lambda x: x / 88.4},
    # Hepatic — converted umol/L -> mg/dL
    'bilirubin':        {'itemids': ['5075','5255'],            'unit': 'mg/dL',
                         'convert': lambda x: x / 17.1},
    'alt':              {'itemids': ['5026','5195'],            'unit': 'IU/L'},
    'ast':              {'itemids': ['5031'],                   'unit': 'IU/L'},
    # Endocrine — converted mmol/L -> mg/dL
    'glucose':          {'itemids': ['5047','5223'],            'unit': 'mg/dL',
                         'convert': lambda x: x * 18.0},
    # Immunologic
    'anc':              {'itemids': ['5094'],                   'unit': '10^9/L'},
    'alc':              {'itemids': ['5110'],                   'unit': '10^9/L'},
    'wbc':              {'itemids': ['5141'],                   'unit': '10^9/L'},
    # Blood gas
    'ph':               {'itemids': ['5237','5238'],            'unit': 'pH'},
    'pco2':             {'itemids': ['5235','5236'],            'unit': 'mmHg'},
    'pao2':             {'itemids': ['5239','5244'],            'unit': 'mmHg'},
    'base_excess':      {'itemids': ['5211','5249'],            'unit': 'mmol/L'},
    'bicarbonate':      {'itemids': ['5248','5224'],            'unit': 'mmol/L'},
    'anion_gap':        {'itemids': ['5212','5213'],            'unit': 'mmol/L'},
    'spo2_lab':         {'itemids': ['5252'],                   'unit': '%'},
    # Haematology — haemoglobin converted g/L -> g/dL
    'hemoglobin':       {'itemids': ['5099','5257'],            'unit': 'g/dL',
                         'convert': lambda x: x * 0.1},
    'hematocrit':       {'itemids': ['5097','5225'],            'unit': '%'},
    'pt':               {'itemids': ['5186'],                   'unit': 'seconds'},
    'ptt':              {'itemids': ['5161'],                   'unit': 'seconds'},
    # Chemistry
    'sodium':           {'itemids': ['5062','5230'],            'unit': 'mmol/L'},
    'potassium':        {'itemids': ['5056','5226'],            'unit': 'mmol/L'},
    'calcium':          {'itemids': ['5034','5215'],            'unit': 'mmol/L'},
    'albumin':          {'itemids': ['5024'],                   'unit': 'g/L'},
    'crp':              {'itemids': ['5626','5821'],            'unit': 'mg/L'},
    'ldh':              {'itemids': ['5057'],                   'unit': 'IU/L'},
    'urea':             {'itemids': ['5033'],                   'unit': 'mmol/L'},
    'uric_acid':        {'itemids': ['5083'],                   'unit': 'umol/L'},
    'ck':               {'itemids': ['5038'],                   'unit': 'IU/L'},
}

# Variables for which we compute trend (last - first)
TREND_VARS = [
    'lactate', 'platelets', 'creatinine', 'bilirubin',
    'inr', 'wbc', 'hemoglobin', 'glucose', 'ph',
    'base_excess', 'sodium', 'potassium',
]

# =============================================================================
# CHART ITEMID CONFIGURATION
# =============================================================================

CHART_FEATURES = {
    'spo2':         {'itemid': '1006', 'unit': '%'},
    'systolic_bp':  {'itemid': '1016', 'unit': 'mmHg'},
    'diastolic_bp': {'itemid': '1015', 'unit': 'mmHg'},
    'heart_rate':   {'itemid': '1003', 'unit': 'bpm'},
    'resp_rate':    {'itemid': '1004', 'unit': 'breaths/min'},
    'temperature':  {'itemid': '1001', 'unit': 'Celsius'},
}

# =============================================================================
# VASOACTIVE AND ANTIBIOTIC KEYWORDS
# =============================================================================

VASOACTIVE_KEYWORDS = [
    'dopamine', 'epinephrine', 'adrenaline',
    'norepinephrine', 'noradrenaline',
    'vasopressin', 'dobutamine', 'milrinone',
]

VASOACTIVE_INDIVIDUAL = {
    'dopamine':       ['dopamine'],
    'epinephrine':    ['epinephrine', 'adrenaline'],
    'norepinephrine': ['norepinephrine', 'noradrenaline'],
    'vasopressin':    ['vasopressin'],
    'dobutamine':     ['dobutamine'],
    'milrinone':      ['milrinone'],
}

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

# =============================================================================
# EMR SYMPTOMS
# =============================================================================

SEPSIS_SYMPTOMS = [
    'fever', 'cough', 'rale', 'twitching', 'cyanosis',
    'anhelation', 'listless', 'cool extremities', 'warm extremities',
    'edema', 'hemorrhage', 'dysphoria', 'diarrhea', 'skin jaundice',
    'arrhythmia', 'vomiting', 'swelling', 'tenderness',
    'phlegm sound', 'three depressions sign', 'abdominal distension',
    'erythra', 'anhelation and cyanosis', 'moist rale',
    'dry and moist rales', 'infection', 'pharyngeal red',
    'headache', 'chest tightness', 'rebound tenderness',
    'abdominal tenderness',
]

# Cardiac surgery keywords for surgical context feature
CARDIAC_SURGERY_KEYWORDS = [
    'cardiac', 'heart', 'aorta', 'valve', 'septal', 'ventricle',
    'atrial', 'bypass', 'coronary', 'congenital heart',
]

# =============================================================================
# AGE-ADJUSTED VITAL REFERENCE RANGES
# =============================================================================
# Source: Fleming et al. (2011) Arch Dis Child — pediatric reference ranges
# Format: age_months_upper -> (hr_low, hr_high, rr_low, rr_high, sbp_low, sbp_high)

AGE_VITAL_RANGES = [
    (1,    (100, 180, 30, 60, 60,  90)),   # 0-1 month
    (12,   (100, 170, 25, 55, 70, 100)),   # 1-12 months
    (24,   (90,  160, 20, 40, 75, 105)),   # 1-2 years
    (60,   (80,  140, 20, 35, 80, 110)),   # 2-5 years
    (144,  (70,  120, 15, 30, 85, 115)),   # 5-12 years
    (216,  (60,  100, 12, 25, 90, 120)),   # 12-18 years
]


def get_age_vital_range(age_months):
    for upper, ranges in AGE_VITAL_RANGES:
        if age_months <= upper:
            return ranges
    return AGE_VITAL_RANGES[-1][1]


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


def in_window(t, intime, hours=24):
    return intime <= t <= intime + timedelta(hours=hours)


def apply_cap(val, cap_info):
    """Apply conversion factor then check floor/cap. Return None if out of range."""
    if val is None:
        return None
    v = val * cap_info.get('convert_factor', 1.0)
    if v < cap_info['floor'] or v > cap_info['cap']:
        return None
    return v


def compute_stats(values):
    """5 statistics from a list. All None if empty."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {'max': None, 'min': None, 'mean': None,
                'first': None, 'last': None, 'count': 0}
    return {
        'max':   max(clean),
        'min':   min(clean),
        'mean':  round(sum(clean) / len(clean), 4),
        'first': clean[0],
        'last':  clean[-1],
        'count': len(clean),
    }


def clean_col(s):
    return s.lower().replace(' ', '_').replace('(', '').replace(')', '')\
            .replace('/', '_').replace('-', '_').replace('.', '_')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_caps(caps_file):
    with open(caps_file) as f:
        return json.load(f)


def load_cohort(cohort_file):
    """Returns list of stay dicts and set of subject_ids."""
    stays = []
    with open(cohort_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            intime = parse_dt(row['intime'])
            if intime is None:
                continue
            stays.append({
                'subject_id':   row['subject_id'],
                'hadm_id':      row['hadm_id'],
                'icustay_id':   row['icustay_id'],
                'intime':       intime,
                'outtime':      parse_dt(row['outtime']),
                'los_hours':    safe_float(row['los_hours']),
                'age_years':    safe_float(row['age_years']),
                'age_months':   safe_float(row['age_months']),
                'gender':       row['gender'],
                'expire_flag':  row['expire_flag'],
                'hosp_expire':  row['hospital_expire_flag'],
                'diagnosis':    row['admission_diagnosis'],
                'icd10':        row['icd10_code'],
                'ethnicity':    row['ethnicity'],
                'care_unit':    row['care_unit'],
            })
    subjects = {s['subject_id'] for s in stays}
    return stays, subjects


def load_labevents(data_dir, subjects, caps):
    """Load relevant lab events for cohort subjects, apply caps."""
    print("  Loading LABEVENTS (large file)...")

    # Build itemid -> (feature_name, cap_key, convert_fn)
    itemid_map = {}
    for feat, info in LAB_FEATURES.items():
        cap_key = feat if feat in caps else None
        # Some features use a different key in caps
        # map feature name to caps key
        caps_key_map = {
            'creatinine': 'creatinine_umol',
            'bilirubin':  'bilirubin_umol',
            'glucose':    'glucose_mmol',
            'hemoglobin': 'hemoglobin',
            'anion_gap':  'anion_gap',
            'spo2_lab':   'spo2_lab',
        }
        cap_key = caps_key_map.get(feat, feat)
        for iid in info['itemids']:
            itemid_map[iid] = {
                'feat':     feat,
                'cap_key':  cap_key,
                'convert':  info.get('convert'),
            }

    target_ids = set(itemid_map.keys())

    # subject_id -> list of (charttime, feat, value) sorted by time
    events = defaultdict(list)
    count = 0

    with open(os.path.join(data_dir, 'LABEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if row['ITEMID'] not in target_ids:
                continue
            raw = safe_float(row['VALUENUM'])
            t   = parse_dt(row['CHARTTIME'])
            if raw is None or t is None:
                continue

            mapping  = itemid_map[row['ITEMID']]
            cap_key  = mapping['cap_key']
            cap_info = caps.get(cap_key)

            # Apply physiological cap on RAW value first (before conversion)
            # This is critical for glucose where cap is in mmol/L but
            # we convert to mg/dL. Checking cap after conversion would
            # wrongly null normal glucose values (5.6 mmol/L -> 100.8 mg/dL
            # which exceeds the mmol/L cap of 80).
            if cap_info:
                raw_cap = {
                    'floor': cap_info['floor'],
                    'cap':   cap_info['cap'],
                    'convert_factor': 1.0,
                }
                raw_checked = apply_cap(raw, raw_cap)
            else:
                raw_checked = raw

            if raw_checked is None:
                continue  # outside physiological range, skip

            # Now apply unit conversion for storage
            if mapping['convert']:
                val = mapping['convert'](raw_checked)
            else:
                val = raw_checked

            if val is None:
                continue

            events[row['SUBJECT_ID']].append((t, mapping['feat'], val))
            count += 1

    # Sort by time
    for sid in events:
        events[sid].sort(key=lambda x: x[0])

    print(f"    -> {count:,} relevant lab events")
    return events


def load_chartevents(data_dir, subjects, caps):
    """Load relevant chart events, apply caps."""
    print("  Loading CHARTEVENTS (large file)...")

    itemid_map = {}
    caps_key_map = {
        'spo2':        'spo2_chart',
        'systolic_bp': 'systolic_bp',
        'diastolic_bp':'diastolic_bp',
        'heart_rate':  'heart_rate',
        'resp_rate':   'resp_rate',
        'temperature': 'temperature',
    }
    for feat, info in CHART_FEATURES.items():
        cap_key = caps_key_map.get(feat, feat)
        itemid_map[info['itemid']] = {'feat': feat, 'cap_key': cap_key}

    target_ids = set(itemid_map.keys())

    events = defaultdict(list)
    count = 0

    with open(os.path.join(data_dir, 'CHARTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if row['ITEMID'] not in target_ids:
                continue
            raw = safe_float(row['VALUENUM'])
            t   = parse_dt(row['CHARTTIME'])
            if raw is None or t is None:
                continue

            mapping  = itemid_map[row['ITEMID']]
            cap_info = caps.get(mapping['cap_key'])
            val = apply_cap(raw, cap_info) if cap_info else raw
            if val is None:
                continue

            events[row['SUBJECT_ID']].append((t, mapping['feat'], val))
            count += 1

    for sid in events:
        events[sid].sort(key=lambda x: x[0])

    print(f"    -> {count:,} relevant chart events")
    return events


def load_prescriptions(data_dir, subjects):
    """Load vasoactives and antibiotics."""
    print("  Loading PRESCRIPTIONS (large file)...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            drug    = row['DRUG_NAME_EN'].lower()
            is_vaso = any(k in drug for k in VASOACTIVE_KEYWORDS)
            is_abx  = any(k in drug for k in ANTIBIOTIC_KEYWORDS)
            if not (is_vaso or is_abx):
                continue
            t = parse_dt(row['STARTDATE'])
            if t is None:
                continue
            out[row['SUBJECT_ID']].append({
                'drug':     row['DRUG_NAME_EN'],
                'drug_low': drug,
                'start':    t,
                'is_vaso':  is_vaso,
                'is_abx':   is_abx,
            })
            count += 1
    print(f"    -> {count:,} relevant prescriptions")
    return out


def load_microbiologyevents(data_dir, subjects):
    """Load culture events."""
    print("  Loading MICROBIOLOGYEVENTS...")
    NEG_KEYWORDS = [
        'no bacterial','no bacteria','no growth','no fungus',
        'normal flora','uncultured','not detected','not found',
        'no pathogenic','无细菌','无生长','正常菌群',
    ]
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'MICROBIOLOGYEVENTS.csv'),
              encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            t = parse_dt(row['CHARTTIME'])
            if t is None:
                continue
            org = row['ORG_NAME'].lower()
            has_org = len(org) > 0 and not any(k in org for k in NEG_KEYWORDS)
            out[row['SUBJECT_ID']].append({
                'charttime': t,
                'has_org':   has_org,
            })
            count += 1
    print(f"    -> {count:,} microbiology events")
    return out


def load_outputevents(data_dir, subjects):
    """Load urine output (ITEMID 1034)."""
    print("  Loading OUTPUTEVENTS...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'OUTPUTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if row['ITEMID'] != '1034':
                continue
            val = safe_float(row['VALUE'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append((t, val))
            count += 1
    print(f"    -> {count:,} urine output events")
    return out


def load_inputevents(data_dir, subjects):
    """Load fluid input."""
    print("  Loading INPUTEVENTS...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'INPUTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            val = safe_float(row['AMOUNT'])
            t   = parse_dt(row['CHARTTIME'])
            if val is None or t is None:
                continue
            out[row['SUBJECT_ID']].append((t, val))
            count += 1
    print(f"    -> {count:,} fluid input events")
    return out


def load_emr_symptoms(data_dir, subjects):
    """Load first EMR symptom record per patient."""
    print("  Loading EMR_SYMPTOMS...")
    raw = defaultdict(list)
    with open(os.path.join(data_dir, 'EMR_SYMPTOMS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            t = parse_dt(row['RECORDTIME'])
            if t is None:
                continue
            raw[row['SUBJECT_ID']].append({
                'time':      t,
                'symptom':   row['SYMPTOM_NAME'].strip().lower(),
                'attribute': row['SYMPTOM_ATTRIBUTE'].strip(),
            })

    out = {}
    for sid, recs in raw.items():
        recs.sort(key=lambda x: x['time'])
        earliest = recs[0]['time']
        out[sid] = {}
        for r in recs:
            if r['time'] == earliest:
                out[sid][r['symptom']] = r['attribute']
    print(f"    -> {len(out):,} patients with symptoms")
    return out


def load_surgery_info(data_dir, subjects):
    """Load surgery information — had surgery flag and cardiac surgery."""
    print("  Loading SURGERY_INFO...")
    out = defaultdict(list)
    count = 0
    with open(os.path.join(data_dir, 'SURGERY_INFO.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            t = parse_dt(row['SURGERY_BEGIN_TIME'])
            if t is None:
                continue
            name = row['SURGERY_NAME'].lower()
            is_cardiac = any(k in name for k in CARDIAC_SURGERY_KEYWORDS)
            out[row['SUBJECT_ID']].append({
                'time':       t,
                'is_cardiac': is_cardiac,
            })
            count += 1
    print(f"    -> {count:,} surgery records")
    return out


# =============================================================================
# PER-STAY FEATURE EXTRACTION
# =============================================================================

def extract_stay_features(stay, lab_events, chart_events,
                           prescriptions, micro, output_ev,
                           input_ev, symptoms, surgery_info):

    sid      = stay['subject_id']
    intime   = stay['intime']
    age_m    = stay['age_months'] or 0
    end_time = intime + timedelta(hours=WINDOW_HOURS)

    record = {}

    # ── A: Lab features ───────────────────────────────────────────────────────
    # Filter to window, group by feature name
    lab_by_feat = defaultdict(list)
    for (t, feat, val) in lab_events.get(sid, []):
        if intime <= t <= end_time:
            lab_by_feat[feat].append(val)

    for feat in LAB_FEATURES:
        stats = compute_stats(lab_by_feat[feat])
        for stat, val in stats.items():
            if stat != 'last':  # we use last only for trend
                record[f'{feat}_{stat}'] = val

    # ── B: Chart vital features ───────────────────────────────────────────────
    chart_by_feat = defaultdict(list)
    for (t, feat, val) in chart_events.get(sid, []):
        if intime <= t <= end_time:
            chart_by_feat[feat].append(val)

    for feat in CHART_FEATURES:
        stats = compute_stats(chart_by_feat[feat])
        for stat, val in stats.items():
            if stat != 'last':
                record[f'{feat}_{stat}'] = val

    # ── C: Derived vitals ─────────────────────────────────────────────────────
    # MAP — pair SBP and DBP by computing per-reading MAP
    sys_vals = chart_by_feat.get('systolic_bp', [])
    dia_vals = chart_by_feat.get('diastolic_bp', [])

    # Compute MAP from all chart events with both SBP and DBP at same time
    # For stats: use paired times where both exist
    sys_times = {}
    dia_times = {}
    for (t, feat, val) in chart_events.get(sid, []):
        if not (intime <= t <= end_time):
            continue
        if feat == 'systolic_bp':
            sys_times[t] = val
        elif feat == 'diastolic_bp':
            dia_times[t] = val

    map_vals = []
    for t in sorted(sys_times):
        if t in dia_times:
            m = dia_times[t] + (sys_times[t] - dia_times[t]) / 3.0
            if 10 <= m <= 200:  # basic sanity
                map_vals.append(round(m, 1))

    map_stats = compute_stats(map_vals)
    for stat, val in map_stats.items():
        if stat != 'last':
            record[f'map_{stat}'] = val

    # Shock index = HR / SBP (mean values)
    hr_mean  = record.get('heart_rate_mean')
    sbp_mean = record.get('systolic_bp_mean')
    if hr_mean and sbp_mean and sbp_mean > 0:
        record['shock_index'] = round(hr_mean / sbp_mean, 4)
    else:
        record['shock_index'] = None

    # ── D: Trend features (last - first) ─────────────────────────────────────
    for feat in TREND_VARS:
        vals = lab_by_feat.get(feat, [])
        if len(vals) >= 2:
            record[f'{feat}_trend'] = round(vals[-1] - vals[0], 4)
        else:
            record[f'{feat}_trend'] = None

    # Vital trends
    for feat in ['heart_rate', 'resp_rate', 'temperature', 'spo2']:
        vals = chart_by_feat.get(feat, [])
        if len(vals) >= 2:
            record[f'{feat}_trend'] = round(vals[-1] - vals[0], 4)
        else:
            record[f'{feat}_trend'] = None

    # ── E: Age-adjusted vital flags ───────────────────────────────────────────
    vr = get_age_vital_range(age_m)
    hr_low, hr_high, rr_low, rr_high, sbp_low, sbp_high = vr

    hr_mean_val  = record.get('heart_rate_mean')
    rr_mean_val  = record.get('resp_rate_mean')
    sbp_mean_val = record.get('systolic_bp_mean')

    record['hr_abnormal_for_age'] = (
        1 if hr_mean_val is not None and
        (hr_mean_val < hr_low or hr_mean_val > hr_high) else 0
    )
    record['rr_abnormal_for_age'] = (
        1 if rr_mean_val is not None and
        (rr_mean_val < rr_low or rr_mean_val > rr_high) else 0
    )
    record['sbp_abnormal_for_age'] = (
        1 if sbp_mean_val is not None and
        (sbp_mean_val < sbp_low or sbp_mean_val > sbp_high) else 0
    )

    # ── F: Fluid balance ──────────────────────────────────────────────────────
    urine_total = sum(
        v for (t, v) in output_ev.get(sid, [])
        if intime <= t <= end_time
    )
    fluid_total = sum(
        v for (t, v) in input_ev.get(sid, [])
        if intime <= t <= end_time
    )
    los_h = min(stay['los_hours'] or WINDOW_HOURS, WINDOW_HOURS)
    urine_per_hr = round(urine_total / max(los_h, 1), 2)

    record['urine_output_total_ml'] = urine_total
    record['urine_output_per_hour'] = urine_per_hr
    record['fluid_input_total_ml']  = fluid_total
    record['fluid_balance_ml']      = round(fluid_total - urine_total, 1)

    # ── G: Vasoactive drugs ───────────────────────────────────────────────────
    vaso_in_window = [
        rx for rx in prescriptions.get(sid, [])
        if rx['is_vaso'] and intime <= rx['start'] <= end_time
    ]
    found_vasos = set()
    for rx in vaso_in_window:
        for kw in VASOACTIVE_KEYWORDS:
            if kw in rx['drug_low']:
                found_vasos.add(kw)

    record['n_vasoactives_24h'] = len(found_vasos)
    for name, keywords in VASOACTIVE_INDIVIDUAL.items():
        record[f'vaso_{name}'] = int(any(k in found_vasos for k in keywords))

    # ── H: Infection markers ──────────────────────────────────────────────────
    abx_in_window = [
        rx for rx in prescriptions.get(sid, [])
        if rx['is_abx'] and intime <= rx['start'] <= end_time
    ]
    cultures_in_window = [
        m for m in micro.get(sid, [])
        if intime <= m['charttime'] <= end_time
    ]
    pos_cultures = [m for m in cultures_in_window if m['has_org']]

    record['n_antibiotics_24h']   = len(abx_in_window)
    record['n_cultures_24h']      = len(cultures_in_window)
    record['n_positive_cultures'] = len(pos_cultures)
    record['suspected_infection'] = int(
        len(abx_in_window) > 0 and len(cultures_in_window) > 0
    )

    # ── I: EMR symptoms ───────────────────────────────────────────────────────
    pt_symptoms = symptoms.get(sid, {})
    for symptom in SEPSIS_SYMPTOMS:
        col  = f'symptom_{clean_col(symptom)}'
        attr = pt_symptoms.get(symptom.lower())
        record[col] = 1 if attr == '+' else 0

    # ── J: Surgical context ───────────────────────────────────────────────────
    surgeries = surgery_info.get(sid, [])
    had_surgery = int(len(surgeries) > 0)
    had_cardiac = int(any(s['is_cardiac'] for s in surgeries))

    record['had_surgery_flag']         = had_surgery
    record['had_cardiac_surgery_flag'] = had_cardiac

    return record


# =============================================================================
# MAIN
# =============================================================================

def run_step3(data_dir, output_dir, log_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 3: Feature Extraction")
    print("="*60)

    # Load caps
    caps = load_caps(CAPS_FILE)
    print(f"\nLoaded caps from {CAPS_FILE}")

    # Load cohort
    print(f"Loading cohort from {COHORT_FILE}...")
    stays, subjects = load_cohort(COHORT_FILE)
    print(f"  {len(stays):,} stays, {len(subjects):,} subjects")

    # Load all tables
    print("\nLoading tables...")
    lab_ev    = load_labevents(data_dir, subjects, caps)
    chart_ev  = load_chartevents(data_dir, subjects, caps)
    rx        = load_prescriptions(data_dir, subjects)
    micro     = load_microbiologyevents(data_dir, subjects)
    output_ev = load_outputevents(data_dir, subjects)
    input_ev  = load_inputevents(data_dir, subjects)
    symp      = load_emr_symptoms(data_dir, subjects)
    surg      = load_surgery_info(data_dir, subjects)

    # Extract features per stay
    print(f"\nExtracting features for {len(stays):,} stays...")
    records = []
    for i, stay in enumerate(stays):
        if i % 2000 == 0 and i > 0:
            print(f"  {i:,}/{len(stays):,} processed...")

        feat = extract_stay_features(
            stay, lab_ev, chart_ev, rx, micro,
            output_ev, input_ev, symp, surg
        )

        # Build final row — identifiers first, then features, no label yet
        row = {
            'subject_id':   stay['subject_id'],
            'hadm_id':      stay['hadm_id'],
            'icustay_id':   stay['icustay_id'],
            'age_years':    stay['age_years'],
            'age_months':   stay['age_months'],
            'gender':       stay['gender'],
            'los_hours':    stay['los_hours'],
            'expire_flag':  stay['expire_flag'],
            'hosp_expire':  stay['hosp_expire'],
            'care_unit':    stay['care_unit'],
            'diagnosis':    stay['diagnosis'],
            'icd10':        stay['icd10'],
            'ethnicity':    stay['ethnicity'],
        }
        row.update(feat)
        records.append(row)

    # Write CSV
    print(f"\nWriting output...")
    if records:
        fieldnames = list(records[0].keys())
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved {len(records):,} rows -> {OUT_CSV}")
        print(f"  Columns: {len(fieldnames)}")

    # Missing data summary
    print("\nComputing missing data rates for key columns...")
    import statistics as stats_mod

    key_cols = [
        'lactate_max', 'platelets_min', 'inr_max', 'ddimer_max',
        'creatinine_max', 'bilirubin_max', 'glucose_max',
        'map_min', 'heart_rate_max', 'resp_rate_max',
        'temperature_max', 'ph_min', 'base_excess_min',
        'hemoglobin_mean', 'shock_index',
    ]

    log_lines = [
        "=" * 60,
        "STEP 3 SUMMARY — Feature Extraction",
        "=" * 60,
        f"Total stays processed:  {len(records):,}",
        f"Total feature columns:  {len(fieldnames) if records else 0}",
        "",
        f"{'Column':<30} {'Present':>8} {'Missing':>8} {'Missing%':>10}",
        "-" * 60,
    ]

    n = len(records)
    for col in key_cols:
        if col not in (records[0].keys() if records else []):
            log_lines.append(f"  {col:<28} NOT FOUND")
            continue
        present = sum(
            1 for r in records
            if r.get(col) is not None and r.get(col) != ''
        )
        missing = n - present
        log_lines.append(
            f"{col:<30} {present:>8,} {missing:>8,} {missing/n*100:>9.1f}%"
        )

    log_lines += [
        "",
        "NOTE: No imputation applied here.",
        "Imputation happens in Step 6 after train/test split.",
        f"\nOutput: {OUT_CSV}",
    ]

    log_str = '\n'.join(log_lines)
    print('\n' + log_str)

    log_path = os.path.join(log_dir, 'step3.1_log.txt')
    with open(log_path, 'w') as f:
        f.write(log_str)
    print(f"\nLog saved -> {log_path}")
    print("\nStep 3.1 complete. Fixed glucose cap bug. Ready for Step 4.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step3(DATA_DIR, OUTPUT_DIR, LOG_DIR)
