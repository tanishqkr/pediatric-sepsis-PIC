"""
STEP 2 — Physiological Caps Definition and Validation
======================================================
Authors : Uzair & Tanish | NMIMS | 2026

What this script does
---------------------
1. Defines physiologically justified caps for every continuous variable
   we will extract in Step 3.
2. Scans the raw V1.1.0 files to count how many values exceed each cap.
3. Saves the caps as a JSON config file that Step 3 will import and apply
   during feature extraction.
4. Saves a validation report showing exactly how many values will be
   nulled per variable.

Philosophy
----------
Caps are NOT the normal clinical range.
Caps are the MAXIMUM PHYSIOLOGICALLY POSSIBLE value in a living patient.
Example:
  - Normal lactate: 0.5 - 2.2 mmol/L
  - Critical lactate: >4 mmol/L
  - Cap: 30 mmol/L  (highest ever documented in surviving patient)
  - Value of 22 billion: clearly a data entry error -> set to null

Values ABOVE cap  -> set to null (impossible, data error)
Values BELOW floor -> set to null (impossible, data error)
Values within range -> kept AS IS, even if extreme

Note on buffers
---------------
You asked about buffer zones for extreme-but-real values.
Answer: the caps already include generous buffers.
Example heart rate cap is 350 bpm — the absolute maximum ever recorded
in any human being. A real patient with SVT at 280 bpm is kept.
Only the impossible values (HR 9718) are nulled.

Outputs
-------
  output2/logs/step2_caps.json         — caps config imported by Step 3
  output2/logs/step2_validation.txt    — count of values affected per variable
  output2/logs/step2_raw_samples.txt   — sample of actual outlier values found
"""

import os
import csv
import json
import math
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR   = "V1.1.0"
OUTPUT_DIR = "output2"
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")

# Cohort subject IDs from Step 1 — we only check values for eligible patients
COHORT_FILE = os.path.join(OUTPUT_DIR, "cohort_step1.csv")


# =============================================================================
# PHYSIOLOGICAL CAPS
# =============================================================================
# Format: 'variable_name': {'floor': X, 'cap': Y, 'unit': '...', 'rationale': '...'}
#
# floor = minimum possible value (values below this are impossible)
# cap   = maximum possible value (values above this are impossible)
#
# Sources:
#   - Phoenix criteria paper (Sanchez-Pinto et al. 2024)
#   - Pediatric Reference Intervals (Soldin et al.)
#   - ICU data quality literature (Goldberger et al.)
#   - Clinical judgment for extreme-but-documented cases

PHYSIOLOGICAL_CAPS = {

    # ── Blood Gas / Metabolic ─────────────────────────────────────────────────
    'lactate': {
        'floor': 0.0, 'cap': 30.0,
        'unit': 'mmol/L',
        'rationale': 'Highest documented survivable lactate ~25 mmol/L. Cap 30 with buffer.',
        'itemids': ['5227'],
    },
    'ph': {
        'floor': 6.5, 'cap': 7.9,
        'unit': 'pH units',
        'rationale': 'pH outside 6.5-7.9 incompatible with life. Values in billions are data errors.',
        'itemids': ['5237', '5238'],
    },
    'pco2': {
        'floor': 5.0, 'cap': 200.0,
        'unit': 'mmHg',
        'rationale': 'Extreme hypercapnia documented up to ~150-180 mmHg in severe COPD/asthma.',
        'itemids': ['5235', '5236'],
    },
    'pao2': {
        'floor': 0.0, 'cap': 600.0,
        'unit': 'mmHg',
        'rationale': 'Maximum on 100% FiO2 ~500-600 mmHg. Higher values are calibration errors.',
        'itemids': ['5239', '5244'],
    },
    'bicarbonate': {
        'floor': 0.0, 'cap': 60.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme metabolic alkalosis documented up to ~50 mmol/L.',
        'itemids': ['5248', '5224'],
    },
    'base_excess': {
        'floor': -50.0, 'cap': 50.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme BE values documented at ±40. Cap ±50 with buffer.',
        'itemids': ['5211', '5249'],
    },
    'anion_gap': {
        'floor': 0.0, 'cap': 60.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme AG in severe metabolic acidosis up to ~50.',
        'itemids': ['5212', '5213'],
    },
    'spo2_lab': {
        'floor': 50.0, 'cap': 100.0,
        'unit': '%',
        'rationale': 'SpO2 is a percentage. Cannot exceed 100. Below 50 patient would be dead.',
        'itemids': ['5252'],
    },

    # ── Coagulation ───────────────────────────────────────────────────────────
    'platelets': {
        'floor': 0.0, 'cap': 2000.0,
        'unit': '10^9/L',
        'rationale': 'Extreme thrombocytosis documented up to ~1500. Cap 2000 with buffer.',
        'itemids': ['5129'],
    },
    'inr': {
        'floor': 0.5, 'cap': 20.0,
        'unit': 'ratio',
        'rationale': 'INR >20 essentially unmeasurable — extreme coagulopathy. Real values rarely exceed 15.',
        'itemids': ['5174'],
    },
    'ddimer': {
        'floor': 0.0, 'cap': 200.0,
        'unit': 'mg/L FEU',
        'rationale': 'Extreme D-dimer in severe DIC documented up to ~150. Cap 200 with buffer.',
        'itemids': ['5163'],
    },
    'fibrinogen': {
        'floor': 0.0, 'cap': 15.0,
        'unit': 'g/L',
        'rationale': 'Maximum fibrinogen in acute phase response ~10-12 g/L.',
        'itemids': ['5164'],
    },
    'pt': {
        'floor': 0.0, 'cap': 200.0,
        'unit': 'seconds',
        'rationale': 'Extreme PT in severe coagulopathy documented up to ~150s.',
        'itemids': ['5186'],
    },
    'ptt': {
        'floor': 0.0, 'cap': 300.0,
        'unit': 'seconds',
        'rationale': 'Extreme PTT in severe coagulopathy documented up to ~250s.',
        'itemids': ['5161'],
    },

    # ── Renal ─────────────────────────────────────────────────────────────────
    'creatinine_umol': {
        'floor': 0.0, 'cap': 2000.0,
        'unit': 'umol/L',
        'rationale': '2000 umol/L = ~22.6 mg/dL. Extreme renal failure with dialysis. Higher values are errors.',
        'itemids': ['5032', '5041', '6954'],
    },

    # ── Hepatic ───────────────────────────────────────────────────────────────
    'bilirubin_umol': {
        'floor': 0.0, 'cap': 1800.0,
        'unit': 'umol/L',
        'rationale': '1800 umol/L = ~105 mg/dL. Extreme in biliary atresia neonates. Negative values are errors.',
        'itemids': ['5075', '5255'],
    },
    'alt': {
        'floor': 0.0, 'cap': 50000.0,
        'unit': 'IU/L',
        'rationale': 'Extreme ALT in acute liver failure documented >30,000. Cap 50,000 with buffer.',
        'itemids': ['5026', '5195'],
    },
    'ast': {
        'floor': 0.0, 'cap': 50000.0,
        'unit': 'IU/L',
        'rationale': 'Same as ALT.',
        'itemids': ['5031'],
    },

    # ── Endocrine ─────────────────────────────────────────────────────────────
    'glucose_mmol': {
        'floor': 0.5, 'cap': 80.0,
        'unit': 'mmol/L',
        'rationale': '80 mmol/L = ~1440 mg/dL. Extreme hyperglycaemic crisis. Higher values are errors.',
        'itemids': ['5047', '5223'],
    },

    # ── Immunologic ───────────────────────────────────────────────────────────
    'anc': {
        'floor': 0.0, 'cap': 100.0,
        'unit': '10^9/L',
        'rationale': 'Extreme neutrophilia in leukaemoid reaction documented up to ~80.',
        'itemids': ['5094'],
    },
    'alc': {
        'floor': 0.0, 'cap': 100.0,
        'unit': '10^9/L',
        'rationale': 'Extreme lymphocytosis in CLL-like picture documented up to ~80.',
        'itemids': ['5110'],
    },
    'wbc': {
        'floor': 0.0, 'cap': 300.0,
        'unit': '10^9/L',
        'rationale': 'Extreme leukocytosis in leukaemia documented up to ~200-250.',
        'itemids': ['5141'],
    },

    # ── Haematology ───────────────────────────────────────────────────────────
    'hemoglobin': {
        'floor': 0.0, 'cap': 25.0,
        'unit': 'g/dL',
        'rationale': 'Highest documented haemoglobin ~23-24 g/dL in extreme polycythaemia.',
        'itemids': ['5099', '5257'],
    },
    'hematocrit': {
        'floor': 0.0, 'cap': 75.0,
        'unit': '%',
        'rationale': 'Cannot exceed 100%. Extreme polycythaemia documented up to ~70%.',
        'itemids': ['5097', '5225'],
    },

    # ── Chemistry ─────────────────────────────────────────────────────────────
    'sodium': {
        'floor': 100.0, 'cap': 200.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypo/hypernatraemia documented at 100-190 mmol/L.',
        'itemids': ['5062', '5230'],
    },
    'potassium': {
        'floor': 1.0, 'cap': 12.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypo/hyperkalaemia documented at 1.5-10 mmol/L.',
        'itemids': ['5056', '5226'],
    },
    'calcium': {
        'floor': 0.0, 'cap': 5.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypercalcaemia documented up to ~4.5 mmol/L.',
        'itemids': ['5034', '5215'],
    },
    'albumin': {
        'floor': 0.0, 'cap': 70.0,
        'unit': 'g/L',
        'rationale': 'Normal range 35-50 g/L. Values above 70 are errors.',
        'itemids': ['5024'],
    },
    'crp': {
        'floor': 0.0, 'cap': 500.0,
        'unit': 'mg/L',
        'rationale': 'Extreme CRP in severe sepsis/inflammation documented up to ~400 mg/L.',
        'itemids': ['5626', '5821'],
    },
    'ldh': {
        'floor': 0.0, 'cap': 100000.0,
        'unit': 'IU/L',
        'rationale': 'Extreme LDH in haemolysis/malignancy documented up to ~50,000.',
        'itemids': ['5057'],
    },
    'urea': {
        'floor': 0.0, 'cap': 100.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme uraemia documented up to ~80 mmol/L.',
        'itemids': ['5033'],
    },
    'uric_acid': {
        'floor': 0.0, 'cap': 2000.0,
        'unit': 'umol/L',
        'rationale': 'Extreme in tumour lysis syndrome documented up to ~1500 umol/L.',
        'itemids': ['5083'],
    },
    'ck': {
        'floor': 0.0, 'cap': 1000000.0,
        'unit': 'IU/L',
        'rationale': 'Extreme CK in rhabdomyolysis documented up to ~500,000 IU/L.',
        'itemids': ['5038'],
    },

    # ── Vitals (from CHARTEVENTS) ─────────────────────────────────────────────
    'heart_rate': {
        'floor': 10.0, 'cap': 350.0,
        'unit': 'bpm',
        'rationale': 'Absolute maximum in documented SVT/VT ~320 bpm. Cap 350 with buffer. Value 9718 is a data error.',
        'itemids': ['1003'],
    },
    'resp_rate': {
        'floor': 0.0, 'cap': 100.0,
        'unit': 'breaths/min',
        'rationale': 'Maximum documented RR in severe distress ~80-90/min. Value 2694 is a data error.',
        'itemids': ['1004'],
    },
    'temperature': {
        'floor': 30.0, 'cap': 43.0,
        'unit': 'Celsius',
        'rationale': 'Lowest documented survival hypothermia ~27C. Cap floor 30 with buffer. Maximum hyperthermia ~43C. Value 337 is a data error.',
        'itemids': ['1001'],
    },
    'systolic_bp': {
        'floor': 20.0, 'cap': 300.0,
        'unit': 'mmHg',
        'rationale': 'Extreme hypertensive crisis documented up to ~300 mmHg. Value 1108 is a data error.',
        'itemids': ['1016'],
    },
    'diastolic_bp': {
        'floor': 5.0, 'cap': 200.0,
        'unit': 'mmHg',
        'rationale': 'Extreme diastolic hypertension documented up to ~180 mmHg.',
        'itemids': ['1015'],
    },
    'spo2_chart': {
        'floor': 50.0, 'cap': 100.0,
        'unit': '%',
        'rationale': 'SpO2 is a percentage. Cannot exceed 100. Below 50 patient would be dead.',
        'itemids': ['1006'],
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def safe_float(s):
    try:
        v = float(s)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def load_cohort_subjects(cohort_file):
    """Load subject IDs from Step 1 cohort — only check these patients."""
    subjects = set()
    with open(cohort_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            subjects.add(row['subject_id'])
    return subjects


# =============================================================================
# SCAN RAW FILES FOR OUTLIERS
# =============================================================================

def scan_labevents(data_dir, subjects, caps):
    """Scan LABEVENTS for values outside caps."""
    print("  Scanning LABEVENTS (large file, please wait)...")

    # Build itemid -> variable_name lookup
    itemid_to_var = {}
    for var_name, info in caps.items():
        if 'itemids' in info:
            for iid in info['itemids']:
                itemid_to_var[iid] = var_name

    # Only scan itemids we care about
    target_itemids = set(itemid_to_var.keys())

    results = defaultdict(lambda: {
        'total': 0, 'below_floor': 0, 'above_cap': 0,
        'below_samples': [], 'above_samples': []
    })

    with open(os.path.join(data_dir, 'LABEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if row['ITEMID'] not in target_itemids:
                continue
            val = safe_float(row['VALUENUM'])
            if val is None:
                continue

            var = itemid_to_var[row['ITEMID']]
            info = caps[var]
            results[var]['total'] += 1

            if val < info['floor']:
                results[var]['below_floor'] += 1
                if len(results[var]['below_samples']) < 5:
                    results[var]['below_samples'].append(val)
            elif val > info['cap']:
                results[var]['above_cap'] += 1
                if len(results[var]['above_samples']) < 5:
                    results[var]['above_samples'].append(val)

    return results


def scan_chartevents(data_dir, subjects, caps):
    """Scan CHARTEVENTS for values outside caps."""
    print("  Scanning CHARTEVENTS (large file, please wait)...")

    itemid_to_var = {}
    for var_name, info in caps.items():
        if 'itemids' in info:
            for iid in info['itemids']:
                itemid_to_var[iid] = var_name

    target_itemids = set(itemid_to_var.keys())

    results = defaultdict(lambda: {
        'total': 0, 'below_floor': 0, 'above_cap': 0,
        'below_samples': [], 'above_samples': []
    })

    with open(os.path.join(data_dir, 'CHARTEVENTS.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['SUBJECT_ID'] not in subjects:
                continue
            if row['ITEMID'] not in target_itemids:
                continue
            val = safe_float(row['VALUENUM'])
            if val is None:
                continue

            var = itemid_to_var[row['ITEMID']]
            info = caps[var]
            results[var]['total'] += 1

            if val < info['floor']:
                results[var]['below_floor'] += 1
                if len(results[var]['below_samples']) < 5:
                    results[var]['below_samples'].append(val)
            elif val > info['cap']:
                results[var]['above_cap'] += 1
                if len(results[var]['above_samples']) < 5:
                    results[var]['above_samples'].append(val)

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_step2(data_dir, output_dir, log_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 2: Physiological Caps Definition and Validation")
    print("="*60)

    # Load cohort subjects
    print(f"\nLoading Step 1 cohort subjects from {COHORT_FILE}...")
    subjects = load_cohort_subjects(COHORT_FILE)
    print(f"  {len(subjects):,} subjects in cohort")

    # Scan raw files
    print("\nScanning raw files for outliers...")
    lab_results   = scan_labevents(data_dir, subjects, PHYSIOLOGICAL_CAPS)
    chart_results = scan_chartevents(data_dir, subjects, PHYSIOLOGICAL_CAPS)

    # Merge results
    all_results = {}
    for var in PHYSIOLOGICAL_CAPS:
        lr = lab_results.get(var, {'total':0,'below_floor':0,'above_cap':0,
                                    'below_samples':[],'above_samples':[]})
        cr = chart_results.get(var, {'total':0,'below_floor':0,'above_cap':0,
                                      'below_samples':[],'above_samples':[]})
        all_results[var] = {
            'total':         lr['total'] + cr['total'],
            'below_floor':   lr['below_floor'] + cr['below_floor'],
            'above_cap':     lr['above_cap'] + cr['above_cap'],
            'below_samples': lr['below_samples'] + cr['below_samples'],
            'above_samples': lr['above_samples'] + cr['above_samples'],
        }

    # ── Save caps JSON (imported by Step 3) ──────────────────────────────────
    caps_export = {}
    for var, info in PHYSIOLOGICAL_CAPS.items():
        caps_export[var] = {
            'floor':     info['floor'],
            'cap':       info['cap'],
            'unit':      info['unit'],
            'rationale': info['rationale'],
            'itemids':   info.get('itemids', []),
        }

    caps_json = os.path.join(log_dir, 'step2_caps.json')
    with open(caps_json, 'w') as f:
        json.dump(caps_export, f, indent=2)
    print(f"\nCaps config saved -> {caps_json}")

    # ── Build validation report ───────────────────────────────────────────────
    report_lines = [
        "=" * 70,
        "STEP 2 VALIDATION REPORT — Physiological Caps",
        "=" * 70,
        "",
        f"{'Variable':<22} {'Total':>8} {'Below floor':>12} "
        f"{'Above cap':>10} {'% affected':>11}  Cap range",
        "-" * 90,
    ]

    total_affected = 0
    total_readings = 0

    for var in sorted(PHYSIOLOGICAL_CAPS.keys()):
        info = PHYSIOLOGICAL_CAPS[var]
        res  = all_results[var]
        n         = res['total']
        n_below   = res['below_floor']
        n_above   = res['above_cap']
        affected  = n_below + n_above
        pct       = affected / n * 100 if n > 0 else 0
        total_affected += affected
        total_readings += n

        flag = " ⚠" if affected > 0 else ""
        report_lines.append(
            f"{var:<22} {n:>8,} {n_below:>12,} {n_above:>10,} "
            f"{pct:>10.2f}%  "
            f"[{info['floor']} – {info['cap']}] {info['unit']}{flag}"
        )

    report_lines += [
        "-" * 90,
        f"{'TOTAL':<22} {total_readings:>8,} "
        f"{'':>12} {'':>10} "
        f"{total_affected/total_readings*100 if total_readings else 0:>10.2f}%  "
        f"total values affected: {total_affected:,}",
        "",
        "=" * 70,
        "SAMPLE OUTLIER VALUES",
        "=" * 70,
    ]

    for var in sorted(PHYSIOLOGICAL_CAPS.keys()):
        info = PHYSIOLOGICAL_CAPS[var]
        res  = all_results[var]
        if res['above_cap'] > 0 or res['below_floor'] > 0:
            report_lines.append(f"\n{var} [{info['unit']}]:")
            if res['below_samples']:
                report_lines.append(
                    f"  Below floor ({info['floor']}): "
                    f"{res['below_samples']}"
                )
            if res['above_samples']:
                report_lines.append(
                    f"  Above cap ({info['cap']}): "
                    f"{res['above_samples']}"
                )

    report_lines += [
        "",
        "=" * 70,
        "HOW CAPS ARE APPLIED",
        "=" * 70,
        "Values outside caps are set to NULL during Step 3 feature extraction.",
        "They are NOT replaced here — replacement with median happens in Step 6.",
        "This ensures imputation statistics are computed on clean data only.",
        "",
        f"Caps config file: {caps_json}",
        "Step 3 imports this file automatically.",
    ]

    report_str = '\n'.join(report_lines)
    print('\n' + report_str)

    report_txt = os.path.join(log_dir, 'step2_validation.txt')
    with open(report_txt, 'w') as f:
        f.write(report_str)
    print(f"\nValidation report saved -> {report_txt}")
    print("\nStep 2 complete. Caps defined and validated. Ready for Step 3.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step2(DATA_DIR, OUTPUT_DIR, LOG_DIR)
