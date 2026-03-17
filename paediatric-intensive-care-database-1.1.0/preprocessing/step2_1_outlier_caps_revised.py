"""
STEP 2.1 — Physiological Caps Revised
======================================
Authors : Uzair & Tanish | NMIMS | 2026

Changes from Step 2
-------------------
Three fixes applied after reviewing Step 2 output:

  FIX 1 — Haemoglobin unit error
    PIC stores haemoglobin in g/L, not g/dL.
    Values like 145 are normal (145 g/L = 14.5 g/dL).
    Fix: added convert_factor 0.1 (divide by 10) in caps config.
    Step 3 will apply this conversion before checking caps.

  FIX 2 — Anion gap floor too strict
    Floor changed from 0 to -5.
    Anion gap is a derived value (Na - Cl - HCO3).
    Measurement variation in individually measured electrolytes
    can produce slightly negative results (-1 to -4).
    These are valid readings, not errors.
    Only values below -5 are genuine data errors.

  FIX 3 — SpO2 lab cap too strict
    Cap raised from 100.0 to 100.5.
    Blood gas analysers can report 100.1-100.5% due to rounding.
    These are valid 100% readings, not errors.

All other caps unchanged from Step 2.

Outputs
-------
  output2/logs/step2.1_caps.json          — revised caps (used by Step 3)
  output2/logs/step2.1_validation.txt     — revised validation report
"""

import os
import csv
import json
import math
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = "V1.1.0"
OUTPUT_DIR  = "output2"
LOG_DIR     = os.path.join(OUTPUT_DIR, "logs")
COHORT_FILE = os.path.join(OUTPUT_DIR, "cohort_step1.csv")


# =============================================================================
# REVISED PHYSIOLOGICAL CAPS
# =============================================================================
# New keys vs Step 2:
#   convert_factor : multiply raw value by this before applying caps
#                    1.0 means no conversion (default)
#                    0.1 means divide by 10 (haemoglobin g/L -> g/dL)

PHYSIOLOGICAL_CAPS = {

    # ── Blood Gas / Metabolic ─────────────────────────────────────────────────
    'lactate': {
        'floor': 0.0, 'cap': 30.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Highest documented survivable lactate ~25 mmol/L. Cap 30 with buffer.',
        'itemids': ['5227'],
    },
    'ph': {
        'floor': 6.5, 'cap': 7.9, 'convert_factor': 1.0,
        'unit': 'pH units',
        'rationale': 'pH outside 6.5-7.9 incompatible with life.',
        'itemids': ['5237', '5238'],
    },
    'pco2': {
        'floor': 5.0, 'cap': 200.0, 'convert_factor': 1.0,
        'unit': 'mmHg',
        'rationale': 'Extreme hypercapnia documented up to ~150-180 mmHg.',
        'itemids': ['5235', '5236'],
    },
    'pao2': {
        'floor': 0.0, 'cap': 600.0, 'convert_factor': 1.0,
        'unit': 'mmHg',
        'rationale': 'Maximum on 100% FiO2 ~500-600 mmHg.',
        'itemids': ['5239', '5244'],
    },
    'bicarbonate': {
        'floor': 0.0, 'cap': 60.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme metabolic alkalosis documented up to ~50 mmol/L.',
        'itemids': ['5248', '5224'],
    },
    'base_excess': {
        'floor': -50.0, 'cap': 50.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme BE documented at ±40. Cap ±50 with buffer.',
        'itemids': ['5211', '5249'],
    },
    'anion_gap': {
        # FIX 2: floor changed from 0 to -5
        'floor': -5.0, 'cap': 60.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': (
            'Floor changed from 0 to -5 (Step 2.1 fix). '
            'Anion gap = Na - Cl - HCO3. Measurement variation in '
            'independently measured electrolytes can produce values '
            'down to -4 mmol/L. These are valid readings. '
            'Only below -5 are genuine errors.'
        ),
        'itemids': ['5212', '5213'],
    },
    'spo2_lab': {
        # FIX 3: cap raised from 100.0 to 100.5
        'floor': 50.0, 'cap': 100.5, 'convert_factor': 1.0,
        'unit': '%',
        'rationale': (
            'Cap raised from 100.0 to 100.5 (Step 2.1 fix). '
            'Blood gas analysers report 100.1-100.5 due to rounding. '
            'These are valid 100% readings. Below 50 patient would be dead.'
        ),
        'itemids': ['5252'],
    },

    # ── Coagulation ───────────────────────────────────────────────────────────
    'platelets': {
        'floor': 0.0, 'cap': 2000.0, 'convert_factor': 1.0,
        'unit': '10^9/L',
        'rationale': 'Extreme thrombocytosis documented up to ~1500.',
        'itemids': ['5129'],
    },
    'inr': {
        'floor': 0.5, 'cap': 20.0, 'convert_factor': 1.0,
        'unit': 'ratio',
        'rationale': 'INR >20 essentially unmeasurable.',
        'itemids': ['5174'],
    },
    'ddimer': {
        'floor': 0.0, 'cap': 200.0, 'convert_factor': 1.0,
        'unit': 'mg/L FEU',
        'rationale': 'Extreme D-dimer in severe DIC documented up to ~150.',
        'itemids': ['5163'],
    },
    'fibrinogen': {
        'floor': 0.0, 'cap': 15.0, 'convert_factor': 1.0,
        'unit': 'g/L',
        'rationale': 'Maximum fibrinogen in acute phase response ~10-12 g/L.',
        'itemids': ['5164'],
    },
    'pt': {
        'floor': 0.0, 'cap': 200.0, 'convert_factor': 1.0,
        'unit': 'seconds',
        'rationale': 'Extreme PT in severe coagulopathy documented up to ~150s.',
        'itemids': ['5186'],
    },
    'ptt': {
        'floor': 0.0, 'cap': 300.0, 'convert_factor': 1.0,
        'unit': 'seconds',
        'rationale': 'Extreme PTT in severe coagulopathy documented up to ~250s.',
        'itemids': ['5161'],
    },

    # ── Renal ─────────────────────────────────────────────────────────────────
    'creatinine_umol': {
        'floor': 0.0, 'cap': 2000.0, 'convert_factor': 1.0,
        'unit': 'umol/L',
        'rationale': '2000 umol/L = ~22.6 mg/dL. Extreme renal failure.',
        'itemids': ['5032', '5041', '6954'],
    },

    # ── Hepatic ───────────────────────────────────────────────────────────────
    'bilirubin_umol': {
        'floor': 0.0, 'cap': 1800.0, 'convert_factor': 1.0,
        'unit': 'umol/L',
        'rationale': '1800 umol/L = ~105 mg/dL. Extreme in biliary atresia.',
        'itemids': ['5075', '5255'],
    },
    'alt': {
        'floor': 0.0, 'cap': 50000.0, 'convert_factor': 1.0,
        'unit': 'IU/L',
        'rationale': 'Extreme ALT in acute liver failure documented >30,000.',
        'itemids': ['5026', '5195'],
    },
    'ast': {
        'floor': 0.0, 'cap': 50000.0, 'convert_factor': 1.0,
        'unit': 'IU/L',
        'rationale': 'Same as ALT.',
        'itemids': ['5031'],
    },

    # ── Endocrine ─────────────────────────────────────────────────────────────
    'glucose_mmol': {
        'floor': 0.5, 'cap': 80.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': '80 mmol/L = ~1440 mg/dL. Extreme hyperglycaemic crisis.',
        'itemids': ['5047', '5223'],
    },

    # ── Immunologic ───────────────────────────────────────────────────────────
    'anc': {
        'floor': 0.0, 'cap': 100.0, 'convert_factor': 1.0,
        'unit': '10^9/L',
        'rationale': 'Extreme neutrophilia in leukaemoid reaction up to ~80.',
        'itemids': ['5094'],
    },
    'alc': {
        'floor': 0.0, 'cap': 100.0, 'convert_factor': 1.0,
        'unit': '10^9/L',
        'rationale': 'Extreme lymphocytosis documented up to ~80.',
        'itemids': ['5110'],
    },
    'wbc': {
        'floor': 0.0, 'cap': 300.0, 'convert_factor': 1.0,
        'unit': '10^9/L',
        'rationale': 'Extreme leukocytosis in leukaemia documented up to ~200-250.',
        'itemids': ['5141'],
    },

    # ── Haematology ───────────────────────────────────────────────────────────
    'hemoglobin': {
        # FIX 1: convert_factor 0.1 — PIC stores in g/L, Phoenix uses g/dL
        'floor': 0.0, 'cap': 25.0, 'convert_factor': 0.1,
        'unit': 'g/dL (raw values in PIC are g/L, divided by 10)',
        'rationale': (
            'Unit conversion added (Step 2.1 fix). '
            'PIC stores haemoglobin in g/L (Chinese lab standard). '
            'Normal range 120-180 g/L = 12-18 g/dL. '
            'Values like 145 are normal (145 g/L = 14.5 g/dL). '
            'Divide by 10 before applying cap of 25 g/dL.'
        ),
        'itemids': ['5099', '5257'],
    },
    'hematocrit': {
        'floor': 0.0, 'cap': 75.0, 'convert_factor': 1.0,
        'unit': '%',
        'rationale': 'Extreme polycythaemia documented up to ~70%.',
        'itemids': ['5097', '5225'],
    },

    # ── Chemistry ─────────────────────────────────────────────────────────────
    'sodium': {
        'floor': 100.0, 'cap': 200.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypo/hypernatraemia documented at 100-190 mmol/L.',
        'itemids': ['5062', '5230'],
    },
    'potassium': {
        'floor': 1.0, 'cap': 12.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypo/hyperkalaemia documented at 1.5-10 mmol/L.',
        'itemids': ['5056', '5226'],
    },
    'calcium': {
        'floor': 0.0, 'cap': 5.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme hypercalcaemia documented up to ~4.5 mmol/L.',
        'itemids': ['5034', '5215'],
    },
    'albumin': {
        'floor': 0.0, 'cap': 70.0, 'convert_factor': 1.0,
        'unit': 'g/L',
        'rationale': 'Normal range 35-50 g/L. Values above 70 are errors.',
        'itemids': ['5024'],
    },
    'crp': {
        'floor': 0.0, 'cap': 500.0, 'convert_factor': 1.0,
        'unit': 'mg/L',
        'rationale': 'Extreme CRP in severe sepsis up to ~400 mg/L.',
        'itemids': ['5626', '5821'],
    },
    'ldh': {
        'floor': 0.0, 'cap': 100000.0, 'convert_factor': 1.0,
        'unit': 'IU/L',
        'rationale': 'Extreme LDH in haemolysis/malignancy up to ~50,000.',
        'itemids': ['5057'],
    },
    'urea': {
        'floor': 0.0, 'cap': 100.0, 'convert_factor': 1.0,
        'unit': 'mmol/L',
        'rationale': 'Extreme uraemia documented up to ~80 mmol/L.',
        'itemids': ['5033'],
    },
    'uric_acid': {
        'floor': 0.0, 'cap': 2000.0, 'convert_factor': 1.0,
        'unit': 'umol/L',
        'rationale': 'Extreme in tumour lysis syndrome up to ~1500 umol/L.',
        'itemids': ['5083'],
    },
    'ck': {
        'floor': 0.0, 'cap': 1000000.0, 'convert_factor': 1.0,
        'unit': 'IU/L',
        'rationale': 'Extreme CK in rhabdomyolysis up to ~500,000 IU/L.',
        'itemids': ['5038'],
    },

    # ── Vitals ────────────────────────────────────────────────────────────────
    'heart_rate': {
        'floor': 10.0, 'cap': 350.0, 'convert_factor': 1.0,
        'unit': 'bpm',
        'rationale': 'Absolute maximum in SVT/VT ~320 bpm. Cap 350 with buffer.',
        'itemids': ['1003'],
    },
    'resp_rate': {
        'floor': 0.0, 'cap': 100.0, 'convert_factor': 1.0,
        'unit': 'breaths/min',
        'rationale': 'Maximum documented RR in severe distress ~80-90/min.',
        'itemids': ['1004'],
    },
    'temperature': {
        'floor': 30.0, 'cap': 43.0, 'convert_factor': 1.0,
        'unit': 'Celsius',
        'rationale': 'Lowest survival hypothermia ~27C. Maximum hyperthermia ~43C.',
        'itemids': ['1001'],
    },
    'systolic_bp': {
        'floor': 20.0, 'cap': 300.0, 'convert_factor': 1.0,
        'unit': 'mmHg',
        'rationale': 'Extreme hypertensive crisis documented up to ~300 mmHg.',
        'itemids': ['1016'],
    },
    'diastolic_bp': {
        'floor': 5.0, 'cap': 200.0, 'convert_factor': 1.0,
        'unit': 'mmHg',
        'rationale': 'Extreme diastolic hypertension documented up to ~180 mmHg.',
        'itemids': ['1015'],
    },
    'spo2_chart': {
        'floor': 50.0, 'cap': 100.5, 'convert_factor': 1.0,
        'unit': '%',
        'rationale': (
            'Cap raised to 100.5 (Step 2.1 fix, consistent with spo2_lab). '
            'Monitors can report 100.1-100.5 due to rounding.'
        ),
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
    subjects = set()
    with open(cohort_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            subjects.add(row['subject_id'])
    return subjects


# =============================================================================
# SCAN RAW FILES
# =============================================================================

def scan_file(filepath, subject_col, itemid_col, value_col,
              subjects, caps, label):
    print(f"  Scanning {label} (please wait)...")

    itemid_to_var = {}
    for var_name, info in caps.items():
        for iid in info.get('itemids', []):
            itemid_to_var[iid] = var_name

    target_itemids = set(itemid_to_var.keys())

    results = defaultdict(lambda: {
        'total': 0, 'below_floor': 0, 'above_cap': 0,
        'below_samples': [], 'above_samples': []
    })

    with open(filepath, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row[subject_col] not in subjects:
                continue
            if row[itemid_col] not in target_itemids:
                continue
            raw = safe_float(row[value_col])
            if raw is None:
                continue

            var   = itemid_to_var[row[itemid_col]]
            info  = caps[var]
            # Apply conversion factor before checking caps
            val   = raw * info.get('convert_factor', 1.0)

            results[var]['total'] += 1

            if val < info['floor']:
                results[var]['below_floor'] += 1
                if len(results[var]['below_samples']) < 5:
                    results[var]['below_samples'].append(round(val, 4))
            elif val > info['cap']:
                results[var]['above_cap'] += 1
                if len(results[var]['above_samples']) < 5:
                    results[var]['above_samples'].append(round(val, 4))

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_step2_1(data_dir, output_dir, log_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("STEP 2.1: Physiological Caps Revised")
    print("="*60)
    print("\nFixes applied vs Step 2:")
    print("  FIX 1 - Haemoglobin: convert_factor=0.1 (g/L -> g/dL)")
    print("  FIX 2 - Anion gap:   floor changed 0 -> -5")
    print("  FIX 3 - SpO2:        cap changed 100.0 -> 100.5 (both lab + chart)")

    print(f"\nLoading cohort subjects from {COHORT_FILE}...")
    subjects = load_cohort_subjects(COHORT_FILE)
    print(f"  {len(subjects):,} subjects")

    print("\nScanning raw files...")
    lab_results = scan_file(
        os.path.join(data_dir, 'LABEVENTS.csv'),
        'SUBJECT_ID', 'ITEMID', 'VALUENUM',
        subjects, PHYSIOLOGICAL_CAPS, 'LABEVENTS'
    )
    chart_results = scan_file(
        os.path.join(data_dir, 'CHARTEVENTS.csv'),
        'SUBJECT_ID', 'ITEMID', 'VALUENUM',
        subjects, PHYSIOLOGICAL_CAPS, 'CHARTEVENTS'
    )

    # Merge
    all_results = {}
    for var in PHYSIOLOGICAL_CAPS:
        lr = lab_results.get(var,   {'total':0,'below_floor':0,'above_cap':0,
                                      'below_samples':[],'above_samples':[]})
        cr = chart_results.get(var, {'total':0,'below_floor':0,'above_cap':0,
                                      'below_samples':[],'above_samples':[]})
        all_results[var] = {
            'total':         lr['total']       + cr['total'],
            'below_floor':   lr['below_floor'] + cr['below_floor'],
            'above_cap':     lr['above_cap']   + cr['above_cap'],
            'below_samples': lr['below_samples'][:3] + cr['below_samples'][:2],
            'above_samples': lr['above_samples'][:3] + cr['above_samples'][:2],
        }

    # ── Save caps JSON ────────────────────────────────────────────────────────
    caps_export = {
        var: {
            'floor':          info['floor'],
            'cap':            info['cap'],
            'convert_factor': info.get('convert_factor', 1.0),
            'unit':           info['unit'],
            'rationale':      info['rationale'],
            'itemids':        info.get('itemids', []),
        }
        for var, info in PHYSIOLOGICAL_CAPS.items()
    }

    caps_json = os.path.join(log_dir, 'step2.1_caps.json')
    with open(caps_json, 'w') as f:
        json.dump(caps_export, f, indent=2)
    print(f"\nRevised caps config saved -> {caps_json}")

    # ── Build report ──────────────────────────────────────────────────────────
    report_lines = [
        "=" * 75,
        "STEP 2.1 VALIDATION REPORT — Revised Physiological Caps",
        "=" * 75,
        "",
        "CHANGES FROM STEP 2:",
        "  FIX 1 - hemoglobin:  convert_factor=0.1 (PIC stores in g/L, cap in g/dL)",
        "  FIX 2 - anion_gap:   floor 0 -> -5 (measurement variation is valid)",
        "  FIX 3 - spo2_lab:    cap 100.0 -> 100.5 (rounding artefact)",
        "  FIX 3 - spo2_chart:  cap 100.0 -> 100.5 (same reason)",
        "",
        f"{'Variable':<22} {'Total':>8} {'Below':>8} {'Above':>8} "
        f"{'% bad':>8}  {'Cap range'}",
        "-" * 90,
    ]

    total_affected = 0
    total_readings = 0
    changed_vars   = ['hemoglobin', 'anion_gap', 'spo2_lab', 'spo2_chart']

    for var in sorted(PHYSIOLOGICAL_CAPS.keys()):
        info  = PHYSIOLOGICAL_CAPS[var]
        res   = all_results[var]
        n     = res['total']
        below = res['below_floor']
        above = res['above_cap']
        aff   = below + above
        pct   = aff / n * 100 if n > 0 else 0
        total_affected += aff
        total_readings += n

        flag  = " ✱ FIXED" if var in changed_vars else ""
        warn  = " ⚠" if aff > 0 else ""
        cf    = info.get('convert_factor', 1.0)
        cf_note = f" [×{cf}]" if cf != 1.0 else ""

        report_lines.append(
            f"{var:<22} {n:>8,} {below:>8,} {above:>8,} "
            f"{pct:>7.2f}%  "
            f"[{info['floor']} – {info['cap']}] "
            f"{info['unit']}{cf_note}{warn}{flag}"
        )

    report_lines += [
        "-" * 90,
        f"{'TOTAL':<22} {total_readings:>8,} {'':>8} {'':>8} "
        f"{total_affected/total_readings*100 if total_readings else 0:>7.2f}%  "
        f"values affected: {total_affected:,}",
        "",
        "=" * 75,
        "SAMPLE OUTLIER VALUES (after conversion)",
        "=" * 75,
    ]

    for var in sorted(PHYSIOLOGICAL_CAPS.keys()):
        info = PHYSIOLOGICAL_CAPS[var]
        res  = all_results[var]
        if res['above_cap'] > 0 or res['below_floor'] > 0:
            cf = info.get('convert_factor', 1.0)
            cf_note = f" [raw ÷{int(1/cf)} = converted]" if cf != 1.0 else ""
            report_lines.append(f"\n{var} [{info['unit']}]{cf_note}:")
            if res['below_samples']:
                report_lines.append(
                    f"  Below floor ({info['floor']}): {res['below_samples']}")
            if res['above_samples']:
                report_lines.append(
                    f"  Above cap ({info['cap']}): {res['above_samples']}")

    report_lines += [
        "",
        "=" * 75,
        "INTERPRETATION",
        "=" * 75,
        "Remaining outliers after fixes are genuine data errors in PIC.",
        "Step 3 will set these to NULL during feature extraction.",
        "Median imputation replaces NULLs in Step 6 (after train/test split).",
        "",
        f"Caps config: {caps_json}",
        "Step 3 imports step2.1_caps.json automatically.",
    ]

    report_str = '\n'.join(report_lines)
    print('\n' + report_str)

    report_txt = os.path.join(log_dir, 'step2.1_validation.txt')
    with open(report_txt, 'w') as f:
        f.write(report_str)

    print(f"\nValidation report saved -> {report_txt}")
    print("\nStep 2.1 complete. Ready for Step 3.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_step2_1(DATA_DIR, OUTPUT_DIR, LOG_DIR)
