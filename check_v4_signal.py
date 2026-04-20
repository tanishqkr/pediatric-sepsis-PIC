"""
check_v4_signal.py
Run AFTER v4 generation to instantly verify label signal is preserved.

Usage: python check_v4_signal.py
Expected: ratio should be between 0.5 and 2.0 for all top features,
          and sign should ALWAYS match.
"""
import pandas as pd
import numpy as np

r = pd.read_csv("model_datasets/B_train_model_ready.csv")
s = pd.read_csv("model_datasets/synthetic/B_synthetic_train_v4.csv")

KEY = [
    "inr_max", "pt_max", "ddimer_max", "lactate_max", "platelets_min",
    "ast_max", "ldh_max", "bicarbonate_min", "hematocrit_min",
    "base_excess_min", "crp_max", "glucose_max", "creatinine_max",
    "ph_min", "ptt_max", "heart_rate_max", "resp_rate_max",
    "age_years", "shock_index", "wbc_max",
]

print(f"{'Feature':30s} {'Real diff':>10} {'Synth diff':>11} {'Ratio':>7}  {'Sign OK':>8}")
print("-" * 72)
all_ok = True
for f in KEY:
    if f not in r.columns: continue
    rd = r[r.sepsis_label==1][f].mean() - r[r.sepsis_label==0][f].mean()
    sd = s[s.sepsis_label==1][f].mean() - s[s.sepsis_label==0][f].mean()
    ratio = sd / rd if abs(rd) > 1e-9 else float("nan")
    sign_ok = (rd * sd > 0) or abs(rd) < 0.01
    flag = "" if sign_ok else "  <== SIGN WRONG"
    if not sign_ok: all_ok = False
    print(f"{f:30s} {rd:10.3f} {sd:11.3f} {ratio:7.3f}  {'YES' if sign_ok else 'NO':>8}{flag}")

print()
print(f"Synthetic prevalence: {s.sepsis_label.mean():.4f} (real: {r.sepsis_label.mean():.4f})")
print()
if all_ok:
    print("RESULT: SIGNAL PRESERVED — v4 looks correct. Run CatBoost training.")
else:
    print("RESULT: SIGNAL ISSUES FOUND — check label_signal_v4.csv for details.")

