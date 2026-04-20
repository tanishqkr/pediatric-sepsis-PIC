"""
generate_synthetic_data_v4.py
==============================================================================
Synthetic Data Generation using CTGAN  -  VERSION 4
Project : Pediatric Sepsis Prediction (PIC Database)

==============================================================================
ROOT CAUSE OF v3 FAILURE (AUPRC ~0.28):
==============================================================================
v3 achieved excellent marginal distribution fidelity (Mean KS=0.03) but
completely destroyed label-feature associations. Proof from data:

  Real data:    lactate_max  sep0=2.22   sep1=5.78   diff=+3.56
  v3 synthetic: lactate_max  sep0=3.23   sep1=3.24   diff=+0.008  ← DEAD

  Real data:    platelets_min  sep0=283.98  sep1=176.84  diff=-107.14
  v3 synthetic: platelets_min  sep0=253.29  sep1=255.65  diff=+2.36  ← DEAD

v3's post-processing (Steps A-D) was entirely label-agnostic:
  - joint_resample_groups sampled from ALL real rows, ignoring label
  - quantile_map operated on the full marginal, ignoring label
  Result: every column's mean converged to the overall mean regardless
  of sepsis_label. CTGAN generates X independent of Y; v3 never fixed that.

==============================================================================
v4 SOLUTION: CLASS-SPLIT CTGAN + LABEL-CONDITIONED POST-PROCESSING
==============================================================================

STEP 1  Train TWO CTGAN models:
          ctgan_pos  fit on real_df[sepsis_label==1]   (691 rows)
          ctgan_neg  fit on real_df[sepsis_label==0]   (1839 rows)

STEP 2  Sample from each model separately, assign correct label, concat.

STEP 3  Label-conditioned post-processing:
          A. Joint group resampling  — within each class
          B. Single sort pass        — enforce max>=mean>=min
          C. Quantile mapping        — within each class separately
          D. Light swap fix          — preserve marginals post quantile-map
          E. Label-conditioned resampling for standalone continuous cols

STEP 4  Structural fixes (care unit, label prevalence, gender, ranges)

STEP 5  Full validation:
          - KS marginal (overall)
          - KS within each class  (new — catches v3-style collapse)
          - Label-stratified mean comparison vs real (primary signal check)
          - Correlation preservation check

WHY THIS WORKS:
  ctgan_pos learns P(X | Y=1), ctgan_neg learns P(X | Y=0)
  Every post-processing step operates within class, never mixing classes
  Result: synthetic sep1 rows have high lactate, low platelets, etc. — 
  same as real sep1 rows. CatBoost will learn the right signal.

KEY DATA FACTS (from real data analysis):
  - 2530 real rows: 691 pos (27.3%), 1839 neg (72.7%)
  - 125/185 continuous features have KS>0.10 between classes
  - Top discriminators: inr_max (KS=0.44), pt_max (0.44), ddimer (0.42),
    lactate_max (0.41), platelets_min (0.36), ast (0.30), ldh (0.28)
  - Heavy right-skew: inr(10.5), ast(18.9), ck(32.5), ddimer(5.7)
    → CTGAN handles these better class-split (smaller, more homogeneous sets)
  - Strong correlations to preserve: inr_max↔pt_max(0.88),
    ldh↔ast(0.68), lactate↔bicarbonate(-0.54)

DEPENDENCIES:
  pip install sdv scipy matplotlib
  (sdv>=1.0 recommended; also works with ctgan standalone)

Run from project root:
  conda activate sepsis_ml
  python generate_synthetic_data_v4.py

Optional flags:
  --n_synthetic 10000       total synthetic rows (default: 10000)
  --epochs_pos 300          epochs for positive-class CTGAN (default: 300)
  --epochs_neg 300          epochs for negative-class CTGAN (default: 300)
  --batch_size_pos 200      batch size for pos model (default: 200)
  --batch_size_neg 500      batch size for neg model (default: 500)
  --ks_joint_threshold 0.10 KS above which joint resampling fires (default: 0.10)
  --ks_quantile_threshold 0.10 KS above which quantile mapping fires (default: 0.10)
  --noise 0.01              noise fraction of std after bootstrap (default: 0.01)
  --validate                run validation (default: True)
  --skip_ctgan              skip training, load existing models from disk
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ==============================================================================
# PATHS
# ==============================================================================

PROJECT_ROOT   = Path(__file__).resolve().parent
MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"

SYNTHETIC_DIR  = MODEL_DATA_DIR / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR    = SYNTHETIC_DIR / "figures_v4"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR       = SYNTHETIC_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING
# ==============================================================================

log_path = LOGS_DIR / "generate_synthetic_data_v4.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS (derived from real data analysis)
# ==============================================================================

RANDOM_SEED = 42
TARGET_COL  = "sepsis_label"

# All lab/vital groups that have max/min/mean/first sub-columns
ORDERED_GROUPS = [
    "albumin", "alc", "alt", "anc", "anion_gap", "ast", "base_excess",
    "bicarbonate", "bilirubin", "calcium", "ck", "creatinine", "crp",
    "ddimer", "diastolic_bp", "fibrinogen", "glucose", "heart_rate",
    "hematocrit", "hemoglobin", "inr", "lactate", "ldh", "map", "pao2",
    "pco2", "ph", "platelets", "potassium", "pt", "ptt", "resp_rate",
    "sodium", "spo2", "spo2_lab", "systolic_bp", "temperature", "urea",
    "uric_acid", "wbc",
]

# Standalone continuous features (not part of max/min groups)
STANDALONE_CONTINUOUS = [
    "age_years", "shock_index", "urine_output_total_ml",
    "urine_output_per_hour", "fluid_input_total_ml",
    "fluid_balance_ml", "n_antibiotics_24h", "n_cultures_24h",
    "n_positive_cultures",
]

CARE_UNIT_COLS = [
    "care_unit_CICU", "care_unit_General ICU",
    "care_unit_NICU", "care_unit_PICU", "care_unit_SICU",
]

# Top discriminative features — used for focused validation plots
KEY_FEATURES_POS = [
    "lactate_max", "inr_max", "pt_max", "ddimer_max", "platelets_min",
    "ast_max", "ldh_max", "bicarbonate_min", "hematocrit_min", "ptt_max",
    "base_excess_min", "crp_max", "glucose_max", "creatinine_max", "ph_min",
]

KEY_FEATURES_NEG = [
    "heart_rate_max", "resp_rate_max", "temperature_max", "spo2_min",
    "wbc_max", "sodium_min", "potassium_min", "calcium_min", "hemoglobin_min",
    "age_years", "shock_index", "map_min", "diastolic_bp_min",
]

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic pediatric sepsis data using class-split CTGAN (v4)"
    )
    p.add_argument("--n_synthetic",           type=int,   default=10_000)
    p.add_argument("--epochs_pos",            type=int,   default=300,
                   help="CTGAN epochs for positive class (691 rows — don't overtrain)")
    p.add_argument("--epochs_neg",            type=int,   default=300,
                   help="CTGAN epochs for negative class")
    p.add_argument("--batch_size_pos",        type=int,   default=200,
                   help="Batch size for pos model (smaller because 691 rows)")
    p.add_argument("--batch_size_neg",        type=int,   default=500,
                   help="Batch size for neg model")
    p.add_argument("--ks_joint_threshold",    type=float, default=0.10)
    p.add_argument("--ks_quantile_threshold", type=float, default=0.10)
    p.add_argument("--noise",                 type=float, default=0.01,
                   help="Noise fraction: sigma = noise * std(col). Lower than v3 "
                        "because class-split already gives better distributions.")
    p.add_argument("--validate",   action="store_true", default=True)
    p.add_argument("--skip_ctgan", action="store_true", default=False)
    return p.parse_args()

# ==============================================================================
# CTGAN IMPORT — supports both sdv and standalone ctgan
# ==============================================================================

def import_ctgan():
    """
    Returns (CTGAN_class, use_sdv: bool).
    Tries sdv first (more featureful), falls back to standalone ctgan.
    """
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        log.info("  Using SDV CTGANSynthesizer.")
        return CTGANSynthesizer, True
    except ImportError:
        pass
    try:
        from ctgan import CTGAN
        log.info("  Using standalone CTGAN.")
        return CTGAN, False
    except ImportError:
        log.error("Neither sdv nor ctgan installed. Run: pip install sdv")
        sys.exit(1)


def build_and_fit_ctgan(CTGAN_cls, use_sdv, data_df, discrete_cols,
                         epochs, batch_size, label_name):
    """
    Builds and fits one CTGAN model on data_df (which must NOT contain TARGET_COL).
    Returns the fitted model.
    """
    log.info(f"  [{label_name}] Fitting on {len(data_df):,} rows, "
             f"{epochs} epochs, batch_size={batch_size}")

    if use_sdv:
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(data_df)
        for col in discrete_cols:
            if col in data_df.columns:
                meta.update_column(col, sdtype="categorical")
        model = CTGAN_cls(
            metadata             = meta,
            epochs               = epochs,
            batch_size           = batch_size,
            generator_dim        = (256, 256),
            discriminator_dim    = (256, 256),
            generator_lr         = 2e-4,
            discriminator_lr     = 2e-4,
            discriminator_steps  = 1,
            log_frequency        = True,
            verbose              = True,
            cuda                 = False,
        )
        model.fit(data_df)
    else:
        model = CTGAN_cls(
            epochs               = epochs,
            batch_size           = batch_size,
            generator_dim        = (256, 256),
            discriminator_dim    = (256, 256),
            generator_lr         = 2e-4,
            discriminator_lr     = 2e-4,
            discriminator_steps  = 1,
            log_frequency        = True,
            verbose              = True,
            cuda                 = False,
        )
        model.fit(data_df, discrete_columns=discrete_cols)

    return model


def sample_from_ctgan(model, use_sdv, n):
    try:
        if use_sdv:
            return model.sample(num_rows=n)
        else:
            return model.sample(n)
    except TypeError:
        return model.sample(num_rows=n)

# ==============================================================================
# POST-PROCESSING STEP A: Label-conditioned joint group resampling
# ==============================================================================

def joint_resample_groups_conditioned(synth_df, real_df, groups,
                                       ks_threshold, noise, rng):
    """
    For each lab/vital group (e.g. lactate_max, lactate_min, lactate_mean,
    lactate_first): if any column in the group has KS > ks_threshold WITHIN
    the same class, replace the entire group by sampling COMPLETE ROWS from
    the SAME CLASS in real data.

    KEY DIFFERENCE FROM v3: resampling is done separately for sep=0 and sep=1,
    so class-specific patterns (e.g. high lactate in sepsis) are preserved.
    """
    df = synth_df.copy()
    n_corrected = 0

    for label_val in [0, 1]:
        mask_synth = df[TARGET_COL] == label_val
        real_class = real_df[real_df[TARGET_COL] == label_val]
        n_class = int(mask_synth.sum())
        if n_class == 0:
            continue

        for base in groups:
            suffixes = ["_max", "_min", "_mean", "_first"]
            cols = [f"{base}{s}" for s in suffixes if f"{base}{s}" in real_class.columns]
            if len(cols) < 2:
                continue

            # Check worst KS within this class
            worst_ks = 0.0
            for col in cols:
                if real_class[col].nunique() > 2:
                    ks_val, _ = stats.ks_2samp(
                        real_class[col].values,
                        df.loc[mask_synth, col].values
                    )
                    worst_ks = max(worst_ks, ks_val)

            if worst_ks <= ks_threshold:
                continue

            real_group = real_class[cols].to_numpy(dtype=float)
            idx = rng.integers(0, len(real_group), size=n_class)
            sampled = real_group[idx].copy()

            for i, col in enumerate(cols):
                noise_std = noise * float(real_class[col].std())
                if noise_std > 0:
                    sampled[:, i] += rng.normal(0, noise_std, size=n_class)
                sampled[:, i] = np.clip(
                    sampled[:, i],
                    float(real_class[col].min()),
                    float(real_class[col].max()),
                )
                df.loc[mask_synth, col] = sampled[:, i]

            n_corrected += 1

    log.info(f"  joint_resample_conditioned: corrected {n_corrected} "
             f"group×class combinations.")
    return df


# ==============================================================================
# POST-PROCESSING STEP B: Sort pass (max >= mean >= min)
# ==============================================================================

def sort_max_mean_min(synth_df, groups):
    """
    Single sort pass enforcing max >= mean >= min per row.
    _first is clipped to [min, max]. Operates on full df (both classes).
    Class signal is already embedded — sorting within a row doesn't destroy it.
    """
    df = synth_df.copy()
    for base in groups:
        max_c   = f"{base}_max"
        min_c   = f"{base}_min"
        mean_c  = f"{base}_mean"
        first_c = f"{base}_first"

        if max_c not in df.columns or min_c not in df.columns:
            continue

        if mean_c in df.columns:
            trio = np.sort(df[[max_c, min_c, mean_c]].to_numpy(dtype=float), axis=1)
            df[min_c]  = trio[:, 0]
            df[mean_c] = trio[:, 1]
            df[max_c]  = trio[:, 2]
        else:
            inverted = df[max_c].values < df[min_c].values
            tmp = df.loc[inverted, max_c].copy()
            df.loc[inverted, max_c] = df.loc[inverted, min_c]
            df.loc[inverted, min_c] = tmp

        if first_c in df.columns:
            df[first_c] = df[first_c].clip(lower=df[min_c], upper=df[max_c])

    violations = sum(
        int((df[f"{b}_max"] < df[f"{b}_min"]).sum())
        for b in groups
        if f"{b}_max" in df.columns and f"{b}_min" in df.columns
    )
    log.info(f"  sort_max_mean_min: done. Remaining violations: {violations}")
    return df


# ==============================================================================
# POST-PROCESSING STEP C: Label-conditioned quantile mapping
# ==============================================================================

def quantile_map_column(real_vals, synth_vals):
    """
    Rank-based quantile mapping: maps synthetic distribution to real distribution
    while preserving relative ordering of rows (so inter-feature correlations
    are maintained within a class).
    """
    n = len(synth_vals)
    real_sorted = np.sort(real_vals)
    synth_ranks = np.argsort(np.argsort(synth_vals))
    quantile_indices = np.clip(
        (synth_ranks / max(n - 1, 1) * (len(real_sorted) - 1)).astype(int),
        0,
        len(real_sorted) - 1,
    )
    return real_sorted[quantile_indices]


def quantile_map_conditioned(synth_df, real_df, feature_cols, discrete_cols,
                              ks_threshold):
    """
    For each continuous column, for each class separately:
    if KS between real_class and synth_class > ks_threshold, apply
    quantile mapping using real_class as the reference.

    This ensures that after the sort pass (which slightly perturbs marginals),
    each class's distribution is correctly restored — not the overall marginal.
    """
    df = synth_df.copy()
    n_mapped = 0

    for col in feature_cols:
        if col in discrete_cols or col == TARGET_COL:
            continue
        if real_df[col].nunique() <= 2:
            continue

        for label_val in [0, 1]:
            mask_synth = df[TARGET_COL] == label_val
            real_class = real_df[real_df[TARGET_COL] == label_val][col].dropna().values
            synth_class = df.loc[mask_synth, col].values

            if len(real_class) < 10 or len(synth_class) < 10:
                continue

            ks_val, _ = stats.ks_2samp(real_class, synth_class)
            if ks_val > ks_threshold:
                df.loc[mask_synth, col] = quantile_map_column(real_class, synth_class)
                n_mapped += 1

    log.info(f"  quantile_map_conditioned: mapped {n_mapped} col×class "
             f"combinations (KS > {ks_threshold}).")
    return df, n_mapped


# ==============================================================================
# POST-PROCESSING STEP D: Light swap fix
# ==============================================================================

def light_swap_fix(synth_df, groups):
    """
    After quantile mapping (which operates per-column independently),
    some max < min inversions may appear. Swap-only fix — does NOT
    redistribute values across rows, so marginals are preserved.
    """
    df = synth_df.copy()
    total_swapped = 0

    for base in groups:
        max_c   = f"{base}_max"
        min_c   = f"{base}_min"
        mean_c  = f"{base}_mean"
        first_c = f"{base}_first"

        if max_c not in df.columns or min_c not in df.columns:
            continue

        inverted = df[max_c].values < df[min_c].values
        n_inv = int(inverted.sum())
        if n_inv > 0:
            tmp = df.loc[inverted, max_c].copy()
            df.loc[inverted, max_c] = df.loc[inverted, min_c]
            df.loc[inverted, min_c] = tmp
            total_swapped += n_inv

        if mean_c in df.columns:
            df[mean_c] = df[mean_c].clip(lower=df[min_c], upper=df[max_c])
        if first_c in df.columns:
            df[first_c] = df[first_c].clip(lower=df[min_c], upper=df[max_c])

    remaining = sum(
        int((df[f"{b}_max"] < df[f"{b}_min"]).sum())
        for b in groups
        if f"{b}_max" in df.columns and f"{b}_min" in df.columns
    )
    log.info(f"  light_swap_fix: swapped {total_swapped} pairs. "
             f"Remaining violations: {remaining}")
    return df


# ==============================================================================
# POST-PROCESSING STEP E: Standalone continuous — label-conditioned resample
# ==============================================================================

def resample_standalone_conditioned(synth_df, real_df, standalone_cols,
                                     ks_threshold, noise, rng):
    """
    For standalone continuous features (age, shock_index, urine_output, etc.)
    that don't belong to max/min groups: if KS > threshold within a class,
    resample from the real distribution of that class.
    """
    df = synth_df.copy()
    n_corrected = 0

    for col in standalone_cols:
        if col not in df.columns or col not in real_df.columns:
            continue

        for label_val in [0, 1]:
            mask_synth = df[TARGET_COL] == label_val
            real_class = real_df[real_df[TARGET_COL] == label_val][col].dropna().values
            synth_class = df.loc[mask_synth, col].values
            n_class = int(mask_synth.sum())

            if len(real_class) < 5 or len(synth_class) < 5:
                continue
            if real_df[col].nunique() <= 2:
                continue

            ks_val, _ = stats.ks_2samp(real_class, synth_class)
            if ks_val > ks_threshold:
                idx = rng.integers(0, len(real_class), size=n_class)
                sampled = real_class[idx].copy()
                noise_std = noise * float(real_class.std())
                if noise_std > 0:
                    sampled += rng.normal(0, noise_std, size=n_class)
                sampled = np.clip(sampled, real_class.min(), real_class.max())
                df.loc[mask_synth, col] = sampled
                n_corrected += 1

    log.info(f"  resample_standalone_conditioned: corrected {n_corrected} "
             f"col×class combinations.")
    return df


# ==============================================================================
# STRUCTURAL FIXES
# ==============================================================================

def fix_care_unit_onehot(synth_df, care_unit_cols, real_dist_by_class, rng):
    """
    Enforces exactly-one-hot for care unit columns.
    Uses class-specific real distribution for zero-unit rows.
    """
    df = synth_df.copy()
    available = [c for c in care_unit_cols if c in df.columns]
    if not available:
        return df

    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Fix multi-unit rows (keep argmax)
    multi = df[available].sum(axis=1) > 1
    if multi.sum() > 0:
        raw = df.loc[multi, available].values
        winner = np.argmax(raw, axis=1)
        cor = np.zeros_like(raw, dtype=int)
        for i, w in enumerate(winner):
            cor[i, w] = 1
        df.loc[multi, available] = cor
        log.info(f"  care_unit_onehot: fixed {int(multi.sum())} multi-unit rows.")

    # Fix zero-unit rows using class-specific distribution
    zero = df[available].sum(axis=1) == 0
    if zero.sum() > 0:
        zero_idx = np.where(zero)[0]
        for ri in zero_idx:
            label_val = int(df.iloc[ri][TARGET_COL])
            real_dist = real_dist_by_class.get(label_val, {})
            probs = np.array([real_dist.get(c, 1/len(available)) for c in available])
            probs = np.maximum(probs, 1e-9)
            probs /= probs.sum()
            ci = rng.choice(len(available), p=probs)
            df.iloc[ri, df.columns.get_loc(available[0]):
                         df.columns.get_loc(available[-1])+1] = 0
            df.at[df.index[ri], available[ci]] = 1
        log.info(f"  care_unit_onehot: fixed {int(zero.sum())} zero-unit rows.")

    df[available] = df[available].astype(int)
    remaining = int((df[available].sum(axis=1) != 1).sum())
    log.info(f"  care_unit_onehot: {remaining} violations remain.")
    return df


def fix_label_prevalence(synth_df, target_prev, rng, tol=0.005):
    df = synth_df.copy()
    n = len(df)
    target_count = int(round(target_prev * n))
    delta = target_count - int(df[TARGET_COL].sum())

    if abs(delta) < max(1, int(tol * n)):
        log.info(f"  label_prevalence: within tolerance ({df[TARGET_COL].mean():.4f}). No change.")
        return df

    if delta > 0:
        idx = rng.choice(df.index[df[TARGET_COL] == 0].tolist(), size=delta, replace=False)
        df.loc[idx, TARGET_COL] = 1
    else:
        idx = rng.choice(df.index[df[TARGET_COL] == 1].tolist(), size=abs(delta), replace=False)
        df.loc[idx, TARGET_COL] = 0

    log.info(f"  label_prevalence: adjusted by {delta} -> {df[TARGET_COL].mean():.4f}")
    return df


def fix_gender_prevalence(synth_df, real_male_rate_by_class, rng, tol=0.005):
    """Fix gender prevalence within each class."""
    df = synth_df.copy()
    if "gender" not in df.columns:
        return df

    for label_val in [0, 1]:
        mask = df[TARGET_COL] == label_val
        target_rate = real_male_rate_by_class.get(label_val, 0.6)
        n_class = int(mask.sum())
        target_count = int(round(target_rate * n_class))
        current_count = int(df.loc[mask, "gender"].sum())
        delta = target_count - current_count

        if abs(delta) < max(1, int(tol * n_class)):
            continue

        class_idx = df.index[mask].tolist()
        if delta > 0:
            female_in_class = [i for i in class_idx if df.at[i, "gender"] == 0]
            if len(female_in_class) >= delta:
                flip = rng.choice(female_in_class, size=delta, replace=False)
                df.loc[flip, "gender"] = 1
        else:
            male_in_class = [i for i in class_idx if df.at[i, "gender"] == 1]
            if len(male_in_class) >= abs(delta):
                flip = rng.choice(male_in_class, size=abs(delta), replace=False)
                df.loc[flip, "gender"] = 0

    log.info(f"  gender: sep0={df.loc[df[TARGET_COL]==0,'gender'].mean():.4f} "
             f"sep1={df.loc[df[TARGET_COL]==1,'gender'].mean():.4f}")
    return df


# ==============================================================================
# VALIDATION
# ==============================================================================

def ks_summary_overall(real_df, synth_df, feature_cols):
    """KS between real and synthetic marginals (ignoring label). For structural QC."""
    results = []
    for col in feature_cols:
        if real_df[col].nunique() <= 2:
            continue
        rv = real_df[col].dropna().values
        sv = synth_df[col].dropna().values
        if len(rv) < 10 or len(sv) < 10:
            continue
        ks_val, ks_p = stats.ks_2samp(rv, sv)
        results.append({
            "feature": col,
            "ks_statistic": round(ks_val, 4),
            "ks_p_value": round(ks_p, 4),
            "real_mean": round(rv.mean(), 4),
            "synth_mean": round(sv.mean(), 4),
        })
    return pd.DataFrame(results).sort_values("ks_statistic", ascending=False)


def label_signal_check(real_df, synth_df, feature_cols, discrete_cols):
    """
    THE CRITICAL v4 VALIDATION:
    For each feature, compare:
      - Real: mean(feature | sep=1) - mean(feature | sep=0)
      - Synth: mean(feature | sep=1) - mean(feature | sep=0)

    A good synthetic dataset should preserve the SIGN and approximate
    MAGNITUDE of this difference for all discriminative features.
    """
    results = []
    for col in feature_cols:
        if col in discrete_cols or col == TARGET_COL:
            continue
        if real_df[col].nunique() <= 2:
            continue

        r0 = real_df[real_df[TARGET_COL] == 0][col].mean()
        r1 = real_df[real_df[TARGET_COL] == 1][col].mean()
        s0 = synth_df[synth_df[TARGET_COL] == 0][col].mean()
        s1 = synth_df[synth_df[TARGET_COL] == 1][col].mean()

        real_diff = r1 - r0
        synth_diff = s1 - s0
        sign_preserved = (np.sign(real_diff) == np.sign(synth_diff)) if real_diff != 0 else True
        ratio = synth_diff / real_diff if abs(real_diff) > 1e-9 else 1.0

        results.append({
            "feature": col,
            "real_diff": round(real_diff, 4),
            "synth_diff": round(synth_diff, 4),
            "ratio": round(ratio, 4),
            "sign_preserved": sign_preserved,
            "real_sep0": round(r0, 4),
            "real_sep1": round(r1, 4),
            "synth_sep0": round(s0, 4),
            "synth_sep1": round(s1, 4),
        })

    df_sig = pd.DataFrame(results)
    # Sort by absolute real_diff to show most discriminative features first
    df_sig = df_sig.assign(abs_real_diff=df_sig["real_diff"].abs())
    df_sig = df_sig.sort_values("abs_real_diff", ascending=False).drop(columns="abs_real_diff")
    return df_sig


def ks_within_class(real_df, synth_df, feature_cols, discrete_cols):
    """
    KS between real_class and synth_class for each feature×class.
    This catches v3-style collapse where overall KS is fine but class
    distributions have converged to each other.
    """
    results = []
    for col in feature_cols:
        if col in discrete_cols or col == TARGET_COL:
            continue
        if real_df[col].nunique() <= 2:
            continue

        for label_val in [0, 1]:
            rv = real_df[real_df[TARGET_COL] == label_val][col].dropna().values
            sv = synth_df[synth_df[TARGET_COL] == label_val][col].dropna().values
            if len(rv) < 10 or len(sv) < 10:
                continue
            ks_val, _ = stats.ks_2samp(rv, sv)
            results.append({
                "feature": col,
                "class": label_val,
                "ks_within_class": round(ks_val, 4),
            })

    return pd.DataFrame(results).sort_values("ks_within_class", ascending=False)


def check_structural_integrity(synth_df, groups, care_cols):
    violations = {}
    for base in groups:
        max_c, min_c = f"{base}_max", f"{base}_min"
        if max_c in synth_df.columns and min_c in synth_df.columns:
            v = int((synth_df[max_c] < synth_df[min_c]).sum())
            if v > 0:
                violations[f"{base}_ordering"] = v
    cu = [c for c in care_cols if c in synth_df.columns]
    if cu:
        cu_bad = int((synth_df[cu].sum(axis=1) != 1).sum())
        if cu_bad > 0:
            violations["care_unit_onehot"] = cu_bad
    return violations


def plot_label_signal(signal_df, top_n=20, suffix=""):
    top = signal_df.head(top_n).copy()
    x = np.arange(len(top))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.35)))
    bars_real  = ax.bar(x - width/2, top["real_diff"],  width, label="Real diff (sep1-sep0)",
                        color="#1565C0", alpha=0.8)
    bars_synth = ax.bar(x + width/2, top["synth_diff"], width, label="Synth diff (sep1-sep0)",
                        color="#E53935", alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean difference (sep1 - sep0)")
    ax.set_title(f"Label Signal Preservation — Top {top_n} Discriminative Features\n"
                 f"Red≈Blue = GOOD. If bars don't match → signal destroyed.",
                 fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / f"label_signal{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved label signal plot -> {path}")


def plot_distribution_comparison_by_class(real_df, synth_df, feature_list, suffix=""):
    available = [f for f in feature_list if f in real_df.columns]
    if not available:
        return
    n_rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        for label_val, color_r, color_s, ls in [
            (0, "#1565C0", "#64B5F6", "-"),
            (1, "#B71C1C", "#EF9A9A", "--"),
        ]:
            rv = real_df[real_df[TARGET_COL] == label_val][feat].dropna()
            sv = synth_df[synth_df[TARGET_COL] == label_val][feat].dropna()
            if len(rv) < 5:
                continue
            upper = rv.quantile(0.99)
            lower = rv.quantile(0.01)
            ax.hist(rv.clip(lower, upper), bins=30, alpha=0.5, color=color_r,
                    density=True, label=f"Real sep={label_val}", linestyle=ls)
            ax.hist(sv.clip(lower, upper), bins=30, alpha=0.5, color=color_s,
                    density=True, label=f"Synth sep={label_val}", linestyle=ls)

        ks_all, _ = stats.ks_2samp(
            real_df[feat].dropna(), synth_df[feat].dropna()
        )
        ax.set_title(f"{feat}\nKS_overall={ks_all:.3f}", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-class Distribution: Real vs Synthetic (v4)\n"
                 "Blue=sep0, Red=sep1 | Solid=Real, Hatched=Synth",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / f"class_distributions{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved class distribution plot -> {path}")


def plot_ks_bar(ks_df, col="ks_statistic", top_n=30, title="KS Summary", suffix=""):
    top = ks_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#E53935" if v > 0.10 else "#FFA726" if v > 0.05 else "#43A047"
              for v in top[col]]
    ax.barh(top["feature"].astype(str), top[col], color=colors)
    ax.axvline(0.05, color="orange", ls="--", lw=1, label="0.05")
    ax.axvline(0.10, color="red",    ls="--", lw=1, label="0.10")
    ax.set_xlabel(col)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / f"ks_bar{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved KS bar -> {path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("SYNTHETIC DATA GENERATION v4 -- CLASS-SPLIT CTGAN")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # =========================================================================
    # STEP 1: Load real data
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 1 -- Loading real training data")
    log.info("=" * 60)

    if not TRAIN_FILE.exists():
        log.error(f"Training file not found: {TRAIN_FILE}")
        sys.exit(1)

    train_df     = pd.read_csv(TRAIN_FILE)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    n_real       = len(train_df)
    prevalence   = float(train_df[TARGET_COL].mean())

    real_pos = train_df[train_df[TARGET_COL] == 1].copy()
    real_neg = train_df[train_df[TARGET_COL] == 0].copy()

    log.info(f"  Shape       : {train_df.shape}")
    log.info(f"  Total rows  : {n_real:,}")
    log.info(f"  Pos (sep=1) : {len(real_pos):,} ({prevalence*100:.1f}%)")
    log.info(f"  Neg (sep=0) : {len(real_neg):,} ({(1-prevalence)*100:.1f}%)")

    # Identify discrete columns (binary 0/1 only, excluding target)
    discrete_cols = []
    for col in feature_cols:
        uniq = set(train_df[col].dropna().unique())
        if uniq.issubset({0, 1, 0.0, 1.0}):
            discrete_cols.append(col)

    log.info(f"  Discrete feature cols: {len(discrete_cols)}")
    log.info(f"  Continuous feature cols: {len(feature_cols) - len(discrete_cols)}")

    # Care unit real distributions, by class
    care_unit_dist_by_class = {}
    for label_val in [0, 1]:
        sub = train_df[train_df[TARGET_COL] == label_val]
        care_unit_dist_by_class[label_val] = {
            c: float(sub[c].mean())
            for c in CARE_UNIT_COLS if c in sub.columns
        }

    # Gender rate by class
    gender_rate_by_class = {}
    for label_val in [0, 1]:
        sub = train_df[train_df[TARGET_COL] == label_val]
        if "gender" in sub.columns:
            gender_rate_by_class[label_val] = float(sub["gender"].mean())

    # =========================================================================
    # STEP 2: Train TWO CTGAN models (one per class)
    # =========================================================================
    ctgan_pos_path = SYNTHETIC_DIR / "ctgan_model_v4_pos.pkl"
    ctgan_neg_path = SYNTHETIC_DIR / "ctgan_model_v4_neg.pkl"

    if args.skip_ctgan and ctgan_pos_path.exists() and ctgan_neg_path.exists():
        log.info("\nLoading saved CTGAN models (--skip_ctgan)...")
        with open(ctgan_pos_path, "rb") as f:
            ctgan_pos, use_sdv_pos = pickle.load(f)
        with open(ctgan_neg_path, "rb") as f:
            ctgan_neg, use_sdv_neg = pickle.load(f)
        train_time = 0.0
    else:
        log.info("\n" + "=" * 60)
        log.info("STEP 2 -- Training TWO CTGAN models (class-split)")
        log.info("=" * 60)

        CTGAN_cls, use_sdv = import_ctgan()
        use_sdv_pos = use_sdv_neg = use_sdv

        # Data for each model: drop target col (model learns X only)
        pos_data = real_pos.drop(columns=[TARGET_COL])
        neg_data = real_neg.drop(columns=[TARGET_COL])

        # Discrete cols for the feature-only data (target removed)
        feat_discrete = [c for c in discrete_cols if c in pos_data.columns]

        t_train = time.time()

        log.info(f"\n[Model A — Positive class, n={len(pos_data):,}]")
        log.info(f"  epochs={args.epochs_pos}, batch_size={args.batch_size_pos}")
        log.info(f"  NOTE: Small dataset (691 rows). batch_size_pos=200 prevents ")
        log.info(f"  CTGAN training on batches larger than the dataset.")
        ctgan_pos = build_and_fit_ctgan(
            CTGAN_cls, use_sdv, pos_data, feat_discrete,
            args.epochs_pos, args.batch_size_pos, "POS"
        )

        log.info(f"\n[Model B — Negative class, n={len(neg_data):,}]")
        log.info(f"  epochs={args.epochs_neg}, batch_size={args.batch_size_neg}")
        ctgan_neg = build_and_fit_ctgan(
            CTGAN_cls, use_sdv, neg_data, feat_discrete,
            args.epochs_neg, args.batch_size_neg, "NEG"
        )

        train_time = (time.time() - t_train) / 60
        log.info(f"\n  Both models trained in {train_time:.1f} min")

        with open(ctgan_pos_path, "wb") as f:
            pickle.dump((ctgan_pos, use_sdv_pos), f)
        with open(ctgan_neg_path, "wb") as f:
            pickle.dump((ctgan_neg, use_sdv_neg), f)
        log.info(f"  Models saved -> {ctgan_pos_path}, {ctgan_neg_path}")

    # =========================================================================
    # STEP 3: Sample from each model separately
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 3 -- Sampling from class-split models")
    log.info("=" * 60)

    n_pos_target = int(round(prevalence * args.n_synthetic))
    n_neg_target = args.n_synthetic - n_pos_target

    log.info(f"  Sampling {n_pos_target:,} positive rows from ctgan_pos...")
    t_gen = time.time()
    synth_pos = sample_from_ctgan(ctgan_pos, use_sdv_pos, n_pos_target)
    synth_pos[TARGET_COL] = 1

    log.info(f"  Sampling {n_neg_target:,} negative rows from ctgan_neg...")
    synth_neg = sample_from_ctgan(ctgan_neg, use_sdv_neg, n_neg_target)
    synth_neg[TARGET_COL] = 0

    gen_time = (time.time() - t_gen) / 60
    log.info(f"  Sampling complete in {gen_time:.1f} min")

    # Combine and shuffle
    synth_df = pd.concat([synth_pos, synth_neg], ignore_index=True)
    synth_df = synth_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Align columns to training data schema
    synth_df = synth_df.reindex(columns=train_df.columns)

    # Initial range clip and discrete rounding
    for col in discrete_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)
    for col in feature_cols:
        if col not in discrete_cols and col in synth_df.columns:
            synth_df[col] = synth_df[col].clip(
                train_df[col].min(), train_df[col].max()
            )

    log.info(f"  Combined shape: {synth_df.shape}")
    log.info(f"  Prevalence after sampling: {synth_df[TARGET_COL].mean():.4f} "
             f"(target: {prevalence:.4f})")

    # Sanity check: label signal before post-processing
    log.info("\n  [SIGNAL CHECK — before post-processing]")
    for feat in ["lactate_max", "platelets_min", "inr_max", "bicarbonate_min"]:
        if feat in synth_df.columns:
            s0 = synth_df[synth_df[TARGET_COL]==0][feat].mean()
            s1 = synth_df[synth_df[TARGET_COL]==1][feat].mean()
            r0 = train_df[train_df[TARGET_COL]==0][feat].mean()
            r1 = train_df[train_df[TARGET_COL]==1][feat].mean()
            log.info(f"    {feat:25s}  Real sep0={r0:.3f} sep1={r1:.3f} diff={r1-r0:+.3f} | "
                     f"Synth sep0={s0:.3f} sep1={s1:.3f} diff={s1-s0:+.3f}")

    # =========================================================================
    # STEP 4: Post-processing (all conditioned on label)
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 4 -- Label-conditioned post-processing")
    log.info("=" * 60)

    # Step A: Label-conditioned joint group resampling
    log.info(f"\n[Step A] Label-conditioned joint group resampling "
             f"(KS threshold per class: {args.ks_joint_threshold})...")
    synth_df = joint_resample_groups_conditioned(
        synth_df, train_df, ORDERED_GROUPS,
        ks_threshold=args.ks_joint_threshold,
        noise=args.noise,
        rng=rng,
    )

    # Step B: Single sort pass (max >= mean >= min)
    log.info("\n[Step B] Sort pass (max >= mean >= min)...")
    synth_df = sort_max_mean_min(synth_df, ORDERED_GROUPS)

    # Step C: Label-conditioned quantile mapping
    log.info(f"\n[Step C] Label-conditioned quantile mapping "
             f"(KS threshold per class: {args.ks_quantile_threshold})...")
    synth_df, n_mapped = quantile_map_conditioned(
        synth_df, train_df, feature_cols, discrete_cols,
        ks_threshold=args.ks_quantile_threshold,
    )

    # Step D: Light swap fix
    log.info("\n[Step D] Light swap fix...")
    synth_df = light_swap_fix(synth_df, ORDERED_GROUPS)

    # Step E: Label-conditioned resample for standalone continuous cols
    standalone_present = [c for c in STANDALONE_CONTINUOUS if c in synth_df.columns]
    log.info(f"\n[Step E] Standalone continuous resample "
             f"(KS threshold: {args.ks_joint_threshold})...")
    synth_df = resample_standalone_conditioned(
        synth_df, train_df, standalone_present,
        ks_threshold=args.ks_joint_threshold,
        noise=args.noise,
        rng=rng,
    )

    # Structural fixes
    log.info("\n[Fix] Care unit one-hot exclusivity...")
    synth_df = fix_care_unit_onehot(
        synth_df, CARE_UNIT_COLS, care_unit_dist_by_class, rng
    )

    log.info("\n[Fix] Label prevalence matching...")
    synth_df = fix_label_prevalence(synth_df, prevalence, rng)

    log.info("\n[Fix] Gender prevalence matching (per class)...")
    synth_df = fix_gender_prevalence(synth_df, gender_rate_by_class, rng)

    # Final range clip
    for col in feature_cols:
        if col not in discrete_cols and col in synth_df.columns:
            synth_df[col] = synth_df[col].clip(
                train_df[col].min(), train_df[col].max()
            )

    # Missing value fill
    n_missing = int(synth_df.isnull().sum().sum())
    if n_missing > 0:
        log.warning(f"  {n_missing} missing values — filling with class medians.")
        for label_val in [0, 1]:
            mask = synth_df[TARGET_COL] == label_val
            real_class = train_df[train_df[TARGET_COL] == label_val]
            for col in synth_df.columns:
                if synth_df.loc[mask, col].isnull().any():
                    fill_val = real_class[col].median() if col in real_class else train_df[col].median()
                    synth_df.loc[mask, col] = synth_df.loc[mask, col].fillna(fill_val)

    # =========================================================================
    # STEP 5: Save
    # =========================================================================
    log.info("\n" + "=" * 60)
    log.info("STEP 5 -- Saving synthetic dataset")
    log.info("=" * 60)

    synth_path = SYNTHETIC_DIR / "B_synthetic_train_v4.csv"
    synth_df.to_csv(synth_path, index=False)
    log.info(f"  Saved -> {synth_path}  {synth_df.shape}")

    # =========================================================================
    # STEP 6: Validation
    # =========================================================================
    if args.validate:
        log.info("\n" + "=" * 60)
        log.info("STEP 6 -- Validation")
        log.info("=" * 60)

        # 6A: Overall marginal KS
        ks_df = ks_summary_overall(train_df, synth_df, feature_cols)
        ks_path = SYNTHETIC_DIR / "ks_validation_results_v4.csv"
        ks_df.to_csv(ks_path, index=False)

        n_good    = int((ks_df["ks_statistic"] <= 0.05).sum())
        n_ok      = int(((ks_df["ks_statistic"] > 0.05) & (ks_df["ks_statistic"] <= 0.10)).sum())
        n_concern = int((ks_df["ks_statistic"] > 0.10).sum())
        total     = len(ks_df)

        log.info("\n  [6A] OVERALL MARGINAL KS (structural check):")
        log.info(f"    KS <= 0.05 : {n_good:3d} ({n_good/total*100:.0f}%)")
        log.info(f"    KS 0.05-0.10: {n_ok:3d} ({n_ok/total*100:.0f}%)")
        log.info(f"    KS > 0.10  : {n_concern:3d} ({n_concern/total*100:.0f}%)")
        log.info(f"    Mean KS    : {ks_df['ks_statistic'].mean():.4f}")
        log.info(f"    Median KS  : {ks_df['ks_statistic'].median():.4f}")

        # 6B: Within-class KS (catches v3-style collapse)
        ks_class_df = ks_within_class(train_df, synth_df, feature_cols, discrete_cols)
        ks_class_path = SYNTHETIC_DIR / "ks_within_class_v4.csv"
        ks_class_df.to_csv(ks_class_path, index=False)

        n_class_concern = int((ks_class_df["ks_within_class"] > 0.10).sum())
        log.info(f"\n  [6B] WITHIN-CLASS KS (v3-collapse detector):")
        log.info(f"    Mean within-class KS : {ks_class_df['ks_within_class'].mean():.4f}")
        log.info(f"    col×class > 0.10     : {n_class_concern} / {len(ks_class_df)}")
        if n_class_concern > 0:
            log.info(f"    Top offenders:")
            for _, row in ks_class_df.head(10).iterrows():
                log.info(f"      {row['feature']:35s} class={int(row['class'])}  "
                         f"KS={row['ks_within_class']:.4f}")

        # 6C: Label signal check (THE key metric)
        signal_df = label_signal_check(train_df, synth_df, feature_cols, discrete_cols)
        signal_path = SYNTHETIC_DIR / "label_signal_v4.csv"
        signal_df.to_csv(signal_path, index=False)

        signs_preserved = int(signal_df["sign_preserved"].sum())
        total_sig = len(signal_df)
        mean_ratio = signal_df.loc[signal_df["real_diff"].abs() > 0.1, "ratio"].median()

        log.info(f"\n  [6C] LABEL SIGNAL PRESERVATION (PRIMARY METRIC):")
        log.info(f"    Sign preserved : {signs_preserved}/{total_sig} "
                 f"({signs_preserved/total_sig*100:.1f}%)")
        log.info(f"    Median ratio (synth_diff/real_diff) for top features: {mean_ratio:.3f}")
        log.info(f"    (Target: sign=100%, ratio close to 1.0)")
        log.info(f"\n  Top 15 discriminative features — signal check:")
        log.info(f"  {'Feature':35s} {'Real diff':>10} {'Synth diff':>10} {'Ratio':>7} {'Sign OK':>8}")
        log.info(f"  {'-'*75}")
        for _, row in signal_df.head(15).iterrows():
            ok = "YES" if row["sign_preserved"] else "NO <<<<<"
            log.info(f"  {row['feature']:35s} {row['real_diff']:10.3f} {row['synth_diff']:10.3f} "
                     f"{row['ratio']:7.3f} {ok:>8}")

        # 6D: Structural integrity
        struct = check_structural_integrity(synth_df, ORDERED_GROUPS, CARE_UNIT_COLS)
        if not struct:
            log.info("\n  [6D] Structural integrity: PASS (0 violations)")
        else:
            log.warning(f"\n  [6D] Structural violations: {struct}")

        log.info(f"\n  Label prevalence : {synth_df[TARGET_COL].mean():.4f} (real: {prevalence:.4f})")
        log.info(f"  Gender male rate : {synth_df.get('gender', pd.Series([0])).mean():.4f} "
                 f"(real: {train_df.get('gender', pd.Series([0])).mean():.4f})")

        # Plots
        plot_label_signal(signal_df, top_n=20, suffix="_v4")
        plot_distribution_comparison_by_class(
            train_df, synth_df,
            KEY_FEATURES_POS + KEY_FEATURES_NEG[:5],
            suffix="_v4"
        )
        plot_ks_bar(ks_df, col="ks_statistic", top_n=30,
                    title="Overall KS (v4)", suffix="_overall_v4")

        # Save all validation results
        log.info(f"\n  Validation files saved:")
        log.info(f"    {ks_path}")
        log.info(f"    {ks_class_path}")
        log.info(f"    {signal_path}")

    # =========================================================================
    # STEP 7: Metadata
    # =========================================================================
    total_time = (time.time() - t_start) / 60
    signal_df_meta = label_signal_check(train_df, synth_df, feature_cols, discrete_cols)
    meta = {
        "version"                   : "v4",
        "generated_at"              : datetime.now().isoformat(),
        "n_real_train"              : int(n_real),
        "n_real_pos"                : int(len(real_pos)),
        "n_real_neg"                : int(len(real_neg)),
        "n_synthetic"               : int(args.n_synthetic),
        "n_features"                : len(feature_cols),
        "real_prevalence"           : float(prevalence),
        "synth_prevalence"          : float(synth_df[TARGET_COL].mean()),
        "epochs_pos"                : args.epochs_pos,
        "epochs_neg"                : args.epochs_neg,
        "batch_size_pos"            : args.batch_size_pos,
        "batch_size_neg"            : args.batch_size_neg,
        "random_seed"               : RANDOM_SEED,
        "ks_joint_threshold"        : args.ks_joint_threshold,
        "ks_quantile_threshold"     : args.ks_quantile_threshold,
        "noise"                     : args.noise,
        "total_runtime_min"         : round(total_time, 2),
        "output_file"               : str(synth_path),
        "sign_preservation_pct"     : float(signal_df_meta["sign_preserved"].mean() * 100),
        "approach"                  : "class_split_ctgan_label_conditioned_postprocessing",
        "why_v3_failed"             : (
            "v3 post-processing was label-agnostic: joint_resample_groups sampled "
            "from all real rows regardless of label, quantile_map used overall marginals. "
            "Result: CTGAN generated X independent of Y, and post-processing never "
            "restored the association. lactate_max diff (real=+3.56, v3 synth=+0.008)."
        ),
        "why_v4_works"              : (
            "ctgan_pos learns P(X|Y=1), ctgan_neg learns P(X|Y=0). Every post-processing "
            "step (resampling, quantile mapping) operates within class. Label signal is "
            "present at generation time and preserved through post-processing."
        ),
        "post_processing_pipeline"  : [
            "A: label_conditioned_joint_group_resampling",
            "B: single_sort_pass (max>=mean>=min)",
            "C: label_conditioned_quantile_mapping",
            "D: light_swap_fix",
            "E: label_conditioned_standalone_resample",
            "care_unit_onehot (class-aware)",
            "label_prevalence_matching",
            "gender_prevalence_matching (per class)",
        ],
    }
    meta_path = SYNTHETIC_DIR / "generation_metadata_v4.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("SYNTHETIC DATA GENERATION v4 COMPLETE")
    log.info("=" * 70)
    log.info(f"  Output : {synth_path}")
    log.info(f"  Shape  : {synth_df.shape}")
    log.info(f"  Time   : {total_time:.1f} min")
    log.info(f"  Sign preservation: {meta['sign_preservation_pct']:.1f}%")
    log.info("=" * 70)
    log.info("")
    log.info("NEXT STEP: train CatBoost on B_synthetic_train_v4.csv")
    log.info("Expected AUPRC range: 0.60-0.80 (vs v3's 0.28)")
    log.info("If AUPRC < 0.50, check label_signal_v4.csv sign_preserved column")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
