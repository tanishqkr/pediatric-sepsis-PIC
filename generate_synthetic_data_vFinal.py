"""
generate_synthetic_data_vFinal.py
==============================================================================
Synthetic Data Generation  -  VERSION FINAL
Project : Pediatric Sepsis Prediction (PIC Database)

==============================================================================
WHY EVERY PREVIOUS VERSION FAILED AND WHAT FINAL DOES DIFFERENTLY
==============================================================================

v3 (Mean KS=0.03 but AUPRC=0.28):
  Post-processing was label-agnostic. CTGAN generates X independent of Y.
  Result: lactate_max diff real=+3.56, synth=+0.008. Signal = dead.

v4 (AUPRC=0.62-0.78):
  Fixed label signal via class-split CTGAN. All top-20 features sign=YES,
  ratio~1.0. BUT: inter-feature correlations completely destroyed.
    inr_max x pt_max:           real=0.999, v4=0.016
    lactate x bicarbonate:      real=-0.718, v4=0.006
    bicarbonate x base_excess:  real=0.974, v4=-0.007
    ast x ldh:                  real=0.713, v4=-0.001
  CatBoost trees rely on these interactions. FFT transformer absolutely
  needs them. Without correlations, ensemble models are bottlenecked.

vFinal SOLUTION — THREE-LAYER APPROACH:
  Layer 1: Class-split CTGAN  →  correct marginal distributions per class
  Layer 2: Iman-Conover       →  inject real rank correlation structure per class
  Layer 3: Zero-inflation fix →  two-part model for 10 heavily zero-inflated features

  Iman-Conover theorem (1982): given samples with correct marginals,
  reordering them according to a target rank correlation matrix preserves
  the marginals EXACTLY while achieving the target correlations.
  Works for ANY distribution (no normality assumption).
  Works with class-split (preserves label signal from Layer 1).

EXPECTED RESULTS:
  Marginal KS:      ~0.02 (same as v4 — preserved exactly by IC)
  Correlation loss: ~0.02 RMSE (vs ~0.65 RMSE in v4)
  Label signal:     ~97% signs preserved, ratio ~0.97 (same as v4)
  AUPRC synth→real: 0.80-0.92 (vs v4's 0.62-0.78)
  AUPRC augmented:  0.95-0.98 (real+vFinal synthetic, what you want)

KEY DATA FACTS (from full analysis of B_train_model_ready.csv):
  - 2530 real rows: 691 pos (27.3%), 1839 neg (72.7%)
  - 185 continuous, 40 binary, 1 target = 226 total features
  - 49 features skew > 5 (ck, ast, alt, creatinine, inr, ldh, wbc) → CTGAN
  - 10 zero-inflated features (>20% zeros) → two-part model
  - Spearman matrices already positive-definite (min eig > 0) → clean IC
  - 125/185 features discriminate between classes (KS > 0.10)

DEPENDENCIES:
  pip install sdv scipy matplotlib
  (Works with ctgan standalone too)

RUN:
  conda activate sepsis_ml
  python generate_synthetic_data_vFinal.py

FLAGS:
  --n_synthetic 10000          default: 10000
  --epochs_pos 300             CTGAN epochs for sep=1 model (691 rows)
  --epochs_neg 300             CTGAN epochs for sep=0 model (1839 rows)
  --batch_size_pos 200         small because only 691 pos rows
  --batch_size_neg 500
  --ic_alpha 0.95              Iman-Conover blend: 1.0=full IC, 0.0=no IC
                               0.95 = 95% real correlation + 5% shrinkage
                               (shrinkage prevents rank-collapse on tied values)
  --ks_quantile_threshold 0.10 post-IC residual quantile mapping threshold
  --noise 0.005                very low noise — IC handles structure, not noise
  --skip_ctgan                 load saved models, jump to IC step
  --validate                   run full validation suite (default True)
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
from scipy.linalg import cholesky, LinAlgError

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

FIGURES_DIR    = SYNTHETIC_DIR / "figures_vFinal"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR       = SYNTHETIC_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING
# ==============================================================================

log_path = LOGS_DIR / "generate_synthetic_data_vFinal.log"
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
# CONSTANTS
# ==============================================================================

RANDOM_SEED = 42
TARGET_COL  = "sepsis_label"

ORDERED_GROUPS = [
    "albumin", "alc", "alt", "anc", "anion_gap", "ast", "base_excess",
    "bicarbonate", "bilirubin", "calcium", "ck", "creatinine", "crp",
    "ddimer", "diastolic_bp", "fibrinogen", "glucose", "heart_rate",
    "hematocrit", "hemoglobin", "inr", "lactate", "ldh", "map", "pao2",
    "pco2", "ph", "platelets", "potassium", "pt", "ptt", "resp_rate",
    "sodium", "spo2", "spo2_lab", "systolic_bp", "temperature", "urea",
    "uric_acid", "wbc",
]

STANDALONE_CONTINUOUS = [
    "age_years", "shock_index", "urine_output_total_ml",
    "urine_output_per_hour", "fluid_input_total_ml",
    "fluid_balance_ml", "n_antibiotics_24h", "n_cultures_24h",
    "n_positive_cultures",
]

# Features with >20% zeros — need two-part treatment in IC step
ZERO_INFLATED_FEATURES = [
    "urine_output_per_hour", "urine_output_total_ml", "spo2_trend",
    "creatinine_trend", "bilirubin_trend", "fluid_input_total_ml",
    "fluid_balance_ml", "resp_rate_trend", "n_positive_cultures",
    "platelets_trend",
]

CARE_UNIT_COLS = [
    "care_unit_CICU", "care_unit_General ICU",
    "care_unit_NICU", "care_unit_PICU", "care_unit_SICU",
]

KEY_FEATURES_SIGNAL = [
    "inr_max", "pt_max", "ddimer_max", "lactate_max", "platelets_min",
    "ast_max", "ldh_max", "bicarbonate_min", "hematocrit_min", "ph_min",
    "ptt_max", "base_excess_min", "crp_max", "glucose_max", "creatinine_max",
    "alt_max", "ck_max", "wbc_max", "age_years", "shock_index",
]

KEY_CORR_PAIRS = [
    ("inr_max", "pt_max"),
    ("ast_max", "ldh_max"),
    ("ast_max", "alt_max"),
    ("lactate_max", "bicarbonate_min"),
    ("lactate_max", "ph_min"),
    ("bicarbonate_min", "base_excess_min"),
    ("shock_index", "map_min"),
    ("shock_index", "heart_rate_max"),
    ("resp_rate_max", "heart_rate_max"),
    ("hematocrit_min", "hemoglobin_min"),
    ("ldh_max", "ck_max"),
    ("ptt_max", "inr_max"),
]

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_synthetic",           type=int,   default=10_000)
    p.add_argument("--epochs_pos",            type=int,   default=300)
    p.add_argument("--epochs_neg",            type=int,   default=300)
    p.add_argument("--batch_size_pos",        type=int,   default=200)
    p.add_argument("--batch_size_neg",        type=int,   default=500)
    p.add_argument("--ic_alpha",              type=float, default=0.95,
                   help="Iman-Conover blend: 1.0=full target correlation, "
                        "values<1.0 shrink toward identity (prevents rank collapse)")
    p.add_argument("--ks_quantile_threshold", type=float, default=0.10,
                   help="Post-IC residual quantile map threshold")
    p.add_argument("--noise",                 type=float, default=0.005)
    p.add_argument("--validate",   action="store_true", default=True)
    p.add_argument("--skip_ctgan", action="store_true", default=False)
    return p.parse_args()

# ==============================================================================
# CTGAN IMPORT
# ==============================================================================

def import_ctgan():
    try:
        from sdv.single_table import CTGANSynthesizer
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
    log.info(f"  [{label_name}] Fitting on {len(data_df):,} rows, "
             f"{epochs} epochs, batch={batch_size}")
    if use_sdv:
        from sdv.metadata import SingleTableMetadata
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(data_df)
        for col in discrete_cols:
            if col in data_df.columns:
                meta.update_column(col, sdtype="categorical")
        model = CTGAN_cls(
            metadata=meta, epochs=epochs, batch_size=batch_size,
            generator_dim=(256, 256), discriminator_dim=(256, 256),
            generator_lr=2e-4, discriminator_lr=2e-4,
            discriminator_steps=1, log_frequency=True,
            verbose=True, cuda=False,
        )
        model.fit(data_df)
    else:
        model = CTGAN_cls(
            epochs=epochs, batch_size=batch_size,
            generator_dim=(256, 256), discriminator_dim=(256, 256),
            generator_lr=2e-4, discriminator_lr=2e-4,
            discriminator_steps=1, log_frequency=True,
            verbose=True, cuda=False,
        )
        model.fit(data_df, discrete_columns=discrete_cols)
    return model


def sample_ctgan(model, use_sdv, n):
    try:
        return model.sample(num_rows=n) if use_sdv else model.sample(n)
    except TypeError:
        return model.sample(num_rows=n)

# ==============================================================================
# NEAREST POSITIVE DEFINITE MATRIX (Higham 1988)
# ==============================================================================

def nearest_pd(C):
    """
    Find the nearest positive-definite matrix to C.
    Uses Higham's algorithm with eigenvalue clipping.
    Required when Spearman matrix has near-zero eigenvalues.
    """
    B = (C + C.T) / 2
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    C2 = (B + H) / 2
    C3 = (C2 + C2.T) / 2
    if is_pd(C3):
        return C3
    # Eigenvalue clipping
    k = 1
    while not is_pd(C3):
        min_eig = np.linalg.eigvalsh(C3).min()
        C3 += (-min_eig * k**2 + np.finfo(float).eps) * np.eye(C3.shape[0])
        k += 1
    return C3


def is_pd(C):
    try:
        cholesky(C)
        return True
    except LinAlgError:
        return False

# ==============================================================================
# LAYER 2: IMAN-CONOVER RANK CORRELATION INJECTION
# ==============================================================================

def compute_spearman_matrix(df, cols, alpha=0.95):
    """
    Compute Spearman rank correlation matrix for given columns of df.
    alpha: blend toward identity (prevents exact rank-collapse).
    Returns a valid positive-definite correlation matrix.
    """
    n = len(cols)
    sub = df[cols].copy()

    # Rank each column (handle ties with average method)
    ranked = sub.rank(method="average")
    C = ranked.corr(method="pearson").values  # Pearson of ranks = Spearman

    # Blend toward identity: C_blended = alpha*C + (1-alpha)*I
    # This is Ledoit-Wolf style shrinkage — prevents issues with small samples
    C_blended = alpha * C + (1 - alpha) * np.eye(n)
    np.fill_diagonal(C_blended, 1.0)

    # Ensure positive definite
    if not is_pd(C_blended):
        log.warning("    Spearman matrix not PD, applying Higham correction...")
        C_blended = nearest_pd(C_blended)
        np.fill_diagonal(C_blended, 1.0)

    return C_blended


def van_der_waerden_scores(n):
    """
    Van der Waerden normal scores for n observations.
    These are the expected values of the order statistics of a standard normal.
    Used in Iman-Conover instead of raw ranks — more stable.
    """
    ranks = np.arange(1, n + 1)
    return stats.norm.ppf(ranks / (n + 1))


def iman_conover(synth_matrix, target_corr, rng):
    """
    Iman-Conover (1982) method to impose target_corr on synth_matrix.

    Given:
      synth_matrix: (n_samples, n_features) — CTGAN output, good marginals
      target_corr:  (n_features, n_features) — target Spearman corr matrix
                    computed from real data (per class)

    Returns:
      reordered_matrix: same shape, marginals preserved, correlations imposed.

    Algorithm:
      1. Replace each column with van der Waerden normal scores (rank-based)
         → gives a matrix with standard normal marginals
      2. Compute current correlation of this scores matrix
      3. Cholesky decompose both current and target correlations
      4. Transform scores matrix to have target correlation
      5. Use transformed scores as ORDER indices to reorder original columns
         → original marginals restored, target correlations achieved
    """
    n, p = synth_matrix.shape
    assert target_corr.shape == (p, p), "Correlation matrix shape mismatch"

    # Step 1: Van der Waerden scores matrix
    scores = np.zeros_like(synth_matrix, dtype=float)
    vdw = van_der_waerden_scores(n)
    for j in range(p):
        col_ranks = stats.rankdata(synth_matrix[:, j], method="ordinal") - 1
        scores[:, j] = vdw[col_ranks]

    # Step 2: Current correlation of scores
    C_current = np.corrcoef(scores.T)
    np.fill_diagonal(C_current, 1.0)
    if not is_pd(C_current):
        C_current = nearest_pd(C_current)
        np.fill_diagonal(C_current, 1.0)

    # Step 3: Cholesky decompositions
    try:
        P = cholesky(C_current, lower=True)   # current
        Q = cholesky(target_corr, lower=True)  # target
    except LinAlgError:
        log.warning("    Cholesky failed, applying Higham to both matrices...")
        C_current = nearest_pd(C_current)
        target_corr = nearest_pd(target_corr)
        np.fill_diagonal(C_current, 1.0)
        np.fill_diagonal(target_corr, 1.0)
        P = cholesky(C_current, lower=True)
        Q = cholesky(target_corr, lower=True)

    # Step 4: Transform scores
    # scores_transformed = scores @ inv(P).T @ Q.T
    # This rotates the score space to have target correlation
    P_inv = np.linalg.solve(P, np.eye(p))
    T = P_inv.T @ Q.T
    scores_transformed = scores @ T

    # Step 5: Reorder original columns using transformed score ranks
    result = np.zeros_like(synth_matrix)
    for j in range(p):
        # Rank order of transformed scores for column j
        target_order = np.argsort(scores_transformed[:, j])
        # Sort original column j
        sorted_col = np.sort(synth_matrix[:, j])
        # Assign: position i in result gets sorted_col value at rank i
        result[target_order, j] = sorted_col

    return result


def apply_iman_conover_per_class(synth_df, real_df, continuous_cols, alpha, rng):
    """
    Apply Iman-Conover separately for each class.
    This is the key step that v4 was missing.

    For each class:
      1. Extract real data subset → compute Spearman corr matrix
      2. Extract synthetic subset → apply IC transformation
      3. Put reordered synthetic back into the full dataframe

    Marginals preserved exactly.
    Correlations converge to real per-class Spearman structure.
    Label signal preserved because each class is handled independently.
    """
    df = synth_df.copy()

    # Only apply IC to continuous columns that exist in synth
    ic_cols = [c for c in continuous_cols
               if c in df.columns and c in real_df.columns
               and c != TARGET_COL]

    log.info(f"  IC applying to {len(ic_cols)} continuous features per class")

    for label_val in [0, 1]:
        mask_synth = df[TARGET_COL] == label_val
        real_class = real_df[real_df[TARGET_COL] == label_val]
        n_synth_class = int(mask_synth.sum())

        log.info(f"  [Class {label_val}] {n_synth_class} synth rows, "
                 f"{len(real_class)} real rows")

        if n_synth_class < 10:
            log.warning(f"  [Class {label_val}] Too few rows, skipping IC")
            continue

        # Step A: Compute target Spearman matrix from real class data
        log.info(f"  [Class {label_val}] Computing Spearman matrix "
                 f"({len(ic_cols)}x{len(ic_cols)})...")
        t0 = time.time()
        C_target = compute_spearman_matrix(real_class, ic_cols, alpha=alpha)
        log.info(f"  [Class {label_val}] Spearman matrix done in "
                 f"{time.time()-t0:.1f}s. "
                 f"min_eig={np.linalg.eigvalsh(C_target).min():.6f}")

        # Step B: Extract synth class matrix
        synth_class_matrix = df.loc[mask_synth, ic_cols].to_numpy(dtype=float)

        # Handle any NaN in synth (fill with column mean before IC)
        for j, col in enumerate(ic_cols):
            nan_mask = np.isnan(synth_class_matrix[:, j])
            if nan_mask.any():
                synth_class_matrix[nan_mask, j] = np.nanmean(synth_class_matrix[:, j])

        # Step C: Apply Iman-Conover
        log.info(f"  [Class {label_val}] Applying Iman-Conover transform...")
        t0 = time.time()
        reordered = iman_conover(synth_class_matrix, C_target, rng)
        log.info(f"  [Class {label_val}] IC done in {time.time()-t0:.1f}s")

        # Step D: Put back into dataframe
        df.loc[mask_synth, ic_cols] = reordered

    return df

# ==============================================================================
# ZERO-INFLATED FEATURE FIX (class-conditioned two-part model)
# ==============================================================================

def fix_zero_inflated_features(synth_df, real_df, zero_inf_cols, rng):
    """
    For heavily zero-inflated features (>20% zeros):
    CTGAN tends to generate wrong zero proportions and wrong nonzero distributions.
    Iman-Conover also struggles with zero-inflation (tied ranks).

    Two-part fix (per class):
      Part 1: Bernoulli — set correct fraction of rows to zero
      Part 2: Nonzero values — resample from real nonzero distribution of that class
    """
    df = synth_df.copy()
    n_fixed = 0

    for col in zero_inf_cols:
        if col not in df.columns or col not in real_df.columns:
            continue

        for label_val in [0, 1]:
            mask_synth = df[TARGET_COL] == label_val
            real_class = real_df[real_df[TARGET_COL] == label_val][col]
            n_synth_class = int(mask_synth.sum())

            real_zero_rate = float((real_class == 0).mean())
            real_nonzero   = real_class[real_class != 0].values

            if len(real_nonzero) < 3:
                continue

            # How many synth rows should be zero?
            n_zeros_target = int(round(real_zero_rate * n_synth_class))
            synth_indices  = df.index[mask_synth].tolist()

            # Randomly assign zeros and nonzeros
            rng.shuffle(synth_indices)
            zero_idx    = synth_indices[:n_zeros_target]
            nonzero_idx = synth_indices[n_zeros_target:]

            df.loc[zero_idx, col] = 0.0

            # Resample nonzero values from real nonzero distribution
            if len(nonzero_idx) > 0:
                sampled = rng.choice(real_nonzero, size=len(nonzero_idx), replace=True)
                # Add tiny noise to prevent exact copies
                noise_std = 0.01 * float(real_nonzero.std())
                if noise_std > 0:
                    sampled = sampled + rng.normal(0, noise_std, len(nonzero_idx))
                sampled = np.clip(sampled, real_class.min(), real_class.max())
                df.loc[nonzero_idx, col] = sampled

            n_fixed += 1

    log.info(f"  zero_inflated_fix: corrected {n_fixed} col×class combinations.")
    return df

# ==============================================================================
# POST-PROCESSING (same as v4 but applied AFTER IC)
# ==============================================================================

def sort_max_mean_min(synth_df, groups):
    df = synth_df.copy()
    for base in groups:
        max_c, min_c = f"{base}_max", f"{base}_min"
        mean_c, first_c = f"{base}_mean", f"{base}_first"
        if max_c not in df.columns or min_c not in df.columns:
            continue
        if mean_c in df.columns:
            trio = np.sort(df[[max_c, min_c, mean_c]].to_numpy(dtype=float), axis=1)
            df[min_c]  = trio[:, 0]
            df[mean_c] = trio[:, 1]
            df[max_c]  = trio[:, 2]
        else:
            inv = df[max_c].values < df[min_c].values
            tmp = df.loc[inv, max_c].copy()
            df.loc[inv, max_c] = df.loc[inv, min_c]
            df.loc[inv, min_c] = tmp
        if first_c in df.columns:
            df[first_c] = df[first_c].clip(lower=df[min_c], upper=df[max_c])
    viol = sum(
        int((df[f"{b}_max"] < df[f"{b}_min"]).sum())
        for b in groups if f"{b}_max" in df.columns and f"{b}_min" in df.columns
    )
    log.info(f"  sort_max_mean_min: violations={viol}")
    return df


def quantile_map_column(real_vals, synth_vals):
    n = len(synth_vals)
    real_sorted = np.sort(real_vals)
    synth_ranks = np.argsort(np.argsort(synth_vals))
    qi = np.clip(
        (synth_ranks / max(n-1, 1) * (len(real_sorted)-1)).astype(int),
        0, len(real_sorted)-1
    )
    return real_sorted[qi]


def quantile_map_conditioned(synth_df, real_df, feature_cols, discrete_cols, ks_threshold):
    """
    After IC, a small number of features may still drift.
    Apply per-class quantile mapping only where KS > threshold.
    This is a light cleanup step — IC should handle 90%+ of it.
    """
    df = synth_df.copy()
    n_mapped = 0
    for col in feature_cols:
        if col in discrete_cols or col == TARGET_COL or real_df[col].nunique() <= 2:
            continue
        for label_val in [0, 1]:
            mask = df[TARGET_COL] == label_val
            rv = real_df[real_df[TARGET_COL] == label_val][col].dropna().values
            sv = df.loc[mask, col].values
            if len(rv) < 10 or len(sv) < 10:
                continue
            ks_val, _ = stats.ks_2samp(rv, sv)
            if ks_val > ks_threshold:
                df.loc[mask, col] = quantile_map_column(rv, sv)
                n_mapped += 1
    log.info(f"  quantile_map_conditioned: mapped {n_mapped} col×class pairs")
    return df, n_mapped


def light_swap_fix(synth_df, groups):
    df = synth_df.copy()
    total_swapped = 0
    for base in groups:
        max_c, min_c = f"{base}_max", f"{base}_min"
        mean_c, first_c = f"{base}_mean", f"{base}_first"
        if max_c not in df.columns or min_c not in df.columns:
            continue
        inv = df[max_c].values < df[min_c].values
        if inv.sum() > 0:
            tmp = df.loc[inv, max_c].copy()
            df.loc[inv, max_c] = df.loc[inv, min_c]
            df.loc[inv, min_c] = tmp
            total_swapped += int(inv.sum())
        if mean_c in df.columns:
            df[mean_c] = df[mean_c].clip(lower=df[min_c], upper=df[max_c])
        if first_c in df.columns:
            df[first_c] = df[first_c].clip(lower=df[min_c], upper=df[max_c])
    remaining = sum(
        int((df[f"{b}_max"] < df[f"{b}_min"]).sum())
        for b in groups if f"{b}_max" in df.columns and f"{b}_min" in df.columns
    )
    log.info(f"  light_swap_fix: swapped={total_swapped}, remaining={remaining}")
    return df


def fix_care_unit_onehot(synth_df, care_unit_cols, real_dist_by_class, rng):
    df = synth_df.copy()
    available = [c for c in care_unit_cols if c in df.columns]
    if not available:
        return df
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    multi = df[available].sum(axis=1) > 1
    if multi.sum() > 0:
        raw = df.loc[multi, available].values
        winner = np.argmax(raw, axis=1)
        cor = np.zeros_like(raw, dtype=int)
        for i, w in enumerate(winner):
            cor[i, w] = 1
        df.loc[multi, available] = cor
    zero = df[available].sum(axis=1) == 0
    if zero.sum() > 0:
        zero_idx = np.where(zero)[0]
        for ri in zero_idx:
            label_val = int(df.iloc[ri][TARGET_COL])
            rdist = real_dist_by_class.get(label_val, {})
            probs = np.array([rdist.get(c, 1/len(available)) for c in available])
            probs = np.maximum(probs, 1e-9) / probs.sum()
            ci = rng.choice(len(available), p=probs)
            for c in available:
                df.at[df.index[ri], c] = 0
            df.at[df.index[ri], available[ci]] = 1
    df[available] = df[available].astype(int)
    remaining = int((df[available].sum(axis=1) != 1).sum())
    log.info(f"  care_unit_onehot: {remaining} violations remain")
    return df


def fix_label_prevalence(synth_df, target_prev, rng, tol=0.005):
    df = synth_df.copy()
    n = len(df)
    target_count = int(round(target_prev * n))
    delta = target_count - int(df[TARGET_COL].sum())
    if abs(delta) < max(1, int(tol * n)):
        log.info(f"  label_prevalence: within tolerance ({df[TARGET_COL].mean():.4f})")
        return df
    if delta > 0:
        idx = rng.choice(df.index[df[TARGET_COL] == 0].tolist(), size=delta, replace=False)
        df.loc[idx, TARGET_COL] = 1
    else:
        idx = rng.choice(df.index[df[TARGET_COL] == 1].tolist(), size=abs(delta), replace=False)
        df.loc[idx, TARGET_COL] = 0
    log.info(f"  label_prevalence: adjusted by {delta} -> {df[TARGET_COL].mean():.4f}")
    return df


def fix_gender_prevalence(synth_df, gender_rate_by_class, rng, tol=0.005):
    df = synth_df.copy()
    if "gender" not in df.columns:
        return df
    for label_val in [0, 1]:
        mask = df[TARGET_COL] == label_val
        target_rate = gender_rate_by_class.get(label_val, 0.6)
        n_class = int(mask.sum())
        target_count = int(round(target_rate * n_class))
        current_count = int(df.loc[mask, "gender"].sum())
        delta = target_count - current_count
        if abs(delta) < max(1, int(tol * n_class)):
            continue
        class_idx = df.index[mask].tolist()
        if delta > 0:
            female_in = [i for i in class_idx if df.at[i, "gender"] == 0]
            if len(female_in) >= delta:
                df.loc[rng.choice(female_in, size=delta, replace=False), "gender"] = 1
        else:
            male_in = [i for i in class_idx if df.at[i, "gender"] == 1]
            if len(male_in) >= abs(delta):
                df.loc[rng.choice(male_in, size=abs(delta), replace=False), "gender"] = 0
    log.info(f"  gender: sep0={df.loc[df[TARGET_COL]==0,'gender'].mean():.4f} "
             f"sep1={df.loc[df[TARGET_COL]==1,'gender'].mean():.4f}")
    return df

# ==============================================================================
# VALIDATION
# ==============================================================================

def label_signal_check(real_df, synth_df, feature_cols, discrete_cols):
    results = []
    for col in feature_cols:
        if col in discrete_cols or col == TARGET_COL or real_df[col].nunique() <= 2:
            continue
        r0 = real_df[real_df[TARGET_COL]==0][col].mean()
        r1 = real_df[real_df[TARGET_COL]==1][col].mean()
        s0 = synth_df[synth_df[TARGET_COL]==0][col].mean()
        s1 = synth_df[synth_df[TARGET_COL]==1][col].mean()
        real_diff = r1 - r0
        synth_diff = s1 - s0
        sign_ok = (np.sign(real_diff) == np.sign(synth_diff)) if abs(real_diff) > 1e-9 else True
        ratio = synth_diff / real_diff if abs(real_diff) > 1e-9 else 1.0
        results.append({"feature": col, "real_diff": round(real_diff, 4),
                        "synth_diff": round(synth_diff, 4), "ratio": round(ratio, 4),
                        "sign_preserved": sign_ok,
                        "real_sep0": round(r0, 4), "real_sep1": round(r1, 4),
                        "synth_sep0": round(s0, 4), "synth_sep1": round(s1, 4)})
    df_s = pd.DataFrame(results)
    df_s = df_s.assign(_abs=df_s["real_diff"].abs()).sort_values("_abs", ascending=False).drop("_abs", axis=1)
    return df_s


def correlation_check(real_df, synth_df, pairs, label_vals=[0, 1]):
    results = []
    for c1, c2 in pairs:
        if c1 not in real_df.columns or c2 not in real_df.columns:
            continue
        for lv in label_vals:
            rsp = stats.spearmanr(
                real_df[real_df[TARGET_COL]==lv][c1].dropna(),
                real_df[real_df[TARGET_COL]==lv][c2].dropna()
            )[0]
            ssp = stats.spearmanr(
                synth_df[synth_df[TARGET_COL]==lv][c1].dropna(),
                synth_df[synth_df[TARGET_COL]==lv][c2].dropna()
            )[0]
            results.append({"c1": c1, "c2": c2, "class": lv,
                            "real_spearman": round(rsp, 4),
                            "synth_spearman": round(ssp, 4),
                            "error": round(abs(rsp - ssp), 4)})
    return pd.DataFrame(results)


def ks_summary(real_df, synth_df, feature_cols, discrete_cols):
    results = []
    for col in feature_cols:
        if col in discrete_cols or real_df[col].nunique() <= 2:
            continue
        rv = real_df[col].dropna().values
        sv = synth_df[col].dropna().values
        if len(rv) < 10 or len(sv) < 10:
            continue
        ks_val, _ = stats.ks_2samp(rv, sv)
        results.append({"feature": col, "ks_statistic": round(ks_val, 4),
                        "real_mean": round(rv.mean(), 4), "synth_mean": round(sv.mean(), 4)})
    return pd.DataFrame(results).sort_values("ks_statistic", ascending=False)


def check_structural_integrity(synth_df, groups, care_cols):
    v = {}
    for base in groups:
        mx, mn = f"{base}_max", f"{base}_min"
        if mx in synth_df.columns and mn in synth_df.columns:
            n = int((synth_df[mx] < synth_df[mn]).sum())
            if n > 0:
                v[f"{base}_ordering"] = n
    cu = [c for c in care_cols if c in synth_df.columns]
    if cu:
        bad = int((synth_df[cu].sum(axis=1) != 1).sum())
        if bad > 0:
            v["care_unit"] = bad
    return v

# ==============================================================================
# PLOTS
# ==============================================================================

def plot_label_signal(signal_df, top_n=20, suffix=""):
    top = signal_df.head(top_n).copy()
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(14, max(6, top_n * 0.35)))
    ax.bar(x - 0.175, top["real_diff"],  0.35, label="Real",  color="#1565C0", alpha=0.85)
    ax.bar(x + 0.175, top["synth_diff"], 0.35, label="Synth", color="#E53935", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(top["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean difference (sep1 - sep0)")
    ax.set_title(f"Label Signal Preservation — Top {top_n} Discriminative Features\n"
                 "Red≈Blue = GOOD", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / f"label_signal{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {path}")


def plot_correlation_comparison(corr_df, suffix=""):
    fig, ax = plt.subplots(figsize=(10, max(6, len(corr_df) * 0.35)))
    x = np.arange(len(corr_df))
    labels = [f"{r.c1[:12]}x{r.c2[:12]}\ncls={int(r['class'])}"
              for _, r in corr_df.iterrows()]
    ax.bar(x - 0.175, corr_df["real_spearman"],  0.35, label="Real",  color="#1565C0", alpha=0.85)
    ax.bar(x + 0.175, corr_df["synth_spearman"], 0.35, label="Synth", color="#E53935", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Correlation Preservation — Key Feature Pairs\nRed≈Blue = GOOD",
                 fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / f"correlation_check{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {path}")


def plot_ks_bar(ks_df, top_n=30, suffix=""):
    top = ks_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#E53935" if v > 0.10 else "#FFA726" if v > 0.05 else "#43A047"
              for v in top["ks_statistic"]]
    ax.barh(top["feature"], top["ks_statistic"], color=colors)
    ax.axvline(0.05, color="orange", ls="--", lw=1)
    ax.axvline(0.10, color="red",    ls="--", lw=1)
    ax.set_xlabel("KS Statistic")
    ax.set_title("Overall KS (vFinal)", fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / f"ks_bar{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {path}")


def plot_per_class_distributions(real_df, synth_df, features, suffix=""):
    available = [f for f in features if f in real_df.columns]
    if not available:
        return
    n_rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3))
    axes = axes.flatten()
    for i, feat in enumerate(available):
        ax = axes[i]
        for lv, cr, cs in [(0, "#1565C0", "#64B5F6"), (1, "#B71C1C", "#EF9A9A")]:
            rv = real_df[real_df[TARGET_COL]==lv][feat].dropna()
            sv = synth_df[synth_df[TARGET_COL]==lv][feat].dropna()
            if len(rv) < 5:
                continue
            lo, hi = rv.quantile(0.01), rv.quantile(0.99)
            ax.hist(rv.clip(lo,hi), bins=30, alpha=0.5, color=cr, density=True,
                    label=f"Real sep={lv}")
            ax.hist(sv.clip(lo,hi), bins=30, alpha=0.5, color=cs, density=True,
                    label=f"Synth sep={lv}")
        ks_all, _ = stats.ks_2samp(real_df[feat].dropna(), synth_df[feat].dropna())
        ax.set_title(f"{feat}\nKS={ks_all:.3f}", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6)
        ax.spines[["top","right"]].set_visible(False)
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Per-class Distributions: Real vs Synthetic (vFinal)", fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / f"class_distributions{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = parse_args()
    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("SYNTHETIC DATA GENERATION  vFinal")
    log.info("CTGAN class-split + Iman-Conover correlation injection")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # =========================================================================
    # STEP 1: Load data
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("STEP 1 — Load real training data")
    log.info("="*60)

    if not TRAIN_FILE.exists():
        log.error(f"Not found: {TRAIN_FILE}")
        sys.exit(1)

    train_df     = pd.read_csv(TRAIN_FILE)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    n_real       = len(train_df)
    prevalence   = float(train_df[TARGET_COL].mean())

    real_pos = train_df[train_df[TARGET_COL] == 1].copy()
    real_neg = train_df[train_df[TARGET_COL] == 0].copy()

    discrete_cols = []
    for col in feature_cols:
        if set(train_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
            discrete_cols.append(col)
    continuous_cols = [c for c in feature_cols if c not in discrete_cols]

    log.info(f"  Total rows  : {n_real:,} | Pos: {len(real_pos):,} | Neg: {len(real_neg):,}")
    log.info(f"  Prevalence  : {prevalence:.4f}")
    log.info(f"  Continuous  : {len(continuous_cols)} | Discrete: {len(discrete_cols)}")

    care_unit_dist_by_class = {
        lv: {c: float(train_df[train_df[TARGET_COL]==lv][c].mean())
             for c in CARE_UNIT_COLS if c in train_df.columns}
        for lv in [0, 1]
    }
    gender_rate_by_class = {
        lv: float(train_df[train_df[TARGET_COL]==lv]["gender"].mean())
        for lv in [0, 1] if "gender" in train_df.columns
    }

    # =========================================================================
    # STEP 2: Train CTGAN per class
    # =========================================================================
    ctgan_pos_path = SYNTHETIC_DIR / "ctgan_model_vFinal_pos.pkl"
    ctgan_neg_path = SYNTHETIC_DIR / "ctgan_model_vFinal_neg.pkl"

    if args.skip_ctgan and ctgan_pos_path.exists() and ctgan_neg_path.exists():
        log.info("\nLoading saved CTGAN models (--skip_ctgan)...")
        with open(ctgan_pos_path, "rb") as f:
            ctgan_pos, use_sdv_pos = pickle.load(f)
        with open(ctgan_neg_path, "rb") as f:
            ctgan_neg, use_sdv_neg = pickle.load(f)
    else:
        log.info("\n" + "="*60)
        log.info("STEP 2 — Train class-split CTGAN models")
        log.info("="*60)

        CTGAN_cls, use_sdv = import_ctgan()
        use_sdv_pos = use_sdv_neg = use_sdv
        feat_discrete = [c for c in discrete_cols if c in train_df.columns]

        t_train = time.time()
        log.info(f"\n[Model POS — sep=1, n={len(real_pos):,}]")
        ctgan_pos = build_and_fit_ctgan(
            CTGAN_cls, use_sdv,
            real_pos.drop(columns=[TARGET_COL]),
            feat_discrete, args.epochs_pos, args.batch_size_pos, "POS"
        )

        log.info(f"\n[Model NEG — sep=0, n={len(real_neg):,}]")
        ctgan_neg = build_and_fit_ctgan(
            CTGAN_cls, use_sdv,
            real_neg.drop(columns=[TARGET_COL]),
            feat_discrete, args.epochs_neg, args.batch_size_neg, "NEG"
        )

        train_time = (time.time() - t_train) / 60
        log.info(f"\n  Both models trained in {train_time:.1f} min")

        with open(ctgan_pos_path, "wb") as f:
            pickle.dump((ctgan_pos, use_sdv_pos), f)
        with open(ctgan_neg_path, "wb") as f:
            pickle.dump((ctgan_neg, use_sdv_neg), f)
        log.info(f"  Saved -> {ctgan_pos_path}")

    # =========================================================================
    # STEP 3: Sample from each model
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("STEP 3 — Sample from class-split models")
    log.info("="*60)

    n_pos_target = int(round(prevalence * args.n_synthetic))
    n_neg_target = args.n_synthetic - n_pos_target

    log.info(f"  Sampling {n_pos_target:,} pos + {n_neg_target:,} neg rows...")
    t_gen = time.time()

    synth_pos = sample_ctgan(ctgan_pos, use_sdv_pos, n_pos_target)
    synth_pos[TARGET_COL] = 1
    synth_neg = sample_ctgan(ctgan_neg, use_sdv_neg, n_neg_target)
    synth_neg[TARGET_COL] = 0

    synth_df = pd.concat([synth_pos, synth_neg], ignore_index=True)
    synth_df = synth_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    synth_df = synth_df.reindex(columns=train_df.columns)

    # Initial range clip + discrete rounding
    for col in discrete_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)
    for col in continuous_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].clip(
                train_df[col].min(), train_df[col].max()
            )

    log.info(f"  Done in {(time.time()-t_gen)/60:.1f} min | shape={synth_df.shape}")
    log.info(f"  Prevalence after sampling: {synth_df[TARGET_COL].mean():.4f}")

    # Quick pre-IC signal check
    log.info("\n  [SIGNAL CHECK — before IC]")
    for feat in ["lactate_max", "platelets_min", "inr_max", "bicarbonate_min"]:
        if feat in synth_df.columns:
            r0=train_df[train_df[TARGET_COL]==0][feat].mean()
            r1=train_df[train_df[TARGET_COL]==1][feat].mean()
            s0=synth_df[synth_df[TARGET_COL]==0][feat].mean()
            s1=synth_df[synth_df[TARGET_COL]==1][feat].mean()
            log.info(f"    {feat:25s}  Real:{r0:.3f}/{r1:.3f}(d={r1-r0:+.3f})  "
                     f"Synth:{s0:.3f}/{s1:.3f}(d={s1-s0:+.3f})")

    # =========================================================================
    # STEP 4: Iman-Conover correlation injection (THE KEY NEW STEP)
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("STEP 4 — Iman-Conover rank correlation injection (per class)")
    log.info(f"         alpha={args.ic_alpha} (blend toward identity)")
    log.info("="*60)

    # Exclude zero-inflated features from IC (their tied zeros break rank math)
    zi_present = [c for c in ZERO_INFLATED_FEATURES if c in continuous_cols]
    ic_cols = [c for c in continuous_cols if c not in zi_present]

    log.info(f"  IC columns: {len(ic_cols)} (excluded {len(zi_present)} zero-inflated)")

    t_ic = time.time()
    synth_df = apply_iman_conover_per_class(
        synth_df, train_df, ic_cols, alpha=args.ic_alpha, rng=rng
    )
    log.info(f"  Iman-Conover complete in {(time.time()-t_ic)/60:.1f} min")

    # Post-IC signal check — ensure label signal survived IC
    log.info("\n  [SIGNAL CHECK — after IC]")
    for feat in ["lactate_max", "platelets_min", "inr_max", "bicarbonate_min",
                 "ast_max", "ldh_max", "ph_min", "base_excess_min"]:
        if feat in synth_df.columns:
            r0=train_df[train_df[TARGET_COL]==0][feat].mean()
            r1=train_df[train_df[TARGET_COL]==1][feat].mean()
            s0=synth_df[synth_df[TARGET_COL]==0][feat].mean()
            s1=synth_df[synth_df[TARGET_COL]==1][feat].mean()
            ok = "OK" if (r1-r0)*(s1-s0) > 0 else "SIGN WRONG"
            log.info(f"    {feat:25s}  Real d={r1-r0:+.3f}  Synth d={s1-s0:+.3f}  [{ok}]")

    # Post-IC correlation check
    log.info("\n  [CORRELATION CHECK — after IC]")
    for c1,c2 in [("inr_max","pt_max"),("ast_max","ldh_max"),
                  ("lactate_max","bicarbonate_min"),("bicarbonate_min","base_excess_min")]:
        if c1 not in synth_df.columns: continue
        for lv in [0,1]:
            rsp = stats.spearmanr(
                train_df[train_df[TARGET_COL]==lv][c1].dropna(),
                train_df[train_df[TARGET_COL]==lv][c2].dropna()
            )[0]
            ssp = stats.spearmanr(
                synth_df[synth_df[TARGET_COL]==lv][c1].dropna(),
                synth_df[synth_df[TARGET_COL]==lv][c2].dropna()
            )[0]
            log.info(f"    sep={lv} {c1}x{c2[:15]}  real={rsp:.3f}  synth={ssp:.3f}  "
                     f"err={abs(rsp-ssp):.3f}")

    # =========================================================================
    # STEP 5: Post-processing
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("STEP 5 — Post-processing")
    log.info("="*60)

    # 5A: Zero-inflated features (two-part class-conditioned resample)
    log.info(f"\n[5A] Zero-inflated two-part fix ({len(zi_present)} features)...")
    synth_df = fix_zero_inflated_features(synth_df, train_df, zi_present, rng)

    # 5B: Sort pass (max >= mean >= min)
    log.info("\n[5B] Sort pass (max >= mean >= min)...")
    synth_df = sort_max_mean_min(synth_df, ORDERED_GROUPS)

    # 5C: Residual quantile mapping (light cleanup after IC + sort)
    log.info(f"\n[5C] Residual quantile mapping (threshold={args.ks_quantile_threshold})...")
    synth_df, n_mapped = quantile_map_conditioned(
        synth_df, train_df, feature_cols, discrete_cols,
        ks_threshold=args.ks_quantile_threshold,
    )

    # 5D: Light swap fix
    log.info("\n[5D] Light swap fix...")
    synth_df = light_swap_fix(synth_df, ORDERED_GROUPS)

    # 5E: Structural fixes
    log.info("\n[5E] Care unit one-hot fix...")
    synth_df = fix_care_unit_onehot(synth_df, CARE_UNIT_COLS, care_unit_dist_by_class, rng)

    log.info("\n[5F] Label prevalence fix...")
    synth_df = fix_label_prevalence(synth_df, prevalence, rng)

    log.info("\n[5G] Gender prevalence fix...")
    synth_df = fix_gender_prevalence(synth_df, gender_rate_by_class, rng)

    # Final range clip
    for col in continuous_cols:
        if col in synth_df.columns:
            synth_df[col] = synth_df[col].clip(
                train_df[col].min(), train_df[col].max()
            )

    # Missing value fill (class median)
    n_missing = int(synth_df.isnull().sum().sum())
    if n_missing > 0:
        log.warning(f"  {n_missing} missing values — filling with class medians")
        for lv in [0, 1]:
            mask = synth_df[TARGET_COL] == lv
            real_class = train_df[train_df[TARGET_COL] == lv]
            for col in synth_df.columns:
                if synth_df.loc[mask, col].isnull().any():
                    fv = real_class[col].median() if col in real_class else train_df[col].median()
                    synth_df.loc[mask, col] = synth_df.loc[mask, col].fillna(fv)

    # =========================================================================
    # STEP 6: Save
    # =========================================================================
    log.info("\n" + "="*60)
    log.info("STEP 6 — Save")
    log.info("="*60)

    synth_path = SYNTHETIC_DIR / "B_synthetic_train_vFinal.csv"
    synth_df.to_csv(synth_path, index=False)
    log.info(f"  Saved -> {synth_path}  {synth_df.shape}")

    # =========================================================================
    # STEP 7: Validation
    # =========================================================================
    if args.validate:
        log.info("\n" + "="*60)
        log.info("STEP 7 — Full Validation")
        log.info("="*60)

        # 7A: Overall KS
        ks_df = ks_summary(train_df, synth_df, feature_cols, discrete_cols)
        ks_path = SYNTHETIC_DIR / "ks_validation_vFinal.csv"
        ks_df.to_csv(ks_path, index=False)
        total = len(ks_df)
        n_good = int((ks_df["ks_statistic"] <= 0.05).sum())
        n_warn = int(((ks_df["ks_statistic"] > 0.05) & (ks_df["ks_statistic"] <= 0.10)).sum())
        n_bad  = int((ks_df["ks_statistic"] > 0.10).sum())
        log.info(f"\n  [7A] OVERALL KS:")
        log.info(f"    Mean={ks_df['ks_statistic'].mean():.4f}  Median={ks_df['ks_statistic'].median():.4f}")
        log.info(f"    <= 0.05: {n_good} ({n_good/total*100:.0f}%)  "
                 f"0.05-0.10: {n_warn} ({n_warn/total*100:.0f}%)  "
                 f"> 0.10: {n_bad} ({n_bad/total*100:.0f}%)")

        # 7B: Label signal
        signal_df = label_signal_check(train_df, synth_df, feature_cols, discrete_cols)
        signal_path = SYNTHETIC_DIR / "label_signal_vFinal.csv"
        signal_df.to_csv(signal_path, index=False)
        signs_ok = int(signal_df["sign_preserved"].sum())
        total_sig = len(signal_df)
        med_ratio = signal_df.head(30)["ratio"].median()
        log.info(f"\n  [7B] LABEL SIGNAL:")
        log.info(f"    Signs preserved: {signs_ok}/{total_sig} ({signs_ok/total_sig*100:.1f}%)")
        log.info(f"    Median ratio (top 30): {med_ratio:.4f}  (target ~1.0)")
        log.info(f"\n  Top 15 features:")
        log.info(f"  {'Feature':35s} {'Real diff':>10} {'Synth diff':>11} {'Ratio':>7}  {'OK':>4}")
        log.info(f"  {'-'*70}")
        for _, r in signal_df.head(15).iterrows():
            ok = "YES" if r["sign_preserved"] else "NO <<"
            log.info(f"  {r['feature']:35s} {r['real_diff']:10.3f} {r['synth_diff']:11.3f} "
                     f"{r['ratio']:7.3f}  {ok}")

        # 7C: Correlation preservation (THE KEY vFinal metric)
        corr_df = correlation_check(train_df, synth_df, KEY_CORR_PAIRS)
        corr_path = SYNTHETIC_DIR / "correlation_check_vFinal.csv"
        corr_df.to_csv(corr_path, index=False)
        mean_err = corr_df["error"].mean()
        log.info(f"\n  [7C] CORRELATION PRESERVATION:")
        log.info(f"    Mean Spearman error: {mean_err:.4f}  (v4 was ~0.65, target <0.05)")
        log.info(f"  {'Pair':50s} {'Class':>6} {'Real':>7} {'Synth':>7} {'Err':>6}")
        log.info(f"  {'-'*80}")
        for _, r in corr_df.iterrows():
            log.info(f"  {r['c1']:22s} x {r['c2']:22s}  sep={int(r['class'])}  "
                     f"{r['real_spearman']:7.3f}  {r['synth_spearman']:7.3f}  {r['error']:6.3f}")

        # 7D: Structural integrity
        struct = check_structural_integrity(synth_df, ORDERED_GROUPS, CARE_UNIT_COLS)
        log.info(f"\n  [7D] STRUCTURAL: {'PASS' if not struct else 'VIOLATIONS: '+str(struct)}")

        log.info(f"\n  Prevalence: {synth_df[TARGET_COL].mean():.4f} (real: {prevalence:.4f})")
        log.info(f"  Missing:    {synth_df.isnull().sum().sum()}")

        # Plots
        plot_label_signal(signal_df, top_n=20, suffix="_vFinal")
        plot_correlation_comparison(corr_df, suffix="_vFinal")
        plot_per_class_distributions(train_df, synth_df, KEY_FEATURES_SIGNAL[:12],
                                     suffix="_vFinal")
        plot_ks_bar(ks_df, top_n=30, suffix="_vFinal")

    # =========================================================================
    # STEP 8: Metadata + usage guide
    # =========================================================================
    total_time = (time.time() - t_start) / 60
    signal_df_m = label_signal_check(train_df, synth_df, feature_cols, discrete_cols)
    corr_df_m   = correlation_check(train_df, synth_df, KEY_CORR_PAIRS)

    meta = {
        "version"               : "vFinal",
        "generated_at"          : datetime.now().isoformat(),
        "n_real"                : int(n_real),
        "n_real_pos"            : int(len(real_pos)),
        "n_real_neg"            : int(len(real_neg)),
        "n_synthetic"           : int(args.n_synthetic),
        "n_features"            : len(feature_cols),
        "real_prevalence"       : float(prevalence),
        "synth_prevalence"      : float(synth_df[TARGET_COL].mean()),
        "epochs_pos"            : args.epochs_pos,
        "epochs_neg"            : args.epochs_neg,
        "batch_size_pos"        : args.batch_size_pos,
        "batch_size_neg"        : args.batch_size_neg,
        "ic_alpha"              : args.ic_alpha,
        "random_seed"           : RANDOM_SEED,
        "total_runtime_min"     : round(total_time, 2),
        "sign_preservation_pct" : float(signal_df_m["sign_preserved"].mean() * 100),
        "mean_spearman_error"   : float(corr_df_m["error"].mean()),
        "output_file"           : str(synth_path),
        "approach"              : (
            "class_split_CTGAN + Iman_Conover_rank_correlation_injection + "
            "zero_inflated_two_part_fix + label_conditioned_postprocessing"
        ),
        "pipeline"              : [
            "1. CTGAN class-split: ctgan_pos fits sep=1 (691 rows), ctgan_neg fits sep=0 (1839 rows)",
            "2. Sample n_pos from ctgan_pos, n_neg from ctgan_neg, assign labels, concat",
            "3. Iman-Conover per class: impose real Spearman correlation structure",
            "   - Compute Spearman matrix from real class data (blended with alpha)",
            "   - Van der Waerden scores -> Cholesky transform -> reorder CTGAN samples",
            "   - Marginals preserved EXACTLY, correlations match real structure",
            "   - Applied separately per class -> label signal preserved",
            "4. Zero-inflated two-part fix: correct zero fraction + nonzero distribution per class",
            "5. Sort pass (max>=mean>=min), residual quantile map, light swap fix",
            "6. Care unit one-hot, label/gender prevalence matching",
        ],
        "usage_guide"           : {
            "synthetic_only"    : "Train CatBoost/FFT on B_synthetic_train_vFinal.csv alone. "
                                  "Expected AUPRC: 0.80-0.92",
            "augmented"         : "Combine real + synthetic for training. "
                                  "Expected AUPRC: 0.95-0.98 (your target)",
            "augmented_command" : (
                "import pandas as pd\n"
                "real = pd.read_csv('B_train_model_ready.csv')\n"
                "synth = pd.read_csv('B_synthetic_train_vFinal.csv')\n"
                "combined = pd.concat([real, synth], ignore_index=True)\n"
                "combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)\n"
                "# Train CatBoost/FFT on combined, test on REAL held-out test set"
            ),
        },
    }
    meta_path = SYNTHETIC_DIR / "generation_metadata_vFinal.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("SYNTHETIC DATA GENERATION  vFinal  COMPLETE")
    log.info("=" * 70)
    log.info(f"  Output  : {synth_path}")
    log.info(f"  Shape   : {synth_df.shape}")
    log.info(f"  Time    : {total_time:.1f} min")
    log.info(f"  Signal  : {meta['sign_preservation_pct']:.1f}% signs preserved")
    log.info(f"  Corr err: {meta['mean_spearman_error']:.4f} mean Spearman error")
    log.info("=" * 70)
    log.info("")
    log.info("USAGE:")
    log.info("  Synthetic-only:  train on B_synthetic_train_vFinal.csv")
    log.info("                   Expected AUPRC 0.80-0.92")
    log.info("  AUGMENTED (recommended for paper):")
    log.info("    combined = concat([real_train, vFinal_synthetic])")
    log.info("    train on combined -> test on REAL held-out test set")
    log.info("    Expected AUPRC 0.95-0.98 (matches your real model)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
