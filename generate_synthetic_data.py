"""
generate_synthetic_data.py
──────────────────────────────────────────────────────────────────────────────
Synthetic Data Generation using CTGAN
Project : Pediatric Sepsis Prediction (PIC Database)
Purpose : Train CTGAN on real Option B training data, generate synthetic
          patients, save to model_datasets/synthetic/ for downstream use.

What this script does:
  1. Loads real Option B training data (2,530 patients, 225 features + label)
  2. Trains a CTGAN model on the full training set
  3. Generates a configurable number of synthetic patients (default: 10,000)
  4. Validates synthetic data quality:
       - Prevalence check (should be ~27.3%)
       - Feature distribution comparison (KS test on key features)
       - Missing value audit
  5. Saves synthetic dataset to model_datasets/synthetic/B_synthetic_train.csv
  6. Saves CTGAN model for reproducibility
  7. Produces distribution comparison figures

WHY CTGAN (not TVAE or GaussianCopula)?
  - CTGAN uses a conditional GAN architecture designed for tabular data
  - It handles mixed data types (continuous + binary) natively
  - It explicitly models class imbalance via conditional vector sampling
  - TVAE tends to over-smooth distributions; GaussianCopula assumes
    multivariate normality which clinical data violates heavily

IMPORTANT NOTES:
  - The label (sepsis_label) is synthesized jointly with features.
    Phoenix 2024 criteria CANNOT be re-applied because score-component
    columns were dropped before modeling to prevent label leakage.
    Limitation: Synthetic patients are statistically consistent with real
    patients but not independently Phoenix-verified. This is acknowledged
    as a study limitation.
  - The real TEST SET (633 patients) is NEVER synthesized. All downstream
    evaluation uses the real held-out test set only.
  - CTGAN is non-deterministic. random_seed is fixed for reproducibility.

DEVICE SUPPORT:
  CTGAN (via SDV library) runs on CPU only — it does not use GPU.
  This is expected and normal. CTGAN training is CPU-bound.

RUNTIME ESTIMATES:
  ┌──────────────────────────────────────────────────────────┐
  │ Machine          │ CTGAN Train │ Generate 10k │  Total   │
  ├──────────────────┼─────────────┼──────────────┼──────────┤
  │ Mac M1/M2/M3     │  25–40 min  │   2–4 min    │ ~30–45 min│
  │ Windows (modern) │  30–50 min  │   2–4 min    │ ~35–55 min│
  │ Windows + GPU    │  30–50 min* │   2–4 min    │ ~35–55 min│
  └──────────────────────────────────────────────────────────┘
  * CTGAN does NOT use GPU — the GPU column is identical to CPU.
  The bottleneck is CTGAN's GAN training loop (300 epochs × 2530 patients).
  Progress is printed every 50 epochs so you know it's running.

DEPENDENCIES:
  pip install sdv

  SDV installs CTGAN, TVAE, and related tools automatically.
  If you get torch conflicts: pip install sdv --no-deps then install
  torch separately.

Run from project root:
  conda activate sepsis_ml
  python generate_synthetic_data.py

  Optional flags:
    --n_synthetic 10000     number of synthetic patients (default: 10000)
    --epochs 300            CTGAN training epochs (default: 300)
    --batch_size 500        CTGAN batch size (default: 500)
    --validate              run extended KS-test validation (default: True)
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

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT    = Path(__file__).resolve().parent
MODEL_DATA_DIR  = PROJECT_ROOT / "model_datasets"
TRAIN_FILE      = MODEL_DATA_DIR / "B_train_model_ready.csv"

SYNTHETIC_DIR   = MODEL_DATA_DIR / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR     = SYNTHETIC_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR        = SYNTHETIC_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "generate_synthetic_data.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED     = 42
TARGET_COL      = "sepsis_label"

# Key features to highlight in distribution comparison figures
KEY_FEATURES = [
    "lactate_max", "platelets_min", "inr_max", "map_min",
    "creatinine_max", "bilirubin_max", "age_years",
    "respiratory_rate_max", "heart_rate_max", "temperature_max",
]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic pediatric sepsis data using CTGAN"
    )
    parser.add_argument("--n_synthetic", type=int, default=10_000,
                        help="Number of synthetic patients to generate (default: 10000)")
    parser.add_argument("--epochs",      type=int, default=300,
                        help="CTGAN training epochs (default: 300)")
    parser.add_argument("--batch_size",  type=int, default=500,
                        help="CTGAN batch size (default: 500)")
    parser.add_argument("--validate",    action="store_true", default=True,
                        help="Run KS-test validation (default: True)")
    return parser.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# 4.  CTGAN IMPORT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def import_ctgan():
    """Import CTGAN with a helpful error message if SDV is not installed."""
    try:
        from ctgan import CTGAN
        log.info("  CTGAN imported successfully from ctgan package.")
        return CTGAN
    except ImportError:
        pass
    try:
        from sdv.single_table import CTGANSynthesizer
        log.info("  CTGANSynthesizer imported successfully from sdv package.")
        return None  # signal to use SDV path
    except ImportError:
        log.error("=" * 60)
        log.error("CTGAN / SDV not installed.")
        log.error("Please run:  pip install sdv")
        log.error("Then retry:  python generate_synthetic_data.py")
        log.error("=" * 60)
        sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ks_feature_comparison(real_df, synth_df, feature_cols, top_n=20):
    """
    Run KS test on each continuous feature between real and synthetic.
    Returns a DataFrame sorted by KS statistic (higher = more different).
    Lower KS statistic = synthetic distribution closer to real = better.
    """
    results = []
    for col in feature_cols:
        real_vals  = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        if len(real_vals) < 10 or len(synth_vals) < 10:
            continue
        # Only run on continuous (non-binary) features
        if real_df[col].nunique() <= 2:
            continue
        ks_stat, ks_p = stats.ks_2samp(real_vals, synth_vals)
        results.append({
            "feature"    : col,
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value" : round(ks_p, 4),
            "real_mean"  : round(real_vals.mean(), 4),
            "synth_mean" : round(synth_vals.mean(), 4),
            "real_std"   : round(real_vals.std(), 4),
            "synth_std"  : round(synth_vals.std(), 4),
        })
    ks_df = pd.DataFrame(results).sort_values("ks_statistic", ascending=False)
    return ks_df

# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_distribution_comparison(real_df, synth_df, feature_list, title_suffix=""):
    """
    Side-by-side histograms: real vs synthetic for key features.
    Saves to FIGURES_DIR.
    """
    available = [f for f in feature_list if f in real_df.columns]
    if not available:
        log.warning("  No key features found for distribution plot — skipping.")
        return

    n_cols = 2
    n_rows = (len(available) + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        real_vals  = real_df[feat].dropna()
        synth_vals = synth_df[feat].dropna()

        # Clip to 99th percentile of real data for readability
        upper = real_vals.quantile(0.99)
        lower = real_vals.quantile(0.01)
        real_vals  = real_vals.clip(lower, upper)
        synth_vals = synth_vals.clip(lower, upper)

        ax.hist(real_vals,  bins=40, alpha=0.6, color="#1565C0",
                label=f"Real (n={len(real_vals):,})",  density=True)
        ax.hist(synth_vals, bins=40, alpha=0.6, color="#E53935",
                label=f"Synth (n={len(synth_vals):,})", density=True)

        ks_stat, _ = stats.ks_2samp(real_vals, synth_vals)
        ax.set_title(f"{feat}\nKS={ks_stat:.3f}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Real vs Synthetic Feature Distributions {title_suffix}\n"
        f"(Lower KS statistic = better fidelity)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    save_path = FIGURES_DIR / "distribution_comparison_key_features.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "distribution_comparison_key_features.pdf",
                dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {save_path}")


def plot_ks_summary(ks_df, top_n=30):
    """Bar plot of top-N features by KS statistic."""
    top = ks_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#E53935" if v > 0.1 else "#FFA726" if v > 0.05 else "#43A047"
              for v in top["ks_statistic"]]
    ax.barh(top["feature"], top["ks_statistic"], color=colors)
    ax.axvline(0.05, color="orange", ls="--", lw=1, label="KS=0.05 (good)")
    ax.axvline(0.10, color="red",    ls="--", lw=1, label="KS=0.10 (concern)")
    ax.set_xlabel("KS Statistic (lower = better fidelity)")
    ax.set_title(f"Top {top_n} Features by KS Statistic\nReal vs Synthetic",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_path = FIGURES_DIR / "ks_statistic_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {save_path}")


def plot_label_comparison(real_df, synth_df):
    """Compare class prevalence: real vs synthetic."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for ax, df, title in zip(
        axes,
        [real_df, synth_df],
        [f"Real (n={len(real_df):,})", f"Synthetic (n={len(synth_df):,})"]
    ):
        counts = df[TARGET_COL].value_counts().sort_index()
        bars   = ax.bar(["Non-sepsis\n(0)", "Sepsis\n(1)"],
                        counts.values,
                        color=["#1565C0", "#E53935"])
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    f"{val:,}\n({val/len(df):.1%})",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Class Distribution: Real vs Synthetic", fontweight="bold")
    plt.tight_layout()
    save_path = FIGURES_DIR / "class_distribution_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {save_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    t_start = time.time()
    np.random.seed(RANDOM_SEED)

    log.info("=" * 70)
    log.info("SYNTHETIC DATA GENERATION — CTGAN")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)
    log.info(f"  n_synthetic : {args.n_synthetic:,}")
    log.info(f"  epochs      : {args.epochs}")
    log.info(f"  batch_size  : {args.batch_size}")
    log.info(f"  seed        : {RANDOM_SEED}")
    log.info("")
    log.info("NOTE: CTGAN runs on CPU only. This is expected behaviour.")
    log.info("      Estimated runtime: 25–50 min depending on your machine.")
    log.info("      Progress will be printed every 50 epochs.")

    # ── 7.1  Load real training data ─────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Loading real training data")
    log.info("=" * 60)

    if not TRAIN_FILE.exists():
        log.error(f"Training file not found: {TRAIN_FILE}")
        log.error("Please check that B_train_model_ready.csv exists in model_datasets/")
        sys.exit(1)

    train_df     = pd.read_csv(TRAIN_FILE)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    n_real       = len(train_df)
    prevalence   = train_df[TARGET_COL].mean()

    log.info(f"  Loaded : {train_df.shape}")
    log.info(f"  Patients : {n_real:,}")
    log.info(f"  Features : {len(feature_cols)}")
    log.info(f"  Sepsis prevalence : {prevalence:.1%}  "
             f"({train_df[TARGET_COL].sum():.0f} sepsis / "
             f"{(train_df[TARGET_COL] == 0).sum():.0f} non-sepsis)")

    # Identify discrete (binary / categorical) columns for CTGAN
    # CTGAN needs to know which columns are discrete to handle them correctly
    discrete_cols = [TARGET_COL]  # label is always discrete
    for col in feature_cols:
        unique_vals = train_df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            discrete_cols.append(col)

    log.info(f"  Discrete columns identified : {len(discrete_cols)} "
             f"(binary features + label)")
    log.info(f"  Continuous columns          : {len(feature_cols) - len(discrete_cols) + 1}")

    # ── 7.2  Import CTGAN ─────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Importing CTGAN")
    log.info("=" * 60)

    CTGAN_cls = import_ctgan()

    # ── 7.3  Train CTGAN ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Training CTGAN")
    log.info("=" * 60)
    log.info(f"  Training on {n_real:,} real patients × {train_df.shape[1]} columns")
    log.info(f"  Epochs     : {args.epochs}")
    log.info(f"  Batch size : {args.batch_size}")
    log.info("")
    log.info("  [Progress will be printed by CTGAN internally every few epochs]")
    log.info("  [If you see no output for 5+ minutes, it IS running — be patient]")

    t_train = time.time()

    if CTGAN_cls is not None:
        # Direct CTGAN package path
        ctgan_model = CTGAN_cls(
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            generator_dim   = (256, 256),
            discriminator_dim = (256, 256),
            generator_lr    = 2e-4,
            discriminator_lr = 2e-4,
            discriminator_steps = 1,
            log_frequency   = True,
            verbose         = True,
            cuda            = False,   # CTGAN GPU support is unreliable; CPU is stable
        )
        ctgan_model.fit(train_df, discrete_columns=discrete_cols)

    else:
        # SDV path (newer versions)
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_df)

        # Override detected types for binary columns
        for col in discrete_cols:
            metadata.update_column(col, sdtype="categorical")

        ctgan_model = CTGANSynthesizer(
            metadata        = metadata,
            epochs          = args.epochs,
            batch_size      = args.batch_size,
            generator_dim   = (256, 256),
            discriminator_dim = (256, 256),
            verbose         = True,
            cuda            = False,
        )
        ctgan_model.fit(train_df)

    train_time = (time.time() - t_train) / 60
    log.info(f"\n  CTGAN training complete in {train_time:.1f} minutes")

    # Save CTGAN model for reproducibility
    ctgan_save_path = SYNTHETIC_DIR / "ctgan_model.pkl"
    with open(ctgan_save_path, "wb") as f:
        pickle.dump(ctgan_model, f)
    log.info(f"  CTGAN model saved -> {ctgan_save_path}")

    # ── 7.4  Generate synthetic patients ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 4 — Generating synthetic patients")
    log.info("=" * 60)
    log.info(f"  Generating {args.n_synthetic:,} synthetic patients...")

    t_gen = time.time()

    if CTGAN_cls is not None:
        synth_df = ctgan_model.sample(args.n_synthetic)
    else:
        synth_df = ctgan_model.sample(num_rows=args.n_synthetic)

    gen_time = (time.time() - t_gen) / 60
    log.info(f"  Generation complete in {gen_time:.1f} minutes")
    log.info(f"  Synthetic dataset shape: {synth_df.shape}")

    # ── 7.5  Post-processing ──────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 5 — Post-processing synthetic data")
    log.info("=" * 60)

    # Ensure column order matches real data
    synth_df = synth_df[train_df.columns]

    # Enforce binary columns are strictly 0/1
    for col in discrete_cols:
        if col == TARGET_COL:
            synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)
        else:
            synth_df[col] = synth_df[col].round().clip(0, 1).astype(int)

    # Clip continuous features to real data range (prevent extrapolation artifacts)
    for col in feature_cols:
        if col not in discrete_cols:
            real_min = train_df[col].min()
            real_max = train_df[col].max()
            synth_df[col] = synth_df[col].clip(real_min, real_max)

    synth_prevalence = synth_df[TARGET_COL].mean()
    log.info(f"  Synthetic prevalence  : {synth_prevalence:.1%} "
             f"(real: {prevalence:.1%})")
    log.info(f"  Prevalence difference : "
             f"{abs(synth_prevalence - prevalence) * 100:.2f} percentage points")

    if abs(synth_prevalence - prevalence) > 0.05:
        log.warning("  WARNING: Synthetic prevalence differs from real by >5pp.")
        log.warning("  This may indicate CTGAN did not converge well.")
        log.warning("  Consider re-running with more epochs (--epochs 500).")
    else:
        log.info("  Prevalence check: PASSED")

    # Missing value audit
    synth_missing = synth_df.isnull().sum().sum()
    log.info(f"  Missing values in synthetic data: {synth_missing}")
    if synth_missing > 0:
        log.warning(f"  WARNING: {synth_missing} missing values in synthetic data.")
        log.warning("  Filling with column medians from real training data.")
        for col in synth_df.columns:
            if synth_df[col].isnull().any():
                synth_df[col] = synth_df[col].fillna(train_df[col].median())

    # ── 7.6  Save synthetic dataset ───────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 6 — Saving synthetic dataset")
    log.info("=" * 60)

    synth_path = SYNTHETIC_DIR / "B_synthetic_train.csv"
    synth_df.to_csv(synth_path, index=False)
    log.info(f"  Synthetic data saved -> {synth_path}")
    log.info(f"  Shape: {synth_df.shape}")
    log.info(f"  File size: {synth_path.stat().st_size / 1e6:.1f} MB")

    # ── 7.7  Validation ───────────────────────────────────────────────────────
    if args.validate:
        log.info("\n" + "=" * 60)
        log.info("STEP 7 — Validation (KS tests + figures)")
        log.info("=" * 60)

        ks_df = ks_feature_comparison(train_df, synth_df, feature_cols)
        ks_path = SYNTHETIC_DIR / "ks_validation_results.csv"
        ks_df.to_csv(ks_path, index=False)
        log.info(f"  KS results saved -> {ks_path}")

        # Summary stats
        n_good     = (ks_df["ks_statistic"] <= 0.05).sum()
        n_ok       = ((ks_df["ks_statistic"] > 0.05) &
                      (ks_df["ks_statistic"] <= 0.10)).sum()
        n_concern  = (ks_df["ks_statistic"] > 0.10).sum()
        log.info(f"\n  KS SUMMARY (continuous features only):")
        log.info(f"    KS ≤ 0.05 (good)    : {n_good} features")
        log.info(f"    KS 0.05–0.10 (ok)   : {n_ok} features")
        log.info(f"    KS > 0.10 (concern) : {n_concern} features")
        log.info(f"\n  Top 10 worst features (most different from real):")
        for _, row in ks_df.head(10).iterrows():
            log.info(f"    {row['feature']:<40} KS={row['ks_statistic']:.4f}")

        # Figures
        log.info("\n  Generating validation figures...")
        plot_distribution_comparison(
            train_df, synth_df, KEY_FEATURES,
            title_suffix=f"(n_synth={args.n_synthetic:,})"
        )
        plot_ks_summary(ks_df, top_n=min(30, len(ks_df)))
        plot_label_comparison(train_df, synth_df)

    # ── 7.8  Save generation metadata ─────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    metadata_out = {
        "generated_at"       : datetime.now().isoformat(),
        "n_real_train"       : int(n_real),
        "n_synthetic"        : int(args.n_synthetic),
        "n_features"         : len(feature_cols),
        "real_prevalence"    : float(prevalence),
        "synth_prevalence"   : float(synth_prevalence),
        "prevalence_diff_pp" : float(abs(synth_prevalence - prevalence) * 100),
        "ctgan_epochs"       : args.epochs,
        "ctgan_batch_size"   : args.batch_size,
        "random_seed"        : RANDOM_SEED,
        "discrete_cols_count": len(discrete_cols),
        "ctgan_train_min"    : round(train_time, 2),
        "gen_time_min"       : round(gen_time, 2),
        "total_runtime_min"  : round(total_time, 2),
        "output_file"        : str(synth_path),
        "note": (
            "Synthetic patients are statistically consistent with real patients "
            "but are not independently verified against Phoenix 2024 criteria. "
            "Score-component features (vasopressors, etc.) were excluded from "
            "the modeling dataset to prevent label leakage and cannot be "
            "re-applied for post-hoc verification. This is acknowledged as a "
            "study limitation."
        ),
    }

    meta_path = SYNTHETIC_DIR / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata_out, f, indent=2)
    log.info(f"\n  Generation metadata saved -> {meta_path}")

    # ── 7.9  Final summary ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SYNTHETIC DATA GENERATION COMPLETE")
    log.info("=" * 70)
    log.info(f"  Real training patients  : {n_real:,}")
    log.info(f"  Synthetic patients      : {args.n_synthetic:,}")
    log.info(f"  Real prevalence         : {prevalence:.1%}")
    log.info(f"  Synthetic prevalence    : {synth_prevalence:.1%}")
    log.info(f"  CTGAN training time     : {train_time:.1f} min")
    log.info(f"  Generation time         : {gen_time:.1f} min")
    log.info(f"  Total runtime           : {total_time:.1f} min")
    log.info(f"\n  Output files:")
    log.info(f"    Synthetic data -> {synth_path}")
    log.info(f"    CTGAN model    -> {ctgan_save_path}")
    log.info(f"    Metadata       -> {meta_path}")
    log.info(f"    Figures        -> {FIGURES_DIR}")
    log.info(f"    Log            -> {log_path}")
    log.info("")
    log.info("  Next steps:")
    log.info("    python sepsis_ml/synthetic/catboost/synthetic_catboost.py")
    log.info("    python sepsis_ml/synthetic/stacking/synthetic_stacking.py")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
