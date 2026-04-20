"""
make_hybrid_dataset.py
──────────────────────────────────────────────────────────────────────────────
Concatenates real training data (Option B) + synthetic training data into a
single hybrid training CSV for Experiment C.

Output: model_datasets/synthetic/C_hybrid_train.csv

Run from project root:
  conda activate sepsis_ml
  python sepsis_ml/synthetic/make_hybrid_dataset.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).resolve().parent          # sepsis_ml/synthetic/
SEPSIS_ML    = SCRIPT_DIR.parent                        # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                         # pediatric_sepsis_prediction_PIC_XAI/

MODEL_DATA_DIR   = PROJECT_ROOT / "model_datasets"
REAL_TRAIN_FILE  = MODEL_DATA_DIR / "B_train_model_ready.csv"
SYNTH_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train_vFinal.csv"
OUTPUT_FILE      = MODEL_DATA_DIR / "synthetic" / "C_hybrid_train.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger()

TARGET_COL = "sepsis_label"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("HYBRID DATASET BUILDER — Real + Synthetic (Option B)")
    log.info("=" * 60)

    # Load
    log.info(f"\nLoading real train : {REAL_TRAIN_FILE}")
    if not REAL_TRAIN_FILE.exists():
        log.error(f"  Not found: {REAL_TRAIN_FILE}")
        sys.exit(1)
    real_train = pd.read_csv(REAL_TRAIN_FILE)

    log.info(f"Loading synthetic  : {SYNTH_TRAIN_FILE}")
    if not SYNTH_TRAIN_FILE.exists():
        log.error(f"  Not found: {SYNTH_TRAIN_FILE}")
        sys.exit(1)
    synth_train = pd.read_csv(SYNTH_TRAIN_FILE)

    # Sanity checks
    real_cols  = set(real_train.columns)
    synth_cols = set(synth_train.columns)

    extra_in_synth = synth_cols - real_cols
    missing_in_synth = real_cols - synth_cols

    if extra_in_synth:
        log.warning(f"  Columns in synthetic but NOT in real (will be dropped): {extra_in_synth}")
        synth_train = synth_train.drop(columns=list(extra_in_synth))

    if missing_in_synth:
        log.error(f"  Columns in real but MISSING from synthetic: {missing_in_synth}")
        log.error("  Cannot concatenate — fix synthetic data first.")
        sys.exit(1)

    # Align column order to real train
    synth_train = synth_train[real_train.columns]

    # Stats before concat
    log.info(f"\n  Real train  : {real_train.shape}  | "
             f"Sepsis: {real_train[TARGET_COL].mean():.1%} "
             f"({real_train[TARGET_COL].sum()} / {len(real_train)})")
    log.info(f"  Synth train : {synth_train.shape}  | "
             f"Sepsis: {synth_train[TARGET_COL].mean():.1%} "
             f"({synth_train[TARGET_COL].sum()} / {len(synth_train)})")

    # Concatenate — real first, then synthetic
    hybrid = pd.concat([real_train, synth_train], axis=0, ignore_index=True)

    log.info(f"\n  Hybrid train: {hybrid.shape}  | "
             f"Sepsis: {hybrid[TARGET_COL].mean():.1%} "
             f"({hybrid[TARGET_COL].sum()} / {len(hybrid)})")
    log.info(f"  Real rows   : {len(real_train):,}  ({len(real_train)/len(hybrid):.1%} of hybrid)")
    log.info(f"  Synth rows  : {len(synth_train):,}  ({len(synth_train)/len(hybrid):.1%} of hybrid)")

    # Save
    hybrid.to_csv(OUTPUT_FILE, index=False)
    log.info(f"\n  Saved → {OUTPUT_FILE}")
    log.info("\nDone. Run hybrid_catboost.py next.")


if __name__ == "__main__":
    main()
