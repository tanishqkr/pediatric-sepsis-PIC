"""
save_catboost_probs.py
======================
Run this ONCE before any Phase 6 script.
Loads the tuned CatBoost model and saves test set probabilities
to sepsis_ml/results/catboost_test_probs.npy

All three Phase 6 DeLong tests read from this file.

Usage:
    conda activate sepsis_ml
    cd sepsis_ml
    python dl/save_catboost_probs.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import json

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT.parent / "model_datasets"
MODEL_PATH   = PROJECT_ROOT / "models" / "run_tuned" / "catboost_tuned.cbm"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUT_PATH     = RESULTS_DIR / "catboost_test_probs.npy"

print(f"Loading CatBoost model from: {MODEL_PATH}")
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

# Load test data
test_df      = pd.read_csv(DATA_DIR / "B_test_model_ready.csv")
TARGET       = "sepsis_label"
feature_cols = [c for c in test_df.columns if c != TARGET]
X_test       = test_df[feature_cols]
y_test       = test_df[TARGET].values

# Load model and predict
model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

probs = model.predict_proba(X_test)[:, 1]

# Save
np.save(OUT_PATH, probs)
print(f"Saved {len(probs)} test probabilities to: {OUT_PATH}")
print(f"CatBoost AUROC verification: ", end="")

from sklearn.metrics import roc_auc_score
auroc = roc_auc_score(y_test, probs)
print(f"{auroc:.4f}  (expected ~0.9868)")
print("Done. You can now run the Phase 6 scripts.")
