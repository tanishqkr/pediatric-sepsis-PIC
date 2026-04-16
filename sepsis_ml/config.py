"""
config.py
---------
Single source of truth for all paths, constants, and experiment settings.
All other scripts import from here. Never hardcode paths elsewhere.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent          # sepsis_ml/
REPO_ROOT      = PROJECT_ROOT.parent                      # pediatric_sepsis_prediction_PIC_XAI/
DB_ROOT        = REPO_ROOT / "paediatric-intensive-care-database-1.1.0"
DATA_DIR       = DB_ROOT / "output2"

# ── Input files ───────────────────────────────────────────────────────────────
FILES = {
    "A_train" : DATA_DIR / "option_A_train.csv",
    "A_test"  : DATA_DIR / "option_A_test.csv",
    "B_train" : DATA_DIR / "option_B_train.csv",
    "B_test"  : DATA_DIR / "option_B_test.csv",
    "A_meta"  : DATA_DIR / "option_A_meta_train.csv",
    "B_meta"  : DATA_DIR / "option_B_meta_train.csv",
}

# ── Top-level output directories ──────────────────────────────────────────────
DIAG_DIR    = PROJECT_ROOT / "diagnostics"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_DIR    = PROJECT_ROOT / "shap"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR    = PROJECT_ROOT / "logs"

# ── Phase-specific subdirectories ─────────────────────────────────────────────
# diagnostics/
DIAG_AUDIT  = DIAG_DIR / "phase0_audit"
DIAG_DIAG   = DIAG_DIR / "phase0_diagnostics"

# figures/
FIG_AUDIT       = FIGURES_DIR / "phase0_audit"
FIG_DIAG        = FIGURES_DIR / "phase0_diagnostics"
FIG_BASELINE    = FIGURES_DIR / "phase1_baseline"
FIG_TUNING      = FIGURES_DIR / "phase2_tuning"
FIG_EVALUATION  = FIGURES_DIR / "phase3_evaluation"
FIG_SHAP        = FIGURES_DIR / "phase4_shap"

# ── Create all directories on import ──────────────────────────────────────────
for d in [
    DIAG_DIR, MODELS_DIR, RESULTS_DIR, SHAP_DIR, FIGURES_DIR, LOGS_DIR,
    DIAG_AUDIT, DIAG_DIAG,
    FIG_AUDIT, FIG_DIAG, FIG_BASELINE, FIG_TUNING, FIG_EVALUATION, FIG_SHAP,
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Results logger files ──────────────────────────────────────────────────────
EXPERIMENT_LOG  = RESULTS_DIR / "experiment_log.json"
RESULTS_SUMMARY = RESULTS_DIR / "results_summary.csv"

# ── Target and identifier columns ─────────────────────────────────────────────
TARGET_COL   = "sepsis_label"
DROP_COLS    = [
    "subject_id", "hadm_id", "stay_id",
    "phoenix_core_score", "phoenix_8_score",
]

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Cross-validation ──────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Models to train ───────────────────────────────────────────────────────────
MODEL_NAMES = [
    "catboost",
    "xgboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
    "mlp",
]

# ── Optuna tuning ─────────────────────────────────────────────────────────────
OPTUNA_TRIALS   = 100
OPTUNA_METRIC   = "auprc"

# ── Evaluation metrics to record ──────────────────────────────────────────────
EVAL_METRICS = [
    "auroc", "auprc", "f1",
    "sensitivity", "specificity", "ppv", "npv",
    "brier_score",
]

# ── Clinical threshold strategy ───────────────────────────────────────────────
TARGET_SENSITIVITY = 0.85

# ── Bootstrap CI ──────────────────────────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 1000

# ── SHAP ──────────────────────────────────────────────────────────────────────
SHAP_TOP_N_FEATURES = 20
SHAP_DEPENDENCE_TOP = 5

# ── Datasets ──────────────────────────────────────────────────────────────────
DATASETS = {
    "A": {"label": "Option A (Full cohort)",           "train": "A_train", "test": "A_test"},
    "B": {"label": "Option B (Infection-only cohort)", "train": "B_train", "test": "B_test"},
}