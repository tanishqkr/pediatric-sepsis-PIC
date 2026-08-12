"""
config.py
Single source of truth for paths, node structure, and run settings.

Wired against your ACTUAL parent-project layout (from project_context.md):
- Preprocessed data:      model_datasets/B_train_model_ready.csv, B_test_model_ready.csv
- FL training script:     sepsis_ml/fl/phase7_federated_learning.py
- FL results (existing):  sepsis_ml/fl/results/phase7_fedavg_results.json, phase7_fedprox_results.json
- Results/logging util:   sepsis_ml/logger.py  (log_experiment, load_all_results, print_leaderboard)
- SHAP display helpers:   sepsis_ml/phase4_shap.py  (get_display_name, shap_color_map)
- Sensitivity utilities:  sepsis_ml/phase5_sensitivity_analysis.py (bootstrap_auroc, bootstrap_auprc, metrics_at_threshold)
- FT-Transformer arch:    sepsis_ml/dl/phase6_fttransformer_windows.py (build_fttransformer, scale_df)

ONE CRITICAL GAP: phase7_federated_learning.py's results/ folder only has
JSON metrics (phase7_fedavg_results.json etc.) — no models/ folder was found
in the context report. If it isn't already saving the trained global model's
weights, federated SHAP has nothing to explain. See checkpoint_patch.py.
"""
import os

# ---------------------------------------------------------------- ADAPT ----
PARENT_PROJECT_ROOT = "D:/pediatric-sepsis-PIC"  # confirmed from your PS prompt
SEPSIS_ML_ROOT = os.path.join(PARENT_PROJECT_ROOT, "sepsis_ml")

MODEL_READY_TRAIN = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/B_train_model_ready.csv")
MODEL_READY_TEST = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/B_test_model_ready.csv")
FEATURE_LIST_JSON = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/feature_list.json")

# phase7_federated_learning.py lives here; its results/ subfolder already exists
FL_DIR = os.path.join(SEPSIS_ML_ROOT, "fl")
FL_RESULTS_DIR = os.path.join(FL_DIR, "results")
FL_SCRIPT = os.path.join(FL_DIR, "phase7_federated_learning.py")

# NEW — sepsis_ml/fl/ currently has only figures/ and results/, no models/ folder
# (confirmed from your directory listing) — the phase7 patch now creates this.
FL_CHECKPOINT_DIR = os.path.join(FL_DIR, "models")
GLOBAL_MODEL_CHECKPOINT_FEDAVG = os.path.join(FL_CHECKPOINT_DIR, "fedavg_global_final.pt")
GLOBAL_MODEL_CHECKPOINT_FEDPROX = os.path.join(FL_CHECKPOINT_DIR, "fedprox_global_final.pt")
SCALER_PATH = os.path.join(FL_CHECKPOINT_DIR, "scaler.pkl")
MODEL_METADATA_PATH = os.path.join(FL_CHECKPOINT_DIR, "model_metadata.json")

# This new pipeline's own home — a new sibling folder next to fl/, dl/,
# phase7_stacking/, phase8_attention_catboost/, sensitivity_analysis/ etc.,
# following your project's existing phaseN_name/ convention.
PHASE9_ROOT = os.path.join(SEPSIS_ML_ROOT, "phase9_federated_shap")
# ------------------------------------------------------------------------- #

FEATURE_COUNT = 225  # from model_datasets/README.md

# 5 simulated ICU-type nodes — training sizes confirmed by phase7_per_node_metrics.csv
NODES = {
    "CICU":    144,
    "NICU":    421,
    "PICU":    213,
    "SICU":    501,
    "General": 1251,
}
TOTAL_N = sum(NODES.values())  # 2530

# node column is a one-hot dummy in the model-ready CSV: care_unit_CICU, care_unit_SICU, etc.
CARE_UNIT_PREFIX = "care_unit_"

# Results root for the NEW federated-SHAP + sensitivity-analysis work.
# Copy this whole folder to move results between machines.
RESULTS_ROOT = os.environ.get("SEPSIS_RESULTS_ROOT", os.path.join(PHASE9_ROOT, "results"))

TOP_K = 5  # for the mentor's "top-5 features" dashboard requirement
SHAP_BACKGROUND_SIZE = 100
RANDOM_SEED = 42

