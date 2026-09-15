"""
fl_shap_config.py
Single source of truth for paths, node structure, and run settings.

FIXES APPLIED (per repo audit findings):
  1. General ICU column bug: the NODES dict key "General" does NOT match
     the actual data column, which is "care_unit_General ICU" (with a
     space + "ICU" suffix) -- confirmed directly from B_test_model_ready.csv.
     This caused federated_shap.py to raise KeyError on "care_unit_General"
     and silently drop the largest node (1,251 patients, 49.4% of training
     data) from every downstream result. Fixed via NODE_COLUMN_SUFFIX below
     -- node keys/filenames stay clean ("General"), only the column lookup
     is corrected.
  2. Portability: PARENT_PROJECT_ROOT was hardcoded to a Windows path,
     which fails on your partner's Mac. Now auto-detected from this file's
     own location (repo_root/sepsis_ml/phase9_federated_shap/fl_shap_config.py
     -> walk up two levels), with an environment variable override for
     edge cases. Works unmodified on Windows, Mac, or Linux.
"""
import os

# ---------------------------------------------------------------- ADAPT ----
# Auto-detected from this file's own location:
#   <repo_root>/sepsis_ml/phase9_federated_shap/fl_shap_config.py
# walk up two levels to get <repo_root>. Override with the SEPSIS_PIC_ROOT
# environment variable if your checkout is structured differently.
_THIS_FILE = os.path.abspath(__file__)
_SEPSIS_ML_ROOT_AUTO = os.path.dirname(os.path.dirname(_THIS_FILE))
_PARENT_PROJECT_ROOT_AUTO = os.path.dirname(_SEPSIS_ML_ROOT_AUTO)

PARENT_PROJECT_ROOT = os.environ.get("SEPSIS_PIC_ROOT", _PARENT_PROJECT_ROOT_AUTO)
SEPSIS_ML_ROOT = os.path.join(PARENT_PROJECT_ROOT, "sepsis_ml")

MODEL_READY_TRAIN = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/B_train_model_ready.csv")
MODEL_READY_TEST = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/B_test_model_ready.csv")
FEATURE_LIST_JSON = os.path.join(PARENT_PROJECT_ROOT, "model_datasets/feature_list.json")

FL_DIR = os.path.join(SEPSIS_ML_ROOT, "fl")
FL_RESULTS_DIR = os.path.join(FL_DIR, "results")
FL_SCRIPT = os.path.join(FL_DIR, "phase7_federated_learning.py")

FL_CHECKPOINT_DIR = os.path.join(FL_DIR, "models")
GLOBAL_MODEL_CHECKPOINT_FEDAVG = os.path.join(FL_CHECKPOINT_DIR, "fedavg_global_final.pt")
GLOBAL_MODEL_CHECKPOINT_FEDPROX = os.path.join(FL_CHECKPOINT_DIR, "fedprox_global_final.pt")
SCALER_PATH = os.path.join(FL_CHECKPOINT_DIR, "scaler.pkl")
MODEL_METADATA_PATH = os.path.join(FL_CHECKPOINT_DIR, "model_metadata.json")

PHASE9_ROOT = os.path.join(SEPSIS_ML_ROOT, "phase9_federated_shap")
# ------------------------------------------------------------------------- #

FEATURE_COUNT = 225

# 5 simulated ICU-type nodes, with training-set sizes confirmed by
# phase7_per_node_metrics.csv.
NODES = {
    "CICU":    144,
    "NICU":    421,
    "PICU":    213,
    "SICU":    501,
    "General": 1251,
}
TOTAL_N = sum(NODES.values())  # 2530

CARE_UNIT_PREFIX = "care_unit_"

# THE FIX: maps each node's short key (used for filenames/dict keys
# everywhere in this pipeline) to its ACTUAL column-name suffix in
# B_train_model_ready.csv / B_test_model_ready.csv. Confirmed via:
#   python3 -c "import pandas as pd; df = pd.read_csv('model_datasets/B_test_model_ready.csv', nrows=1); print([c for c in df.columns if 'care_unit' in c])"
# -> ['care_unit_CICU', 'care_unit_General ICU', 'care_unit_NICU',
#     'care_unit_PICU', 'care_unit_SICU']
# Only General ICU's suffix differs from its node key -- the other four
# match exactly, which is why this went unnoticed until General was the
# only one that broke.
NODE_COLUMN_SUFFIX = {
    "CICU": "CICU",
    "NICU": "NICU",
    "PICU": "PICU",
    "SICU": "SICU",
    "General": "General ICU",
}


def care_unit_column(node_name: str) -> str:
    """The single source of truth for turning a node key into its real
    column name. Use this everywhere instead of hand-building the string
    -- this is exactly the function whose absence caused the bug."""
    return f"{CARE_UNIT_PREFIX}{NODE_COLUMN_SUFFIX[node_name]}"


# Results root for the federated-SHAP + sensitivity-analysis work.
RESULTS_ROOT = os.environ.get("SEPSIS_RESULTS_ROOT", os.path.join(PHASE9_ROOT, "results"))

TOP_K = 5
SHAP_BACKGROUND_SIZE = 100
RANDOM_SEED = 42
