import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# ===== PATHS (same as your main script) =====
SCRIPT_DIR   = Path(__file__).resolve().parent
SYNTHETIC_ML = SCRIPT_DIR.parent
SEPSIS_ML    = SYNTHETIC_ML.parent
PROJECT_ROOT = SEPSIS_ML.parent

MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"

SYNTH_TRAIN_FILE = MODEL_DATA_DIR / "synthetic" / "B_synthetic_train_vFinal.csv"
REAL_TEST_FILE   = MODEL_DATA_DIR / "B_test_model_ready.csv"

TARGET = "sepsis_label"

# ===== LOAD =====
synth = pd.read_csv(SYNTH_TRAIN_FILE)
test  = pd.read_csv(REAL_TEST_FILE)

feature_cols = [c for c in synth.columns if c != TARGET]

X_train = synth[feature_cols]
y_train = synth[TARGET]

X_test  = test[feature_cols]
y_test  = test[TARGET]

# ===== PASTE ANY OPTUNA TRIAL PARAMS HERE =====
params = {
    "iterations": 793,
    "learning_rate": 0.1444,
    "depth": 5,
    "l2_leaf_reg": 5.62,
    "bagging_temperature": 0.59,
    "random_strength": 0.09,
    "border_count": 168,
    "class_weights": [1.0, 2.09],

    "eval_metric": "Logloss",
    "custom_metric": ["AUC"],
    "verbose": False,
    "allow_writing_files": False,
    "random_seed": 42
}

# ===== TRAIN =====
model = CatBoostClassifier(**params)
model.fit(X_train, y_train)

# ===== TEST ON REAL =====
probs = model.predict_proba(X_test)[:, 1]

print("AUROC:", round(roc_auc_score(y_test, probs), 4))
print("AUPRC:", round(average_precision_score(y_test, probs), 4))