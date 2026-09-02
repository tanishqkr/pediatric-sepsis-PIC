"""
generate_phase6_comparison.py
=============================================================
Generates the combined ROC and Precision-Recall comparison figure
for all three deep learning architectures against tuned CatBoost.

Output file: sepsis_ml/dl/figures/phase6_roc_pr_comparison.png
(This name matches the includegraphics call in main.tex exactly.)

Run from project root:
    python generate_phase6_comparison.py

Requirements: catboost, scikit-learn, matplotlib, pandas, numpy
=============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from catboost import CatBoostClassifier

# ─── Paths (all relative to project root) ────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
TEST_CSV    = os.path.join(ROOT, "model_datasets", "B_test_model_ready.csv")
CBM_PATH    = os.path.join(ROOT, "sepsis_ml", "models", "run_tuned", "catboost_tuned.cbm")
RESULTS_DIR = os.path.join(ROOT, "sepsis_ml", "dl", "results")
OUT_DIR     = os.path.join(ROOT, "sepsis_ml", "dl", "figures")
OUT_FILE    = os.path.join(OUT_DIR, "phase6_roc_pr_comparison.png")
TARGET_COL  = "sepsis_label"

os.makedirs(OUT_DIR, exist_ok=True)


# ─── Step 1: Load test labels ────────────────────────────────
print("Loading test set...")
if not os.path.exists(TEST_CSV):
    sys.exit(f"ERROR: Test CSV not found at:\n  {TEST_CSV}")

df_test = pd.read_csv(TEST_CSV)
y_true  = df_test[TARGET_COL].values
X_test  = df_test.drop(columns=[TARGET_COL])
print(f"  Test set: {len(y_true)} patients, prevalence = {y_true.mean():.3f}")


# ─── Step 2: Load CatBoost model and predict ─────────────────
print("\nLoading CatBoost model...")
if not os.path.exists(CBM_PATH):
    sys.exit(f"ERROR: CatBoost model not found at:\n  {CBM_PATH}")

cb_model = CatBoostClassifier()
cb_model.load_model(CBM_PATH)
y_prob_catboost = cb_model.predict_proba(X_test)[:, 1]
print("  CatBoost predictions generated.")


# ─── Step 3: Load DL probabilities from .npz files ───────────
def load_dl_probs(model_key):
    """
    Loads test-set probabilities from the .npz prediction files
    saved by the phase6 scripts.

    The .npz file is expected to contain one of these keys:
      y_prob, probs, predictions, y_pred_proba, or the first key found.

    Falls back to CSV or plain .npy if .npz is not found.
    """
    candidates = [
        # Primary: .npz files (what your scripts actually saved)
        os.path.join(RESULTS_DIR, f"{model_key}_predictions.npz"),
        # Fallbacks
        os.path.join(RESULTS_DIR, f"{model_key}_probs.npz"),
        os.path.join(RESULTS_DIR, f"{model_key}_probs.npy"),
        os.path.join(RESULTS_DIR, f"{model_key}_probs.csv"),
        os.path.join(RESULTS_DIR, f"{model_key}_test_probs.csv"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        print(f"  Found: {os.path.basename(path)}")

        if path.endswith(".npz"):
            data = np.load(path, allow_pickle=True)
            print(f"  Keys in file: {list(data.files)}")
            # Try common key names in order
            for key in ["y_prob", "probs", "predictions",
                        "y_pred_proba", "y_pred", "prob"]:
                if key in data:
                    arr = data[key].ravel()
                    # If probabilities are 2-column (neg, pos), take col 1
                    if arr.ndim == 2 and arr.shape[1] == 2:
                        arr = arr[:, 1]
                    print(f"  Using key '{key}', shape {arr.shape}, "
                          f"range [{arr.min():.3f}, {arr.max():.3f}]")
                    return arr.ravel()
            # If none of the above keys match, just use the first key
            first_key = data.files[0]
            arr = data[first_key].ravel()
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr[:, 1]
            print(f"  Using first key '{first_key}', shape {arr.shape}")
            return arr.ravel()

        elif path.endswith(".npy"):
            arr = np.load(path).ravel()
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr[:, 1]
            return arr

        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            for col in ["prob", "y_prob", "proba", "probability",
                        df.columns[0]]:
                if col in df.columns:
                    return df[col].values.ravel()

    # Nothing found
    print(f"\n  ERROR: No prediction file found for {model_key}.")
    print(f"  Checked:")
    for p in candidates:
        print(f"    {p}")
    return None


print("\nLoading DL model probabilities from .npz files...")

DL_MODELS = {
    "FT-Transformer": "phase6_fttransformer",
    "ResNet":         "phase6_resnet",
    "TabNet":         "phase6_tabnet",
}

dl_probs = {}
missing  = []
for display_name, key in DL_MODELS.items():
    print(f"\n  {display_name}...")
    arr = load_dl_probs(key)
    if arr is None:
        missing.append(display_name)
    else:
        dl_probs[display_name] = arr

if missing:
    print(f"\nStopping: could not load probabilities for: {missing}\n")
    sys.exit(1)

print("\nAll probabilities loaded successfully.")


# ─── Step 4: Build model dict and plot config ─────────────────
all_models = {
    "Tuned CatBoost":  y_prob_catboost,
    "FT-Transformer":  dl_probs["FT-Transformer"],
    "ResNet":          dl_probs["ResNet"],
    "TabNet":          dl_probs["TabNet"],
}

STYLE = {
    "Tuned CatBoost": {"color": "#7B2D8B", "lw": 2.5, "ls": "-",  "zorder": 5},
    "FT-Transformer": {"color": "#2196F3", "lw": 1.8, "ls": "-",  "zorder": 4},
    "ResNet":         {"color": "#FF9800", "lw": 1.8, "ls": "--", "zorder": 3},
    "TabNet":         {"color": "#4CAF50", "lw": 1.8, "ls": "-.", "zorder": 3},
}

prevalence = y_true.mean()


# ─── Step 5: Generate figure ──────────────────────────────────
print("\nGenerating figure...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    "Phase 6 — Deep Learning Benchmark vs Tuned CatBoost\n"
    "Option B (Infection-only)  |  Test Set  n = 633",
    fontsize=11, fontweight="bold", y=1.01
)

for model_name, y_prob in all_models.items():
    s = STYLE[model_name]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = auc(fpr, tpr)
    axes[0].plot(
        fpr, tpr,
        color=s["color"], lw=s["lw"], linestyle=s["ls"], zorder=s["zorder"],
        label=f"{model_name}  (AUROC = {auroc:.4f})"
    )

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec, prec)
    axes[1].plot(
        rec, prec,
        color=s["color"], lw=s["lw"], linestyle=s["ls"], zorder=s["zorder"],
        label=f"{model_name}  (AUPRC = {auprc:.4f})"
    )

# ROC axis
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random classifier")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — All Models\nOption B", fontsize=11)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25)

# PR axis
ax = axes[1]
ax.axhline(
    y=prevalence, color="gray", linestyle=":", lw=1.5, alpha=0.7,
    label=f"Baseline (prevalence = {prevalence:.2f})"
)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([0.0, 1.01])
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves — All Models\nOption B", fontsize=11)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
print(f"\nDone. Figure saved to:\n  {OUT_FILE}")
print(
    "\nNext step: upload phase6_roc_pr_comparison.png to the\n"
    "figures/ folder in Overleaf."
)
plt.close()
