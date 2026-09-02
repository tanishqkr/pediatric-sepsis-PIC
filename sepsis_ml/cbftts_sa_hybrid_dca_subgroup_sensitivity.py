"""
cbftts_sa_hybrid_dca_subgroup_sensitivity.py
─────────────────────────────────────────────────────────────────────────────
Extended clinical evaluation for the HYBRID CBFTTS-SA model
(CatBoost [Exp D, weighted hybrid] + FT-Transformer [hybrid weighted],
stacked via logistic-regression meta-learner — i.e. the model produced by
synthetic_stacking.py).

This script fills three gaps that synthetic_stacking.py does NOT cover:

  1. Decision Curve Analysis (DCA)          — net benefit vs treat-all/treat-none
  2. Subgroup analysis by paediatric age band (neonate/infant/child/adolescent)
  3. Clinical threshold sensitivity analysis — named operating points
     (80/85/~90/92/95% sensitivity) with a "recommended" threshold flagged,
     mirroring Table 7 / Table 13 in the paper — NOT the same as the raw
     7-row sweep table synthetic_stacking.py already writes.

Nothing here re-trains anything. It consumes:
  - stacking_predictions.npz   (ensemble_probs, catboost_probs, ftt_probs, y_test)
    written by synthetic_stacking.py -> sepsis_ml/synthetic/stacking/results/
  - B_test_model_ready.csv     (for age_years, to build age bands)
    model_datasets/B_test_model_ready.csv

Run AFTER synthetic_stacking.py has completed successfully.

Outputs (sepsis_ml/synthetic/stacking/):
  results/  cbftts_sa_hybrid_dca.csv
            cbftts_sa_hybrid_subgroup_by_age.csv
            cbftts_sa_hybrid_threshold_sensitivity.csv
            cbftts_sa_hybrid_extended_eval.json
  figures/  cbftts_sa_hybrid_dca.png / .pdf
            cbftts_sa_hybrid_subgroup_by_age.png / .pdf
            cbftts_sa_hybrid_threshold_sensitivity.png / .pdf

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/synthetic/stacking/cbftts_sa_hybrid_dca_subgroup_sensitivity.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, f1_score
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# 0. PATHS  (mirrors synthetic_stacking.py layout)
# ══════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════

from pathlib import Path

# Absolute project root
PROJECT_ROOT = Path(r"D:\Projects\pediatric-sepsis-PIC")

# Main folders
SEPSIS_ML = PROJECT_ROOT / "sepsis_ml"
SYNTHETIC = SEPSIS_ML / "synthetic"
STACKING = SYNTHETIC / "stacking"

MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"
REAL_TEST_FILE = MODEL_DATA_DIR / "B_test_model_ready.csv"

RESULTS_DIR = STACKING / "results"
FIGURES_DIR = STACKING / "figures"
LOGS_DIR = STACKING / "logs"

for d in [RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STACKING_PREDICTIONS_PATH = RESULTS_DIR / "stacking_predictions.npz"

TARGET_COL = "sepsis_label"
AGE_COL    = "age_years"          # must exist in B_test_model_ready.csv

# ══════════════════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════════════════
log_path = LOGS_DIR / "cbftts_sa_hybrid_dca_subgroup_sensitivity.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# 2. METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════
def find_threshold_at_sensitivity(y_true, y_prob, target):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "threshold": float(threshold), "sensitivity": float(sens),
        "specificity": float(spec), "ppv": float(ppv), "npv": float(npv),
        "f1": float(f1), "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. DECISION CURVE ANALYSIS (DCA)
# ══════════════════════════════════════════════════════════════════════════
def decision_curve_analysis(y_true, y_prob, thresholds=None):
    """
    Standard net-benefit DCA (Vickers & Elkin formulation):
      NB_model    = (TP/N) - (FP/N) * (pt / (1 - pt))
      NB_treat_all= prevalence - (1 - prevalence) * (pt / (1 - pt))
      NB_treat_none = 0
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.96, 0.01)

    y_true = np.asarray(y_true)
    N = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for pt in thresholds:
        y_pred = (np.asarray(y_prob) >= pt).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())

        odds = pt / (1 - pt) if pt < 1.0 else np.inf
        nb_model     = (tp / N) - (fp / N) * odds
        nb_treat_all = prevalence - (1 - prevalence) * odds
        nb_treat_none = 0.0

        rows.append({
            "threshold": round(float(pt), 4),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_treat_all),
            "net_benefit_treat_none": float(nb_treat_none),
        })
    return pd.DataFrame(rows)


def plot_dca(dca_df, out_prefix, model_label="CBFTTS-SA (Hybrid, real+synthetic)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"],
            color="#E53935", lw=2.2, label=model_label)
    ax.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"],
            color="#F39C12", lw=1.4, ls=":", label="Treat all")
    ax.axhline(0.0, color="black", lw=1.0, label="Treat none (NB=0)")
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"Decision Curve Analysis\n{model_label}", fontweight="bold")
    ax.set_xlim(0, 1)
    ymin = min(dca_df["net_benefit_model"].min(),
               dca_df["net_benefit_treat_all"].min(), -0.05)
    ax.set_ylim(ymin, dca_df["net_benefit_model"].max() * 1.15)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{out_prefix}.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {out_prefix}.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. SUBGROUP ANALYSIS BY AGE BAND
# ══════════════════════════════════════════════════════════════════════════
AGE_BANDS = [
    ("Neonate (0-28d)",   0.0,          28 / 365.25),
    ("Infant (29d-1y)",   28 / 365.25,  1.0),
    ("Child (1-12y)",     1.0,          12.0),
    ("Adolescent (12-18y)", 12.0,       18.0 + 1e-6),
]


def subgroup_analysis_by_age(y_true, y_prob, age_years, threshold):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    age_years = np.asarray(age_years)

    rows = []
    for label, lo, hi in AGE_BANDS:
        mask = (age_years >= lo) & (age_years < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"age_group": label, "n": 0, "sepsis_n": 0,
                          "prevalence": np.nan, "auroc": np.nan,
                          "auprc": np.nan, "sensitivity": np.nan,
                          "specificity": np.nan, "f1": np.nan})
            continue

        yt = y_true[mask]
        yp = y_prob[mask]
        sepsis_n = int(yt.sum())
        prev = yt.mean()

        if len(np.unique(yt)) < 2:
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(yt, yp)
            auprc = average_precision_score(yt, yp)

        m = compute_metrics(yt, yp, threshold)
        rows.append({
            "age_group": label, "n": n, "sepsis_n": sepsis_n,
            "prevalence": float(prev),
            "auroc": float(auroc) if not np.isnan(auroc) else None,
            "auprc": float(auprc) if not np.isnan(auprc) else None,
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "f1": m["f1"],
        })
    return pd.DataFrame(rows)


def plot_subgroup(subgroup_df, out_prefix, model_label="CBFTTS-SA (Hybrid)"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Subgroup Performance by Age Band — {model_label}",
                 fontsize=12, fontweight="bold")

    labels = subgroup_df["age_group"].values
    aurocs = subgroup_df["auroc"].astype(float).values

    axes[0].barh(labels, aurocs, color="#1565C0", alpha=0.85)
    for i, v in enumerate(aurocs):
        if not np.isnan(v):
            axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0.5, 1.02)
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("AUROC by Age Band", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    x = np.arange(len(labels))
    w = 0.35
    sens = subgroup_df["sensitivity"].astype(float).values
    spec = subgroup_df["specificity"].astype(float).values
    axes[1].bar(x - w/2, sens, w, label="Sensitivity", color="#2ECC71")
    axes[1].bar(x + w/2, spec, w, label="Specificity", color="#8E44AD")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8, rotation=15)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_title("Sensitivity & Specificity by Age Band", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{out_prefix}.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {out_prefix}.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. CLINICAL THRESHOLD SENSITIVITY ANALYSIS (Table-7 / Table-13 style)
# ══════════════════════════════════════════════════════════════════════════
OPERATING_POINTS = [
    ("80% Sensitivity", 0.80),
    ("85% Sensitivity", 0.85),
    ("~90% (recommended)", 0.90),
    ("92% Sensitivity", 0.92),
    ("95% Sensitivity", 0.95),
]


def clinical_threshold_sensitivity(y_true, y_prob):
    rows = []
    for label, target in OPERATING_POINTS:
        thr = find_threshold_at_sensitivity(y_true, y_prob, target)
        m = compute_metrics(y_true, y_prob, thr)
        rows.append({
            "operating_point": label,
            "target_sensitivity": target,
            "threshold": round(thr, 4),
            "sensitivity": round(m["sensitivity"], 4),
            "specificity": round(m["specificity"], 4),
            "ppv": round(m["ppv"], 4),
            "npv": round(m["npv"], 4),
            "f1": round(m["f1"], 4),
            "recommended": label.startswith("~90%"),
        })
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(thr_df, out_prefix, model_label="CBFTTS-SA (Hybrid)"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Threshold Sensitivity Analysis — {model_label}",
                 fontsize=12, fontweight="bold")

    x = thr_df["threshold"].values
    rec_idx = thr_df.index[thr_df["recommended"]].tolist()

    axes[0].plot(x, thr_df["sensitivity"], "o-", color="#E53935", label="Sensitivity")
    axes[0].plot(x, thr_df["specificity"], "s-", color="#1565C0", label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(x, thr_df["ppv"], "o-", color="#2ECC71", label="PPV")
    axes[1].plot(x, thr_df["npv"], "s-", color="#F39C12", label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    axes[2].plot(x, thr_df["f1"], "o-", color="#6A1B9A")
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].grid(alpha=0.3)
    axes[2].spines[["top", "right"]].set_visible(False)

    for ax in axes:
        for i in rec_idx:
            ax.axvline(x[i], color="gray", ls="--", lw=1.0, alpha=0.7)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{out_prefix}.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved -> {out_prefix}.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("CBFTTS-SA HYBRID (WEIGHTED SYNTHETIC) — EXTENDED EVALUATION")
    log.info("DCA | Subgroup-by-age | Clinical threshold sensitivity")
    log.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── Load ensemble predictions from synthetic_stacking.py ────────────────
    if not STACKING_PREDICTIONS_PATH.exists():
        log.error(f"MISSING: {STACKING_PREDICTIONS_PATH}")
        log.error("Run synthetic_stacking.py first to produce this file.")
        sys.exit(1)

    preds = np.load(STACKING_PREDICTIONS_PATH)
    ensemble_probs = preds["ensemble_probs"]
    y_test         = preds["y_test"]
    log.info(f"Loaded ensemble predictions: n={len(y_test)}, "
             f"prevalence={y_test.mean():.1%}")
    log.info(f"Ensemble AUROC (sanity check): "
             f"{roc_auc_score(y_test, ensemble_probs):.4f}")

    # ── Load age_years for subgroup analysis ────────────────────────────────
    if not REAL_TEST_FILE.exists():
        log.error(f"MISSING: {REAL_TEST_FILE}")
        sys.exit(1)
    real_test_df = pd.read_csv(REAL_TEST_FILE)
    if AGE_COL not in real_test_df.columns:
        log.error(f"Column '{AGE_COL}' not found in {REAL_TEST_FILE.name}. "
                  f"Available columns include: {list(real_test_df.columns)[:15]} ...")
        sys.exit(1)
    if len(real_test_df) != len(y_test):
        log.error(f"Row count mismatch: real_test_df has {len(real_test_df)} rows, "
                  f"but stacking predictions have {len(y_test)}. "
                  f"Ensure both come from the same, unshuffled B_test_model_ready.csv.")
        sys.exit(1)
    age_years = real_test_df[AGE_COL].values

    # ── Recommended clinical threshold (90% sensitivity, matching paper convention) ──
    recommended_threshold = find_threshold_at_sensitivity(y_test, ensemble_probs, 0.90)
    log.info(f"Recommended threshold (targeting 90% sensitivity): "
             f"{recommended_threshold:.4f}")

    # ── 1. Decision Curve Analysis ──────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Decision Curve Analysis")
    log.info("=" * 60)
    dca_df = decision_curve_analysis(y_test, ensemble_probs)
    dca_df.to_csv(RESULTS_DIR / "cbftts_sa_hybrid_dca.csv", index=False)
    plot_dca(dca_df, "cbftts_sa_hybrid_dca")
    positive_nb_range = dca_df.loc[dca_df["net_benefit_model"] > 0, "threshold"]
    log.info(f"  Net benefit > 0 across thresholds "
             f"{positive_nb_range.min():.2f}-{positive_nb_range.max():.2f}"
             if len(positive_nb_range) else "  Model never exceeds NB=0")

    # ── 2. Subgroup analysis by age band ────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Subgroup Analysis by Age Band")
    log.info("=" * 60)
    subgroup_df = subgroup_analysis_by_age(
        y_test, ensemble_probs, age_years, recommended_threshold
    )
    subgroup_df.to_csv(RESULTS_DIR / "cbftts_sa_hybrid_subgroup_by_age.csv", index=False)
    plot_subgroup(subgroup_df, "cbftts_sa_hybrid_subgroup_by_age")
    log.info("\n" + subgroup_df.to_string(index=False))

    # ── 3. Clinical threshold sensitivity analysis ──────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Clinical Threshold Sensitivity Analysis")
    log.info("=" * 60)
    thr_df = clinical_threshold_sensitivity(y_test, ensemble_probs)
    thr_df.to_csv(RESULTS_DIR / "cbftts_sa_hybrid_threshold_sensitivity.csv", index=False)
    plot_threshold_sensitivity(thr_df, "cbftts_sa_hybrid_threshold_sensitivity")
    log.info("\n" + thr_df.to_string(index=False))

    # ── Save combined JSON ───────────────────────────────────────────────────
    combined = {
        "model": "CBFTTS-SA (hybrid, real+synthetic weighted)",
        "timestamp": datetime.now().isoformat(),
        "n_test": int(len(y_test)),
        "prevalence": float(y_test.mean()),
        "recommended_threshold": float(recommended_threshold),
        "ensemble_auroc": float(roc_auc_score(y_test, ensemble_probs)),
        "ensemble_auprc": float(average_precision_score(y_test, ensemble_probs)),
        "dca": dca_df.to_dict("records"),
        "subgroup_by_age": subgroup_df.to_dict("records"),
        "threshold_sensitivity": thr_df.to_dict("records"),
    }
    with open(RESULTS_DIR / "cbftts_sa_hybrid_extended_eval.json", "w") as f:
        json.dump(combined, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("EXTENDED EVALUATION COMPLETE")
    log.info(f"  Results -> {RESULTS_DIR}")
    log.info(f"  Figures -> {FIGURES_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
