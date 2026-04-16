"""
phase0_diagnostics.py
---------------------
Run BEFORE any modelling. Produces all diagnostic outputs that inform:
  1. Whether to use SMOTE or class_weights only
  2. Which features to drop (near-perfect correlation / zero variance)
  3. Missingness summary for the methods section
  4. Class distribution stats

Outputs saved to:  sepsis_ml/diagnostics/
Figures saved to:  sepsis_ml/figures/
Log saved to:      sepsis_ml/logs/phase0_diagnostics.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase0_diagnostics.py
"""

import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FILES, DIAG_DIAG, FIG_DIAG, LOGS_DIR,
    TARGET_COL, DROP_COLS, RANDOM_SEED,
)

# ── Logging ───────────────────────────────────────────────────────────────────
log_path = LOGS_DIR / "phase0_diagnostics.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()

np.random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(key_train: str, key_test: str, label: str):
    train = pd.read_csv(FILES[key_train])
    test  = pd.read_csv(FILES[key_test])
    log.info(f"Loaded {label}: train={train.shape}, test={test.shape}")
    return train, test


def get_X_y(df: pd.DataFrame):
    drop = [c for c in DROP_COLS if c in df.columns]
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL] + drop)
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        log.info(f"    Non-numeric columns present (kept as-is): {non_numeric}")
    return X, y


def class_distribution(y: pd.Series, label: str) -> dict:
    counts = y.value_counts().sort_index()
    total  = len(y)
    pos    = counts.get(1, 0)
    neg    = counts.get(0, 0)
    ratio  = neg / pos if pos > 0 else float("inf")
    log.info(f"\n  [{label}] Class distribution:")
    log.info(f"    Negative (0): {neg:,}  ({100*neg/total:.1f}%)")
    log.info(f"    Positive (1): {pos:,}  ({100*pos/total:.1f}%)")
    log.info(f"    Imbalance ratio (neg:pos): {ratio:.1f}:1")
    return {"neg": int(neg), "pos": int(pos), "total": int(total), "ratio": round(ratio, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 1 — Class distribution plots
# ══════════════════════════════════════════════════════════════════════════════

def diag_class_distribution(datasets: dict) -> dict:
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 1: Class Distribution")
    log.info("="*60)

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Class distribution — sepsis label", fontsize=13, fontweight="bold")

    for ax, (key, (train, test, label)) in zip(axes, datasets.items()):
        _, y = get_X_y(train)
        stats = class_distribution(y, label)
        results[key] = stats

        colors = ["#5B8DB8", "#C0392B"]
        bars = ax.bar(["Non-sepsis (0)", "Sepsis (1)"],
                      [stats["neg"], stats["pos"]],
                      color=colors, width=0.5, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, [stats["neg"], stats["pos"]]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + stats["total"]*0.01,
                    f"{val:,}\n({100*val/stats['total']:.1f}%)",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{label}\nRatio {stats['ratio']:.1f}:1", fontsize=10)
        ax.set_ylabel("Patients")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, stats["total"] * 0.95)

    plt.tight_layout()
    out = FIG_DIAG / "diag1_class_distribution.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close()
    log.info(f"  Saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 2 — Missingness summary
# ══════════════════════════════════════════════════════════════════════════════

def diag_missingness(datasets: dict) -> dict:
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 2: Missingness Summary")
    log.info("="*60)

    results = {}
    for key, (train, test, label) in datasets.items():
        X, y = get_X_y(train)
        missing_pct = (X.isnull().sum() / len(X) * 100).sort_values(ascending=False)
        n_zero_var   = int((X.select_dtypes(include=[np.number]).std() == 0).sum())
        n_any_miss   = int((X.isnull().any()).sum())
        n_high_miss  = int((missing_pct > 50).sum())

        log.info(f"\n  [{label}]")
        log.info(f"    Total features:              {X.shape[1]}")
        log.info(f"    Features with any missing:   {n_any_miss}")
        log.info(f"    Features >50% missing:       {n_high_miss}")
        log.info(f"    Zero-variance features:      {n_zero_var}")
        log.info(f"    Top 10 highest missing %:")
        for col, pct in missing_pct.head(10).items():
            log.info(f"      {col:<45} {pct:.1f}%")

        miss_df = missing_pct[missing_pct > 0].reset_index()
        miss_df.columns = ["feature", "missing_pct"]
        miss_path = DIAG_DIAG / f"diag2_missingness_{key}.csv"
        miss_df.to_csv(miss_path, index=False)

        results[key] = {
            "n_features"    : int(X.shape[1]),
            "n_any_missing" : n_any_miss,
            "n_high_missing": n_high_miss,
            "n_zero_var"    : n_zero_var,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 3 — Correlation / redundancy heatmap (top features only)
# ══════════════════════════════════════════════════════════════════════════════

def diag_correlation(datasets: dict) -> dict:
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 3: Correlation / Redundancy Analysis")
    log.info("="*60)

    results = {}
    for key, (train, test, label) in datasets.items():
        X, y = get_X_y(train)
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        # Find near-perfect correlations (|r| > 0.98) — these are candidates to drop
        corr = X_num.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = [(col, row, round(upper.loc[row, col], 4))
                           for col in upper.columns
                           for row in upper.index
                           if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.98]

        log.info(f"\n  [{label}] Near-perfect correlations (|r|>0.98): {len(high_corr_pairs)}")
        for c1, c2, r in high_corr_pairs[:20]:
            log.info(f"    {c1:<40} ↔  {c2:<40}  r={r}")

        # Save full list
        corr_path = DIAG_DIAG / f"diag3_high_corr_pairs_{key}.csv"
        pd.DataFrame(high_corr_pairs, columns=["feature_1", "feature_2", "pearson_r"])\
          .to_csv(corr_path, index=False)

        # Heatmap of top-40 most variable features
        top40 = X_num.std().nlargest(40).index.tolist()
        corr40 = X_num[top40].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr40, dtype=bool))
        sns.heatmap(corr40, mask=mask, cmap="coolwarm", center=0,
                    vmin=-1, vmax=1, square=True, linewidths=0.2,
                    cbar_kws={"shrink": 0.6}, ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"Feature correlation heatmap — top 40 variables by variance\n{label}",
                     fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=5, rotation=90)
        ax.tick_params(axis="y", labelsize=5)
        plt.tight_layout()
        out = FIG_DIAG / f"diag3_correlation_heatmap_{key}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"  Saved heatmap → {out}")

        results[key] = {"n_high_corr_pairs": len(high_corr_pairs)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 4 — Feature-space overlap via t-SNE (SMOTE decision)
# ══════════════════════════════════════════════════════════════════════════════

def diag_tsne(datasets: dict) -> dict:
    """
    t-SNE on a stratified sample (max 2000 rows for speed).
    If the two classes are visually well-separated → SMOTE is safe.
    If classes heavily overlap → class_weights only (SMOTE would synthesise
    ambiguous boundary points that hurt precision).
    """
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 4: Feature-Space Overlap (t-SNE)")
    log.info("="*60)

    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    results = {}
    for key, (train, test, label) in datasets.items():
        X, y = get_X_y(train)
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        # Stratified sample
        sample_size = min(2000, len(X_num))
        pos_idx = y[y == 1].index
        neg_idx = y[y == 0].index
        n_pos = min(len(pos_idx), sample_size // 2)
        n_neg = min(len(neg_idx), sample_size - n_pos)
        idx = np.concatenate([
            np.random.choice(pos_idx, n_pos, replace=False),
            np.random.choice(neg_idx, n_neg, replace=False),
        ])
        X_sample = X_num.loc[idx]
        y_sample = y.loc[idx]

        log.info(f"\n  [{label}] Running t-SNE on {len(X_sample)} samples "
                 f"({n_pos} sepsis, {n_neg} non-sepsis)...")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)

        tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                    perplexity=30, max_iter=1000, learning_rate="auto",
                    init="pca", n_jobs=-1)
        X_2d = tsne.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = {0: "#5B8DB8", 1: "#C0392B"}
        labels_map = {0: "Non-sepsis", 1: "Sepsis"}
        for cls in [0, 1]:
            mask = y_sample.values == cls
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=colors[cls], label=labels_map[cls],
                       alpha=0.5, s=8, linewidths=0)
        ax.legend(markerscale=3, fontsize=10)
        ax.set_title(f"t-SNE feature-space visualisation — {label}\n"
                     f"(sample n={len(X_sample)}; interpret overlap to decide on SMOTE)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        out = FIG_DIAG / f"diag4_tsne_{key}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"  Saved t-SNE → {out}")
        log.info(f"  *** Inspect diag4_tsne_{key}.png to decide on SMOTE ***")
        log.info(f"      Well-separated clusters → SMOTE safe to test")
        log.info(f"      Heavy overlap            → use class_weights only")

        results[key] = {"tsne_sample_size": len(X_sample)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 5 — Zero-variance and near-zero-variance features to drop
# ══════════════════════════════════════════════════════════════════════════════

def diag_zero_variance(datasets: dict) -> dict:
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 5: Zero / Near-Zero Variance Features")
    log.info("="*60)

    results = {}
    for key, (train, test, label) in datasets.items():
        X, y = get_X_y(train)
        X_num = X.select_dtypes(include=[np.number])

        zero_var  = X_num.columns[X_num.std() == 0].tolist()
        near_zero = X_num.columns[X_num.std() < 0.001].tolist()

        log.info(f"\n  [{label}]")
        log.info(f"    Zero-variance features ({len(zero_var)}):  {zero_var[:10]}")
        log.info(f"    Near-zero-var features ({len(near_zero)}): {near_zero[:10]}")

        drop_path = DIAG_DIAG / f"diag5_drop_candidates_{key}.json"
        with open(drop_path, "w") as f:
            json.dump({"zero_variance": zero_var, "near_zero_variance": near_zero}, f, indent=2)
        log.info(f"  Saved → {drop_path}")

        results[key] = {
            "n_zero_var"  : len(zero_var),
            "n_near_zero" : len(near_zero),
            "drop_list"   : near_zero,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic 6 — Key Phoenix variable distributions (sepsis vs non-sepsis)
# ══════════════════════════════════════════════════════════════════════════════

def diag_key_variable_distributions(datasets: dict) -> None:
    log.info("\n" + "="*60)
    log.info("DIAGNOSTIC 6: Key Variable Distributions (sepsis vs non-sepsis)")
    log.info("="*60)

    KEY_VARS = [
        "lactate_max", "platelets_min", "inr_max",
        "map_min", "creatinine_max", "bilirubin_max",
        "age_years",
    ]

    for key, (train, test, label) in datasets.items():
        available = [v for v in KEY_VARS if v in train.columns]
        if not available:
            log.info(f"  [{label}] No key variables found — skipping")
            continue

        n_cols = min(4, len(available))
        n_rows = (len(available) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        axes = np.array(axes).flatten()

        for ax, var in zip(axes, available):
            for cls, color, lbl in [(0, "#5B8DB8", "Non-sepsis"), (1, "#C0392B", "Sepsis")]:
                vals = train.loc[train[TARGET_COL] == cls, var].dropna()
                ax.hist(vals, bins=40, alpha=0.55, color=color,
                        label=lbl, density=True, edgecolor="none")
            ax.set_title(var, fontsize=9, fontweight="bold")
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7)

        for ax in axes[len(available):]:
            ax.set_visible(False)

        fig.suptitle(f"Key variable distributions — {label}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = FIG_DIAG / f"diag6_key_distributions_{key}.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=150)
        plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
        plt.close()
        log.info(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("PHASE 0 — DIAGNOSTICS")
    log.info(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    log.info("=" * 60)

    # Load all four splits
    train_A, test_A = load_dataset("A_train", "A_test", "Option A")
    train_B, test_B = load_dataset("B_train", "B_test", "Option B")

    datasets = {
        "A": (train_A, test_A, "Option A — Full cohort"),
        "B": (train_B, test_B, "Option B — Infection-only"),
    }

    # ── Run all diagnostics ───────────────────────────────────────────────────
    class_stats  = diag_class_distribution(datasets)
    miss_stats   = diag_missingness(datasets)
    corr_stats   = diag_correlation(datasets)
    tsne_stats   = diag_tsne(datasets)
    var_stats    = diag_zero_variance(datasets)
    diag_key_variable_distributions(datasets)

    # ── Save consolidated summary ─────────────────────────────────────────────
    summary = {
        "timestamp"        : datetime.now().isoformat(timespec="seconds"),
        "class_distribution": class_stats,
        "missingness"      : miss_stats,
        "correlation"      : corr_stats,
        "tsne"             : tsne_stats,
        "zero_variance"    : var_stats,
    }
    summary_path = DIAG_DIAG / "phase0_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Decision guidance ─────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("DECISION GUIDANCE — READ BEFORE PROCEEDING TO PHASE 1")
    log.info("="*60)
    for key in ["A", "B"]:
        r = class_stats[key]
        log.info(f"\n  Dataset {key}: imbalance ratio = {r['ratio']:.1f}:1")
        if r["ratio"] > 10:
            log.info(f"    → Severe imbalance. class_weights REQUIRED.")
            log.info(f"    → SMOTE: only if t-SNE shows clear separation (check diag4_tsne_{key}.png)")
        else:
            log.info(f"    → Moderate imbalance. class_weights recommended.")
            log.info(f"    → SMOTE: reasonable to test (check t-SNE first)")

    log.info(f"\n  Zero-variance features to drop before modelling:")
    for key in ["A", "B"]:
        n = var_stats[key]["n_zero_var"]
        log.info(f"    Dataset {key}: {n} zero-variance features "
                 f"(see diag5_drop_candidates_{key}.json)")

    log.info(f"\n  Near-perfect correlations (|r|>0.98):")
    for key in ["A", "B"]:
        n = corr_stats[key]["n_high_corr_pairs"]
        log.info(f"    Dataset {key}: {n} pairs (see diag3_high_corr_pairs_{key}.csv)")

    log.info(f"\n  Full summary → {summary_path}")
    log.info(f"  All figures  → {FIG_DIAG}/")
    log.info(f"  Full log     → {log_path}")
    log.info("\nPhase 0 complete. Review outputs before running phase1_train_all_models.py")


if __name__ == "__main__":
    main()