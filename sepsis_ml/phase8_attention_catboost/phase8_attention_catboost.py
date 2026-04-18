"""
phase8_attention_catboost.py  — FIXED for rtdl_revisiting_models
─────────────────────────────────────────────────────────────────────────────
FIX: rtdl_revisiting_models FTTransformer does NOT use .blocks — it uses
     .backbone.blocks (ModuleDict list, not ModuleList of nn.Module).
     The attention sub-module is a custom class (NOT nn.MultiheadAttention)
     with signature forward(x_q, x_kv) -> Tensor — no need_weights support.

     Correct extraction approach (validated):
       1. Build token sequence: CLS + feature embeddings
       2. Run all backbone blocks except the last
       3. In the last block, manually compute Q = W_q(x_cls), K = W_k(x_all)
       4. Compute multi-head attention logits -> softmax -> average heads
       5. Drop CLS self-attention col -> shape (batch, n_features)

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase8_attention_catboost/phase8_attention_catboost.py
"""

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

import sys
import json
import pickle
import logging
import warnings
import time
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from catboost import CatBoostClassifier
from rtdl_revisiting_models import FTTransformer
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix, f1_score
)
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR   = Path(__file__).resolve().parent
SEPSIS_ML    = SCRIPT_DIR.parent
PROJECT_ROOT = SEPSIS_ML.parent

MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

FTT_MODEL_DIR    = SEPSIS_ML / "dl" / "models" / "run_phase6_fttransformer"
FTT_WEIGHTS_PATH = FTT_MODEL_DIR / "fttransformer_best.pt"
FTT_PARAMS_PATH  = FTT_MODEL_DIR / "best_params_final.json"
FTT_SCALER_PATH  = FTT_MODEL_DIR / "scaler.pkl"

CB_PARAMS_PATH     = SEPSIS_ML / "results" / "best_params.json"
CB_TEST_PROBS_PATH = SEPSIS_ML / "results" / "catboost_test_probs.npy"

PHASE8_DIR  = SCRIPT_DIR
MODELS_DIR  = PHASE8_DIR / "models"
RESULTS_DIR = PHASE8_DIR / "results"
FIGURES_DIR = PHASE8_DIR / "figures"
LOGS_DIR    = PHASE8_DIR / "logs"

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOGGING
# ══════════════════════════════════════════════════════════════════════════════

log_path = LOGS_DIR / "phase8_attention_catboost.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED        = 42
TARGET_COL         = "sepsis_label"
TARGET_SENSITIVITY = 0.90

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    DEVICE   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"Device : CUDA — {gpu_name} ({vram_gb:.1f} GB VRAM)")
else:
    DEVICE = torch.device("cpu")
    log.info("Device : CPU")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ATTENTION EXTRACTION — FIXED FOR rtdl_revisiting_models
#
# Architecture confirmed by inspection:
#   model.cls_embedding           — CLS token
#   model.cont_embeddings         — feature linear projections
#   model.backbone.blocks         — ModuleList of ModuleDict
#     block["attention"]          — custom MultiheadAttention(x_q, x_kv)->Tensor
#     block["attention_normalization"] — optional pre-norm LayerNorm
#
# Last block special case (from backbone.forward source):
#   x_q  = x[:, :1]   (CLS token only queries the sequence)
#   x_kv = x          (all tokens serve as keys/values)
#
# We manually compute Q @ K^T / sqrt(d_head), softmax, average heads,
# drop CLS self-attention column -> (batch, n_features) attention map.
# ══════════════════════════════════════════════════════════════════════════════

def _reshape_multihead(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(batch, seq, d) -> (batch*n_heads, seq, d_head)"""
    b, s, d = t.shape
    d_h = d // n_heads
    return t.reshape(b, s, n_heads, d_h).permute(0, 2, 1, 3).reshape(b * n_heads, s, d_h)


@torch.no_grad()
def extract_attention_weights(
    ftt_model: FTTransformer,
    x_cont: torch.Tensor,
) -> torch.Tensor:
    """
    Extract CLS-token attention weights from the last FTTransformer block.

    Args:
        ftt_model : frozen FTTransformer on DEVICE in eval mode
        x_cont    : (batch, n_features) float32 tensor on DEVICE

    Returns:
        attn_feat : (batch, n_features) cpu tensor, head-averaged,
                    CLS self-attention column dropped.
    """
    # Step 1: build token sequence exactly as FTTransformer.forward does
    x_embeddings = [
        ftt_model.cls_embedding(x_cont.shape[:-1]),   # (batch, d)
        ftt_model.cont_embeddings(x_cont),             # (batch, n_feat, d)
    ]
    x = torch.cat(x_embeddings, dim=1)                 # (batch, n_feat+1, d)

    # Step 2: run all backbone blocks except the last
    all_blocks = list(ftt_model.backbone.blocks)

    for block in all_blocks[:-1]:
        x_identity = x
        if "attention_normalization" in block:
            x = block["attention_normalization"](x)
        x = block["attention"](x, x)                   # self-attention
        x = block["attention_residual_dropout"](x)
        x = x_identity + x

        x_identity = x
        x = block["ffn_normalization"](x)
        x = block["ffn"](x)
        x = block["ffn_residual_dropout"](x)
        x = x_identity + x
        x = block["output"](x)

    # Step 3: manually compute attention in last block
    last_block  = all_blocks[-1]
    attn_module = last_block["attention"]

    x_normed = x
    if "attention_normalization" in last_block:
        x_normed = last_block["attention_normalization"](x)

    # Last block: CLS-only queries, all tokens as keys/values
    x_q  = x_normed[:, :1]   # (batch, 1, d)
    x_kv = x_normed           # (batch, n_feat+1, d)

    q = attn_module.W_q(x_q)    # (batch, 1, d)
    k = attn_module.W_k(x_kv)   # (batch, n_feat+1, d)

    n_heads    = attn_module._n_heads
    d_head_key = k.shape[-1] // n_heads

    q_mh = _reshape_multihead(q, n_heads)  # (batch*n_heads, 1, d_head)
    k_mh = _reshape_multihead(k, n_heads)  # (batch*n_heads, n_feat+1, d_head)

    attn_logits = q_mh @ k_mh.transpose(1, 2) / math.sqrt(d_head_key)
    attn_probs  = F.softmax(attn_logits, dim=-1)       # (batch*n_heads, 1, n_feat+1)

    batch      = x_cont.shape[0]
    attn_probs = attn_probs.reshape(batch, n_heads, -1) # (batch, n_heads, n_feat+1)
    attn_avg   = attn_probs.mean(dim=1)                 # (batch, n_feat+1)
    attn_feat  = attn_avg[:, 1:].cpu()                  # (batch, n_feat) — drop CLS col

    return attn_feat


def extract_attention_dataset(ftt_model, X_scaled_np, batch_size=256):
    """Batch-wise extraction over full numpy array. Returns (n, n_features)."""
    ftt_model.eval()
    all_attn = []
    n = len(X_scaled_np)
    for start in range(0, n, batch_size):
        end     = min(start + batch_size, n)
        X_batch = torch.tensor(X_scaled_np[start:end], dtype=torch.float32).to(DEVICE)
        attn    = extract_attention_weights(ftt_model, X_batch)
        all_attn.append(attn.numpy())
        if (start // batch_size) % 5 == 0:
            log.info(f"    Processed {end}/{n} samples...")
    return np.concatenate(all_attn, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. METRIC HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def find_threshold_at_sensitivity(y_true, y_prob, target=TARGET_SENSITIVITY):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.where(tpr >= target)[0]
    return float(thresholds[idx[0]]) if len(idx) > 0 else 0.5


def compute_metrics(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = find_threshold_at_sensitivity(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "auroc"      : float(roc_auc_score(y_true, y_prob)),
        "auprc"      : float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "f1"         : float(f1_score(y_true, y_pred)),
        "sensitivity": float(sens), "specificity": float(spec),
        "ppv"        : float(ppv),  "npv": float(npv),
        "threshold"  : float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_iter=1000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def delong_test(y_true, probs_a, probs_b):
    def compute_midrank(x):
        J = np.argsort(x); Z = x[J]; N = len(x)
        T = np.zeros(N); i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]: j += 1
            T[i:j] = 0.5 * (i + j - 1); i = j
        T2 = np.empty(N); T2[J] = T + 1
        return T2

    def fast_delong(yt, ys):
        m = int(yt.sum()); n = len(yt) - m
        pos = ys[yt == 1]; neg = ys[yt == 0]
        ranks = compute_midrank(np.concatenate([pos, neg]))
        pos_ranks = ranks[:m]
        auc = (pos_ranks.sum() - m * (m + 1) / 2) / (m * n)
        v10 = (pos_ranks - np.arange(1, m + 1)) / n
        v01 = np.array([(pos > ns).mean() + 0.5*(pos == ns).mean() for ns in neg])
        var = np.var(v10, ddof=1) / m + np.var(v01, ddof=1) / n
        return auc, var, v10, v01

    yt = np.array(y_true); pa = np.array(probs_a); pb = np.array(probs_b)
    auc_a, var_a, v10_a, v01_a = fast_delong(yt, pa)
    auc_b, var_b, v10_b, v01_b = fast_delong(yt, pb)
    m = int(yt.sum()); n = len(yt) - m
    cov = (np.cov(v10_a, v10_b, ddof=1)[0, 1] / m +
           np.cov(v01_a, v01_b, ddof=1)[0, 1] / n)
    z = (auc_a - auc_b) / np.sqrt(max(var_a + var_b - 2 * cov, 1e-12))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(auc_a), float(auc_b), float(z), float(p)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    log.info("=" * 70)
    log.info("PHASE 8 — ATTENTION-AUGMENTED CATBOOST HYBRID (FIXED)")
    log.info("FT-Transformer attention weights -> CatBoost augmented features")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("\nLoading Option B datasets...")
    train_df     = pd.read_csv(TRAIN_FILE)
    test_df      = pd.read_csv(TEST_FILE)
    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    n_features   = len(feature_cols)

    X_train_df = train_df[feature_cols]
    y_train    = train_df[TARGET_COL].values
    X_test_df  = test_df[feature_cols]
    y_test     = test_df[TARGET_COL].values

    log.info(f"  Train : {X_train_df.shape} | Sepsis: {y_train.mean():.1%}")
    log.info(f"  Test  : {X_test_df.shape}  | Sepsis: {y_test.mean():.1%}")
    log.info(f"  Features: {n_features}")

    # ── Load FT-T params and scaler ────────────────────────────────────────────
    log.info("\nLoading FT-Transformer params and scaler (Phase 6)...")
    with open(FTT_PARAMS_PATH) as f:
        ftt_params = json.load(f)
    with open(FTT_SCALER_PATH, "rb") as f:
        ftt_scaler = pickle.load(f)

    binary_cols = [c for c in feature_cols
                   if set(X_train_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]

    def scale_for_ftt(df):
        d = df.copy()
        d[num_cols] = ftt_scaler.transform(d[num_cols])
        return d.values.astype(np.float32)

    # ── Build and load FT-T model ──────────────────────────────────────────────
    log.info("\nBuilding FT-Transformer and loading Phase 6 weights...")
    d_block = ftt_params["d_block"]
    n_heads = ftt_params.get("actual_attention_n_heads", ftt_params["attention_n_heads"])
    while d_block % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1: n_heads = 1; break

    ftt_model = FTTransformer(
        n_cont_features         = n_features,
        cat_cardinalities       = None,
        d_out                   = 1,
        n_blocks                = ftt_params["n_blocks"],
        d_block                 = d_block,
        attention_n_heads       = n_heads,
        attention_dropout       = ftt_params["attention_dropout"],
        ffn_d_hidden_multiplier = ftt_params["ffn_d_hidden_multiplier"],
        ffn_dropout             = ftt_params["ffn_dropout"],
        residual_dropout        = ftt_params["residual_dropout"],
    )
    state_dict = torch.load(FTT_WEIGHTS_PATH, map_location=DEVICE)
    ftt_model.load_state_dict(state_dict)
    ftt_model = ftt_model.to(DEVICE)
    ftt_model.eval()
    log.info(f"  Loaded. n_blocks={ftt_params['n_blocks']}, d_block={d_block}, heads={n_heads}")
    log.info("  FT-T FROZEN — read-only.")

    # ── Sanity check ──────────────────────────────────────────────────────────
    log.info("\nSanity check on 8 samples...")
    X_samp  = torch.tensor(scale_for_ftt(X_train_df.iloc[:8]), dtype=torch.float32).to(DEVICE)
    attn_ck = extract_attention_weights(ftt_model, X_samp).numpy()
    log.info(f"  Shape     : {attn_ck.shape}")
    log.info(f"  Row sums  : {attn_ck.sum(axis=1)[:4]}")
    log.info(f"  Range     : [{attn_ck.min():.6f}, {attn_ck.max():.6f}]")
    assert attn_ck.shape == (8, n_features), f"Expected (8,{n_features}), got {attn_ck.shape}"
    assert attn_ck.min() >= 0.0, "Negative attention values"
    log.info("  PASSED.")

    # ── STEP 1: Extract attention features ────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("STEP 1 — Extracting attention features (frozen FT-Transformer)")
    log.info("="*60)

    log.info("  Train set...")
    t0             = time.time()
    X_train_scaled = scale_for_ftt(X_train_df)
    attn_train     = extract_attention_dataset(ftt_model, X_train_scaled)
    log.info(f"  Done: {attn_train.shape} | {time.time()-t0:.1f}s")

    log.info("  Test set...")
    t0            = time.time()
    X_test_scaled = scale_for_ftt(X_test_df)
    attn_test     = extract_attention_dataset(ftt_model, X_test_scaled)
    log.info(f"  Done: {attn_test.shape}  | {time.time()-t0:.1f}s")

    attn_cols = [f"attn_{col}" for col in feature_cols]
    attn_df_tr = pd.DataFrame(attn_train, columns=attn_cols); attn_df_tr["split"] = "train"
    attn_df_te = pd.DataFrame(attn_test,  columns=attn_cols); attn_df_te["split"] = "test"
    pd.concat([attn_df_tr, attn_df_te], ignore_index=True).to_csv(
        RESULTS_DIR / "phase8_attention_features.csv", index=False)
    log.info("  Saved -> phase8_attention_features.csv")

    # ── STEP 2: Augmented feature matrices ─────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("STEP 2 — Augmented matrices: [original 225 || attention 225] = 450")
    log.info("="*60)

    aug_feature_names = feature_cols + attn_cols
    X_train_aug_df    = pd.DataFrame(
        np.concatenate([X_train_df.values, attn_train], axis=1), columns=aug_feature_names)
    X_test_aug_df     = pd.DataFrame(
        np.concatenate([X_test_df.values,  attn_test],  axis=1), columns=aug_feature_names)
    log.info(f"  Total features: {len(aug_feature_names)}")

    with open(RESULTS_DIR / "phase8_augmented_feature_names.json", "w") as f:
        json.dump(aug_feature_names, f, indent=2)

    # ── STEP 3: Train augmented CatBoost ──────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("STEP 3 — Training Attention-Augmented CatBoost")
    log.info("="*60)

    with open(CB_PARAMS_PATH) as f:
        cb_best = json.load(f)["best_params"]
    log.info(f"  Params (Phase 2 Optuna-tuned): {cb_best}")

    augmented_catboost = CatBoostClassifier(
        iterations=cb_best["iterations"], learning_rate=cb_best["learning_rate"],
        depth=cb_best["depth"], l2_leaf_reg=cb_best["l2_leaf_reg"],
        bagging_temperature=cb_best["bagging_temperature"],
        random_strength=cb_best["random_strength"], border_count=cb_best["border_count"],
        class_weights=[1.0, cb_best["class_weight_pos"]],
        eval_metric="AUC", random_seed=RANDOM_SEED, verbose=100, early_stopping_rounds=50,
    )
    t0 = time.time()
    augmented_catboost.fit(X_train_aug_df, y_train, eval_set=(X_test_aug_df, y_test))
    log.info(f"  Done in {(time.time()-t0)/60:.1f} min")
    augmented_catboost.save_model(str(MODELS_DIR / "phase8_catboost_augmented.cbm"))
    log.info("  Saved -> phase8_catboost_augmented.cbm")

    # ── STEP 4: Evaluate ──────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("STEP 4 — Test set evaluation (n=633)")
    log.info("="*60)

    aug_test_probs     = augmented_catboost.predict_proba(X_test_aug_df)[:, 1]
    metrics            = compute_metrics(y_test, aug_test_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, aug_test_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, aug_test_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo;  metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo;  metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier Score : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}  NPV: {metrics['npv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    cb_baseline_probs = np.load(CB_TEST_PROBS_PATH)
    cb_baseline_auroc = roc_auc_score(y_test, cb_baseline_probs)
    log.info(f"\n  Phase 2 Tuned CatBoost AUROC : {cb_baseline_auroc:.4f}")
    log.info(f"  Attention-Aug CatBoost AUROC : {metrics['auroc']:.4f}")
    log.info(f"  Delta                        : {metrics['auroc']-cb_baseline_auroc:+.4f}")

    auc_aug, auc_cb, z_stat, p_val = delong_test(y_test, aug_test_probs, cb_baseline_probs)
    log.info(f"\n  DeLong (Aug-CB vs Tuned-CB): z={z_stat:.4f}, p={p_val:.4f} "
             f"({'YES p<0.05' if p_val<0.05 else 'NO p>=0.05'})")

    delong_results = {
        "aug_catboost_auroc"  : float(auc_aug), "tuned_catboost_auroc": float(auc_cb),
        "z_statistic"         : float(z_stat),  "p_value": float(p_val),
        "significant"         : bool(p_val < 0.05),
        "direction"           : "Aug-CatBoost better" if auc_aug > auc_cb else "Tuned-CatBoost better",
    }

    # Threshold table
    thresh_rows = []
    log.info(f"\n  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7}")
    log.info("  " + "-"*60)
    for target in [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]:
        thr = find_threshold_at_sensitivity(y_test, aug_test_probs, target)
        m   = compute_metrics(y_test, aug_test_probs, thr)
        thresh_rows.append({
            "target_sensitivity": target, "threshold": round(thr,4),
            "sensitivity": round(m["sensitivity"],4), "specificity": round(m["specificity"],4),
            "ppv": round(m["ppv"],4), "npv": round(m["npv"],4), "f1": round(m["f1"],4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
        log.info(f"  {target:>8.2f} {thr:>8.4f} {m['sensitivity']:>7.4f} "
                 f"{m['specificity']:>7.4f} {m['ppv']:>7.4f} {m['npv']:>7.4f} {m['f1']:>7.4f}")
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "phase8_threshold_table.csv", index=False)

    # Feature importances
    mean_attn     = attn_train.mean(axis=0)
    top10_idx     = np.argsort(mean_attn)[::-1][:10]
    cb_imp        = augmented_catboost.get_feature_importance()
    importance_df = pd.DataFrame({"feature": aug_feature_names, "importance": cb_imp})\
                      .sort_values("importance", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "phase8_feature_importance.csv", index=False)

    log.info("\n  Top 10 by mean attention weight:")
    for rank, idx in enumerate(top10_idx):
        log.info(f"    {rank+1:>2}. {feature_cols[idx]:<40} {mean_attn[idx]:.6f}")

    log.info("\n  Top 10 CatBoost importances (augmented):")
    for _, row in importance_df.head(10).iterrows():
        tag = "[ATTN]" if row["feature"].startswith("attn_") else "[ORIG]"
        log.info(f"    {tag} {row['feature']:<45} {row['importance']:.4f}")

    # Save JSON results
    runtime = (time.time() - t_start) / 60
    full_results = {
        "model": "Attention-Augmented CatBoost (Phase 8)",
        "phase": "Phase 8", "timestamp": datetime.now().isoformat(),
        "dataset": "Option B (infection-only)",
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_original_features": int(n_features),
        "n_attention_features": int(n_features),
        "n_total_features": int(n_features * 2),
        "ftt_config": {"n_blocks": ftt_params["n_blocks"], "d_block": d_block,
                       "n_heads": n_heads, "frozen": True, "weights_from": "Phase 6"},
        "top_attention_features": [
            {"rank": i+1, "feature": feature_cols[idx], "mean_attention": float(mean_attn[idx])}
            for i, idx in enumerate(top10_idx)
        ],
        "test_metrics": metrics,
        "delong_vs_tuned_catboost": delong_results,
        "runtime_minutes": float(runtime),
    }
    with open(RESULTS_DIR / "phase8_metrics.json", "w") as f:
        json.dump(full_results, f, indent=2)
    with open(RESULTS_DIR / "phase8_delong_results.json", "w") as f:
        json.dump(delong_results, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nGenerating figures...")
    COLORS = {"aug": "#00897B", "catboost": "#1565C0", "ftt": "#6A1B9A", "random": "#9E9E9E"}

    ftt_npz        = np.load(SEPSIS_ML / "dl" / "results" / "phase6_fttransformer_predictions.npz")
    ftt_test_probs = ftt_npz["test_probs"]

    # Fig 1: ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Phase 8 — Attention-Augmented CatBoost vs Baselines\n"
                 f"Option B (n_test={len(y_test)})", fontsize=13, fontweight="bold")
    for probs, label, color, lw, ls in [
        (aug_test_probs,    f"Attn-Aug CatBoost AUROC={metrics['auroc']:.4f}", COLORS["aug"],      2.5, "-"),
        (cb_baseline_probs, f"Tuned CatBoost     AUROC={cb_baseline_auroc:.4f}", COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs,    f"FT-Transformer     AUROC={roc_auc_score(y_test, ftt_test_probs):.4f}", COLORS["ftt"], 1.2, "-."),
    ]:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)
    axes[0].plot([0,1],[0,1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, aug_test_probs)[:2], alpha=0.07, color=COLORS["aug"])
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    for probs, label, color, lw, ls in [
        (aug_test_probs,    f"Attn-Aug CatBoost AUPRC={average_precision_score(y_test, aug_test_probs):.4f}", COLORS["aug"],      2.5, "-"),
        (cb_baseline_probs, f"Tuned CatBoost     AUPRC={average_precision_score(y_test, cb_baseline_probs):.4f}", COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs,    f"FT-Transformer     AUPRC={average_precision_score(y_test, ftt_test_probs):.4f}", COLORS["ftt"],      1.2, "-."),
    ]:
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls, label=label)
    axes[1].axhline(y_test.mean(), color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({y_test.mean():.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9); axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_roc_pr.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_roc_pr.png")

    # Fig 2: Calibration
    fig, ax = plt.subplots(figsize=(7, 6))
    for probs, label, color in [
        (aug_test_probs,    f"Attn-Aug CatBoost (Brier={metrics['brier_score']:.4f})", COLORS["aug"]),
        (cb_baseline_probs, f"Tuned CatBoost (Brier={brier_score_loss(y_test,cb_baseline_probs):.4f})", COLORS["catboost"]),
        (ftt_test_probs,    f"FT-Transformer (Brier={brier_score_loss(y_test,ftt_test_probs):.4f})", COLORS["ftt"]),
    ]:
        pt, pp = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(pp, pt, "o-", color=color, lw=2, label=label)
    ax.plot([0,1],[0,1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration — Phase 8 vs Baselines", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_calibration.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_calibration.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_calibration.png")

    # Fig 3: Confusion matrix
    y_pred_aug = (aug_test_probs >= metrics["threshold"]).astype(int)
    cm = confusion_matrix(y_test, y_pred_aug)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"]); ax.set_yticklabels(["Non-sepsis","Sepsis"])
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>thresh_cm else "black", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Attn-Aug CatBoost — Confusion Matrix\n"
                 f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
                 f"Thresh={metrics['threshold']:.4f}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_confusion_matrix.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_confusion_matrix.png")

    # Fig 4: Attention heatmap
    top30_idx   = np.argsort(mean_attn)[::-1][:30]
    top30_names = [feature_cols[i] for i in top30_idx]
    attn_top30  = attn_train[:100, top30_idx]
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(attn_top30.T, aspect="auto", cmap="viridis", norm=mcolors.PowerNorm(gamma=0.5))
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_yticks(range(30)); ax.set_yticklabels(top30_names, fontsize=7)
    ax.set_xlabel("Patient index (first 100 training patients)")
    ax.set_ylabel("Feature (top 30 by mean attention)")
    ax.set_title("FT-Transformer Attention Weights — Top 30 Features x First 100 Patients\n"
                 "Last block, CLS-token row, head-averaged", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_attention_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_attention_heatmap.png")

    # Fig 5: Threshold sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 8 — Threshold Sensitivity Analysis", fontsize=12, fontweight="bold")
    thr_vals = thresh_df["threshold"].values
    axes[0].plot(thr_vals, thresh_df["sensitivity"].values, "o-", color=COLORS["aug"],      lw=2, label="Sensitivity")
    axes[0].plot(thr_vals, thresh_df["specificity"].values, "s-", color=COLORS["catboost"], lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3); axes[0].spines[["top","right"]].set_visible(False)
    axes[1].plot(thr_vals, thresh_df["ppv"].values, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, thresh_df["npv"].values, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3); axes[1].spines[["top","right"]].set_visible(False)
    axes[2].plot(thr_vals, thresh_df["f1"].values, "o-", color=COLORS["ftt"], lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1"); axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].grid(True, alpha=0.3); axes[2].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_threshold_sensitivity.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_threshold_sensitivity.png")

    # Fig 6: Feature importance breakdown
    from matplotlib.patches import Patch
    top20_imp  = importance_df.head(20)
    bar_colors = [COLORS["aug"] if r["feature"].startswith("attn_") else COLORS["catboost"]
                  for _, r in top20_imp.iterrows()]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top20_imp)), top20_imp["importance"].values, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top20_imp)))
    ax.set_yticklabels(
        [f"{'[A] ' if r['feature'].startswith('attn_') else '[O] '}{r['feature'].replace('attn_','')}"
         for _, r in top20_imp.iterrows()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("CatBoost Feature Importance")
    ax.set_title("Top 20 Feature Importances\n[A] = Attention  |  [O] = Original", fontweight="bold")
    ax.legend(handles=[Patch(facecolor=COLORS["aug"], label="Attention [A]"),
                        Patch(facecolor=COLORS["catboost"], label="Original [O]")],
              loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_feature_importance.pdf", dpi=150, bbox_inches="tight")
    plt.close(); log.info("  Saved -> phase8_feature_importance.png")

    # ── Final summary ─────────────────────────────────────────────────────────
    runtime = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("PHASE 8 COMPLETE")
    log.info("=" * 70)
    log.info(f"  AUROC      : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC      : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier      : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity: {metrics['specificity']:.4f}")
    log.info(f"  F1         : {metrics['f1']:.4f}")
    log.info(f"  DeLong vs Phase 2 CB: z={z_stat:.4f}, p={p_val:.4f}")
    log.info(f"  Features   : {n_features} orig + {n_features} attn = {n_features*2} total")
    log.info(f"  Runtime    : {runtime:.1f} min")
    log.info(f"\n  Models  -> {MODELS_DIR}")
    log.info(f"  Results -> {RESULTS_DIR}")
    log.info(f"  Figures -> {FIGURES_DIR}")
    log.info(f"  Log     -> {log_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()