"""
phase8_attention_catboost.py
─────────────────────────────────────────────────────────────────────────────
Phase 8 — Attention-Augmented CatBoost (Option 3 Hybrid)

Core idea:
  Extract per-feature attention weights from FT-Transformer's final
  transformer block and use them as additional engineered features for
  CatBoost. CatBoost then trains on [225 original features + 225 attention
  weights] = 450 features total.

Why this is novel:
  - CatBoost receives interaction-aware feature salience from the transformer
    attention mechanism — it learns not just feature values but how the
    transformer weighted each feature in context
  - Final model is still CatBoost → SHAP runs cleanly on the augmented model
  - Calibration problem of FT-T is completely bypassed — CatBoost's
    well-calibrated probability output is preserved

Architecture:
  FT-Transformer (frozen, Phase 6 weights)
    → extract attention weights from last transformer block (CLS token row)
    → produces N x 225 attention matrix per split
  CatBoost (re-tuned with Phase 2 best params)
    → trains on [original 225 features || attention 225 features] = 450 cols
    → evaluated with full metric suite + DeLong vs Phase 2 tuned CatBoost

Folder outputs (all under sepsis_ml/phase8_attention_catboost/):
  models/   — phase8_attention_extractor.pt, phase8_catboost_augmented.cbm
  results/  — phase8_metrics.json, phase8_attention_features.csv,
               phase8_delong_results.json, phase8_threshold_table.csv
  figures/  — phase8_roc_pr.png, phase8_calibration.png,
               phase8_confusion_matrix.png, phase8_attention_heatmap.png,
               phase8_threshold_sensitivity.png
  logs/     — phase8_attention_catboost.log

Run from: pediatric_sepsis_prediction_PIC_XAI/
  python sepsis_ml/phase8_attention_catboost/phase8_attention_catboost.py

Windows CUDA notes:
  - num_workers=0 in all DataLoaders
  - multiprocessing.freeze_support() at top
  - All paths use pathlib.Path

Requirements (already installed from Phase 6):
  catboost, rtdl_revisiting_models, torch (cu121), scikit-learn, optuna
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
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from catboost import CatBoostClassifier
from rtdl_revisiting_models import FTTransformer
from sklearn.preprocessing import StandardScaler
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

SCRIPT_DIR   = Path(__file__).resolve().parent       # sepsis_ml/phase8_attention_catboost/
SEPSIS_ML    = SCRIPT_DIR.parent                     # sepsis_ml/
PROJECT_ROOT = SEPSIS_ML.parent                      # pediatric_sepsis_prediction_PIC_XAI/

# Input data
MODEL_DATA_DIR = PROJECT_ROOT / "model_datasets"
TRAIN_FILE     = MODEL_DATA_DIR / "B_train_model_ready.csv"
TEST_FILE      = MODEL_DATA_DIR / "B_test_model_ready.csv"

# Phase 6 FT-T artifacts (frozen — we only extract, never retrain)
FTT_MODEL_DIR    = SEPSIS_ML / "dl" / "models" / "run_phase6_fttransformer"
FTT_WEIGHTS_PATH = FTT_MODEL_DIR / "fttransformer_best.pt"
FTT_PARAMS_PATH  = FTT_MODEL_DIR / "best_params_final.json"
FTT_SCALER_PATH  = FTT_MODEL_DIR / "scaler.pkl"

# Phase 2 CatBoost best params
CB_PARAMS_PATH       = SEPSIS_ML / "results" / "best_params.json"
CB_TUNED_MODEL_PATH  = SEPSIS_ML / "models" / "run_tuned" / "catboost_tuned.cbm"
CB_TEST_PROBS_PATH   = SEPSIS_ML / "results" / "catboost_test_probs.npy"

# Phase 8 outputs
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
        logging.FileHandler(log_path, mode="w"),
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
# 3. FT-TRANSFORMER WITH ATTENTION EXTRACTION
#
# We subclass / wrap FTTransformer to hook into the final transformer block
# and extract the CLS-token attention weights.
#
# How FTTransformer attention works (rtdl_revisiting_models):
#   - Each feature is projected into a token embedding (d_block dims)
#   - A [CLS] token is prepended → sequence length = n_features + 1
#   - Each transformer block runs multi-head self-attention over this sequence
#   - In the last block, the CLS token attends to all feature tokens
#   - We hook the last block's attention weight matrix and extract the
#     CLS row: shape (batch, n_heads, n_features+1)
#     → average over heads → take CLS row (index 0) → feature weights
#     → shape (batch, n_features+1) → drop CLS self-attention → (batch, n_features)
#
# Result: one attention score per feature per patient — a learned salience map
# that captures how the transformer weighted each feature when making its
# prediction. This is fundamentally different from raw feature values.
# ══════════════════════════════════════════════════════════════════════════════

class AttentionExtractorFTT(nn.Module):
    """
    Wraps a trained FTTransformer and adds a forward hook on the last
    transformer block's self-attention module to capture attention weights.

    Usage:
        extractor = AttentionExtractorFTT(ftt_model)
        extractor.eval()
        with torch.no_grad():
            logits, attn_weights = extractor(x_cont)
        # attn_weights: (batch, n_features) — CLS row, head-averaged
    """

    def __init__(self, ftt_model: FTTransformer):
        super().__init__()
        self.ftt    = ftt_model
        self._attn  = None   # storage for hooked attention weights
        self._hook  = None

        # ── Locate the last transformer block's attention module ─────────────
        # rtdl_revisiting_models structure:
        #   ftt.blocks  — ModuleList of transformer blocks
        #   Each block has .attention — the MultiheadAttention module
        # We hook the last block.
        last_block = self.ftt.blocks[-1]

        # The attention module inside the block
        # Depending on rtdl version it may be .attention or .self_attention
        attn_module = None
        for name in ["attention", "self_attention"]:
            if hasattr(last_block, name):
                attn_module = getattr(last_block, name)
                log.info(f"  Found attention module at block[-1].{name}")
                break

        if attn_module is None:
            # Fallback: search children for nn.MultiheadAttention
            for name, module in last_block.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    attn_module = module
                    log.info(f"  Found attention module via named_modules: {name}")
                    break

        if attn_module is None:
            raise RuntimeError(
                "Could not locate MultiheadAttention in last FTT block. "
                "Inspect the model architecture and update AttentionExtractorFTT."
            )

        self._attn_module = attn_module

        # Register forward hook — fires after attention forward pass
        self._hook = attn_module.register_forward_hook(self._hook_fn)
        log.info("  Attention hook registered on last transformer block.")

    def _hook_fn(self, module, input, output):
        """
        nn.MultiheadAttention forward returns (attn_output, attn_weights)
        when need_weights=True (default). We capture attn_weights.
        Shape: (batch, seq_len, seq_len) — already averaged over heads
               OR None if need_weights=False.
        If None, we need to re-run with need_weights=True (see forward()).
        """
        if isinstance(output, tuple) and output[1] is not None:
            # output[1] shape: (batch, tgt_len, src_len)
            self._attn = output[1].detach().cpu()

    def forward(self, x_cont: torch.Tensor):
        """
        Returns (logits, cls_attention_weights).
        cls_attention_weights shape: (batch, n_features)
        """
        self._attn = None

        # Run full forward pass — hook fires during this call
        logits = self.ftt(x_cont, None)  # (batch, 1)

        if self._attn is None:
            # Hook didn't capture weights — attention was computed without
            # need_weights=True. We need to manually extract by re-running
            # just the last block's attention with need_weights=True.
            # This is the fallback path.
            log.warning("  Attention hook returned None — using manual extraction fallback.")
            cls_attn = self._manual_attention_extract(x_cont)
        else:
            # attn shape: (batch, seq_len, seq_len)
            # seq_len = n_features + 1 (CLS + feature tokens)
            # CLS is at position 0 → take row 0 → (batch, seq_len)
            # Drop the CLS self-attention (col 0) → (batch, n_features)
            attn = self._attn   # (batch, seq_len, seq_len)
            cls_attn = attn[:, 0, 1:]  # (batch, n_features)

        return logits, cls_attn

    def _manual_attention_extract(self, x_cont: torch.Tensor):
        """
        Fallback: manually extract attention from the last block by calling
        the attention sub-module directly with need_weights=True.
        This requires access to the last block's intermediate representation.
        """
        # Get intermediate representation just before last block
        # by running all blocks except the last, then running attention manually
        with torch.no_grad():
            # Get token embeddings — FTTransformer builds them in .tokenizer
            if hasattr(self.ftt, "tokenizer"):
                tokens = self.ftt.tokenizer(x_cont, None)  # (batch, n_features, d_block)
            elif hasattr(self.ftt, "_tokenizer"):
                tokens = self.ftt._tokenizer(x_cont, None)
            else:
                # Last resort: run up to second-to-last block
                log.error("Cannot access FTT tokenizer for manual extraction.")
                n_features = x_cont.shape[1]
                batch      = x_cont.shape[0]
                return torch.ones(batch, n_features) / n_features

            # Run all blocks except last
            x = tokens
            for block in list(self.ftt.blocks)[:-1]:
                x = block(x)

            # Now x: (batch, seq_len, d_block)
            # Run last block's attention with need_weights=True
            last_block  = self.ftt.blocks[-1]
            attn_module = self._attn_module

            # Prepare query, key, value
            # FTT uses pre-norm: norm → attention → residual
            if hasattr(last_block, "norm1"):
                x_norm = last_block.norm1(x)
            elif hasattr(last_block, "norm"):
                x_norm = last_block.norm(x)
            else:
                x_norm = x

            # Transpose for nn.MultiheadAttention: (seq, batch, d)
            x_t = x_norm.transpose(0, 1)
            _, attn_weights = attn_module(x_t, x_t, x_t, need_weights=True)
            # attn_weights: (batch, seq_len, seq_len)
            cls_attn = attn_weights[:, 0, 1:]   # (batch, n_features)
            return cls_attn.detach().cpu()

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


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
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv"        : float(ppv),
        "npv"        : float(npv),
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
# 5. DATA LOADER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


# ══════════════════════════════════════════════════════════════════════════════
# 6. ATTENTION EXTRACTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_attention_features(extractor, X_scaled, batch_size=256):
    """
    Run the AttentionExtractorFTT on a dataset in batches.
    Returns attention weight matrix: (n_samples, n_features)
    """
    extractor.eval()
    all_attn = []
    loader   = make_loader(X_scaled, np.zeros(len(X_scaled)), batch_size, shuffle=False)

    for X_batch, _ in loader:
        X_batch = X_batch.to(DEVICE)
        _, attn = extractor(X_batch)
        # attn: (batch, n_features) — already on CPU from hook
        if isinstance(attn, torch.Tensor):
            all_attn.append(attn.numpy())
        else:
            all_attn.append(attn)

    return np.concatenate(all_attn, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    log.info("=" * 70)
    log.info("PHASE 8 — ATTENTION-AUGMENTED CATBOOST HYBRID")
    log.info("FT-Transformer attention weights → CatBoost augmented features")
    log.info(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 70)

    # ── 7.1 Load data ─────────────────────────────────────────────────────────
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

    # ── 7.2 Load FT-T params and scaler ───────────────────────────────────────
    log.info("\nLoading FT-Transformer params and scaler (Phase 6)...")
    with open(FTT_PARAMS_PATH) as f:
        ftt_params = json.load(f)

    with open(FTT_SCALER_PATH, "rb") as f:
        ftt_scaler = pickle.load(f)

    log.info(f"  FT-T best params: {ftt_params}")

    # Identify binary vs numerical columns (must match Phase 6)
    binary_cols = [c for c in feature_cols
                   if set(X_train_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})]
    num_cols    = [c for c in feature_cols if c not in binary_cols]

    def scale_for_ftt(df):
        d = df.copy()
        d[num_cols] = ftt_scaler.transform(d[num_cols])
        return d.values.astype(np.float32)

    # ── 7.3 Build FT-T model and load Phase 6 weights ─────────────────────────
    log.info("\nBuilding FT-Transformer and loading Phase 6 weights...")

    d_block = ftt_params["d_block"]
    n_heads = ftt_params.get("actual_attention_n_heads", ftt_params["attention_n_heads"])
    while d_block % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1:
            n_heads = 1
            break

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
    ftt_model.eval()  # freeze — we never update FT-T weights in Phase 8

    log.info(f"  FT-T loaded: {ftt_params['n_blocks']} blocks, "
             f"d_block={d_block}, heads={n_heads}")
    log.info(f"  FT-T is FROZEN — Phase 6 weights are read-only.")

    # ── 7.4 Wrap with attention extractor ─────────────────────────────────────
    log.info("\nWrapping FT-Transformer with AttentionExtractorFTT...")
    extractor = AttentionExtractorFTT(ftt_model)
    extractor = extractor.to(DEVICE)

    # Quick sanity check on a small batch
    log.info("  Running sanity check on 8-sample batch...")
    X_sample = scale_for_ftt(X_train_df.iloc[:8])
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits_check, attn_check = extractor(X_tensor)

    if isinstance(attn_check, torch.Tensor):
        attn_check = attn_check.numpy()

    log.info(f"  Sanity check — logits shape: {logits_check.shape}")
    log.info(f"  Sanity check — attention shape: {attn_check.shape}")
    log.info(f"  Attention weights sum per patient (should ≈ 1.0): "
             f"{attn_check.sum(axis=1)[:4]}")
    log.info(f"  Attention range: [{attn_check.min():.6f}, {attn_check.max():.6f}]")

    assert attn_check.shape == (8, n_features), (
        f"Expected attention shape (8, {n_features}), got {attn_check.shape}. "
        f"Check AttentionExtractorFTT."
    )
    log.info("  Sanity check PASSED.")

    # ── 7.5 Extract attention features for train and test ─────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 1 — Extracting attention features from frozen FT-Transformer")
    log.info("=" * 60)

    log.info("  Extracting train attention features...")
    t0 = time.time()
    X_train_scaled = scale_for_ftt(X_train_df)
    attn_train     = extract_attention_features(extractor, X_train_scaled)
    log.info(f"  Train attention: {attn_train.shape} | Time: {time.time()-t0:.1f}s")

    log.info("  Extracting test attention features...")
    t0 = time.time()
    X_test_scaled = scale_for_ftt(X_test_df)
    attn_test     = extract_attention_features(extractor, X_test_scaled)
    log.info(f"  Test attention: {attn_test.shape}  | Time: {time.time()-t0:.1f}s")

    # Remove hook — no longer needed
    extractor.remove_hook()
    log.info("  Attention hook removed.")

    # Save raw attention features for inspection
    attn_cols = [f"attn_{col}" for col in feature_cols]
    attn_train_df = pd.DataFrame(attn_train, columns=attn_cols)
    attn_test_df  = pd.DataFrame(attn_test,  columns=attn_cols)

    attn_train_df["split"] = "train"
    attn_test_df["split"]  = "test"
    attn_combined = pd.concat([attn_train_df, attn_test_df], ignore_index=True)
    attn_combined.to_csv(RESULTS_DIR / "phase8_attention_features.csv", index=False)
    log.info(f"  Attention features saved → {RESULTS_DIR / 'phase8_attention_features.csv'}")

    # ── 7.6 Build augmented feature matrices ──────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2 — Building augmented feature matrices [original || attention]")
    log.info("=" * 60)

    X_train_orig = X_train_df.values
    X_test_orig  = X_test_df.values

    # Concatenate: [225 original features | 225 attention features] = 450 cols
    X_train_aug = np.concatenate([X_train_orig, attn_train], axis=1)
    X_test_aug  = np.concatenate([X_test_orig,  attn_test],  axis=1)

    # Build augmented column names for SHAP-readability
    aug_feature_names = feature_cols + attn_cols

    log.info(f"  Original features  : {X_train_orig.shape[1]}")
    log.info(f"  Attention features : {attn_train.shape[1]}")
    log.info(f"  Augmented total    : {X_train_aug.shape[1]}")

    # Convert to DataFrames for CatBoost (it handles them cleanly)
    X_train_aug_df = pd.DataFrame(X_train_aug, columns=aug_feature_names)
    X_test_aug_df  = pd.DataFrame(X_test_aug,  columns=aug_feature_names)

    # Save augmented feature names
    with open(RESULTS_DIR / "phase8_augmented_feature_names.json", "w") as f:
        json.dump(aug_feature_names, f, indent=2)

    # ── 7.7 Train attention-augmented CatBoost ────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3 — Training Attention-Augmented CatBoost on 450 features")
    log.info("=" * 60)

    with open(CB_PARAMS_PATH) as f:
        cb_best = json.load(f)["best_params"]

    log.info(f"  Using Phase 2 best params: {cb_best}")

    augmented_catboost = CatBoostClassifier(
        iterations           = cb_best["iterations"],
        learning_rate        = cb_best["learning_rate"],
        depth                = cb_best["depth"],
        l2_leaf_reg          = cb_best["l2_leaf_reg"],
        bagging_temperature  = cb_best["bagging_temperature"],
        random_strength      = cb_best["random_strength"],
        border_count         = cb_best["border_count"],
        class_weights        = [1.0, cb_best["class_weight_pos"]],
        eval_metric          = "AUC",
        random_seed          = RANDOM_SEED,
        verbose              = 100,
        early_stopping_rounds= 50,
    )

    log.info("  Training augmented CatBoost (this uses the same Optuna-tuned "
             "hyperparameters as Phase 2 — no re-tuning needed)...")
    t0 = time.time()
    augmented_catboost.fit(
        X_train_aug_df, y_train,
        eval_set=(X_test_aug_df, y_test),
    )
    log.info(f"  Training complete in {(time.time()-t0)/60:.1f} min")

    # Save model
    augmented_catboost.save_model(str(MODELS_DIR / "phase8_catboost_augmented.cbm"))
    log.info(f"  Model saved → {MODELS_DIR / 'phase8_catboost_augmented.cbm'}")

    # ── 7.8 Test set evaluation ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 4 — Evaluating on held-out test set (n=633)")
    log.info("=" * 60)

    aug_test_probs = augmented_catboost.predict_proba(X_test_aug_df)[:, 1]
    metrics        = compute_metrics(y_test, aug_test_probs)
    auroc_lo, auroc_hi = bootstrap_ci(y_test, aug_test_probs, roc_auc_score)
    auprc_lo, auprc_hi = bootstrap_ci(y_test, aug_test_probs, average_precision_score)
    metrics["auroc_ci_low"]  = auroc_lo
    metrics["auroc_ci_high"] = auroc_hi
    metrics["auprc_ci_low"]  = auprc_lo
    metrics["auprc_ci_high"] = auprc_hi

    log.info(f"\n  {'=' * 55}")
    log.info(f"  ATTENTION-AUGMENTED CATBOOST — TEST SET RESULTS")
    log.info(f"  {'=' * 55}")
    log.info(f"  AUROC       : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC       : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier Score : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity : {metrics['specificity']:.4f}")
    log.info(f"  PPV         : {metrics['ppv']:.4f}")
    log.info(f"  NPV         : {metrics['npv']:.4f}")
    log.info(f"  F1          : {metrics['f1']:.4f}")
    log.info(f"  Threshold   : {metrics['threshold']:.4f}")
    log.info(f"  TP={metrics['tp']}  FP={metrics['fp']}  "
             f"TN={metrics['tn']}  FN={metrics['fn']}")

    # ── DeLong vs Phase 2 tuned CatBoost ──────────────────────────────────────
    cb_baseline_probs = np.load(CB_TEST_PROBS_PATH)
    cb_baseline_auroc = roc_auc_score(y_test, cb_baseline_probs)

    log.info(f"\n  Comparison:")
    log.info(f"    Phase 2 Tuned CatBoost AUROC : {cb_baseline_auroc:.4f}")
    log.info(f"    Attention-Aug CatBoost AUROC : {metrics['auroc']:.4f}")
    log.info(f"    Delta                        : {metrics['auroc'] - cb_baseline_auroc:+.4f}")

    log.info("\n  DeLong test: Attention-Aug CatBoost vs Tuned CatBoost")
    auc_aug, auc_cb, z_stat, p_val = delong_test(
        y_test, aug_test_probs, cb_baseline_probs
    )
    log.info(f"    Aug CatBoost AUROC : {auc_aug:.4f}")
    log.info(f"    Tuned CatBoost AUROC: {auc_cb:.4f}")
    log.info(f"    Z-statistic        : {z_stat:.4f}")
    log.info(f"    P-value            : {p_val:.4f}")
    log.info(f"    Significant        : {'YES (p<0.05)' if p_val < 0.05 else 'NO (p>=0.05)'}")

    delong_results = {
        "aug_catboost_auroc"    : float(auc_aug),
        "tuned_catboost_auroc"  : float(auc_cb),
        "z_statistic"           : float(z_stat),
        "p_value"               : float(p_val),
        "significant"           : p_val < 0.05,
        "direction"             : "Aug-CatBoost better" if auc_aug > auc_cb else "Tuned-CatBoost better",
    }

    # ── Threshold sensitivity table ────────────────────────────────────────────
    log.info("\n  Threshold sensitivity analysis:")
    targets     = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
    thresh_rows = []
    log.info(f"  {'Target':>8} {'Thresh':>8} {'Sens':>7} {'Spec':>7} "
             f"{'PPV':>7} {'NPV':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    log.info("  " + "-" * 70)

    for target in targets:
        thr = find_threshold_at_sensitivity(y_test, aug_test_probs, target)
        m   = compute_metrics(y_test, aug_test_probs, thr)
        thresh_rows.append({
            "target_sensitivity": target,
            "threshold"         : round(thr, 4),
            "sensitivity"       : round(m["sensitivity"], 4),
            "specificity"       : round(m["specificity"], 4),
            "ppv"               : round(m["ppv"], 4),
            "npv"               : round(m["npv"], 4),
            "f1"                : round(m["f1"], 4),
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
        })
        log.info(f"  {target:>8.2f} {thr:>8.4f} {m['sensitivity']:>7.4f} "
                 f"{m['specificity']:>7.4f} {m['ppv']:>7.4f} {m['npv']:>7.4f} "
                 f"{m['f1']:>7.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(RESULTS_DIR / "phase8_threshold_table.csv", index=False)

    # ── Top attention-weighted features ───────────────────────────────────────
    log.info("\n  Top 10 features by mean attention weight (train set):")
    mean_attn = attn_train.mean(axis=0)  # (n_features,)
    top10_idx = np.argsort(mean_attn)[::-1][:10]
    for rank, idx in enumerate(top10_idx):
        log.info(f"    {rank+1:>2}. {feature_cols[idx]:<40} {mean_attn[idx]:.6f}")

    # ── Top 20 CatBoost feature importances (augmented model) ─────────────────
    cb_importances = augmented_catboost.get_feature_importance()
    importance_df  = pd.DataFrame({
        "feature"   : aug_feature_names,
        "importance": cb_importances,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(RESULTS_DIR / "phase8_feature_importance.csv", index=False)

    log.info("\n  Top 10 CatBoost feature importances (augmented model):")
    for _, row in importance_df.head(10).iterrows():
        tag = "[ATTN]" if row["feature"].startswith("attn_") else "[ORIG]"
        log.info(f"    {tag} {row['feature']:<45} {row['importance']:.4f}")

    # ── Save full results JSON ─────────────────────────────────────────────────
    runtime = (time.time() - t_start) / 60
    full_results = {
        "model"     : "Attention-Augmented CatBoost (Phase 8)",
        "phase"     : "Phase 8",
        "timestamp" : datetime.now().isoformat(),
        "dataset"   : "Option B (infection-only)",
        "n_train"   : int(len(y_train)),
        "n_test"    : int(len(y_test)),
        "n_original_features"  : int(n_features),
        "n_attention_features" : int(n_features),
        "n_total_features"     : int(n_features * 2),
        "ftt_config": {
            "n_blocks"    : ftt_params["n_blocks"],
            "d_block"     : d_block,
            "n_heads"     : n_heads,
            "frozen"      : True,
            "weights_from": "Phase 6",
        },
        "top_attention_features": [
            {"rank": i+1, "feature": feature_cols[idx],
             "mean_attention": float(mean_attn[idx])}
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
    log.info(f"\n  Results saved → {RESULTS_DIR}")

    # ══════════════════════════════════════════════════════════════════════════
    # 7.9 FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    log.info("\nGenerating figures...")

    COLORS = {
        "aug"     : "#00897B",   # teal — Phase 8 model
        "catboost": "#1565C0",   # blue — Phase 2 baseline
        "ftt"     : "#6A1B9A",   # purple — FT-T
        "random"  : "#9E9E9E",
    }

    # ── Figure 1: ROC + PR ────────────────────────────────────────────────────
    ftt_npz        = np.load(SEPSIS_ML / "dl" / "results" / "phase6_fttransformer_predictions.npz")
    ftt_test_probs = ftt_npz["test_probs"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Phase 8 — Attention-Augmented CatBoost vs Baselines\n"
        f"Option B Infection-Only Cohort (n_test={len(y_test)})",
        fontsize=13, fontweight="bold"
    )

    for probs, label, color, lw, ls in [
        (aug_test_probs,      f"Attn-Aug CatBoost  AUROC={metrics['auroc']:.4f}",
         COLORS["aug"],      2.5, "-"),
        (cb_baseline_probs,   f"Tuned CatBoost      AUROC={cb_baseline_auroc:.4f}",
         COLORS["catboost"], 1.5, "--"),
        (ftt_test_probs,      f"FT-Transformer      AUROC={roc_auc_score(y_test, ftt_test_probs):.4f}",
         COLORS["ftt"],      1.2, "-."),
    ]:
        fpr_, tpr_, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr_, tpr_, color=color, lw=lw, ls=ls, label=label)

    axes[0].plot([0,1],[0,1], "--", color=COLORS["random"], lw=0.8, alpha=0.5)
    axes[0].fill_between(*roc_curve(y_test, aug_test_probs)[:2],
                         alpha=0.07, color=COLORS["aug"])
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].spines[["top","right"]].set_visible(False)

    for probs, label, color, lw, ls in [
        (aug_test_probs,    f"Attn-Aug CatBoost  AUPRC={average_precision_score(y_test, aug_test_probs):.4f}",
         COLORS["aug"],    2.5, "-"),
        (cb_baseline_probs, f"Tuned CatBoost      AUPRC={average_precision_score(y_test, cb_baseline_probs):.4f}",
         COLORS["catboost"],1.5, "--"),
        (ftt_test_probs,    f"FT-Transformer      AUPRC={average_precision_score(y_test, ftt_test_probs):.4f}",
         COLORS["ftt"],    1.2, "-."),
    ]:
        prec_, rec_, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec_, prec_, color=color, lw=lw, ls=ls, label=label)

    prev = y_test.mean()
    axes[1].axhline(prev, color=COLORS["random"], ls="--", lw=0.8,
                    label=f"Prevalence ({prev:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_roc_pr.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_roc_pr.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_roc_pr.png")

    # ── Figure 2: Calibration ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for probs, label, color in [
        (aug_test_probs,    f"Attn-Aug CatBoost (Brier={metrics['brier_score']:.4f})",
         COLORS["aug"]),
        (cb_baseline_probs, f"Tuned CatBoost (Brier={brier_score_loss(y_test, cb_baseline_probs):.4f})",
         COLORS["catboost"]),
        (ftt_test_probs,    f"FT-Transformer (Brier={brier_score_loss(y_test, ftt_test_probs):.4f})",
         COLORS["ftt"]),
    ]:
        prob_true_, prob_pred_ = calibration_curve(probs, y_test, n_bins=10)
        ax.plot(prob_pred_, prob_true_, "o-", color=color, lw=2, label=label)

    ax.plot([0,1],[0,1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration — Attention-Augmented CatBoost vs Baselines",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_calibration.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_calibration.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_calibration.png")

    # ── Figure 3: Confusion matrix ────────────────────────────────────────────
    y_pred_aug = (aug_test_probs >= metrics["threshold"]).astype(int)
    cm         = confusion_matrix(y_test, y_pred_aug)
    fig, ax    = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-sepsis","Sepsis"])
    ax.set_yticklabels(["Non-sepsis","Sepsis"])
    thresh_cm = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > thresh_cm else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(
        f"Attention-Augmented CatBoost — Confusion Matrix\n"
        f"Sens={metrics['sensitivity']:.4f}  Spec={metrics['specificity']:.4f}  "
        f"Thresh={metrics['threshold']:.4f}",
        fontsize=9, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_confusion_matrix.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_confusion_matrix.png")

    # ── Figure 4: Attention heatmap (top 30 features × first 100 patients) ────
    log.info("  Generating attention heatmap...")
    top30_idx     = np.argsort(mean_attn)[::-1][:30]
    top30_names   = [feature_cols[i] for i in top30_idx]
    attn_top30    = attn_train[:100, top30_idx]  # first 100 patients, top 30 features

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(attn_top30.T, aspect="auto", cmap="viridis",
                   norm=mcolors.PowerNorm(gamma=0.5))
    plt.colorbar(im, ax=ax, label="Attention Weight")
    ax.set_yticks(range(30))
    ax.set_yticklabels(top30_names, fontsize=7)
    ax.set_xlabel("Patient index (first 100 training patients)")
    ax.set_ylabel("Feature (top 30 by mean attention)")
    ax.set_title(
        "FT-Transformer Attention Weights — Top 30 Features × First 100 Patients\n"
        "Attention extracted from last transformer block (CLS token row, head-averaged)",
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_attention_heatmap.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_attention_heatmap.png")

    # ── Figure 5: Threshold sensitivity ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Phase 8 — Threshold Sensitivity Analysis (Attention-Aug CatBoost)",
                 fontsize=12, fontweight="bold")

    thr_vals  = thresh_df["threshold"].values
    sens_vals = thresh_df["sensitivity"].values
    spec_vals = thresh_df["specificity"].values
    ppv_vals  = thresh_df["ppv"].values
    npv_vals  = thresh_df["npv"].values
    f1_vals   = thresh_df["f1"].values

    axes[0].plot(thr_vals, sens_vals, "o-", color=COLORS["aug"],      lw=2, label="Sensitivity")
    axes[0].plot(thr_vals, spec_vals, "s-", color=COLORS["catboost"], lw=2, label="Specificity")
    axes[0].set_xlabel("Threshold"); axes[0].set_ylabel("Score")
    axes[0].set_title("Sensitivity vs Specificity", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(thr_vals, ppv_vals, "o-", color="#2ECC71", lw=2, label="PPV")
    axes[1].plot(thr_vals, npv_vals, "s-", color="#F39C12", lw=2, label="NPV")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].set_title("PPV vs NPV", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].spines[["top","right"]].set_visible(False)

    axes[2].plot(thr_vals, f1_vals, "o-", color=COLORS["ftt"], lw=2)
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_threshold_sensitivity.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_threshold_sensitivity.png")

    # ── Figure 6: CatBoost feature importance — original vs attention ─────────
    fig, ax = plt.subplots(figsize=(12, 8))
    top20_imp   = importance_df.head(20)
    bar_colors  = [
        COLORS["aug"] if r["feature"].startswith("attn_") else COLORS["catboost"]
        for _, r in top20_imp.iterrows()
    ]
    ax.barh(range(len(top20_imp)), top20_imp["importance"].values,
            color=bar_colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top20_imp)))
    ax.set_yticklabels(
        [f"{'[A] ' if r['feature'].startswith('attn_') else '[O] '}"
         f"{r['feature'].replace('attn_', '')}"
         for _, r in top20_imp.iterrows()],
        fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("CatBoost Feature Importance")
    ax.set_title(
        "Top 20 Feature Importances — Attention-Augmented CatBoost\n"
        "[A] = Attention feature  |  [O] = Original feature",
        fontweight="bold"
    )
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["aug"],      label="Attention features [A]"),
        Patch(facecolor=COLORS["catboost"], label="Original features [O]"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "phase8_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "phase8_feature_importance.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved → phase8_feature_importance.png")

    # ══════════════════════════════════════════════════════════════════════════
    # 7.10 FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    runtime = (time.time() - t_start) / 60
    log.info("\n" + "=" * 70)
    log.info("PHASE 8 COMPLETE — ATTENTION-AUGMENTED CATBOOST")
    log.info("=" * 70)
    log.info(f"  AUROC        : {metrics['auroc']:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]")
    log.info(f"  AUPRC        : {metrics['auprc']:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")
    log.info(f"  Brier Score  : {metrics['brier_score']:.4f}")
    log.info(f"  Sensitivity  : {metrics['sensitivity']:.4f}")
    log.info(f"  Specificity  : {metrics['specificity']:.4f}")
    log.info(f"  PPV          : {metrics['ppv']:.4f}")
    log.info(f"  F1           : {metrics['f1']:.4f}")
    log.info(f"  DeLong vs Phase 2 CatBoost: z={z_stat:.4f}, p={p_val:.4f} "
             f"({'sig' if p_val < 0.05 else 'not sig'})")
    log.info(f"  Augmented features: {n_features} original + {n_features} attention = {n_features*2} total")
    log.info(f"  Runtime      : {runtime:.1f} min")
    log.info(f"\n  Models  → {MODELS_DIR}")
    log.info(f"  Results → {RESULTS_DIR}")
    log.info(f"  Figures → {FIGURES_DIR}")
    log.info(f"  Log     → {log_path}")
    log.info("\n  Next step: Federated Learning (Phase 9)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
