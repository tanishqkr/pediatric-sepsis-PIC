"""
federated_shap.py

Component 2. Computes SHAP LOCALLY per node on the trained federated global
model -- GradientExplainer, since FT-Transformer is a differentiable net.

FIXES APPLIED (per repo audit):
  - Column lookup now goes through config.care_unit_column(node_name)
    instead of hand-building "care_unit_" + node_name -- this is the exact
    line that raised KeyError on General ICU before.
  - Per-node try/except: one node failing no longer silently kills the
    rest of the run or leaves a gap nobody notices until a manual audit
    finds it. Every node's outcome (success or failure) is logged and
    printed in a final summary.
  - New --node flag: re-run a single node without recomputing the ones
    that already succeeded.

Usage:
    python federated_shap.py --algo fedavg                    # all 5 nodes
    python federated_shap.py --algo fedavg --node General      # just General
    python federated_shap.py --algo fedprox --node General
"""
import argparse
import json
import pickle
import sys
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from rtdl_revisiting_models import FTTransformer

import fl_shap_config as config
from fl_shap_results_io import RunIO

sys.path.insert(0, config.SEPSIS_ML_ROOT)
try:
    from phase4_shap import get_display_name
except ImportError:
    def get_display_name(feat):
        return feat


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class ModelWrapper(nn.Module):
    """shap.GradientExplainer calls model(x) with ONE tensor argument;
    the real FTTransformer needs model(x_num, x_cat)."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x, None)


def load_metadata():
    with open(config.MODEL_METADATA_PATH) as f:
        return json.load(f)


def build_model_from_metadata(meta):
    fl_cfg = meta["fl_config"]
    return FTTransformer(
        n_cont_features=meta["n_features"],
        cat_cardinalities=None,
        d_out=1,
        n_blocks=fl_cfg["n_blocks"],
        d_block=fl_cfg["d_block"],
        attention_n_heads=fl_cfg["attention_n_heads"],
        attention_dropout=fl_cfg["attention_dropout"],
        ffn_d_hidden_multiplier=fl_cfg["ffn_d_hidden_multiplier"],
        ffn_dropout=fl_cfg["ffn_dropout"],
        residual_dropout=fl_cfg["residual_dropout"],
    )


def load_global_model(algo: str, meta: dict):
    ckpt_path = (config.GLOBAL_MODEL_CHECKPOINT_FEDAVG if algo == "fedavg"
                 else config.GLOBAL_MODEL_CHECKPOINT_FEDPROX)
    model = build_model_from_metadata(meta).to(DEVICE)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return ModelWrapper(model).to(DEVICE)


def scale_df(df, scaler, num_cols):
    df = df.copy()
    df[num_cols] = scaler.transform(df[num_cols])
    return df


def load_node_test_data(node_name: str, meta: dict, scaler):
    """Partition B_test_model_ready.csv by its care_unit_<...> one-hot
    column -- THE FIX: uses config.care_unit_column() so General ICU's
    'care_unit_General ICU' suffix resolves correctly instead of the
    naive (and wrong) 'care_unit_General'."""
    df = pd.read_csv(config.MODEL_READY_TEST)
    feature_cols = meta["feature_cols"]

    col = config.care_unit_column(node_name)
    if col not in df.columns:
        available = [c for c in df.columns if c.startswith(config.CARE_UNIT_PREFIX)]
        raise KeyError(
            f"Expected column '{col}' in {config.MODEL_READY_TEST}. "
            f"Available care_unit columns: {available}"
        )

    node_df = df[df[col] == 1].reset_index(drop=True)
    node_df = node_df[feature_cols]
    node_df_scaled = scale_df(node_df, scaler, meta["num_cols"])
    X = node_df_scaled.values.astype(np.float32)
    return X, feature_cols, node_df.shape[0]


def run_node_shap(node_name: str, model, background: torch.Tensor, meta: dict, scaler):
    X_np, feature_names, n_patients = load_node_test_data(node_name, meta, scaler)
    if n_patients == 0:
        raise ValueError(
            f"Node '{node_name}' has 0 test patients under column "
            f"'{config.care_unit_column(node_name)}' -- check the column "
            f"mapping in fl_shap_config.py, this should never be zero."
        )
    X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_tensor)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values).reshape(X_np.shape[0], -1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        zip(feature_names, mean_abs_shap.tolist()),
        key=lambda t: t[1], reverse=True
    )
    return {
        "node": node_name,
        "n_patients": int(n_patients),
        "n_weight": config.NODES[node_name],
        "feature_ranking": [
            {"feature": f, "display_name": get_display_name(f), "mean_abs_shap": v}
            for f, v in ranking
        ],
        "top_k": [
            {"feature": f, "display_name": get_display_name(f), "mean_abs_shap": v}
            for f, v in ranking[:config.TOP_K]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--node", default=None, choices=list(config.NODES.keys()),
                         help="run only this node (fast re-run after fixing a single-node failure)")
    args = parser.parse_args()

    io = RunIO(run_name=args.run_name)
    io.log("federated_shap_start", algo=args.algo, device=str(DEVICE),
            node_filter=args.node or "all")

    meta = load_metadata()
    with open(config.SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = load_global_model(args.algo, meta)

    full_test_df = pd.read_csv(config.MODEL_READY_TEST)
    bg_df = full_test_df[meta["feature_cols"]].sample(
        n=min(config.SHAP_BACKGROUND_SIZE, len(full_test_df)),
        random_state=config.RANDOM_SEED,
    )
    bg_scaled = scale_df(bg_df, scaler, meta["num_cols"])
    background = torch.tensor(bg_scaled.values.astype(np.float32)).to(DEVICE)

    nodes_to_run = [args.node] if args.node else list(config.NODES.keys())
    succeeded, failed = [], []

    for node_name in nodes_to_run:
        io.log("node_shap_start", node=node_name)
        try:
            result = run_node_shap(node_name, model, background, meta, scaler)
            io.save_json(f"node_shap/{args.algo}_{node_name}.json", result)
            io.log_to_leaderboard(
                phase="federated_shap",
                model_name=f"{args.algo}_{node_name}",
                metrics={"n_patients": result["n_patients"]},
                extra={"top_feature": result["top_k"][0]["feature"]},
            )
            io.log("node_shap_done", node=node_name, top_feature=result["top_k"][0]["feature"])
            succeeded.append(node_name)
        except Exception as e:
            tb = traceback.format_exc()
            io.log("node_shap_FAILED", node=node_name, error=str(e), traceback=tb)
            print(f"\n!! FAILED on node '{node_name}': {e}\n{tb}\n", file=sys.stderr)
            failed.append(node_name)

    io.finish()

    print(f"\n=== {args.algo.upper()} SHAP summary: {len(succeeded)}/{len(nodes_to_run)} nodes succeeded ===")
    print(f"  Succeeded: {succeeded}")
    if failed:
        print(f"  FAILED:    {failed}  <-- fix and re-run with --node <name> before trusting any aggregate result")
        sys.exit(1)


if __name__ == "__main__":
    main()
