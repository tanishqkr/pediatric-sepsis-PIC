"""
checkpoint_patch.py

WHY THIS EXISTS:
Your context report shows sepsis_ml/fl/results/ containing
phase7_fedavg_results.json, phase7_fedprox_results.json, phase7_comparison.json,
phase7_convergence.csv, phase7_delong.json, phase7_per_node_metrics.csv —
metrics and numbers only. There is no sepsis_ml/fl/models/ folder in the
tree, which means phase7_federated_learning.py most likely never calls
torch.save() on the final global model. Federated SHAP needs the actual
trained weights, not the metrics, so this has to be added BEFORE you run FL
tomorrow — otherwise you'll finish training and have nothing to explain.

HOW TO APPLY:
1. Open sepsis_ml/fl/phase7_federated_learning.py.
2. Find where it builds/holds the final aggregated global model after the
   last communication round for each algorithm (FedAvg and FedProx). It is
   very likely a variable like `global_model` or `server_model` at the end
   of the training loop, just before results are written to
   phase7_fedavg_results.json / phase7_fedprox_results.json.
3. Right there, add the 3-line snippet below (adjust the variable name and
   the `algo` string to match what the script actually calls it).
4. Re-run only the last ~2 rounds if it's already finished a run without
   this, or just make sure it's in place before your run tomorrow.

--------------------------------------------------------------------------
# --- ADD THIS just before/after each algorithm's results.json is written ---
import os, torch
os.makedirs(FL_CHECKPOINT_DIR, exist_ok=True)          # from config.py
ckpt_path = os.path.join(FL_CHECKPOINT_DIR, f"{algo}_global_final.pt")  # algo = "fedavg" | "fedprox"
torch.save({
    "state_dict": global_model.state_dict(),
    "algo": algo,
    "n_features": FEATURE_COUNT,
}, ckpt_path)
print(f"[checkpoint] saved global {algo} model -> {ckpt_path}")
# -----------------------------------------------------------------------
--------------------------------------------------------------------------

IF you can't find/edit the training loop in time tomorrow, the fallback is:
train the SAME architecture centrally (non-federated) on the pooled training
data as a stand-in just to unblock SHAP development, and swap in the real
federated checkpoint the moment it's saved. Don't let this block your day —
federated_shap.py only needs `model.load_state_dict(...)` to work, it
doesn't care how the checkpoint was produced.
"""
