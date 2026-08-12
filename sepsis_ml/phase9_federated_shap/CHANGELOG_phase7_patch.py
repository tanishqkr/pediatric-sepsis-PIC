"""
CHANGELOG_phase7_patch.md (kept as .py so it doesn't get skipped when you
zip/copy the folder — just documentation, not executable)

WHAT CHANGED in fl/phase7_federated_learning.py:
  A new section "18b. SAVE MODEL CHECKPOINTS + SCALER + METADATA" was
  inserted immediately after the existing "log.info('All results saved.')"
  line (end of section 18), before section 19 (figures). Nothing before
  that line was touched — all of your existing metrics/JSON/CSV/figure
  outputs are unchanged.

WHY:
  sepsis_ml/fl/results/ only ever had metrics JSON — no models/ folder
  existed anywhere under fl/, meaning the trained global model weights
  were never written to disk. Federated SHAP (phase9_federated_shap/)
  needs the actual weights, not the metrics.

WHAT GETS WRITTEN NOW, at fl/models/ (new folder, created automatically):
  - fedavg_global_final.pt   : global_model_fedavg.state_dict()
  - fedprox_global_final.pt  : global_model_fedprox.state_dict()
  - scaler.pkl               : the fitted StandardScaler from section 6
                                (federated SHAP must scale inputs the exact
                                same way, not refit a new scaler)
  - model_metadata.json      : feature_cols (exact column order), num_cols,
                                binary_cols, node_names, node_sizes,
                                n_features, and fl_config (architecture
                                hyperparameters) — everything
                                federated_shap.py needs to rebuild the
                                FTTransformer and reload the weights
                                without re-importing/re-running this
                                training script.

NOTHING in your original script's logic, training loop, aggregation, or
existing outputs was modified — this is a pure addition at the end.
"""
