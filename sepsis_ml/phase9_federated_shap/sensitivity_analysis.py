"""
sensitivity_analysis.py

Component 3. Three robustness checks, reusing the bootstrap/threshold
utilities your parent project already has in
sepsis_ml/phase5_sensitivity_analysis.py (bootstrap_auroc, bootstrap_auprc,
metrics_at_threshold) instead of re-deriving them — that file did the
Option A vs. B (full cohort vs. infection-only) comparison for centralized
CatBoost; this script applies the same statistical machinery to the FL+SHAP
pipeline instead.

  Axis 1 (cohort):     re-run federated_shap.py's aggregation logic on
                        Option A vs Option B test sets (paths in config.py —
                        add MODEL_READY_TEST_OPTION_A if you re-derive it).
  Axis 2 (partition):   compare the ORIGINAL 5-node care_unit split against
                        an alternate non-IID grouping (see
                        `alternate_partition()` below — currently a stub).
  Axis 3 (algorithm):    FedAvg-trained vs. FedProx-trained global model,
                        each explained the same way, ranking compared.

Run this AFTER you have federated_ranking_fedavg.json and
federated_ranking_fedprox.json in the same run folder (i.e. after running
federated_shap.py + aggregate_shap.py once per algorithm).
"""
import argparse
import json
import os
import sys

import pandas as pd
from scipy.stats import spearmanr

import fl_shap_config as config
from fl_shap_results_io import RunIO

sys.path.insert(0, config.SEPSIS_ML_ROOT)  # ADAPT if needed
try:
    from phase5_sensitivity_analysis import bootstrap_auroc, bootstrap_auprc, metrics_at_threshold
    HAVE_PHASE5_UTILS = True
except ImportError:
    HAVE_PHASE5_UTILS = False
    print("[sensitivity_analysis] Could not import phase5_sensitivity_analysis.py — "
          "fix the sys.path above, or implement bootstrap_auroc/bootstrap_auprc locally.")


def axis3_algorithm_agreement(run_dir: str):
    """FedAvg SHAP ranking vs. FedProx SHAP ranking — does the choice of
    federated optimiser itself change what the model appears to have learned?"""
    with open(os.path.join(run_dir, "federated_ranking_fedavg.json")) as f:
        fedavg = json.load(f)
    with open(os.path.join(run_dir, "federated_ranking_fedprox.json")) as f:
        fedprox = json.load(f)

    fa = {d["feature"]: d["federated_mean_abs_shap"] for d in fedavg["federated_ranking"]}
    fp = {d["feature"]: d["federated_mean_abs_shap"] for d in fedprox["federated_ranking"]}
    features = list(fa.keys())
    rho, pval = spearmanr([fa[f] for f in features], [fp[f] for f in features])

    return {
        "check": "FedAvg vs FedProx federated SHAP ranking agreement",
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "fedavg_top_5": [d["feature"] for d in fedavg["top_5_features"]],
        "fedprox_top_5": [d["feature"] for d in fedprox["top_5_features"]],
        "top_5_overlap": len(
            set(d["feature"] for d in fedavg["top_5_features"]) &
            set(d["feature"] for d in fedprox["top_5_features"])
        ),
    }


def alternate_partition():
    """
    STUB — Axis 2. Define a second, different non-IID grouping of the same
    patients (e.g. by admission year, by age band instead of care-unit type)
    and re-run federated_shap.py + aggregate_shap.py against it, then compare
    its federated_ranking JSON to the original 5-node version the same way
    axis3_algorithm_agreement() compares FedAvg vs FedProx above.

    Left as a stub because the right alternate grouping depends on what other
    categorical columns exist in your model-ready dataset (check
    feature_list.json). Age-band is the natural second choice given the base
    paper already reports subgroup AUROC by age (neonate/infant/child/adolescent).
    """
    raise NotImplementedError("Define an alternate non-IID partition and re-run the pipeline against it.")


def axis1_cohort_check(run_dir: str):
    """
    STUB — Axis 1. Re-run the FULL pipeline (FL training -> federated_shap.py
    -> aggregate_shap.py) on the Option A (full intermediate cohort) test set
    instead of Option B (infection-only), then diff the two federated_ranking
    JSONs exactly like axis3_algorithm_agreement() does. This is the most
    expensive axis (needs a second FL training run) — do it last, and only
    if time allows tomorrow.
    """
    raise NotImplementedError("Needs a second FL training run on Option A. Do this last.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    io = RunIO(run_name=os.path.basename(args.run_dir))
    io.run_dir = args.run_dir
    io.log("sensitivity_analysis_start")

    results = {}
    try:
        results["axis3_algorithm"] = axis3_algorithm_agreement(args.run_dir)
        io.log("axis3_done", rho=results["axis3_algorithm"]["spearman_rho"])
    except FileNotFoundError as e:
        io.log("axis3_skipped", reason=str(e))

    io.save_json("sensitivity/summary.json", results)
    io.finish()

    print("\nSensitivity analysis summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
