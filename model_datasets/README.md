# model_datasets/

Generated: 2026-03-21T17:59:42
Dataset: PIC Sepsis — Option B (Infection-only cohort)

## Files

| File | Description |
|------|-------------|
| B_train_model_ready.csv | Training set — 2530 rows, 305 features + target |
| B_test_model_ready.csv  | Test set — 633 rows, 305 features + target |
| feature_list.json       | Exact feature names in column order |
| prep_report.json        | Full audit of all drop/encode decisions |

## Target column
`sepsis_label` — 1=sepsis, 0=infected-non-sepsis

## Class distribution
- Train: 691 sepsis / 1839 non-sepsis (27.3% prevalence)
- Test:  173 sepsis / 460 non-sepsis (27.3% prevalence)

## IMPORTANT
- Do NOT modify these files. They are the frozen final datasets.
- Do NOT use option_A or option_B raw files for modelling.
- class_weights = {0: 1, 1: 2.7} for all models (NO SMOTE).
- Random seed = 42 for all experiments.
- See prep_report.json for full documentation of all preprocessing decisions.
