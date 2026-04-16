# model_datasets/ — v2 leakage-clean

Generated: 2026-03-23T14:49:52
Dataset: PIC Sepsis — Option B (Infection-only cohort)
Version: v2-leakage-clean (drops _count, _was_measured, vaso, los_hours)

## Files
| File | Description |
|------|-------------|
| B_train_model_ready.csv | Training set — 2530 rows, 225 features + target |
| B_test_model_ready.csv  | Test set — 633 rows, 225 features + target |
| feature_list.json       | Exact feature names in column order |
| prep_report.json        | Full audit of all drop/encode decisions |

## Target column
`sepsis_label` — 1=sepsis, 0=infected-non-sepsis

## Class distribution
- Train: 691 sepsis / 1839 non-sepsis
- Test:  173 sepsis / 460 non-sepsis

## What was dropped and why
- `_count` columns (40): measurement frequency — circular with Phoenix labeling
- `_was_measured` columns (33): binary physician-ordering signal — same circularity
- vaso columns (6): part of Phoenix cardiovascular score that defines the label
- `los_hours`: outcome variable — unknown at early prediction time
- Standard: leakage (expire_flag, hosp_expire), zero-variance, free text, duplicates

## IMPORTANT
- class_weights = {0: 1, 1: 2.7} for all models
- Random seed = 42
- Expected realistic AUROC: 0.82–0.90
