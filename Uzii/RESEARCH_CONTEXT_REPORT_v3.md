# RESEARCH PROJECT CONTEXT EXPORT v3
**For:** Claude (new chat) | **Project:** Pediatric Sepsis Prediction
**Students:** Uzair (literature review & writing) & Tanish (codebase)
**Institution:** NMIMS Indore | **Graduating:** 2027
**Last updated:** April 25, 2026

---

## PROJECT IDENTITY

**Final Title:** "Early Prediction of Pediatric Sepsis Using Machine Learning, Deep Learning, and Explainable AI on the PIC Database: A Comparative Study Using Phoenix Sepsis Criteria"

**Target Journals:** Frontiers in Pediatrics, BMC Medical Informatics and Decision Making, PLOS ONE (final selection pending professor guidance)

**GitHub:** tanishqkr/pediatric-sepsis-PIC

---

## DATASET

- **Source:** PIC Database v1.1.0, Children's Hospital Zhejiang University, China
- **Cohort:** Option B (infection-only) — 3,163 patients
- **Split:** 2,530 train / 633 test (80/20, seed 42)
- **Features:** 225 (demographics, lab stats, vitals, shock index, trends, flags, fluid balance, infection metrics, EMR symptoms, surgery flags, ICU type one-hot)
- **Labeling:** Phoenix Sepsis Criteria — suspected infection AND Phoenix core score ≥ 2
- **Class distribution:** ~27.3% sepsis positive, ~72.7% non-sepsis
- **Imbalance handling:** Class weights {0:1.0, 1:2.7}. SMOTE/oversampling deliberately avoided after t-SNE visualisation revealed high class overlap.
- **Critical limitation:** Respiratory and neurological Phoenix components are permanently 0 — FiO₂/ventilation and GCS data missing from PIC. Labels driven by cardiovascular + coagulation only. Disclosed as study limitation.

---

## CONFIRMED RESEARCH GAP

No published study combines **all four** of: CatBoost + Phoenix Sepsis Criteria labeling + SHAP explainability + PIC database for sepsis onset prediction.

**Key competing papers:**
- Chanci et al. (2025) — CatBoost + Phoenix but private US data, no SHAP
- Huang et al. (2025) — PIC + sepsis but survival analysis, not onset, no Phoenix
- Shen et al. (2025) — CatBoost + SHAP + PIC but AKI mortality, not sepsis

---

## COMPLETE MODEL PIPELINE & RESULTS

### Phase 1 — Baseline ML Models (5-fold CV)
Models: CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression, MLP (baseline, not tuned)

### Phase 2 — Tuned CatBoost (PRIMARY MODEL)
- Optuna 100 trials, AUPRC optimized, 5-fold CV
- **AUROC: 0.9868 | AUPRC: 0.9691 | Brier: 0.0385 | Sens: 0.896 | Spec: 0.963 | PPV: 0.901 | F1: 0.899**
- Saved: `sepsis_ml/models/run_tuned/catboost_tuned.cbm`

### Phase 3 — Evaluation Suite
DCA, subgroup analysis, calibration on tuned CatBoost.

### Phase 4 — SHAP on Tuned CatBoost
- SHAP figures: `sepsis_ml/figures/phase4_shap/`
- Top SHAP features: lactate_max, D-dimer_max, platelets_min, INR_max, PT_max, bicarbonate_min, pH_min, glucose_max — directly mapping to cardiovascular and coagulation Phoenix domains
- Key SHAP interactions: Lactate × Platelets axis (mean absolute interaction 0.035), INR × D-dimer coagulopathy axis (0.021)
- Results: `sepsis_ml/results/phase4_shap_results.json`

### Phase 5 — Sensitivity Analysis
Option A vs Option B cohort comparison confirming infection-only cohort choice.

### Phase 6 — Deep Learning Models
All trained on Option B, Optuna-tuned, evaluated with DeLong test vs tuned CatBoost:

| Model | AUROC | AUPRC | Brier | Spec | DeLong p vs CatBoost |
|---|---|---|---|---|---|
| FT-Transformer | 0.9577 | 0.9131 | 0.0733 | 85.2% | p=0.169 (not sig) |
| ResNet | 0.9311 | 0.8744 | 0.0945 | 81.3% | p=0.018 (sig) |
| TabNet | 0.9306 | 0.8657 | 0.1076 | 77.0% | p=0.014 (sig) |

FT-Transformer is best DL model. CatBoost significantly better than ResNet and TabNet, not significantly different from FT-T.

FT-T weights: `sepsis_ml/dl/models/run_phase6_fttransformer/fttransformer_best.pt`
FT-T architecture: 2 blocks, d_block=256, 8 attention heads (from Phase 6 training)

### Phase 7 — TWO COMPONENTS (both in same `sepsis_ml/fl/` and `sepsis_ml/phase7_stacking/` folders)

#### 7A — Stacking Ensemble (CatBoost + FT-Transformer, real data)
- Script: `sepsis_ml/phase7_stacking/phase7_stack_catboost_ftt.py`
- Meta-learner: Logistic Regression on 5-fold OOF probabilities
- CatBoost coefficient: 5.97 | FT-T coefficient: 1.67 (CB trusted ~3.6x more)
- **AUROC: 0.9851 | AUPRC: 0.9658 | Brier: 0.0428 | Sens: 0.902 | Spec: 0.957 | PPV: 0.886 | F1: 0.894**
- DeLong vs tuned CatBoost: p=0.388 (not significant — statistically equivalent)
- DeLong vs FT-T: p=0.0013 (significant — better than FT-T alone)

#### 7B — Federated Learning (FT-Transformer across 5 ICU-type nodes)
- Script: `sepsis_ml/fl/` folder
- Framework: Flower (flwr), simulated federation
- Model federated: FT-Transformer (gradient-descent based; CatBoost cannot be federated via weight averaging)
- **Nodes: 5** (partitioned by ICU type one-hot): CICU (n=144), General ICU (n=1,251), NICU (n=421), PICU (n=213), SICU (n=501)
- Config: 50 rounds, 5 local epochs per round, FedProx μ=0.01
- FT-T architecture same as Phase 6 (2 blocks, d_block=256, 8 heads)

**FedAvg results:**
- AUROC: 0.9507 [0.9340–0.9655] | AUPRC: 0.8870 | Brier: 0.0906
- Sens: 0.896 | Spec: 0.893 | F1: 0.822
- DeLong vs tuned CatBoost: p=0.090 (not significant)
- DeLong vs centralised FT-T: p=0.758 (not significant — federated matches centralised)

**FedProx results:**
- AUROC: 0.9484 [0.9308–0.9636] | AUPRC: 0.8848 | Brier: 0.0995
- Sens: 0.896 | Spec: 0.850 | F1: 0.781
- DeLong vs tuned CatBoost: p=0.070 (not significant)
- DeLong vs centralised FT-T: p=0.682 (not significant)
- DeLong FedAvg vs FedProx: p=0.915 (not significant — equivalent)

**Key FL finding:** Both FedAvg and FedProx match centralised FT-T performance while eliminating raw data sharing across ICU nodes. Neither matches tuned CatBoost, but the gap is not statistically significant.

### Phase 8 — Attention-Augmented CatBoost
- Architecture: Frozen Phase 6 FT-T weights → extract CLS-token attention weights from last transformer block → augment CatBoost with [225 original + 225 attention] = 450 features
- Correct extraction uses `model.backbone.blocks`, manually computing W_q(CLS) @ W_k(all)^T / sqrt(d_head), softmax, average heads, drop CLS col → (batch, 225) attention map
- Script: `sepsis_ml/phase8_attention_catboost/phase8_attention_catboost.py`
- **AUROC: 0.9657 [0.9507–0.9784] | AUPRC: 0.9268 | Brier: 0.0978 | Sens: 0.902 | Spec: 0.915 | F1: 0.848**
- DeLong vs tuned CatBoost: p=0.014 (significant — tuned CatBoost better)
- Top attention features: calcium_max (0.092), pao2_max (0.056), SICU (0.050), lactate_max (0.042), platelets_min (0.040) — clinically coherent but did not add discriminative value
- **Conclusion:** Attention augmentation did not improve over primary CatBoost model

### Synthetic Data Experiments (folder: `sepsis_ml/synthetic/`)

#### Synthetic Data Generation — vFinal Pipeline
- Script: `generate_synthetic_data_vFinal.py`
- Method: **Three-layer approach** (NOT vanilla CTGAN):
  - Layer 1: Class-split CTGAN (Xu et al., 2019) — separate CTGAN for sep=1 (691 rows) and sep=0 (1,839 rows) to preserve label signal
  - Layer 2: Iman-Conover rank correlation injection (Iman & Conover, 1982) — restores real Spearman correlation structure per class using van der Waerden scores + Cholesky decomposition. Previous CTGAN versions destroyed correlations entirely (e.g., INR × PT: real=0.999, v4=0.016)
  - Layer 3: Two-part zero-inflation fix for 10 heavily zero-inflated features (>20% zeros)
- Post-processing: Sort pass (max≥mean≥min), residual quantile mapping, care-unit one-hot fix, label/gender prevalence matching
- Output: 10,000 synthetic patients (`B_synthetic_train_vFinal.csv`)
- Validation: Mean Spearman error ~0.02, >97% label signal signs preserved

#### Experiment A — Real-only CatBoost (baseline reference)
- AUROC: 0.9868 (Phase 2 tuned CatBoost)

#### Experiment B — Synthetic-only CatBoost
- Trained on 10,000 synthetic patients, tested on real 633
- **AUROC: 0.9290 | AUPRC: 0.8693 | Sens: 0.902 | Spec: 0.759 | F1: 0.709**
- DeLong vs real-trained: p<0.0001 (significantly worse)

#### Experiment C — Unweighted Hybrid CatBoost
- Hybrid: 12,530 = 2,530 real + 10,000 synthetic (equal weight)
- **AUROC: 0.9659 | AUPRC: 0.9238**

#### Experiment D — Weighted Hybrid CatBoost (real weight=4)
- Real rows weighted 4x (effective influence: real=50.3%, synthetic=49.7%)
- Optuna 10 trials, AUPRC optimized
- **AUROC: 0.9776 [0.9673–0.9863] | AUPRC: 0.9496 | Brier: 0.0523 | Sens: 0.908 | Spec: 0.926 | F1: 0.863**
- DeLong vs real-trained (Exp A): p=0.0007 (significantly worse)
- DeLong vs unweighted hybrid (Exp C): p=0.0005 (significantly better — weighting helps)

#### Hybrid FT-Transformer
- Trained on hybrid 12,530 with WeightedRandomSampler (real=4x per epoch)
- Optuna 30 trials
- **AUROC: 0.9638 [0.9483–0.9772] | AUPRC: 0.9208 | Brier: 0.0888 | Sens: 0.908 | Spec: 0.861 | F1: 0.797**
- DeLong vs Phase 6 FT-T: p=0.626 (not sig — equivalent to centralised FT-T)
- DeLong vs Exp D CatBoost: p=0.161 (not sig)
- DeLong vs Exp A real-only: p=0.028 (significant — real-only better)

#### Hybrid Stacking Ensemble (Exp D CatBoost + Hybrid FT-T, LR meta-learner)
- CatBoost coefficient: 6.048 | FT-T coefficient: 1.858
- **AUROC: 0.9786 [0.9683–0.9875] | AUPRC: 0.9534 | Brier: 0.0503 | Sens: 0.908 | Spec: 0.935 | F1: 0.872**
- DeLong vs Exp D CatBoost: p=0.577 (not sig)
- DeLong vs Hybrid FT-T: p=0.090 (not sig)
- DeLong vs Exp A real-only: p=0.126 (not sig — statistically equivalent to real CatBoost)
- DELTA AUROC vs real-only: -0.0082 | DELTA AUPRC: -0.0157

**Key synthetic finding:** Synthetic-only training is significantly inferior. Weighted hybrid augmentation narrows but does not close the gap vs real data. The hybrid stacking ensemble, while not significantly different from real CatBoost, achieves this by leaning heavily on the weighted real-data CatBoost (coefficient ~3.3x FT-T).

---

## LEAKAGE AUDIT (Methodological Contribution)
- Conducted systematic leakage audit across 306 features before training
- Measurement frequency features: isolation AUROC 0.878 → excluded
- Vasoactive medication indicators: isolation AUROC 0.863 → excluded (these are Phoenix cardiovascular score components themselves)
- This audit is a methodological contribution of the paper

---

## KNOWN BUGS (must fix before submission)
1. `fillna(0)` in `phase0_1_feature_prep.py` — must be replaced with median imputation from training set only. **HIGH priority** — reviewers will reject this.
2. MLP missing `class_weight` parameter — **MEDIUM priority**
3. Respiratory and neurological Phoenix scores permanently 0 — disclosed as limitation, not a bug

---

## CODEBASE STRUCTURE (key paths)
```
pediatric_sepsis_prediction_PIC_XAI/  (also called pediatric-sepsis-PIC)
├── model_datasets/
│   ├── B_train_model_ready.csv           (2530 patients, 225 features)
│   ├── B_test_model_ready.csv            (633 patients)
│   └── synthetic/
│       ├── B_synthetic_train_vFinal.csv  (10000 synthetic patients)
│       └── C_hybrid_train.csv            (12530 hybrid patients)
├── sepsis_ml/
│   ├── results/
│   │   ├── best_params.json
│   │   └── catboost_test_probs.npy
│   ├── models/run_tuned/
│   │   └── catboost_tuned.cbm
│   ├── figures/phase4_shap/
│   ├── dl/
│   │   ├── models/run_phase6_fttransformer/
│   │   │   ├── fttransformer_best.pt
│   │   │   ├── best_params_final.json
│   │   │   └── scaler.pkl
│   │   └── results/
│   │       └── phase6_fttransformer_predictions.npz
│   ├── fl/                              (federated learning — Phase 7B)
│   │   ├── phase7_fedavg_results.json
│   │   ├── phase7_fedprox_results.json
│   │   ├── phase7_delong.json
│   │   ├── phase7_convergence.xlsx
│   │   └── phase7_per_node_metrics.xlsx
│   ├── phase7_stacking/                 (stacking ensemble — Phase 7A)
│   │   ├── phase7_stack_catboost_ftt.py
│   │   ├── catboost_info/
│   │   ├── figures/
│   │   ├── models/
│   │   └── results/
│   ├── phase8_attention_catboost/
│   │   └── phase8_attention_catboost.py
│   ├── synthetic/
│   │   ├── generate_synthetic_data_vFinal.py
│   │   ├── make_hybrid_dataset.py
│   │   ├── catboost/                    (Exp B: synthetic-only CatBoost)
│   │   ├── whybrid/                     (Exp D: weighted hybrid CatBoost)
│   │   ├── ft/                          (Hybrid FT-Transformer)
│   │   └── stacking/                    (Hybrid Stacking Ensemble)
│   ├── diagnostics/
│   ├── sensitivity_analysis/
│   └── logs/
```

---

## LITERATURE REVIEW STATUS

### Section Structure (6 sections):

| Section | Topic | Status |
|---|---|---|
| 1 | Clinical Background — Pediatric Sepsis | **COMPLETE** (4 paragraphs written and reviewed) |
| 2 | ML for Sepsis Prediction | Knowledge dump given, not yet written |
| 3 | Deep Learning, Hybrid Models & Synthetic Data | Knowledge dump given, not yet written |
| 4 | Explainable AI in Healthcare | Knowledge dump given, not yet written |
| 5 | The PIC Database | Knowledge dump given, not yet written |
| 6 | Research Gap and Study Rationale | Knowledge dump given, not yet written |

### Section 1 Paragraph 4 (final approved version — after minor fixes):
The 2024 Phoenix Sepsis Criteria represent a fundamental reconceptualization of pediatric sepsis, moving away from the inflammation proxy logic of SIRS toward a definition grounded in measurable organ dysfunction. Under Phoenix, pediatric sepsis is defined as life-threatening organ dysfunction arising from a suspected or confirmed infection, quantified across four physiological domains: respiratory, cardiovascular, coagulation and neurological with a composite score of 2 or above meeting the threshold for sepsis (Sanchez-Pinto et al., 2024; Schlapbach et al., 2024). Crucially, this framework was not developed in isolation, but rather was validated across three million patient encounters, giving it a breadth of generalizability that no prior pediatric sepsis definition has achieved (Schlapbach et al., 2024). For these reasons, the present study adopts Phoenix as its labelling criterion: it is the current international standard, and more importantly, it is the only definition that operationalizes organ dysfunction directly rather than treating systemic inflammation as a surrogate (Sanchez-Pinto et al., 2024). An important methodological constraint must nonetheless be acknowledged. The Pediatric Intensive Care (PIC) database, which serves as the data source for this study, does not record FiO₂ or mechanical ventilation parameters, making the respiratory Phoenix component non-computable; similarly, the absence of GCS documentation renders the neurological component effectively zero throughout the cohort (Zeng et al., 2020). Sepsis classification in this study therefore rests on the cardiovascular and coagulation Phoenix domains alone — a limitation acknowledged in prior machine learning studies applying Phoenix criteria to datasets with incomplete physiological coverage (Chanci et al., 2025). While this is a genuine constraint, it does not undermine the validity of the labelling approach, given that cardiovascular instability and coagulation dysfunction are consistently identified as the dominant organ failure signatures driving mortality and deterioration in pediatric sepsis.

---

## WRITING RULES (Uzair)
1. Write independently after knowledge dump — never copy-paste from Claude
2. Cite every specific claim and statistic immediately after it
3. One idea per sentence — split any sentence with more than two commas
4. Formal third person — no "I think," no "we believe"
5. Precise words — "accurate" not "unerring," "implication" not "insinuation"
6. Every paragraph has one job — state what it argues before writing it
7. Connect each section back to why the study matters

---

## MENTOR APPROACH (how Claude should behave)
- Direct, honest, non-sycophantic
- Review writing critically with specific problems and specific fixes
- Give knowledge dumps, let Uzair write independently, then review
- **Never write sections for Uzair to copy** — this is his paper and his professor checks for AI plagiarism
- Tanish handles all code — Uzair handles literature review and writing
- Rate things honestly, flag real problems

---

## REFERENCE BANK (37 papers — original 31 + 6 new)

### Category 1 — Pediatric Sepsis Definition
- Sanchez-Pinto et al. (2024) JAMA — Phoenix criteria
- Schlapbach et al. (2024) JAMA — Phoenix international consensus
- Goldstein et al. (2005) Pediatric Critical Care Med — original SIRS
- Carroll et al. (2023) — operationalizing definitions globally
- Yuniar et al. (2023) Frontiers Pediatrics — prognostic factors

### Category 2 — ML for Sepsis (General)
- Yang et al. (2023) BMC Infectious Diseases — systematic review 23 studies
- Fleuren et al. (2020) Intensive Care Medicine — 150 studies
- Deng et al. (2022) iScience — methodology critique, leakage
- Moor et al. (2021) arXiv — deep self-attention AUROC 0.847

### Category 3 — ML for Pediatric Sepsis
- Le et al. (2019) Frontiers Pediatrics — AUROC 0.916, 4h before onset, UCSF
- Chanci et al. (2025) Pediatric Research — CatBoost AUROC 0.98, Phoenix, no SHAP, private US data
- Alpern et al. (2025) JAMA Pediatrics — multicenter ED
- Han et al. (2024) Acute and Critical Care — CatBoost+XGBoost bloodstream infection
- Kamaleswaran et al. (2018) Pediatric Critical Care Med — CNN 8h ahead
- Marassi et al. (2023) — age-normalization transfer
- Shi et al. (2025) Frontiers Pediatrics — XGBoost+GRU+SHAP AUROC 0.915

### Category 4 — Deep Learning
- Nesaragi & Patidar (2021) Infectious Diseases — explainable ML early sepsis
- Lauritsen et al. (2020) AI in Medicine — LSTM on EHR
- **Gorishniy et al. (2021) NeurIPS — Revisiting Deep Learning Models for Tabular Data (FT-Transformer + ResNet)** ← NEW
- **Arik & Pfister (2021) AAAI — TabNet: Attentive Interpretable Tabular Learning** ← NEW

### Category 5 — XAI
- Athukorala & Ilmini (2026) BMC Medical Informatics — XAI systematic review, pediatric gap
- Shen et al. (2025) BMC Medical Informatics — CatBoost+SHAP+PIC for AKI
- Jiang et al. (2023) MIMIC-IV — SHAP clustered analysis
- Lundberg & Lee (2017) NeurIPS — original SHAP paper

### Category 6 — PIC Database
- Zeng et al. (2020) Scientific Data — PIC database paper
- Zhou et al. (2025) Scientific Reports — SHAP+PIC for DIC
- Huang et al. (2025) Frontiers Pediatrics — PIC+SHAP sepsis survival
- Ding & Mei (2026) — PIC+sepsis+TyG, no ML
- Hong et al. (2021) — PIC+logistic regression+mortality

### Category 7 — Class Imbalance
- Bravo & Fajardo (2025) IEEE — class imbalance on PIC, SHAP beeswarm

### Category 8 — Synthetic Data & Federated Learning (ALL NEW)
- **Xu et al. (2019) NeurIPS — Modeling Tabular Data using Conditional GAN (CTGAN)**
  Full citation: Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. *Advances in Neural Information Processing Systems*, 32.
- **McMahan et al. (2017) AISTATS — Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)**
  Full citation: McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data. *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics*, PMLR 54:1273–1282.
- **Li et al. (2020) MLSys — Federated Optimization in Heterogeneous Networks (FedProx)**
  Full citation: Li, T., Sahu, A.K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *Proceedings of Machine Learning and Systems*, 2, 429–450.
- **Iman & Conover (1982) — A distribution-free approach to inducing rank correlation among input variables**
  Full citation: Iman, R.L., & Conover, W.J. (1982). A distribution-free approach to inducing rank correlation among input variables. *Communications in Statistics — Simulation and Computation*, 11(3), 311–334.

---

## KNOWLEDGE DUMPS GIVEN (for Sections 2–6)
Full knowledge dumps were provided for all six sections in the April 25, 2026 session. These cover:

**Section 2:** Scope of ML sepsis literature (Yang 2023, Fleuren 2020), gradient boosting dominance, CatBoost precedents (Chanci, Han, Shi), class imbalance handling rationale, leakage audit as methodological contribution (Deng 2022).

**Section 3:** DL on tabular clinical data (Lauritsen, Moor, Kamaleswaran), FT-Transformer vs ResNet vs TabNet results and DeLong comparisons, two stacking configurations (real-data and hybrid-data), attention-augmented CatBoost Phase 8 results and interpretation, CTGAN + Iman-Conover synthetic pipeline details, all four synthetic experiment results with DeLong comparisons.

**Section 4:** Clinical trust problem and XAI motivation (Athukorala & Ilmini 2026), SHAP axioms and TreeSHAP (Lundberg & Lee 2017), prior PIC SHAP studies (Shen, Zhou, Huang, Jiang), this study's SHAP findings (top features, interaction axes, Phoenix domain alignment).

**Section 5:** PIC database description (Zeng 2020), prior PIC studies, Option B cohort rationale, PIC limitation disclosure.

**Section 6:** Precise 4-element gap statement, list of 6 study contributions.

---

## IMMEDIATE NEXT STEPS
1. **Literature Review** — Uzair writes Sections 2–6 using knowledge dumps above, sends each paragraph to Claude for review
2. **Bug fix** — `fillna(0)` → median imputation before submission (Tanish)
3. **Architecture diagram** — design together once writing is complete (draw.io or Lucidchart)
4. **Journal selection** — ask professor after paper draft is complete
