# Literature Research Log — ICICC 2027 Submission 359
**Paper Title:** Federated SHAP: Privacy-Preserving, Per-Node Explainability and Sensitivity Analysis for Pediatric Sepsis Prediction  
**Date of Audit & Research:** October 2026  
**Focus Period:** 2024–2026 primary literature (with seminal foundations)

---

## 1. Executive Summary of Literature Findings

This research log provides verified peer-reviewed citations and methodological foundations to address the comments from Reviewer 1 and Reviewer 2. Specifically:
1. **Methodological Novelty & Aggregation Rationale:** Prior assertions of absolute novelty ("no published study combines federated training with federated explainability") were scientifically untenable. Recent primary works, notably **Bonsignori et al. (iFLASH, IEEE Access 2026)** and **Ducange et al. (Federated SHAP, Springer Machine Learning 2026)**, have formally explored federated post-hoc SHAP aggregation and federated background dataset construction. Our paper's distinct contribution is reframed around the **first clinical application and empirical audit of per-node explainability in pediatric critical care sepsis prediction under severe non-IID class and care-unit volume imbalance (PIC database with Phoenix criteria)**.
2. **Privacy Threat Modeling:** Sharing node-level mean absolute SHAP vectors ($\overline{|\text{SHAP}|}_k \in \mathbb{R}^{225}$) does not constitute formal differential privacy. Seminal and recent works (**Shokri et al. 2019/2021**, **Patel et al. FAccT 2022**) demonstrate that feature attributions and aggregate summary vectors can leak membership or patient attributes, particularly in small clinical cohorts ($n=40$ test cases at CICU). Secure Aggregation (**Bonawitz et al. ACM CCS 2017**) and Differential Privacy are essential for formal guarantees.
3. **Target Leakage & Label-Feature Co-occurrence:** Phoenix Sepsis Criteria (**Schlapbach et al. JAMA 2024**, **Sanchez-Pinto et al. JAMA 2024**) explicitly incorporate serum lactate ($\ge 5$ and $\ge 11$ mmol/L) into the cardiovascular dysfunction score. Because predictors and outcomes were extracted across the identical initial 24-hour observation window, `lactate_max` prominence represents partial diagnostic reconstruction rather than purely anticipatory prediction.

---

## 2. Verified Publications Matrix

### Entry 1: Federated Post-Hoc SHAP Aggregation (iFLASH)
- **Title:** *Interpreting Federated Learning by Aggregating SHAP Explanations*
- **Authors:** Valerio Bonsignori, Luca Corbucci, Francesca Naretto, Riccardo Guidotti, Anna Monreale
- **Year:** 2026 (Published online 2025/2026)
- **Venue:** *IEEE Access*
- **DOI / URL:** [10.1109/ACCESS.2026.3696009](https://doi.org/10.1109/ACCESS.2026.3696009)
- **Primary Finding:** Proposes the iFLASH framework for computing local SHAP explanations on client nodes and aggregating them centrally. Evaluates sample-weighted, uniform, and explanation-faithfulness/stability aggregation. Demonstrates that quality-aware aggregation outperforms naive averaging in highly heterogeneous cross-silo environments.
- **Relationship to Our Manuscript:** Challenges our previous absolute novelty claim. Serves as the primary reference for comparing sample-size weighting against uniform and stability-based weighting.
- **Placement in Revised Manuscript:** Sections I (Introduction), II-B (Federated Learning & XAI), and III-B (Aggregation Rationale).

---

### Entry 2: Federated Background Construction for SHAP
- **Title:** *Federated SHAP: Privacy-Preserving and Consistent Post-hoc Explainability in Federated Learning*
- **Authors:** Pietro Ducange, Francesco Marcelloni, Giustino Claudio Miglionico, Alessandro Renda, Fabrizio Ruffini
- **Year:** 2026
- **Venue:** *Machine Learning* (Springer Nature)
- **DOI / URL:** [10.1007/s10994-025-06734-x](https://doi.org/10.1007/s10994-025-06734-x)
- **Primary Finding:** Identifies that post-hoc Shapley value computation across distributed clients suffers from the absence of a shared background dataset. Proposes Federated Fuzzy C-Means (FCM) clustering for tabular data and Federated GANs for images to construct privacy-preserving synthetic background sets, reducing explanation error threefold relative to random baselines.
- **Relationship to Our Manuscript:** Directly addresses Reviewer 1's query regarding background selection in federated GradientSHAP. Shows how background choice interacts with distributed explainability.
- **Placement in Revised Manuscript:** Sections II-C (Explainable AI in Healthcare), III-B (Federated SHAP Pipeline), and V (Limitations & Future Work).

---

### Entry 3: Privacy Risks of Model Explanations & Threat Modeling
- **Title:** *Privacy Risks of Explaining Machine Learning Models*
- **Authors:** Reza Shokri, Martin Strobel, Yair Zick
- **Year:** 2021 (Preliminary version 2019)
- **Venue:** *IEEE Symposium on Security and Privacy (S&P)*
- **DOI / URL:** [10.1109/SP40001.2021.00078](https://doi.org/10.1109/SP40001.2021.00078)
- **Primary Finding:** Establishes that exposing model explanations (such as SHAP vectors and saliency maps) significantly increases vulnerability to Membership Inference Attacks (MIAs) and feature reconstruction attacks, often matching or exceeding the leakage from raw output probabilities.
- **Relationship to Our Manuscript:** Directly supports Reviewer 1 Comment 2. We must not claim that retaining raw records locally provides a "complete privacy guarantee," because transmitting per-node mean absolute SHAP vectors leaves an exploitable statistical attack surface.
- **Placement in Revised Manuscript:** Section III-D (Privacy Boundary and Threat Model).

---

### Entry 4: Differential Privacy for Model Explanations
- **Title:** *Model Explanations with Differential Privacy*
- **Authors:** Neel Patel, Reza Shokri, Yair Zick
- **Year:** 2022
- **Venue:** *ACM Conference on Fairness, Accountability, and Transparency (FAccT 2022)*
- **DOI / URL:** [10.1145/3531146.3533241](https://doi.org/10.1145/3531146.3533241)
- **Primary Finding:** Formulates $(\epsilon, \delta)$-differentially private algorithms for computing feature attributions, deriving sensitivity bounds and demonstrating the utility-privacy trade-off of noise injection into Shapley values.
- **Relationship to Our Manuscript:** Provides the mathematical path forward for our future work and clarifies what differential privacy would require (Laplace/Gaussian perturbation scaled by node sample size $n_k$).
- **Placement in Revised Manuscript:** Section III-D (Privacy Analysis) and Section V (Limitations).

---

### Entry 5: Cryptographic Secure Aggregation in FL
- **Title:** *Practical Secure Aggregation for Privacy-Preserving Machine Learning*
- **Authors:** Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, Karn Seth
- **Year:** 2017
- **Venue:** *ACM SIGSAC Conference on Computer and Communications Security (CCS 2017)*
- **DOI / URL:** [10.1145/3133956.3133982](https://doi.org/10.1145/3133956.3133982)
- **Primary Finding:** Introduces a fault-tolerant multi-party computation protocol allowing a central server to compute the aggregate sum of vectors without inspecting any client's individual vector.
- **Relationship to Our Manuscript:** Provides the exact mechanism needed to prevent the server from inspecting individual care-unit SHAP vectors $\overline{|\text{SHAP}|}_k$, ensuring only the global federated aggregate is revealed.
- **Placement in Revised Manuscript:** Section III-D (Privacy Threat Model & Mitigations).

---

### Entry 6: International Consensus Criteria for Pediatric Sepsis (Phoenix)
- **Title:** *International Consensus Criteria for Pediatric Sepsis and Septic Shock*
- **Authors:** Luregn J. Schlapbach, R. Scott Watson, Lauren R. Sorce, Andrew C. Argent, Kusum Menon, Mark W. Hall, et al.
- **Year:** 2024
- **Venue:** *JAMA*, Vol. 331, No. 8, pp. 665–674
- **DOI / URL:** [10.1001/jama.2024.0179](https://doi.org/10.1001/jama.2024.0179)
- **Primary Finding:** Replaces 2005 Goldstein SIRS-based pediatric sepsis definitions with the Phoenix Sepsis Score (organ dysfunction criteria evaluating cardiovascular, respiratory, neurological, and coagulation systems).
- **Relationship to Our Manuscript:** Foundational definition for ground-truth sepsis labelling in our study cohort.
- **Placement in Revised Manuscript:** Retained in Section I & Section III-A.

---

### Entry 7: Development and Validation of Phoenix Criteria
- **Title:** *Development and Validation of the Phoenix Criteria for Pediatric Sepsis and Septic Shock*
- **Authors:** L. Nelson Sanchez-Pinto, Tellen D. Bennett, Peter E. DeWitt, Seth Russell, Margaret N. Rebull, Blake Martin, et al.
- **Year:** 2024
- **Venue:** *JAMA*, Vol. 331, No. 8, pp. 675–686
- **DOI / URL:** [10.1001/jama.2024.0196](https://doi.org/10.1001/jama.2024.0196)
- **Primary Finding:** Validates the Phoenix criteria across >3 million pediatric encounters. Confirms that cardiovascular dysfunction scores include serum lactate ($\ge 5$ mmol/L = 1 pt, $\ge 11$ mmol/L = 2 pts) and coagulation dysfunction includes platelets $<100 \times 10^3/\mu\text{L}$, INR $>1.3$, and D-dimer $>2 \text{ mg/L FEU}$.
- **Relationship to Our Manuscript:** Explains why `lactate_max`, `platelets_min`, `inr_max`, and `ddimer` emerge as top features: they are direct subcomponents of the Phoenix diagnostic score calculated within the same 24h ICU window.
- **Placement in Revised Manuscript:** Section I, Section IV-B, and Section IV-C.

---

### Entry 8: Daily Pediatric Sepsis Machine Learning with Phoenix Criteria
- **Title:** *Machine Learning Model for Daily Prediction of Pediatric Sepsis Using Phoenix Criteria*
- **Authors:** Daniel Chanci, Jocelyn R. Grunwell, Arefin Rafiei, Samuel R. Brown, Michael J. Ripple, Natalie R. Bishop, Priya Rajapreyar, Liliana M. Lima, Rishikesan Kamaleswaran
- **Year:** 2025
- **Venue:** *Pediatric Research*, Vol. 99, No. 2, pp. 631–637
- **DOI / URL:** [10.1038/s41390-025-04221-8](https://doi.org/10.1038/s41390-025-04221-8)
- **Primary Finding:** Evaluates daily Phoenix sepsis prediction using centralized XGBoost models. Notes the high diagnostic salience of laboratory markers while underscoring that centralized training cannot be directly deployed across siloed healthcare institutions.
- **Relationship to Our Manuscript:** Contemporary benchmark of Phoenix-based pediatric ML, highlighting our extension to decentralized, federated training and explainability.
- **Placement in Revised Manuscript:** Section II-A & Section II-D.

---

### Entry 9: Systematic Review of Critical Care Explainable AI
- **Title:** *Explainable AI for Critical Care: A Systematic Review of Interpretable Models for Sepsis and ICU Mortality Prediction*
- **Authors:** V. S. Athukorala, W. M. K. S. Ilmini
- **Year:** 2026
- **Venue:** *BMC Medical Informatics and Decision Making*, Vol. 26, Article 64
- **DOI / URL:** [10.1186/s12911-026-03344-0](https://doi.org/10.1186/s12911-026-03344-0)
- **Primary Finding:** Comprehensive 2026 systematic review showing that while post-hoc explainers (SHAP, LIME) are increasingly applied to ICU risk models, over 95% evaluate centralized models; distributed or federated clinical explainability remains severely under-studied.
- **Relationship to Our Manuscript:** Establishes the clinical motivation and timeliness of our investigation.
- **Placement in Revised Manuscript:** Section I & Section II-C.

---

### Entry 10: Methodological Review of Sepsis Prediction and Data Leakage
- **Title:** *Evaluating Machine Learning Models for Sepsis Prediction: A Systematic Review of Methodologies*
- **Authors:** Hai-Fei Deng, Ming-Wei Sun, Yang Wang, Jie Zeng, Ting Yuan, Tao Li, Dong-Hang Li, Wei Chen, Ping Zhou, Qiang Wang, Hong Jiang
- **Year:** 2022
- **Venue:** *iScience* (Cell Press), Vol. 25, No. 2, Article 103651
- **DOI / URL:** [10.1016/j.isci.2021.103651](https://doi.org/10.1016/j.isci.2021.103651)
- **Primary Finding:** Demonstrates that subtle target leakage, improper temporal windowing (e.g., extracting predictors during or after diagnostic onset), and absence of decision-threshold analysis inflate reported AUROC across critical care publications.
- **Relationship to Our Manuscript:** Directly supports our scientific audit of `lactate_max` and 24-hour observation windows.
- **Placement in Revised Manuscript:** Section II-A and Section V (Limitations).

---

## 3. Summary of Citation Plan for Revised Manuscript

| Topic | Papers Cited | Purpose in Revision |
|:---|:---|:---|
| **Federated Explainability** | Bonsignori et al. (IEEE Access 2026), Ducange et al. (Springer 2026) | Replaces absolute novelty claim; frames methodological contribution relative to state-of-the-art. |
| **Explanation Aggregation** | Bonsignori et al. (2026), McMahan et al. (2017) | Compares sample-weighted, uniform, and stability-weighted aggregation. |
| **Privacy Threat Model** | Shokri et al. (IEEE S&P 2021), Patel et al. (FAccT 2022), Bonawitz et al. (CCS 2017) | Formalizes privacy threat model, reconstruction risks from summaries, and role of SecAgg/DP. |
| **Target Leakage & Phoenix Score** | Schlapbach et al. (2024), Sanchez-Pinto et al. (2024), Deng et al. (2022) | Audits 24-hour windowing and explains why lactate, platelets, and coagulopathy markers dominate. |
| **Pediatric Critical Care ML** | Chanci et al. (Pediatr Res 2025), Athukorala & Ilmini (BMC 2026) | Integrates recent 2024–2026 clinical machine learning benchmarks. |
