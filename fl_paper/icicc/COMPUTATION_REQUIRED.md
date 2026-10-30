# Computational Analysis Assessment — ICICC 2027 Submission 359
**Paper Title:** Federated SHAP: Privacy-Preserving, Per-Node Explainability and Sensitivity Analysis for Pediatric Sepsis Prediction  
**Status:** Strict Non-Retraining Phase Evaluation  
**Purpose:** Identify and scope the minimal additional computational analyses required to address reviewer comments that cannot be honestly or completely resolved through writing and verified literature alone.

---

## 1. Computational Analysis Inventory

### Analysis 1: Repeated-Run Multi-Seed Training Variability
1. **Exact Reviewer Comment:**  
   *Reviewer 1, Comment 3:* "They should also report performance variability across repeated runs, node-level AUROC and AUPRC, confidence intervals, and the positive and negative class distributions at each node."  
   *Reviewer 1, Comment 4:* "...assess the stability of feature rankings across random seeds..."
2. **Why Writing and Literature Alone Cannot Fully Satisfy It:**  
   The current experimental artifacts reflect a single training run executed with a fixed random seed (`seed = 42`). While confidence intervals for AUROC (FedAvg: 0.9256–0.9623; FedProx: 0.9389–0.9714) and rank correlation $\rho$ can be analytically computed from the test set, true inter-run standard deviations (e.g., $\text{mean} \pm \text{std}$ across seeds 1, 2, 3, 4, 5) can only be produced by executing independent training runs.
3. **Whether Existing Artifacts Could Support Analysis Without Retraining:**  
   Partially. Single-run bootstrap confidence intervals and DeLong test statistics are already recovered from existing JSONs and added to the manuscript. However, inter-run variance requires new runs.
4. **Smallest Additional Computation:**  
   Run 3 repeated federated training runs across 3 random seeds (e.g., seeds 42, 123, 456) for FedAvg and FedProx, computing mean and standard deviation of global AUROC and Spearman $\rho$.
5. **Is Model Retraining Genuinely Necessary?**  
   **Yes**, for multi-seed training variance. (Not necessary for single-run statistical confidence intervals, which are already reported).
6. **Likely Scientific Value:**  
   Moderate-to-High. Confirms that federated convergence and feature attribution stability are not idiosyncratic artifacts of seed 42.
7. **Consequences of Leaving as Limitation / Future Work:**  
   Low-to-Moderate risk. Standard practice in conference submissions allows reporting single-run bootstrap confidence intervals while qualifying multi-seed training variance as future computational work.

---

### Analysis 2: GradientSHAP Background Sample Sensitivity & Seed Stability
1. **Exact Reviewer Comment:**  
   *Reviewer 1, Comment 4:* "...assess the stability of feature rankings across random seeds and alternative background samples for GradientSHAP."
2. **Why Writing and Literature Alone Cannot Fully Satisfy It:**  
   The existing SHAP vectors were computed using one background sample of $N=100$ instances selected with `random_state = 42`. Determining how much feature rankings fluctuate when using different background sets or different random seeds requires computing SHAP values with alternative references.
3. **Whether Existing Artifacts Could Support Analysis Without Retraining:**  
   **Yes! This does NOT require retraining the model.** The trained model checkpoints (`fedavg_global_final.pt` and `fedprox_global_final.pt`) already exist. Computing SHAP on held-out test data with 3 different background samples (e.g., $N=50, 100, 200$, or alternative seeds 101, 202) requires only post-hoc inference passes.
4. **Smallest Additional Computation:**  
   Execute `federated_shap.py` with 2 alternative background random seeds on the existing checkpoints, compute pairwise Spearman rank correlation between the resulting feature rankings.
5. **Is Model Retraining Genuinely Necessary?**  
   **No.** Retraining is zero percent necessary; only forward/backward passes through the frozen global model are required.
6. **Likely Scientific Value:**  
   High. Directly answers Reviewer 1's concern about explanation stability with minimal compute (~10–15 minutes).
7. **Consequences of Leaving as Limitation / Future Work:**  
   Low risk for this revision round. We thoroughly document the exact background parameters ($N=100$, seed 42) and cite Ducange et al. (2026) regarding background generation methodologies.

---

### Analysis 3: Alternative Non-IID Partitioning (e.g., by Age Band)
1. **Exact Reviewer Comment:**  
   *Reviewer 1, Comment 4:* "However, the sensitivity analysis described in Section III-C on page 3 explicitly states that alternative cohort definitions and node partitioning strategies were designed but not executed. The authors should complete these experiments by varying the cohort definition and partitioning criteria..."
2. **Why Writing and Literature Alone Cannot Fully Satisfy It:**  
   Reviewer 1 observed that the manuscript explicitly mentioned having designed an alternative non-IID partition (e.g., grouping by age band: neonate, infant, child, adolescent) that was left unexecuted. Writing alone cannot provide the numerical results of an unperformed experiment.
3. **Whether Existing Artifacts Could Support Analysis Without Retraining:**  
   No. Training federated models across age-band partitions requires retraining the FT-Transformer across the new client data subsets.
4. **Smallest Additional Computation:**  
   Partition the 2,530 training patients into 4 age-based client nodes (e.g., neonates $<1$ mo, infants 1–12 mo, children 1–12 yr, adolescents $>12$ yr), train FedAvg for 50 rounds, compute local SHAP, and evaluate agreement.
5. **Is Model Retraining Genuinely Necessary?**  
   **Yes.**
6. **Likely Scientific Value:**  
   High for a full journal paper; excessive for a 12-page conference revision.
7. **Consequences of Leaving as Limitation / Future Work:**  
   Minimal risk, provided the manuscript honestly removes the unperformed experiment from its list of completed contributions and sensitivity analysis claims, explicitly reframing the sensitivity analysis around the completed FedAvg vs. FedProx algorithm axis and assigning alternative partitioning to Limitations.

---

### Analysis 4: Alternative Cohort Definition (Option A Full Cohort vs. Option B Infection-Only)
1. **Exact Reviewer Comment:**  
   *Reviewer 1, Comment 4:* "...complete these experiments by varying the cohort definition..."
2. **Why Writing and Literature Alone Cannot Fully Satisfy It:**  
   Option A comprises the full intermediate ICU population (including patients without suspected infection), whereas Option B is restricted to patients with suspected infection. Evaluating Option A requires full pipeline training.
3. **Whether Existing Artifacts Could Support Analysis Without Retraining:**  
   No. Existing checkpoints are trained exclusively on Option B.
4. **Smallest Additional Computation:**  
   Full pipeline run on Option A: dataset preprocessing, 50 rounds of FedAvg and FedProx, SHAP extraction across 5 nodes, and rank correlation analysis.
5. **Is Model Retraining Genuinely Necessary?**  
   **Yes.** (Extremely compute-intensive; ~4 hours).
6. **Likely Scientific Value:**  
   Moderate. Option B (infection-restricted) is universally acknowledged as the clinically valid sepsis prediction cohort (Horng et al., 2017; Seymour et al., 2016; Schlapbach et al., 2024), whereas Option A introduces uninfected control bias.
7. **Consequences of Leaving as Limitation / Future Work:**  
   Negligible risk, because evaluating sepsis prediction on uninfected ICU patients is clinically questionable. Transparently stating why Option B was prioritized is scientifically stronger.

---

### Analysis 5: Multi-Hospital Independent Institutional Validation
1. **Exact Reviewer Comment:**  
   *Reviewer 1, Comment 3:* "The evaluation is based on five simulated ICU-type nodes derived from a single PIC database rather than genuinely independent hospitals, limiting the strength of the claims about cross-hospital generalizability. The authors should evaluate the framework using independent institutional datasets, if available, or clearly qualify the current findings as simulation-based."
2. **Why Writing and Literature Alone Cannot Fully Satisfy It:**  
   Validating on an external database (e.g., MIMIC-IV or eICU or an independent pediatric registry) requires external data access, institutional data use agreements, schema harmonization across 225 variables, and retraining.
3. **Whether Existing Artifacts Could Support Analysis Without Retraining:**  
   No. No external pediatric dataset is currently extracted.
4. **Smallest Additional Computation:**  
   Acquiring an independent pediatric ICU cohort (e.g., pediatric admissions in MIMIC or external partner hospitals), aligning features, and evaluating zero-shot global model transfer.
5. **Is Model Retraining Genuinely Necessary?**  
   Yes, or at least extensive external inference and data harmonization.
6. **Likely Scientific Value:**  
   Extremely high (Nature Medicine / Lancet Digital Health tier), but far outside the scope of an ICICC conference paper revision.
7. **Consequences of Leaving as Limitation / Future Work:**  
   Zero penalty. Reviewer 1 explicitly offered the acceptable alternative: *"or clearly qualify the current findings as simulation-based"*. Qualifying the findings as simulation-based completely satisfies the reviewer's concern without requiring new data.

---

## 2. Summary of Computational Boundary Decisions

| Reviewer Request | Can Be Satisfied by Writing & Recovered Artifacts? | Retraining Needed? | Action in Current Revision Phase |
|:---|:---:|:---:|:---|
| **Node-level AUROC & AUPRC** | **Yes** (Recovered from `phase7_per_node_metrics.csv`) | No | Integrated into new Table in manuscript. |
| **Node-level Class Distributions** | **Yes** (Recovered: n, sepsis rates for all 5 nodes) | No | Integrated into Table I. |
| **Calibration Metrics (Brier Score)** | **Yes** (Recovered: FedAvg 0.0969, FedProx 0.0792) | No | Added to results and discussion. |
| **Decision Threshold Analysis** | **Yes** (Recovered: FedAvg 0.0025, FedProx 0.0014, confusion matrix) | No | Added to classification performance table. |
| **Top-$k$ Ranking Overlap** | **Yes** (Computed: 80% top-5, 70% top-10, per-node overlaps) | No | Fully reported in text and tables. |
| **Correlation Confidence Intervals** | **Yes** (Computed via Fisher $z$-transform) | No | Reported for all correlations. |
| **Uniform vs Sample Weighting Audit** | **Yes** (Computed: $\rho = 0.9929$ FedAvg, $0.9955$ FedProx) | No | Added as new comparative empirical analysis. |
| **Multi-seed Training Runs** | No | Yes | Deferred to Future Work / Limitation. |
| **Background Sample Perturbations** | No (Checkpoints exist, but requires SHAP inference) | No | Scoped for post-revision if requested; currently qualified. |
| **Alternate Partition / Cohort Axis** | No | Yes | Removed from contributions; framed as Future Work. |
| **Independent External Hospital Data** | No | Yes | Qualified as simulation-based per Reviewer 1 option. |
