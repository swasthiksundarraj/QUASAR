# QATNet paper reproduction protocol: PDF-verified audit

## Source and audit scope

This audit is based on the complete local 15-page PDF:

`docs/paper/QATNet_A_Lightweight_QuantumClassical_Tabular_Network_for_Low-Latency_Intrusion_Detection.pdf`

All 15 pages were extracted and reviewed, including the architecture/circuit figures, equations, result tables, runtime table, conclusion, and limitations. No separate supplement was present in `docs/paper/`. Evidence references below use the PDF's printed page numbers (the PDF page number is the same). Reported paper results are described only as the paper's claims and are **not project results**.

## Paper summary

The paper proposes a flow-based IDS pipeline for SDN: train-only imputation/scaling/PCA reduces network-flow features to `D = 2^Q` dimensions; L2-normalized PCA vectors are amplitude-embedded; a shallow, one-time-pretrained `StronglyEntanglingLayers` VQC produces a full-basis probability vector; and an attentive TabNet head classifies that vector. The VQC is frozen for feature extraction/inference. Classical PCA–TabNet and amplitude-only TabNet ablations share preprocessing; random-forest, LightGBM, and XGBoost baselines are also evaluated on amplitude-only features. The article reports balanced five-class CIC-IDS2018 and OD-IDS2022 experiments, Q=4/5/6 dimensional budgets, three-seed evaluation, calibration and runtime measurements, plus a Q=5 backend-noise experiment. (PDF pp. 3–7, 9–13.)

The paper reports that its QATNet configuration performs best among its matched PCA-TabNet and amplitude-only pipelines, with additional cost mainly in one-time VQC pretraining, and claims sub-0.012-ms-per-sample inference. These are paper-reported claims only. They are not verified here and must not be copied into this project's result artifacts as measurements.

## Verified experimental protocol

### CIC-IDS2018

The paper explicitly specifies (PDF p. 7, Section IV-A):

- CIC-IDS2018 flow-level NetFlow/CICFlowMeter data, described as approximately 16.2 million flows.
- Five experiment classes, named `Benign`, `Bot`, `DoS-Hulk`, `FTP-BruteForce`, and `Infiltration`.
- Down-sampling to 200,000 total records, 40,000 per class.
- Removal of obvious identifiers such as IP addresses and ports.
- Missing-value imputation using training-set statistics; StandardScaler and PCA fit only on training data.
- Stratified train/validation/test proportions of 80/10/10.
- Q-dependent PCA dimension `D = 2^Q`, with zero-padding where required.

The methods text does **not** identify exact CIC CSV filenames/capture dates, random sampling algorithm, duplicate handling, or raw CIC label spellings. The class names are paper target names, not evidence that the underlying CSV uses those exact strings.

The repository's separate local audit (`cc-dataset-audit.md`) records ten extracted files and the following raw-label facts:

| Paper target name (PDF p. 7) | Observed exact local `Label` value | Local capture files and counts |
|---|---|---|
| Benign | `Benign` | Present in all ten files; 13,484,708 total |
| Bot | `Bot` | `03-02-2018.csv`; 286,191 |
| DoS-Hulk | `DoS attacks-Hulk` | `02-16-2018.csv`; 461,912 |
| FTP-BruteForce | `FTP-BruteForce` | `02-14-2018.csv`; 193,360 |
| Infiltration | `Infilteration` | `02-28-2018.csv` 68,871; `03-01-2018.csv` 93,063; 161,934 total |

This verifies that the **combined local collection** has enough examples for 40,000/class. It does not verify the exact files used by the paper. All five targets can be sourced from the five listed attack-bearing files collectively, but those dates are not consecutive. The article describes a five-consecutive-day, approximately 16.2-million-flow collection (p. 7), while the local `cc/` audit describes ten dated CSVs totaling 16,233,002 rows. This collection/date-scope discrepancy is unresolved; selecting either all ten files or the five attack-bearing dates is an implementation choice until the authors' source-file manifest is available.

The local source spells the label **`Infilteration`**. The paper names its experiment class **`Infiltration`** (p. 7) but does not print the raw source label or document a spelling conversion. Therefore the repository mapping `Infilteration` → `Infiltration` is a verified local alias to the paper's class name, **not** a paper-verified statement that the paper used that raw spelling. `DoS attacks-Hulk` → `DoS-Hulk` is likewise a local full-label-to-paper-name mapping. No fuzzy mapping or attack-family merge is supported.

### OD-IDS2022 (separate benchmark)

The paper describes OD-IDS2022 as 82 columns (81 numeric flow features plus label), approximately one million labeled flows, with 29 attack classes plus benign. It maps those source classes into five operational macro-classes: `Benign`, `SYN/DoS floods`, `Man-in-the-Middle/Spoofing`, `Fragmented-packet DoS`, and `Brute-Force`, then balances to 200,000 total / 40,000 per class. It says to remove IP/port IDs, fit imputation/scaling on train only, use PCA, and follow the same stratified 80/10/10 protocol (PDF p. 7, Section IV-A).

The PDF does not enumerate the raw OD-IDS2022 labels assigned to each macro-class, nor the exact CSV distribution/version or sampling method. The local project has no verified OD-IDS2022 data or label manifest. This must remain a separate future experiment; do not map CIC labels onto it or claim OD-IDS2022 results from this project.

### Common preprocessing, split, and feature encoding

The paper states that splits are stratified 80/10/10 in a balanced five-class setting; medians, standardization, PCA, and train statistics use the training partition only; identical fitted scaler/PCA are reused across baselines; identifiers (source/destination IPs and ports) are removed; and PCA output is L2-normalized (PDF pp. 5–7, Sections III-A, III-B and IV-A).

For each Q in `{4, 5, 6}`, use `D = 16, 32, 64` PCA coordinates, respectively. If the fitted PCA rank is smaller than D, pad with zeros. Apply amplitude embedding to the normalized D-vector. The paper's amplitude-only ablation measures immediately after amplitude embedding and sends `|x|^2` basis probabilities to TabNet (PDF p. 5, Eq. 1 and Section III-A). The VQC output is the full computational-basis probability vector `p_k = |<k|U_theta U_phi(x)|0^Q>|^2`, summing to one (PDF pp. 5–6, Eqs. 2–3).

### VQC, pretraining, and freezing

The paper specifies one shallow `StronglyEntanglingLayers` block after amplitude embedding. Figure 3 describes RX–RY–RZ rotations on each qubit and ring CNOT entanglers; the parameter tensor is `theta ∈ R^(L×Q×3)`. It says to pretrain once per configuration on approximately 3,000 class-balanced training samples (about 600/class), for 15 epochs, batch size 256, then freeze `theta` and cache the probability features for all splits (PDF pp. 4–7, Fig. 3, Eqs. 1–3 and Section IV-A). PennyLane and its `lightning.gpu` backend are named for the quantum path (p. 7).

The PDF does **not** specify the VQC pretraining loss, optimizer, learning rate, initialization, validation/stopping rule, exact subset sampler, shots/measurement scheme, or whether the VQC objective is supervised. It says the goal is to shape the embedding geometry, not to make the circuit itself the final classifier (p. 4). Do not treat the repository's supervised linear-probe cross-entropy/Adam objective as paper-verified.

### TabNet training and model selection

The paper says the shared TabNet head uses standard class-weighted cross-entropy plus TabNet mask sparsity regularization (PDF p. 6, Eqs. 4–5). It specifies Adam, validation early stopping with patience 15, maximum 60 epochs, batch size 4096 and virtual batch size 512, and class-balanced instance weights (PDF p. 7). It describes a small validation-set grid over depth/width and mask type, selected with seed 101. The chosen configuration is reused across seeds 101, 202, and 402; after hyperparameter selection, it says the final model is retrained on train plus validation before held-out test evaluation (PDF pp. 6–7).

Exact grid values, selection objective, class-weight formula, sparsity coefficient, TabNet implementation/version, and how the retraining uses validation examples while keeping preprocessing train-fit are not specified. The repository's opt-in `PaperTabNetProfile` uses a small candidate grid as an assumption, selects macro-F1 on validation, monitors validation accuracy for early stopping, and uses the selected candidate's best epoch as the fixed number of train-plus-validation refit epochs. The existing train-fitted scaler/PCA matrices are reused for refit; they are not refitted on validation.

The installed `pytorch-tabnet` version 4.1.0 was inspected. Its `fit(weights=...)` parameter is documented/implemented as automatic or custom training-sample balancing through the data loader, not as the class-weight argument to cross-entropy. The paper workflow therefore computes inverse-frequency weights from the original training labels and passes a `torch.nn.CrossEntropyLoss(weight=...)` through the supported `loss_fn` argument; it passes `weights=0` to avoid a second sampler-based balance. In the paper's already balanced 40,000-per-class experiment these loss weights should be equal. TabNet's optimizer is explicitly `torch.optim.Adam`; its installed early-stopping callback monitors the last `eval_metric` and restores the best validation weights. The profile uses accuracy for this stopping metric because the paper does not name one. `device_name="auto"` is the portable profile default; the PDF's one RTX A4500/CUDA 12.4 setup cannot be assumed on this Mac, so actual hardware is recorded and timing comparisons must state the deviation.

### Metrics, seeds, runtime, and reported findings

- Predictive metrics: Accuracy, Macro-F1, and Macro-AUROC; ECE is the calibration measure (PDF pp. 4, 7).
- Repetitions: three independent seeds, explicitly 101, 202, and 402; report means. The paper states standard deviations below 0.2% for core metrics and displays bootstrapped confidence intervals in a figure (PDF pp. 7–8, 12–13). Bootstrap resampling details are not given.
- Runtime: report total training, one-time VQC pretraining, encoding, and end-to-end inference latency in ms/sample (PDF pp. 4, 7, 9, Table 5 p. 10). The stated hardware is one NVIDIA RTX A4500, driver 550.90.07, CUDA 12.4, no multi-GPU; PCA/scikit-learn runs on CPU and TabNet on GPU (PDF p. 7).
- The paper claims that QATNet leads its matched PCA–TabNet and Amp-only baselines, that QATNet's Q=6 CIC result is strongest, and that noise/runtime are favorable (PDF pp. 8–13). Exact paper-table values are deliberately not reproduced in this protocol and are not project results.

### Quantum-noise study

The detailed noise subsection states that it uses Q=5 (32 dimensions), Qiskit Aer noise models derived from IBM FakeNairobi and FakeJakarta backends. It trains/tests XGBoost, LightGBM, Random Forest, and one stacked ensemble on noisy amplitude features. **It explicitly says QATNet itself was not executed under noise**; the experiment studies noisy amplitude-feature representations with classical heads (PDF p. 9, Section IV-D and Table 4). Results are averaged over three runs (Table 4 footnote).

The PDF does not state backend calibration snapshot/version, exact circuit/transpilation, shots, selected data split/sample size, noise execution scheduling, model hyperparameters, or whether noise metrics are for CIC or OD-IDS2022. The abstract and contribution text describe QATNet as robust under noise, while the detailed method clarifies the QATNet VQC+TabNet model is not run under noise. This scope/wording discrepancy must be retained in any reproduction report.

## Implementation gap analysis

The companion [`reproduction-matrix.md`](reproduction-matrix.md) is the authoritative requirement-by-requirement matrix. Most important gaps before a faithful full training run are:

1. **Amplitude-only semantics:** corrected. `amplitude_features()` remains the signed L2-normalized vector for VQC state preparation, while `amplitude_probability_features()` squares that vector for the amplitude-only path as specified by the paper. Focused tests verify nonnegative normalized probabilities, sign invariance, dimensions, zero-vector behavior, and validation; this change does not alter VQC inputs.
2. **VQC objective and circuit details:** current code uses a supervised linear probe with cross-entropy/Adam; the paper does not disclose its pretraining objective. Current PennyLane `StronglyEntanglingLayers` should also be checked against Fig. 3's RX–RY–RZ plus ring-CNOT layout rather than assumed identical.
3. **TabNet configuration:** the legacy training defaults remain 100 epochs and are not the paper profile. The opt-in `PaperTabNetProfile` now uses max 60, patience 15, batch 4096, virtual batch 512, training-only class weights, validation selection and train+validation refits. Its candidate grid, selection metric, early-stopping metric, sparsity coefficient, and fixed refit epoch rule remain implementation assumptions.
4. **Seed/repetition protocol:** `paper-tabnet` orchestrates validation selection at seed 101, final model seeds 101/202/402, separate caller-provided split seed, and per-seed plus aggregate metrics. No full-size repeated-seed run has been launched.
5. **Hardware/backend:** current VQC uses PennyLane `default.qubit` CPU, whereas the paper states `lightning.gpu`; the paper's RTX A4500/CUDA setup is not available in the current macOS environment. Results must be reported with actual local hardware and must not claim hardware parity.
6. **Dataset provenance:** mapping aliases are transparent and local class sufficiency is verified, but exact paper CIC source files/date scope and raw spellings are not reported. The paper's "five consecutive days" description conflicts with this checkout's ten dated files.
7. **Noise experiment:** current code uses a local depolarizing channel on identity gates, then Logistic Regression/Random Forest/SVC/stacking. It does not use FakeNairobi/FakeJakarta-derived Qiskit Aer noise models or the paper's XGBoost/LightGBM/Random Forest/stack heads.
8. **OD-IDS2022:** no dataset, raw-label-to-macro-class map, or separate run is present.

## Paper/report inconsistencies and unspecified details

- Dataset scope: PDF p. 7 says five consecutive capture days; the local audit finds ten dated CSV files, while target attack labels are distributed across dates that are not consecutive. The paper does not list filenames.
- CIC feature count: PDF p. 7 describes 80 flow attributes plus a label. The local audit finds nine files with 80 total columns and `02-20-2018.csv` with 84 total columns. A source-release/schema match has not been established.
- Infiltration: PDF p. 7 reports the normalized class name `Infiltration`; the local raw label is `Infilteration`. The PDF does not state the raw spelling or normalization.
- Noise scope: abstract/contribution wording says QATNet was robust under backend noise, but Section IV-D p. 9 says the QATNet model itself was not executed under noise.
- Latency: abstract p. 1 gives 0.012 ms/sample; contribution 3 p. 3 says approximately 0.71 ms/sample end-to-end and 0.007–0.009 ms/sample for the TabNet head; Table 5 p. 10 gives Q=5 inference values including 0.0115 ms/sample for CIC QATNet. These appear to use different timing scopes or contain a reporting inconsistency; do not target one number without clarifying exact timing boundaries.
- Paper says "five consecutive days" and "80 flow-level attributes" but does not list file manifest, exact source schema, sampling algorithm, or duplicates.
- VQC pretraining objective and TabNet sweep values/weighting are unavailable.
- OD-IDS2022 raw class mapping and noise experiment dataset/split are unavailable.
- Noise backend snapshots, shots, and calibration details are unavailable.
- The article describes fixed seeds but the exact seed list (101/202/402) appears in the methods and should be followed; it does not explain bootstrap CI settings.

## Corrections required before full training

Do not launch the 200,000-row model run yet. First make and test the following scoped changes:

1. Align the Q=4/5/6 circuit gate layout with Fig. 3, or document and justify a deviation. Keep the missing VQC pretraining objective explicitly marked as an implementation assumption until author/code clarification is available; do not claim exact reproduction for it.
2. The explicit `PaperTabNetProfile` configuration and validation-sweep/refit workflow are implemented. Keep its TabNet batch settings separate from VQC batch size 256; the missing sweep/refit specifics remain assumptions.
3. Run the opt-in profile for validation-only candidate selection at seed 101, then final runs at 101/202/402. The split seed is a distinct caller-provided value. Per-seed metrics and a single-dataset/pipeline aggregate are written to a new run directory. The exact candidate grid, validation objective, and refit epoch rule remain implementation assumptions.
4. Keep `cc/` source-file policy and class alias in every run manifest; report per-file class counts, excluded identifiers, duplicate policy, split counts, seeds, hardware/backend. Confirm the paper capture-day list before making an exact-reproduction claim.
5. Keep OD-IDS2022 and CIC-IDS2018 runs separate. Do not start OD-IDS2022 until its source data and explicit raw-to-macro-class map are obtained.
6. Label the current local depolarizing study as a separate noise ablation. To reproduce the paper noise study, use the named FakeNairobi/FakeJakarta backend noise models and paper-listed classical heads after clarifying unavailable run details.
7. Profile memory and runtime on the actual target machine before the first full run. Use the paper's hyperparameters only after implementing them; the paper's RTX A4500 timing claims are not portable to this Mac.

## Evidence references

- Architecture, phases, deployment path, equations (1)–(5), VQC and TabNet objective: PDF pp. 4–6, Sections III-A through III-C, Figs. 1–4.
- CIC/OD-IDS2022 descriptions, balancing, split, PCA widths, seeds, batch/epoch settings, metrics, and baseline list: PDF p. 7, Section IV-A.
- Comparative result tables: PDF pp. 8–9, Tables 1–4 (paper results only).
- Noise method/scope: PDF p. 9, Section IV-D and Table 4.
- Runtime measurements and conflicting latency statements: PDF pp. 3, 9–10, Table 5.
- Interpretability: PDF p. 11, Sections IV-F/G; efficiency and limitations: PDF p. 12, Section IV-H; multi-seed figures and conclusion: PDF p. 13, Section V. References and author biographies occupy PDF pp. 14–15.
- Local CIC counts/schema: [`cc-dataset-audit.md`](cc-dataset-audit.md).
