# QATNet–QUASAR Project Progress

**Status date:** 30 October 2026
**Project:** QATNet research reproduction and QUASAR SDN/QKD security prototype
**Dataset in the latest model runs:** CIC-IDS2018, 5,000-row balanced local pilot

## Executive status

The original Q=4 QATNet pilot had a real inference failure: all three TabNet seeds predicted every test row as Infiltration. Artifact-only diagnosis traced this to a mismatch between TabNet's training-mode BatchNorm statistics and the running statistics used for inference. Recalibrating the final refit model's BatchNorm running statistics from its train-plus-validation feature rows removed the all-one-class collapse in a fresh run on the same split.

The correction is supported by checkpoint evidence and a regression test, and the corrected checkpoints reproduce their saved metrics. It is **not yet independently validated on a new split**. The corrected test set is the same one already examined in the original run; treat the follow-up test scores as descriptive pilot results, not an untouched confirmation.

### Progress estimates

| Area | Estimated completion | Basis |
|---|---:|---|
| QATNet paper reproduction | **About 55%** | Dataset audit, leakage-safe preprocessing, classical/amplitude/Q=4 QATNet implementations, five-split Q=4 representation study, and one corrected Q=4 pilot are complete. Full-scale runs, Q=5/Q=6, verified paper-method details, external dataset, and noise study remain. |
| QUASAR system | **About 35%** | Offline telemetry, synthetic QKD, risk fusion, QUBO, verification, mock actions/rollback, audit and dashboard components exist. A tested live Mininet/OVS/Ryu-to-detector integration and safe deployment validation remain. |

These are **rough planning estimates**, not measured percentages or claims of paper completion. They estimate completion against the end-to-end goals recorded in the project documentation; a working module does not count as a validated end-to-end research result.

## QATNet pilot: before and after

Both experiments used the same CIC-IDS2018 local 5,000-row balanced sample, Q=4 / 16 PCA components, split seed 101, and model seeds 101, 202, and 402. Each split contained 4,000 training, 500 validation, and 500 test rows.

| Result | Original pilot | Corrected pilot |
|---|---:|---:|
| Accuracy, mean ± population SD | 0.2000 ± 0.0000 | 0.8013 ± 0.0236 |
| Macro-F1, mean ± population SD | 0.0667 ± 0.0000 | 0.7983 ± 0.0174 |
| Macro-AUROC, mean ± population SD | 0.5700 ± 0.0846 | 0.9580 ± 0.0077 |
| ECE, mean ± population SD | 0.3649 ± 0.0481 | 0.0538 ± 0.0254 |
| Test prediction behavior | All 500 predictions were Infiltration for every seed | Predictions covered all five classes for each seed |

Corrected-run per-class recall (mean ± population SD across model seeds):

| Class | Recall |
|---|---:|
| Benign | 0.523 ± 0.040 |
| Bot | 0.833 ± 0.222 |
| DoS-Hulk | 0.987 ± 0.005 |
| FTP-BruteForce | 1.000 ± 0.000 |
| Infiltration | 0.663 ± 0.061 |

The corrected model no longer predicts a single class, but Bot recall is variable across seeds. These values describe this small pilot only.

## What was wrong and how it was addressed

The saved original TabNet checkpoints had decreasing recorded training losses, but in ordinary evaluation mode their outputs were nearly constant across examples and their predictions collapsed to Infiltration. The failure was also present on the final-refit training rows, so it was not simply a test-only generalization failure.

Evidence pointed to BatchNorm running-statistic mismatch:

- The saved source-file SHA-256, class order, and reconstructed train/validation/test feature-label fingerprints matched the original run manifest.
- The frozen VQC probability features were finite, normalized, and retained measurable class signal. The saved pretraining probe scored 69.8% validation accuracy and 0.6822 validation Macro-F1.
- With the saved TabNet weights in evaluation mode, all three seeds predicted Infiltration for every row. A cloned network using current-batch statistics instead generated multi-class predictions. This comparison was diagnostic only, not a valid evaluation.
- Reconstructing predictions from the original saved checkpoints reproduced the original reported test accuracy, Macro-F1, and confusion matrices.

The change recalibrates BatchNorm running statistics **after final QATNet refitting** using only the features already present in the train-plus-validation refit set. It uses no test rows or labels and does not update learned model parameters. Classical and amplitude paper-profile runs are not changed by this QATNet-specific behavior. A focused regression test checks that recalibration updates running statistics while preserving learned parameters.

## Current QATNet status

**Verified complete or implemented**

- Local CIC-IDS2018 files and five-class pilot data have been audited; the local dataset is CIC-IDS2018, not CIC-IDS2017.
- Train-only imputation/scaling/PCA, stratified 80/10/10 splitting, identifier exclusions, class mappings, and run manifests are implemented.
- Classical PCA, classical amplitude-probability, frozen-VQC QATNet, paper-profile TabNet workflow, metrics, and artifact persistence are present.
- Q=4 signed PCA vs. normalized signed PCA vs. normalized squared probability features were compared across five deterministic splits. Signed PCA ranked first by mean test Macro-F1 on all five pilot splits.
- Original Q=4 QATNet collapse was diagnosed without retraining; a same-split follow-up with BatchNorm recalibration completed and its checkpoints were verified.
- The latest full test suite passed: **40 tests**.

**Still required before claiming a complete paper reproduction**

1. Validate the corrected QATNet behavior on a predeclared new split with an untouched test set. Revisit validation candidate selection/early stopping: the original selection runs themselves were not recalibrated, and validation best accuracy was low.
2. Resolve or clearly retain as assumptions the VQC objective and pretraining subset choice. The current Q=4 pilot used supervised linear-probe cross-entropy and all 4,000 training rows (800/class), while the paper describes an approximately 3,000-row balanced subset and does not fully specify the objective.
3. Execute properly estimated paper-scale CIC-IDS2018 experiments (40,000 per class) only after the method is stable. Also complete Q=5 and Q=6 configurations and repeated-seed reporting under the validated procedure.
4. Obtain and verify OD-IDS2022 and its raw-to-macro class mapping before implementing that dataset result; it is not available in the current local workflow.
5. Address the paper's noise-study scope and backend details, then finish the reproducibility/results write-up. The current local noisy-feature experiment is not the full paper noise experiment.

The current QATNet implementation is **not** evidence of full CIC-IDS2018 performance, exact reproduction, or quantum advantage. The amplitude-probability branch is a classical feature transformation, not a quantum circuit.

## Current QUASAR status

**Present in the project**

- Offline CIC flow replay and a QATNet detection-service interface.
- Synthetic QKD telemetry simulation, including key-pool/QBER-related state.
- SDN/QKD risk fusion and a small mitigation QUBO.
- Classical exact solution for small instances and optional QAOA sampling.
- Independent mitigation-plan verification, mock-only application/rollback, audit logging, and a dashboard.
- A Mininet/OVS statistics adapter and isolated-lab guidance.

**Still required for an end-to-end QUASAR demonstration**

1. Deploy and test Mininet/Open vSwitch with an SDN controller (for example Ryu) in a Linux VM.
2. Connect controller telemetry to the separate detection service; verify input schema, response, timeout/error handling, and model-version reporting.
3. Correlate controller events and QKD/KMS telemetry by timestamp; keep simulated QKD results explicitly separate from real measurements.
4. Evaluate fusion and mitigation on repeatable emulator scenarios. Compare classical and QAOA on identical QUBO instances and report measured feasibility/runtime, without asserting quantum advantage.
5. Test policy verification, connectivity impact, rollback, and audit logs under failures. Keep live network actions disabled until those checks pass.
6. Complete dashboard integration and end-to-end evaluation. The existing demonstration is offline/synthetic, not a live network deployment.

## Run and evidence locations

- Original collapsed pilot: `outputs/runs/cic-paper-tabnet-pilot-qatnet-q4-seed101/`
- Corrected same-split pilot: `outputs/runs/cic-paper-tabnet-pilot-qatnet-q4-seed101-bn-calibration-20261030/`
- Diagnosis, scripts, distributions, and corrected-run verification: `outputs/diagnostics/qatnet-q4-collapse-20261030-1757/`
- Reproduction tracking: `docs/reproduction-matrix.md` and `docs/paper-reproduction-protocol.md`
- Five-split representation comparison: `outputs/metrics/representation-ablation-split-sensitivity-q4.md`

The corrected pilot took 14.90 seconds and peaked at 592.6 MiB RSS on the current macOS environment. This is measured pilot cost, not a full-scale resource estimate. The original run and its artifacts were preserved.

## Immediate priorities

1. **QATNet:** Run a predeclared, held-out split comparison of recalibration enabled vs. disabled, with all other settings controlled. Do not select or tune using its test results.
2. **Research method:** Confirm paper VQC pretraining details if a reliable source is available; otherwise explicitly report the objective and subset size as deviations.
3. **Paper:** Prepare results and limitations around the five-split classical representation ablation and the QATNet pilot, clearly labeling sample size and known deviations.
4. **QUASAR:** Build and test the controller-to-detector connection in an isolated Linux emulator before enabling any mitigation action.
5. **Do not yet:** Launch the 200,000-row CIC run, claim QATNet full-scale success, or present the synthetic/offline QUASAR replay as live SDN/QKD operation.
