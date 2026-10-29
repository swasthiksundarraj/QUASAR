# Q=4 representation ablation: five-split sensitivity analysis

## Scope and predeclared runs

This report covers five deterministic stratified splits of the **same 5,000-row CIC-IDS2018 local pilot**, not five independent datasets. Split seeds 101 and 202 were existing runs; split seeds **303, 404, and 505 were predeclared before execution**. Each split has model seeds 101, 202, and 402. Per-split ± values are population standard deviations across model seeds. Overall ± values are population standard deviations across the five split-level means; these describe variability among these partitions only and are not inferential uncertainty or statistical significance.

Input CSV: `outputs/runs/cic-paper-pilot-20261029-1313/sampled.csv`

SHA-256 (all runs): `666dfb89ade42bf93a195fc9636b94778e3d1e7c7c8a57403676e7b593a9cda1`

Input: 5,000 rows, 80 columns including `Label`; 1,000 per class: Benign, Bot, DoS-Hulk, FTP-BruteForce, Infiltration.
Q=4 produced 16 PCA components. Imputation, scaling, and PCA were fitted on training data only; `Dst Port` and `Timestamp` were excluded as identifiers. A is signed PCA, B is row-wise L2-normalized signed PCA, and C is row-wise L2-normalized coordinates squared into probabilities. C is a classical feature transformation, not a quantum circuit.

All branches use the same split arrays within each run and the same TabNet `(n_d=8, n_a=8, n_steps=3, sparsemax)`, Adam, training-label-derived class weights, batch size 4096, virtual batch size 512, validation-accuracy early stopping (maximum 60 epochs, patience 15), and fixed 57-epoch train-plus-validation refit. The held-out test partition is used only for final evaluation.

Encoded class order: `0=Benign`, `1=Bot`, `2=DoS-Hulk`, `3=FTP-BruteForce`, `4=Infiltration`.

### Split manifest and measured resources

| Split seed | Train / validation / test | Per-class train / validation / test | PCA components | Exit status | Wall time | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 4000 / 500 / 500 | 800 / 100 / 100 | 16 | 0 | 56.57 s | 526.4 MiB |
| 202 | 4000 / 500 / 500 | 800 / 100 / 100 | 16 | 0 | 54.83 s | 525.6 MiB |
| 303 | 4000 / 500 / 500 | 800 / 100 / 100 | 16 | 0 | 55.46 s | 531.9 MiB |
| 404 | 4000 / 500 / 500 | 800 / 100 / 100 | 16 | 0 | 498.05 s | 539.8 MiB |
| 505 | 4000 / 500 / 500 | 800 / 100 / 100 | 16 | 0 | 57.40 s | 531.4 MiB |

## Per-split test metrics

Mean ± population SD across model seeds (101, 202, 402). ECE is better when lower.

### Accuracy

| Representation | 101 | 202 | 303 | 404 | 505 |
|---|---:|---:|---:|---:|---:|
| A. Signed PCA | 0.804 ± 0.042 | 0.778 ± 0.038 | 0.827 ± 0.012 | 0.816 ± 0.004 | 0.832 ± 0.006 |
| B. L2-normalized signed PCA | 0.405 ± 0.117 | 0.451 ± 0.079 | 0.385 ± 0.006 | 0.351 ± 0.072 | 0.211 ± 0.013 |
| C. Squared normalized probabilities | 0.204 ± 0.047 | 0.284 ± 0.059 | 0.186 ± 0.018 | 0.267 ± 0.104 | 0.204 ± 0.003 |

### Macro-F1

| Representation | 101 | 202 | 303 | 404 | 505 |
|---|---:|---:|---:|---:|---:|
| A. Signed PCA | 0.794 ± 0.043 | 0.767 ± 0.036 | 0.818 ± 0.011 | 0.804 ± 0.004 | 0.822 ± 0.010 |
| B. L2-normalized signed PCA | 0.336 ± 0.146 | 0.385 ± 0.082 | 0.322 ± 0.006 | 0.272 ± 0.091 | 0.107 ± 0.030 |
| C. Squared normalized probabilities | 0.105 ± 0.055 | 0.182 ± 0.062 | 0.076 ± 0.013 | 0.178 ± 0.127 | 0.114 ± 0.044 |

### Macro-AUROC

| Representation | 101 | 202 | 303 | 404 | 505 |
|---|---:|---:|---:|---:|---:|
| A. Signed PCA | 0.9575 ± 0.0060 | 0.9566 ± 0.0052 | 0.9599 ± 0.0018 | 0.9589 ± 0.0036 | 0.9604 ± 0.0032 |
| B. L2-normalized signed PCA | 0.7666 ± 0.0185 | 0.7936 ± 0.0396 | 0.7651 ± 0.0487 | 0.7779 ± 0.0811 | 0.7179 ± 0.0542 |
| C. Squared normalized probabilities | 0.6685 ± 0.1396 | 0.6897 ± 0.0400 | 0.6126 ± 0.0886 | 0.7146 ± 0.0785 | 0.6458 ± 0.0557 |

### ECE

| Representation | 101 | 202 | 303 | 404 | 505 |
|---|---:|---:|---:|---:|---:|
| A. Signed PCA | 0.0908 ± 0.0413 | 0.0699 ± 0.0424 | 0.0467 ± 0.0064 | 0.0334 ± 0.0059 | 0.0564 ± 0.0128 |
| B. L2-normalized signed PCA | 0.2166 ± 0.0847 | 0.2170 ± 0.0539 | 0.2638 ± 0.0362 | 0.2123 ± 0.0816 | 0.3619 ± 0.0039 |
| C. Squared normalized probabilities | 0.3130 ± 0.0480 | 0.3698 ± 0.0212 | 0.4012 ± 0.0075 | 0.3520 ± 0.0274 | 0.2997 ± 0.0376 |

## Overall descriptive summaries across split-level means

Mean ± population SD of each representation’s five split-level model-seed means; this is descriptive split variability on one fixed sampled dataset.

| Representation | Accuracy | Macro-F1 | Macro-AUROC | ECE |
|---|---:|---:|---:|---:|
| A. Signed PCA | 0.8113 ± 0.0192 | 0.8009 ± 0.0197 | 0.9586 ± 0.0014 | 0.0594 ± 0.0197 |
| B. L2-normalized signed PCA | 0.3609 ± 0.0815 | 0.2843 ± 0.0959 | 0.7642 ± 0.0253 | 0.2543 ± 0.0570 |
| C. Squared normalized probabilities | 0.2289 ± 0.0388 | 0.1309 ± 0.0422 | 0.6663 ± 0.0352 | 0.3471 ± 0.0371 |

## Per-class recall

Each cell is mean ± population SD across the three model seeds within that split. Final column gives the mean ± SD across five split-level means.

| Class | Representation | 101 | 202 | 303 | 404 | 505 | Across split means |
|---|---|---:|---:|---:|---:|---:|---:|
| Benign | A. Signed PCA | 0.460 ± 0.014 | 0.410 ± 0.092 | 0.563 ± 0.151 | 0.420 ± 0.036 | 0.453 ± 0.082 | 0.461 ± 0.054 |
| Benign | B. L2-normalized signed PCA | 0.507 ± 0.252 | 0.453 ± 0.180 | 0.567 ± 0.240 | 0.580 ± 0.094 | 0.207 ± 0.154 | 0.463 ± 0.136 |
| Benign | C. Squared normalized probabilities | 0.773 ± 0.170 | 0.077 ± 0.042 | 0.097 ± 0.137 | 0.293 ± 0.220 | 0.163 ± 0.165 | 0.281 ± 0.258 |
| Bot | A. Signed PCA | 0.840 ± 0.226 | 0.847 ± 0.196 | 0.990 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.935 ± 0.075 |
| Bot | B. L2-normalized signed PCA | 0.160 ± 0.226 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.032 ± 0.064 |
| Bot | C. Squared normalized probabilities | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.236 | 0.033 ± 0.067 |
| DoS-Hulk | A. Signed PCA | 0.993 ± 0.005 | 0.997 ± 0.005 | 0.997 ± 0.005 | 1.000 ± 0.000 | 0.967 ± 0.019 | 0.991 ± 0.012 |
| DoS-Hulk | B. L2-normalized signed PCA | 0.000 ± 0.000 | 0.323 ± 0.457 | 0.000 ± 0.000 | 0.007 ± 0.009 | 0.000 ± 0.000 | 0.066 ± 0.129 |
| DoS-Hulk | C. Squared normalized probabilities | 0.003 ± 0.005 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.001 ± 0.001 |
| FTP-BruteForce | A. Signed PCA | 1.000 ± 0.000 | 0.997 ± 0.005 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.001 |
| FTP-BruteForce | B. L2-normalized signed PCA | 0.620 ± 0.442 | 0.970 ± 0.042 | 0.867 ± 0.094 | 0.593 ± 0.420 | 0.000 ± 0.000 | 0.610 ± 0.337 |
| FTP-BruteForce | C. Squared normalized probabilities | 0.000 ± 0.000 | 0.637 ± 0.452 | 0.000 ± 0.000 | 0.300 ± 0.424 | 0.000 ± 0.000 | 0.187 ± 0.253 |
| Infiltration | A. Signed PCA | 0.727 ± 0.012 | 0.640 ± 0.096 | 0.583 ± 0.205 | 0.660 ± 0.049 | 0.740 ± 0.050 | 0.670 ± 0.058 |
| Infiltration | B. L2-normalized signed PCA | 0.740 ± 0.249 | 0.510 ± 0.184 | 0.493 ± 0.275 | 0.577 ± 0.144 | 0.850 ± 0.091 | 0.634 ± 0.139 |
| Infiltration | C. Squared normalized probabilities | 0.243 ± 0.344 | 0.707 ± 0.124 | 0.833 ± 0.229 | 0.740 ± 0.178 | 0.690 ± 0.390 | 0.643 ± 0.206 |

## Ranking and split-to-split changes

| Split | A Macro-F1 | B Macro-F1 | C Macro-F1 | Ranking |
|---:|---:|---:|---:|---|
| 101 | 0.7942 | 0.3356 | 0.1045 | A > B > C |
| 202 | 0.7669 | 0.3851 | 0.1819 | A > B > C |
| 303 | 0.8180 | 0.3222 | 0.0755 | A > B > C |
| 404 | 0.8036 | 0.2718 | 0.1783 | A > B > C |
| 505 | 0.8220 | 0.1066 | 0.1144 | A > C > B |

Macro-F1 change from the preceding listed split seed (descriptive; adjacent seeds are not treated as paired independent samples):

| Transition | A Δ Macro-F1 | B Δ Macro-F1 | C Δ Macro-F1 |
|---|---:|---:|---:|
| 101 → 202 | -0.0273 | +0.0495 | +0.0774 |
| 202 → 303 | +0.0511 | -0.0629 | -0.1064 |
| 303 → 404 | -0.0144 | -0.0504 | +0.1028 |
| 404 → 505 | +0.0183 | -0.1652 | -0.0640 |

## Prediction collapse and stopping stability

A prediction is collapsed for a class when a model predicts zero examples of that class in the 500-row test partition. Counts below cover 15 final models (five splits × three model seeds) per representation.

| Representation | Models with any absent class | Models never predicting each class |
|---|---:|---|
| A. Signed PCA | 0/15 | Benign: 0/15, Bot: 0/15, DoS-Hulk: 0/15, FTP-BruteForce: 0/15, Infiltration: 0/15 |
| B. L2-normalized signed PCA | 15/15 | Benign: 0/15, Bot: 14/15, DoS-Hulk: 12/15, FTP-BruteForce: 4/15, Infiltration: 0/15 |
| C. Squared normalized probabilities | 15/15 | Benign: 2/15, Bot: 14/15, DoS-Hulk: 11/15, FTP-BruteForce: 12/15, Infiltration: 1/15 |

Validation-selected best epochs (zero-based), with each sequence ordered by model seed 101/202/402. The final refit remains 57 epochs for every run.

| Split | A best epochs | B best epochs | C best epochs |
|---:|---|---|---|
| 101 | 56/51/46 | 8/8/18 | 0/25/18 |
| 202 | 46/43/41 | 0/0/2 | 9/11/24 |
| 303 | 52/34/44 | 4/16/14 | 13/10/1 |
| 404 | 58/53/43 | 12/10/18 | 20/12/13 |
| 505 | 57/48/31 | 3/14/6 | 5/8/16 |

### Predicted-class distributions

Counts are ordered Benign / Bot / DoS-Hulk / FTP-BruteForce / Infiltration for each split/model seed.

**Split 101**

- A. Signed PCA: 101: 63/120/103/100/114; 202: 59/64/162/100/115; 402: 57/126/100/100/117
- B. L2-normalized signed PCA: 101: 30/0/0/86/384; 202: 194/60/0/104/142; 402: 394/0/0/0/106
- C. Squared normalized probabilities: 101: 465/0/35/0/0; 202: 91/0/1/0/408; 402: 414/0/0/0/86

**Split 202**

- A. Signed PCA: 101: 45/116/100/104/135; 202: 60/116/100/99/125; 402: 91/62/157/101/89
- B. L2-normalized signed PCA: 101: 109/0/157/110/124; 202: 299/0/0/100/101; 402: 183/0/0/115/202
- C. Squared normalized probabilities: 101: 58/0/0/169/273; 202: 70/0/0/152/278; 402: 81/0/0/0/419

**Split 303**

- A. Signed PCA: 101: 46/105/99/100/150; 202: 120/111/100/100/69; 402: 114/111/100/100/75
- B. L2-normalized signed PCA: 101: 40/0/0/81/379; 202: 206/0/0/80/214; 402: 142/0/0/119/239
- C. Squared normalized probabilities: 101: 0/0/0/0/500; 202: 132/0/0/0/368; 402: 3/0/0/0/497

**Split 404**

- A. Signed PCA: 101: 54/122/101/103/120; 202: 73/121/103/104/99; 402: 60/121/105/101/113
- B. L2-normalized signed PCA: 101: 135/0/0/89/276; 202: 239/0/0/0/261; 402: 280/0/12/96/112
- C. Squared normalized probabilities: 101: 212/0/0/0/288; 202: 58/0/0/0/442; 402: 100/0/0/92/308

**Split 505**

- A. Signed PCA: 101: 84/114/95/100/107; 202: 62/121/102/101/114; 402: 43/123/102/100/132
- B. L2-normalized signed PCA: 101: 208/0/0/0/292; 202: 46/0/17/0/437; 402: 1/0/0/2/497
- C. Squared normalized probabilities: 101: 20/0/49/0/431; 202: 96/188/83/0/133; 402: 0/0/0/0/500


## Integrity, artifacts, and execution notes

All five manifests report the same source hash and 5,000-row input, balanced 800/100/100 class counts, 16 PCA components, train-only preprocessing, split seed, and train/validation/test PCA+label fingerprints. Split 202 fingerprints were independently recomputed and matched. For splits 303/404/505, manifests, dimensions, config, and run artifacts were checked; the reviewed CLI creates the three feature branches from common split arrays, preserving row order. Every mode and split contains three seed metrics files, selection curves, selection and final checkpoints, transformation/run/dataset manifests, test predictions, and predicted-class distributions; metrics include confusion matrices and per-class precision/recall/F1. AUROC and ECE are present for all held-out test runs. No previous artifacts were overwritten.

Runtime and memory were measured using macOS `/usr/bin/time -l`. The seed-404 run took 498.05 seconds wall time despite 47.92 seconds user CPU and 28.95 seconds system CPU; the cause of the unusually high wall/CPU discrepancy is unknown. Other runs took about 55–57 seconds. TabNet warnings that refits do not use early stopping are expected because the refit is intentionally fixed at 57 epochs.

Git metadata is unavailable because the project directory is not a Git repository; it was not initialized. Python 3.11.18, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.14.1, and pytorch-tabnet 4.1.0 were recorded before execution.

## Conclusion and limitations

Signed PCA (A) ranked first on all five splits, but the ordering of B and C was not fully consistent.

The ordering is descriptive on one fixed balanced local pilot. Variation across model seeds within each split is separate from variation across split-level means. Only five overlapping/random partitions of the same pilot were evaluated, so these results do not support statistical-significance claims or external-dataset uncertainty. B and C show class collapse and early-stopping instability; A had the strongest average metrics, but still has low Benign recall and variable seed results. This does not establish full CIC-IDS2018 performance, exact paper reproduction, VQC validation, or quantum advantage.

Recommended next step: freeze this configuration and, if further pilot robustness evidence is needed, pre-register additional split seeds and report all outcomes. Then evaluate on a separately prepared dataset after its protocol is fixed; do not treat resplits of this pilot as external validation.

## Run directories

- Split 101: `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed101-20261029-1413/`
- Split 202: `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed202/`
- Split 303: `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed303/`
- Split 404: `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed404/`
- Split 505: `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed505/`

## Exact commands for the additional runs

```bash
cd /Users/swasthiksundarraj/IdeaProjects/FYP/qatnet-quasar
PYTHONPATH=src /usr/bin/time -l .venv/bin/python -m qatnet.cli representation-ablation --dataset outputs/runs/cic-paper-pilot-20261029-1313/sampled.csv --dataset-name CIC-IDS2018-local-pilot-1k-per-class --qubits 4 --split-seed 303 --output outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed303
```

```bash
cd /Users/swasthiksundarraj/IdeaProjects/FYP/qatnet-quasar
PYTHONPATH=src /usr/bin/time -l .venv/bin/python -m qatnet.cli representation-ablation --dataset outputs/runs/cic-paper-pilot-20261029-1313/sampled.csv --dataset-name CIC-IDS2018-local-pilot-1k-per-class --qubits 4 --split-seed 404 --output outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed404
```

```bash
cd /Users/swasthiksundarraj/IdeaProjects/FYP/qatnet-quasar
PYTHONPATH=src /usr/bin/time -l .venv/bin/python -m qatnet.cli representation-ablation --dataset outputs/runs/cic-paper-pilot-20261029-1313/sampled.csv --dataset-name CIC-IDS2018-local-pilot-1k-per-class --qubits 4 --split-seed 505 --output outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed505
```
