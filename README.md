# QATNet + QUASAR

A research prototype for three leakage-safe tabular intrusion-detection pipelines and a simulated cross-layer SDN/QKD mitigation workflow. The local full paper PDF has now been reviewed, but this project is **not yet an exact paper reproduction**: implementation gaps, PDF page evidence, local dataset discrepancies, and unresolved assumptions are recorded in [`docs/paper-reproduction-protocol.md`](docs/paper-reproduction-protocol.md) and [`docs/reproduction-matrix.md`](docs/reproduction-matrix.md). QKD readings and default SDN demonstrations are synthetic; mitigation is mock-only unless a separately reviewed lab adapter is developed.

## Environment and installation

Python 3.11 is required (verified locally with Python 3.11.18). A CPU-only environment is sufficient; the listed PyTorch package uses the platform's compatible wheel and does not require CUDA.

```bash
cd qatnet-quasar
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

`requirements.txt` lists the same runtime stack. Resolve dependencies on the target OS before large research runs; no GPU or physical quantum hardware is required. Mininet is Linux-only and is not included in the Python package install.

Versions installed and exercised in this macOS ARM environment: Python 3.11.18, NumPy 2.4.6, Pandas 2.3.3, scikit-learn 1.9.1, PyTorch 2.14.1 (CPU), pytorch-tabnet 4.1.0, PennyLane 0.45.1, Qiskit 2.6.1, Qiskit Aer 0.18.2, Matplotlib 3.11.2, pytest 9.1.1, and Streamlit 1.66.1. These are execution-time observed versions, not a guarantee of compatibility for untested platforms.

## Data

Download CSE-CIC-IDS2018 flow CSVs from the [official CIC page](https://www.unb.ca/cic/datasets/ids-2018.html) only if they are not already available. The project does not download the dataset. This checkout already contains extracted day-wise flow CSVs under `cc/`; inspect an individual file with:

```bash
python -m qatnet.cli inspect --dataset cc/02-14-2018.csv --label Label
```

Embedded rows that exactly repeat the CSV header are removed and counted. `paper-smoke` streams CSV files in bounded chunks, harmonizes their union of columns, maps the five requested target classes explicitly, samples a seeded equal number per class without replacement, then performs an 80/10/10 stratified split before train-only imputation/scaling/PCA. Identifier-like fields are excluded from model features. The proposed 40,000-per-class protocol and its paper-verification limits are documented in [`docs/paper-reproduction-protocol.md`](docs/paper-reproduction-protocol.md); the default 200-per-class run is only a development smoke test. The generic `train` command still loads one CSV fully and runs all pipeline modes at Q=4, 5, and 6; do not use it for the 6.89-GB `cc/` collection.

The existing extracted `cc/` files and their measured schemas, classes, data-quality counts, and limitations are documented in [`docs/cc-dataset-audit.md`](docs/cc-dataset-audit.md). No `cc.zip` archive was present at audit time. The paper mapping is restricted to the five explicit source labels in the protocol document; the other observed labels are not silently merged into them.

Run a small Q=4 classical PCA–TabNet smoke test without loading the complete dataset into memory:

```bash
PYTHONPATH=src python -m qatnet.cli paper-smoke \
  --data-dir cc --samples-per-class 200 --seed 42 --split-seed 123 \
  --chunk-size 100000 --epochs 2 --batch-size 64
```

For data preparation without training, create the five-class CIC pilot and its provenance/leakage manifest:

```bash
PYTHONPATH=src python -m qatnet.cli prepare-cic-pilot \
  --data-dir cc --samples-per-class 1000 --seed 20261029 \
  --split-seed 101 --chunk-size 50000 \
  --output outputs/runs/cic-paper-pilot-20261029-1313
```

By default this scans `02-14-2018.csv`, `02-16-2018.csv`, `02-28-2018.csv`, `03-01-2018.csv`, and `03-02-2018.csv`, which contain the five target classes in this local release. This non-consecutive capture selection is an explicit local choice, not a verified match to the paper's unspecified five-day manifest. The command writes `sampled.csv` and `manifest.json` to a new directory and never starts training.

Artifacts go to a new timestamped folder under `outputs/runs/`. The command's default is 200 per class; use `sample-explore` for a separate capped sample of all observed labels without applying the paper mapping:

```bash
PYTHONPATH=src python -m qatnet.cli sample-explore \
  --data-dir cc --samples-per-class 1000 --seed 42
```

The separate `paper-tabnet` command applies the opt-in TabNet profile (validation selection, then model seeds 101/202/402) to a previously sampled single-dataset CSV. It supports the classical, amplitude-probability, or frozen-QATNet feature pipeline and requires an explicit split seed and a new output directory:

```bash
PYTHONPATH=src python -m qatnet.cli paper-tabnet \
  --dataset balanced-sample.csv \
  --dataset-name CIC-IDS2018 --mode classical --qubits 4 \
  --split-seed 123 --output outputs/runs/paper-tabnet-cic-q4-seed123
```

Replace `balanced-sample.csv` with an existing sampled CSV containing one dataset only. This command runs training when invoked; do not use the 200,000-row sample until memory/runtime have been estimated. The paper does not disclose its candidate grid or selection metric; this implementation uses a small documented grid and validation macro-F1. The final train-plus-validation refit uses the best epoch from seed-101 selection. These choices are assumptions, not exact paper settings. Class loss weights come from training labels only and are supplied through TabNet's `loss_fn`; the installed TabNet `weights` option is for sampler balancing, so it is disabled to avoid double balancing. Run each dataset and feature pipeline into its own output directory.

For a fixed-configuration representation ablation, `representation-ablation` uses the same Q=4 PCA split, TabNet candidate, seeds, training-only class weights, validation stopping policy, and 57-epoch train-plus-validation refit for three branches: signed PCA, L2-normalized signed PCA, and squared normalized probabilities. It writes branch/seed manifests, learning curves, checkpoints, and metrics to a new run directory:

```bash
PYTHONPATH=src python -m qatnet.cli representation-ablation \
  --dataset outputs/runs/cic-paper-pilot-20261029-1313/sampled.csv \
  --dataset-name CIC-IDS2018-local-pilot-1k-per-class \
  --qubits 4 --split-seed 101 \
  --output outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed101
```

The measured 5,000-row pilot for this command is saved at `outputs/runs/cic-paper-tabnet-representation-ablation-q4-seed101-20261029-1413/`; its results are local pilot evidence only and do not establish full-dataset performance.

`configs/development.json` is not currently loaded by the CLI; its values are configuration notes, not active command defaults. The CLI's VQC pretraining defaults are 3,000 balanced examples, 15 epochs, and batch size 256. The selected VQC pretraining objective is supervised cross-entropy with Adam; it is a documented implementation choice, not a verified paper detail.

The separate noisy-feature command is implemented, but it still loads its input fully into memory. The experiment is not the paper's full-model noise study and should follow a verified main pipeline:

```bash
python -m qatnet.cli noise --dataset data/raw/Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv \
  --label Label --data-kind real --qubits 4 --shots 1024 --sample-rows 1200
```

Generate synthetic KMS telemetry separately:

```bash
python -m qatnet.cli simulate-qkd --steps 100 --scenario key_pool_exhaustion --seed 42
```

## Mock QUASAR demonstration and dashboard

```bash
python -m qatnet.cli demo
streamlit run src/dashboard/app.py
```

The deterministic demo generates synthetic QKD and flow telemetry, fuses the alert, solves a tiny QUBO by exhaustive enumeration, optionally samples a Qiskit statevector QAOA plan, independently verifies the selected plan, and records the result in SQLite. QAOA can return an infeasible approximate sample; the verifier rejects it and selects the no-op fallback. No command in this demo sends traffic or applies real network rules.

### QATNet-driven offline QUASAR replay

The in-process `QATNetDetectionService` validates the fitted input feature manifest and class/probability dimensions, then returns each flow's predicted class, raw (uncalibrated) probabilities, attack probability, and model version. `quasar-qatnet-smoke` trains a tiny Q=4 QATNet model from an already balanced sample and sends held-out CIC flow-feature records through detection, QKD/SDN risk fusion, exact QUBO selection, independent verification, mock application, and rollback:

```bash
PYTHONPATH=src python -m qatnet.cli quasar-qatnet-smoke \
  --sample-csv outputs/runs/paper-smoke-task1-20261028-1829/sampled.csv \
  --seed 42 --split-seed 123 --pretrain-samples 100 \
  --pretrain-epochs 1 --epochs 2 --batch-size 32 --replay-flows 8
```

The command writes a model, feature cache, run summary, replay JSON, and SQLite audit log under a new `outputs/runs/quasar-qatnet-<timestamp>/` folder. This is an offline CIC replay with synthetic switch/host metadata and synthetic QKD—not an Ryu controller, live telemetry, calibrated confidence, or live mitigation. The current QUBO mitigation is mock-only and is always independently checked and rolled back.

The one-choice QUBO objective, capacity/disruption penalties, and verifier constraints are documented in [`docs/mitigation-objective.md`](docs/mitigation-objective.md). The optional Linux Mininet/Open vSwitch statistics adapter and isolated-VM precautions are in [`docs/mininet-lab.md`](docs/mininet-lab.md).

## Separate noisy-feature experiment

`noise.experiment.qiskit_noisy_amplitude_features` uses Qiskit Aer to measure noisy amplitude-encoded feature vectors. `train_noisy_feature_learners` compares Logistic Regression, Random Forest, SVC, and a stacking ensemble. This separate experiment does **not** run the complete PennyLane QATNet model under noise. It uses a configurable local depolarizing channel; FakeNairobi/FakeJakarta are unavailable in the tested Qiskit 2.6.1 environment (the old `qiskit.providers.fake_provider` exports and optional `qiskit_ibm_runtime` package are absent), so backend-calibrated noise remains unsupported here.

## Tests

```bash
pytest -q
```

Unit tests cover chunked sampling, schema harmonization, repeated headers, label mapping, rare classes, preprocessing boundaries, metrics, deterministic QKD accounting, mock SDN schema, cross-layer detection, QUBO optimality, and verifier fallback. A 200-per-class, two-epoch PCA–TabNet smoke run exists under `outputs/runs/`; its metrics are exploratory only, not final results. The end-to-end synthetic QUASAR workflow is the `demo` command.

## Outputs and reproducibility

The training command stores test predictions, JSON/CSV metrics, confusion-matrix PNGs, a macro-F1 plot, serialized model bundles, class names, seeds, training parameters, timings, and software/hardware information. The `paper-smoke` command stores the sampled CSV, run metadata, logs, metrics, and classical PCA–TabNet model. Use multiple seeds for final studies by preserving each run's output directory. Test rows are reserved for final evaluation and do not enter preprocessing or VQC pretraining. All numeric results must be generated by execution; no paper scores are copied into outputs.

Model inference accepts raw flow-record `DataFrame`s through the saved training-fitted preprocessor. Array inference expects the corresponding PCA feature matrix; the QATNet adapter then produces frozen probability features before the TabNet head.

## Progress checklist

- [x] Inventory existing code, tests, outputs, and extracted CIC-IDS2018 files.
- [x] Verify unit tests and the synthetic QUASAR demo artifacts; no model training is claimed.
- [x] Remove and report embedded repeated CSV header rows during dataset loading.
- [x] Implement chunked, reproducible per-class sampling, schema harmonization, and explicit CIC-to-paper class mapping.
- [x] Run a small five-class Q=4 PCA–TabNet smoke test (not final metrics).
- [x] Run Q=4 QATNet inference through offline QUASAR fusion, verified mock mitigation, audit and rollback.
- [ ] Run the 40,000-per-class PCA–TabNet baseline after reviewing the memory/runtime estimate.
- [ ] Run amplitude/VQC/QATNet experiments, repeated seeds, calibration and runtime evaluation.
- [ ] Validate an isolated Linux Mininet/Ryu controller-to-detection-service bridge; real mitigation remains disabled.

## Safety and scope

Synthetic scenarios are local simulation only. Do not generate attack traffic outside an explicitly authorized isolated environment. The included mock SDN adapter never opens sockets. Real mitigation is deliberately not wired: any future controller adapter must be isolated, allowlisted, independently verified, logged, and support rollback. The seven QUASAR scenario scores are evidence-based prototype indicators, not a claim that QATNet detects all attack classes. High QBER is not, by itself, proof of eavesdropping.
