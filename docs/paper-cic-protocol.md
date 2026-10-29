# QATNet CIC-IDS2018 five-class protocol

**Protocol status:** the local paper PDF was reviewed in full. It explicitly specifies the five target class names, a 200,000-row balanced CIC experiment (40,000/class), stratified 80/10/10 split, training-only imputation/scaling/PCA, and Q-dependent PCA widths. It does not disclose CIC filenames, raw source-label aliases, sampling algorithm, or duplicate policy. Evidence and implementation gaps are in [`paper-reproduction-protocol.md`](paper-reproduction-protocol.md); loader behavior is an implementation protocol, not proof of exact paper data provenance.

## Explicit class mapping

The extracted files were streamed and their original label spellings/counts inspected. Mapping is exact after Unicode NFC normalization and trimming surrounding whitespace; label matching is case-sensitive. No general fuzzy matching or attack-family merging is performed.

| Paper target class | Exact source label | Observed source count | Mapping decision |
|---|---|---:|---|
| Benign | `Benign` | 13,484,708 | Exact match |
| Bot | `Bot` | 286,191 | Exact match |
| DoS-Hulk | `DoS attacks-Hulk` | 461,912 | Local source alias to the paper's target name; raw alias is not stated by the paper |
| FTP-BruteForce | `FTP-BruteForce` | 193,360 | Exact match |
| Infiltration | `Infilteration` | 161,934 | Local source alias to the paper's target name; raw alias is not stated by the paper |

Counts are before repeated-header removal only for the unrelated classes represented by these counts; none of the five selected classes contains an embedded repeated header. Each target has at least 40,000 observed examples. No record is duplicated to meet a target.

The paper names the target class `Infiltration`, while the local CSVs contain `Infilteration`. That explicit local-source alias is not a paper-reported spelling correction. The PDF also names `DoS-Hulk`, while this local source uses `DoS attacks-Hulk`; the paper does not disclose its raw-label alias table.

## Selected local capture files for the pilot

The reproducible pilot uses these five locally audited CSVs:

| File | Reason included |
|---|---|
| `02-14-2018.csv` | Contains `FTP-BruteForce` and `Benign` |
| `02-16-2018.csv` | Contains `DoS attacks-Hulk` and `Benign` |
| `02-28-2018.csv` | Contains `Infilteration` and `Benign` |
| `03-01-2018.csv` | Contains `Infilteration` and `Benign`; included to increase available source examples |
| `03-02-2018.csv` | Contains `Bot` and `Benign` |

This is the smallest collection of the audited attack-bearing dates that includes all five target classes, retaining both dates for Infiltration. It is not asserted to match the paper's unnamed capture-file manifest: these dates are non-consecutive, and the paper says "five consecutive days." The other five local files are not included in this pilot because they contain none of the four target attack labels. Benign rows are sampled from the selected files together with the target attack rows. The exact selected paths and per-file scanned row counts are written to `manifest.json`.

The mapping is exact after Unicode NFC normalization and trimming surrounding whitespace, case-sensitive, and only the five explicit source labels above map to the five targets. Specifically, `DoS attacks-Hulk` maps to `DoS-Hulk`, and `Infilteration` maps to `Infiltration`; these are local aliases and not spellings verified in the paper. No other attack classes are collapsed into these targets.

## Sampling and split policy

`qatnet.cli paper-smoke` uses a seeded, uniform per-class reservoir over every CSV in the selected directory. This is bounded by the requested sample size plus one input chunk, performs no genuine duplicate removal, and emits source spelling counts and mapped sample counts. The PDF specifies 40,000 samples per class (200,000 total), but does not state its sampling algorithm or duplicate policy; the reservoir/no-deduplication behavior is therefore an implementation choice. The smoke-test default is deliberately smaller (200 per class) and is not paper-result data.

After class-balanced sampling, records are split with stratification into 80/10/10 using a separate split seed. The test partition is not used to fit imputation, scaling, PCA, the classifier, or thresholds. Numeric/categorical imputation, numeric standardization, and PCA are fit on training rows only. Identifier-like fields (Flow ID, source/destination IP addresses and ports, timestamps, and index fields) are excluded by the preprocessor and recorded in the run report.

The reservoir algorithm retains unique source rows only as encountered; it does not deduplicate. Repeated CSV header rows are removed by exact all-fields-equal-header comparison and separately counted.

Use `prepare-cic-pilot` to create a balanced sample and manifest without starting any model training. The pilot command writes a new run directory and fails rather than overwriting it. The manifest includes sampled/split class counts, seeds, source feature columns, excluded identifier columns, missing/infinite handling, train-only preprocessing scope, repeated-header count, and a cross-split duplicate fingerprint report. Duplicate traffic rows are reported but retained. `paper-tabnet` accepts the resulting `sampled.csv`; it independently repeats the stratified split and training-only preprocessing using the supplied split seed.

### Prepared local pilot (2026-10-29)

The pilot was prepared at `outputs/runs/cic-paper-pilot-20261029-1313/` using 1,000 records/class, sampling seed `20261029`, split seed `101`, and 50,000-row input chunks. It contains 5,000 rows total (1,000 in each of Benign, Bot, DoS-Hulk, FTP-BruteForce, and Infiltration). Its stratified split is 4,000 train (800/class), 500 validation (100/class), and 500 test (100/class). Q=4 preprocessing produced 16 PCA components. The manifest records 77 source model feature columns and excludes `Dst Port` and `Timestamp` as identifiers; it preserves the source CSV columns in `sampled.csv` so the existing `paper-tabnet` loader can reapply its own split and train-fitted preprocessing.

No duplicate rows were removed. The 64-bit hash groups candidate matches and full-row equality is checked; it found 5 exact matching row instances across train/validation and 4 across train/test. They are reported and retained as legitimate repeated traffic records. The selected files had 59 repeated header rows removed, 10,710 source missing cells and 13,301 numeric infinities during the scan; the 5,000-row sample contains 9 missing cells and 21 numeric infinities. Infinity is converted to missing before train-only imputation.

Preparation took 97.27 seconds wall-clock (93.24 seconds measured within the preparation function) and peaked at 1,102,610,432 bytes RSS (about 1.03 GiB) on this Mac. The 5,000-row CSV is 1,760,075 bytes and its loaded pandas frame uses 3,828,132 deep bytes. A linear 40x sample-frame estimate for 200,000 rows is about 153 MB (146 MiB) for the loaded raw frame, before split copies, preprocessing arrays, Python overhead, and TabNet. This is an extrapolation, not a measured full-run requirement. The preparation scan itself processes the same five source files, so its scan time should be similar; the full-size sample frame and training behavior need separate benchmarking. No full-size training was launched.

Adding the estimated growth from the pilot frame to the observed 1.03-GiB process peak gives a rough lower-bound preparation peak near 1.2 GiB for 200,000 retained rows; this omits data-frame split copies and transient conversion buffers. Reserve at least 2 GiB of free memory for full-size dataset preparation, but do not treat that as a safe full-training budget. The paper profile can perform three validation candidate fits (up to 60 epochs each) and three final-seed refits; its 200,000-row runtime and peak memory were not measured. Run the 5,000-row paper-profile pilot first and record actual resource use before considering the full experiment.
