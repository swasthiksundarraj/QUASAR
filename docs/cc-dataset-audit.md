# CIC-IDS2018 `cc/` data audit

Audit scope: the already-extracted `qatnet-quasar/cc/` directory. No `cc.zip` was found in the project or its parent directories, so archive metadata could not be inspected. The CSVs were read in chunks; no model training was run.

These are flow-feature records, not PCAP/raw packet captures. Each file has a `Label` column. Nine files have 80 columns; `02-20-2018.csv` has 84 and adds `Flow ID`, `Src IP`, `Src Port`, and `Dst IP`. All files contain CICFlowMeter-style flow statistics and represent separate capture dates. They should not be concatenated blindly until common feature selection and the intended experiment's date policy are explicit.

## Files and observed labels

Sizes are exact bytes. Row counts include 59 embedded repeated-header records until loader sanitation; these are identified in the per-file label detail.

| File | Bytes | Rows | Labels observed |
|---|---:|---:|---|
| `02-14-2018.csv` | 358,223,333 | 1,048,575 | Benign 667,626; FTP-BruteForce 193,360; SSH-Bruteforce 187,589 |
| `02-15-2018.csv` | 375,945,899 | 1,048,575 | Benign 996,077; DoS attacks-GoldenEye 41,508; DoS attacks-Slowloris 10,990 |
| `02-16-2018.csv` | 333,723,605 | 1,048,575 | Benign 446,772; DoS attacks-Hulk 461,912; DoS attacks-SlowHTTPTest 139,890; repeated header 1 |
| `02-20-2018.csv` | 4,054,925,350 | 7,948,748 | Benign 7,372,557; DDoS attacks-LOIC-HTTP 576,191 |
| `02-21-2018.csv` | 328,893,673 | 1,048,575 | Benign 360,833; DDOS attack-HOIC 686,012; DDOS attack-LOIC-UDP 1,730 |
| `02-22-2018.csv` | 382,636,202 | 1,048,575 | Benign 1,048,213; Brute Force -Web 249; Brute Force -XSS 79; SQL Injection 34 |
| `02-23-2018.csv` | 382,840,456 | 1,048,575 | Benign 1,048,009; Brute Force -Web 362; Brute Force -XSS 151; SQL Injection 53 |
| `02-28-2018.csv` | 209,249,758 | 613,104 | Benign 544,200; Infilteration 68,871; repeated header 33 |
| `03-01-2018.csv` | 107,842,858 | 331,125 | Benign 238,037; Infilteration 93,063; repeated header 25 |
| `03-02-2018.csv` | 352,368,373 | 1,048,575 | Benign 762,384; Bot 286,191 |

Total: 16,233,002 rows, 6,886,649,507 bytes (about 6.89 GB / 6.42 GiB), and 15 genuine observed labels after excluding the 59 repeated-header rows. The loader now detects a row only when every field matches its column name, removes it with a `RuntimeWarning`, and includes the count as `repeated_header_rows` in the report.

## Quality and handling notes

- Across the unfiltered files, 59,721 cells parsed as missing and 121,886 numeric cells were infinite. The infinite values are replaced with missing values by the training preprocessor; training-only imputation is fit after splitting.
- A 64-bit pandas row-fingerprint scan found 410,708 repeated row hashes across the raw files. This count includes embedded repeated headers and is not a verified exact-row deduplication count. No duplicate rows have been removed; deduplication policy must be set by the paper protocol and recorded before training.
- All files use `Label`; repeated CSV headers appeared as label `Label` in 02-16, 02-28, and 03-01. Exact row comparison confirmed these are repeated headers, not an attack class.
- The label set is highly imbalanced (13,484,708 Benign versus 87 SQL Injection records after excluding repeated headers). The training pipeline's current split guard requires at least ten instances per class, so a plain proportional 1,200-row sample is not a safe all-class development protocol.
- The local QATNet PDF specifies five target class names and a balanced 40,000-per-class CIC experiment. The paper does not provide raw-label aliases or exact capture filenames. The local aliases `DoS attacks-Hulk` → `DoS-Hulk` and `Infilteration` → `Infiltration` are explicit project mappings, not mappings verified from the paper. See [`paper-reproduction-protocol.md`](paper-reproduction-protocol.md).
- The reproducible local pilot selects `02-14-2018.csv`, `02-16-2018.csv`, `02-28-2018.csv`, `03-01-2018.csv`, and `03-02-2018.csv`: together they contain every required attack label and Benign. These dates are non-consecutive and are not confirmed as the paper's five capture days. The pilot command records the selection and exact scan counts in its manifest.
- The legacy `load_csv` helper and generic `train` command still materialize a single file in memory. The paper-specific `paper-smoke`, `prepare-cic-pilot`, and all-label `sample-explore` commands instead use the chunked sampler. The prepared 5,000-row pilot and its manifest are at `outputs/runs/cic-paper-pilot-20261029-1313/`; preparation did not run training. Do not pass the full collection through the generic loader/`train` CLI.
