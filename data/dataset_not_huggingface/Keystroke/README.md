---
license: cc-by-4.0
pretty_name: Keystroke benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Keystroke — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: dos Reis incremental-ks supplemental repository
- Original source: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Keystroke.zip
- Download endpoint: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Keystroke.zip
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Supplemental repository: dos Reis et al. (2016). Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test. Original dataset attribution is documented in that repository.

## Local file and observed structure

- File: `Keystroke.data` (114,833 bytes).
- Nonempty CSV records: 1,600.
- Column counts and number of records: `{11: 1600}`.
- Observed distinct target strings in last column: 4.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `adb61d7945185ed33f28e29d097a2dbed96c07d95d67f329c6db740d0c0ea845`.
- Parsing note: Headerless CSV. Use header=None and retain the first record. Target is the last column.

## Relation to the paper

Paper Table 1 reports 1,600 rows, 10 total columns and 4 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

Upstream repository distribution: MIT, Copyright (c) 2016 denismr. Original notice is preserved in LICENSE.

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.

## Additional distribution verified

The same 1,600 ordered numeric records and targets were verified against
Keystroke.csv distributed by the USP Data Stream Repository:
https://drive.google.com/file/d/1AmkbMseb0-oBud06Gi26U-wy7DMFRtfL/view

The CC BY 4.0 card metadata records the USP distribution notice:
https://sites.google.com/view/uspdsrepository . Its third-party-source notice
is retained; this does not establish new ownership by this uploader.
The included MIT LICENSE preserves the supplemental repository notice.

Original benchmark reference: Souza et al. (2015), Data Stream Classification
Guided by Clustering on Nonstationary Environments and Extreme Verification Latency.
