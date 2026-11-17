# Posture — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: dos Reis incremental-ks supplemental repository
- Original source: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Posture.zip
- Download endpoint: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Posture.zip
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Supplemental repository: dos Reis et al. (2016). Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test. Original dataset attribution is documented in that repository.

## Local file and observed structure

- File: `Posture.data` (10,038,586 bytes).
- Nonempty CSV records: 164,860.
- Column counts and number of records: `{4: 164860}`.
- Observed distinct target strings in last column: 11.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `400e552cf8d8386c8319fab0428b266a60e2ad88b4802812af8d7666d5ef4a36`.
- Parsing note: Headerless CSV. Use header=None and retain the first record. Target is the last column.

## Relation to the paper

Paper Table 1 reports 164,859 rows, 4 total columns and 11 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

Upstream repository distribution: MIT, Copyright (c) 2016 denismr. Original notice is preserved in LICENSE.

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.
