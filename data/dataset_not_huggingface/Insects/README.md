---
license: mit
pretty_name: Insects benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Insects — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: dos Reis incremental-ks supplemental repository
- Original source: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Insects.zip
- Download endpoint: https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Insects.zip
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Supplemental repository: dos Reis et al. (2016). Fast Unsupervised Online Drift Detection Using Incremental Kolmogorov-Smirnov Test. Original dataset attribution is documented in that repository.

## Local file and observed structure

- File: `Insects.data` (2,485,847 bytes).
- Nonempty CSV records: 5,325.
- Column counts and number of records: `{50: 5325}`.
- Observed distinct target strings in last column: 5.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `5c588472b0e8f4ab4149754192d14a5514d27001a2e55cd64c39f41f2e0c374d`.
- Parsing note: Headerless CSV. Use header=None and retain the first record. Target is the last column.

## Relation to the paper

Paper Table 1 reports 5,325 rows, 50 total columns and 5 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

Upstream repository distribution: MIT, Copyright (c) 2016 denismr. Original notice is preserved in LICENSE.

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.

## Original measurement attribution

Underlying data reference: de Souza, Silva and Batista (2013), Classification of
Data Streams Applied to Insect Recognition: Initial Results, BRACIS.

This mirror preserves the small 5-class, 49-feature benchmark distributed in
the supplemental incremental-ks repository. It is not the later 6-class INSECTS
benchmark collection. The MIT metadata reflects the upstream repository
distribution notice; it introduces no new ownership claim over original measurements.
The upstream license notice is included unchanged in LICENSE.
