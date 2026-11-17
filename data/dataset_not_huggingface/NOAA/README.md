---
license: cc-by-4.0
pretty_name: NOAA weather benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# NOAA — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: USP Data Stream Repository
- Original source: https://drive.google.com/file/d/1lfisx3jZsj7Y1UM5HcYaMr9nOzXsTrEX/view
- Download endpoint: https://drive.usercontent.google.com/download?id=1lfisx3jZsj7Y1UM5HcYaMr9nOzXsTrEX&export=download&confirm=t
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Repository citation: Souza, Reis, Maletzke, Batista (2020). Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. https://doi.org/10.1007/s10618-020-00698-5

Original benchmark reference: Ditzler and Polikar, Incremental Learning of Concept Drift from Streaming Imbalanced Data, IEEE Transactions on Knowledge and Data Engineering (2013).

The CC BY 4.0 metadata records the USP distribution notice; the upstream third-party notice in the Distribution section is retained.

## Local file and observed structure

- File: `NOAA.csv` (700,956 bytes).
- Nonempty CSV records: 18,159.
- Column counts and number of records: `{9: 18159}`.
- Observed distinct target strings in last column: 2.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `31e0c1497dc8a34c7cc0316c02ebb602baa3c54946154f8f499f070be2329ac4`.
- Parsing note: Headerless CSV. Use header=None and retain the first record. Target is the last column.

## Relation to the paper

Paper Table 1 reports 18,159 rows, 8 total columns and 2 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

USP distribution notice: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The repository notes that third-party datasets may have their own licenses/restrictions; retain the original authors and source attribution. Source notice: https://sites.google.com/view/uspdsrepository .

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.
