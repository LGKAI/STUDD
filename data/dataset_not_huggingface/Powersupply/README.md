---
license: cc-by-4.0
pretty_name: Powersupply benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Powersupply — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: USP Data Stream Repository
- Original source: https://drive.google.com/file/d/1YWncempzW_MDMJcRvyh377449ytOSh9g/view
- Download endpoint: https://drive.usercontent.google.com/download?id=1YWncempzW_MDMJcRvyh377449ytOSh9g&export=download&confirm=t
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Repository citation: Souza, Reis, Maletzke, Batista (2020). Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. https://doi.org/10.1007/s10618-020-00698-5

Original benchmark reference: Zhu (2010), Stream Data Mining Repository.

The CC BY 4.0 metadata records the USP distribution notice; the upstream third-party notice below is retained.

## Local file and observed structure

- File: `Powersupply.csv` (396,873 bytes).
- Nonempty CSV records: 29,928.
- Column counts and number of records: `{3: 29927, 4: 1}`.
- Observed distinct target strings in last column: 25.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `e1ee08c09d154577ca9f7d2edc079993c32422dc290c722b1560ee83e9f8647f`.
- Parsing note: Headerless CSV. Record 22,203 has an extra trailing comma: 128.3,93.3,2,. The raw file is preserved; confirm and explicitly document removal of the trailing empty field before treating the third field as target. Raw distinct final-column strings are not a valid class count until this is handled.

## Relation to the paper

Paper Table 1 reports 29,928 rows, 2 total columns and 24 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

USP distribution notice: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The repository notes that third-party datasets may have their own licenses/restrictions; retain the original authors and source attribution. Source notice: https://sites.google.com/view/uspdsrepository .

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.
