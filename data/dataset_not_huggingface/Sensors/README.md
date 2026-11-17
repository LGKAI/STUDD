---
license: cc-by-4.0
pretty_name: Sensor Stream benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Sensors — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: USP Data Stream Repository
- Original source: https://drive.google.com/file/d/1T8QhOH4z6gV2yvB4gA7dyisvnotKcy6t/view
- Download endpoint: https://drive.usercontent.google.com/download?id=1T8QhOH4z6gV2yvB4gA7dyisvnotKcy6t&export=download&confirm=t
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Repository citation: Souza, Reis, Maletzke, Batista (2020). Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. https://doi.org/10.1007/s10618-020-00698-5

Original benchmark reference: Zhu (2010), Stream Data Mining Repository.

The CC BY 4.0 metadata records the USP distribution notice; the upstream third-party notice below is retained.

## Local file and observed structure

- File: `SensorStream.csv` (88,697,101 bytes).
- Nonempty CSV records: 2,219,803.
- Column counts and number of records: `{6: 2219803}`.
- Observed distinct target strings in last column: 55.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `51c4047966380c426da71d7344c1f232474986263028f69cf521637d2e1b2dee`.
- Parsing note: Headerless CSV. Observed 55 distinct target strings, versus 54 classes in the paper. Five feature columns plus target (6 total), versus 5 total columns printed in paper. Investigate before matching paper results.

## Relation to the paper

Paper Table 1 reports 2,219,803 rows, 5 total columns and 54 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

USP distribution notice: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The repository notes that third-party datasets may have their own licenses/restrictions; retain the original authors and source attribution. Source notice: https://sites.google.com/view/uspdsrepository .

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.
