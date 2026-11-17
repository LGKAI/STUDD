---
license: cc-by-4.0
pretty_name: Gas Sensor Array Drift benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Gas — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: USP Data Stream Repository
- Original source: https://drive.google.com/file/d/11igK96DPWn8cgIwHfp2_w8JdpmTYsHVz/view
- Download endpoint: https://drive.usercontent.google.com/download?id=11igK96DPWn8cgIwHfp2_w8JdpmTYsHVz&export=download&confirm=t
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Repository citation: Souza, Reis, Maletzke, Batista (2020). Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. https://doi.org/10.1007/s10618-020-00698-5

## Local file and observed structure

- File: `GasSensorArray.csv` (17,278,995 bytes).
- Nonempty CSV records: 13,910.
- Column counts and number of records: `{129: 13910}`.
- Observed distinct target strings in last column: 6.
- Missing feature values: 0; nonfinite: 0; nonnumeric: 0.
- SHA-256: `e0dcfe689c01b6401edf5865f6dcf62b1a5351e00bdbb038cbd93ee3581783f7`.
- Parsing note: Headerless CSV. Use header=None and retain the first record. Target is the last column.

## Relation to the paper

Paper Table 1 reports 13,910 rows, 17 total columns and 6 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

USP distribution notice: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The repository notes that third-party datasets may have their own licenses/restrictions; retain the original authors and source attribution. Source notice: https://sites.google.com/view/uspdsrepository .

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.

## UCI attribution and inconsistent source notices

Original dataset: Vergara, A. (2012), Gas Sensor Array Drift Dataset.
UCI source: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
DOI: https://doi.org/10.24432/C5RP6W

Checked 2026-11-17: the UCI License section explicitly lists CC BY 4.0 and
sharing/adaptation with attribution. The Additional Information section on the
same page also says research-only and excludes commercial purposes. These notices
are inconsistent. The card metadata records the explicit UCI License section
and the USP distribution notice; this mirror does not resolve the inconsistency
or claim permission beyond the source terms. Consult the original provider to
clarify intended reuse where needed.

The uploader has not changed the downloaded USP CSV or its stream order.
The file has 128 feature columns plus target, not 16 feature columns.
