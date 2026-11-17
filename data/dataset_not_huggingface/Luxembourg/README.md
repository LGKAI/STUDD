---
license: cc-by-4.0
pretty_name: Luxembourg benchmark for STUDD reproduction
task_categories:
- tabular-classification
---

# Luxembourg — upstream benchmark for STUDD reproduction

This is a downloaded upstream dataset, not a new dataset authored by this uploader.
Data bytes and stream order have not been modified. Do not shuffle before drift experiments.

## Source and attribution

- Provider: USP Data Stream Repository
- Original source: https://drive.google.com/file/d/1BwiUIE97rZbZbMHGODbWaGpSJ7DmGwMB/view
- Download endpoint: https://drive.usercontent.google.com/download?id=1BwiUIE97rZbZbMHGODbWaGpSJ7DmGwMB&export=download&confirm=t
- STUDD paper: Cerqueira et al., https://doi.org/10.1007/s10994-022-06188-7
- Repository citation: Souza, Reis, Maletzke, Batista (2020). Challenges in Benchmarking Stream Learning Algorithms with Real-world Data. https://doi.org/10.1007/s10618-020-00698-5

Original benchmark reference: Zliobaite (2011), Combining Similarity in Time and Space for Training Set Formation under Concept Drift.

The CC BY 4.0 metadata records the USP distribution notice; the upstream third-party notice below is retained.

## Local file and observed structure

- File: `Luxembourg.csv` (222,420 bytes).
- Data rows: 1,901, plus one header row (1,902 CSV records total).
- Column counts and number of records: `{32: 1902}`.
- Target classes among data rows: 2; the header text is not a class.
- Data feature values: numeric and finite. The 31 text fields in the header are column names, not invalid samples.
- SHA-256: `cfc6ef67791af901c209c85ae2caa5c69f673941fe31eb4ce12415e0069edc8b`.
- Parsing note: This file HAS a header (at1…at31,label). Use header=0. There are 1,901 data rows after the header, 31 feature columns and 2 classes. The raw record count includes the header; raw class strings include the word label.

## Relation to the paper

Paper Table 1 reports 1,901 rows, 31 total columns and 2 classes.
Observed structure above may differ. This package does not assert identical preprocessing
or promise that paper results can be reproduced without further comparison.
Do not remove the first record or columns just to match the printed table.

## Distribution notice

USP distribution notice: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The repository notes that third-party datasets may have their own licenses/restrictions; retain the original authors and source attribution. Source notice: https://sites.google.com/view/uspdsrepository .

`provenance.json` contains source archive hash (where applicable), final data hash,
observed shape and label distribution. The archive was removed after successful extraction;
only the extracted data is kept here.
