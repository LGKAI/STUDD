# STUDD datasets

This branch stores downloaded source data and reports. No STUDD model runs were performed as part of this data upload.

- `dataset_huggingface/`: Electricity, Bike and InsectsAbrupt prepared data, plus CoverType, Poker, Ozone, Arabic and GMSC candidates downloaded from Hugging Face.
- `dataset_not_huggingface/`: ten original upstream datasets downloaded outside HF. These have now been mirrored to HF; the folder name records where they were originally obtained. Each folder includes source attribution and provenance.
- `hosted_sources.json`: pinned hosted configuration for the original three prepared datasets only. It is not a complete 19-dataset runner configuration.
- `validation_report.json`: validation metadata from the original data preparation.

See [the 19-dataset report](../docs/dataset_reports/STUDD_19_DATASETS_REPORT.md) for HF links, original external links, sizes, verification status and unresolved differences from the paper. Further evidence is in `docs/dataset_reports/` and `docs/data_provenance/`.

The upstream files remain unchanged, including known anomalies. Dataset folders have individual notices; they are not uniformly licensed. Gas has conflicting source notices documented in its README. Original creators and distributors retain their rights.

The instructor requires experimental runs to fetch data programmatically from Hugging Face or Kaggle. Files in this branch provide a reviewable snapshot and provenance; their presence does not replace that run-time requirement.
