"""Gói mã nguồn chính cho thuật toán STUDD (Unsupervised Concept Drift Detection)."""

from .baselines import (
    ALL_METHODS,
    compute_alarm_metrics,
    compute_labeling_efficiency,
    evaluate_all_10_methods,
    evaluate_feature_fixed_ks,
    evaluate_output_ks,
    evaluate_periodic_retrain_baseline,
    evaluate_static_baseline,
    evaluate_supervised_pht,
)
from .detector import PageHinkley
from .studd import (
    PAPER_TABLE2_MODELS,
    STUDD,
    SafeXGBClassifier,
    build_paper_models,
    create_learner,
    select_best_models,
)
from .data_loader import (
    DAT_6_DATASETS,
    load_benchmark_stream,
    load_paper_stream,
    generate_abrupt_drift_stream,
    generate_pure_real_drift_stream,
)

__all__ = [
    "ALL_METHODS",
    "compute_alarm_metrics",
    "compute_labeling_efficiency",
    "evaluate_all_10_methods",
    "evaluate_feature_fixed_ks",
    "evaluate_output_ks",
    "evaluate_periodic_retrain_baseline",
    "evaluate_static_baseline",
    "evaluate_supervised_pht",
    "PageHinkley",
    "PAPER_TABLE2_MODELS",
    "STUDD",
    "SafeXGBClassifier",
    "build_paper_models",
    "create_learner",
    "select_best_models",
    "DAT_6_DATASETS",
    "load_benchmark_stream",
    "load_paper_stream",
    "generate_abrupt_drift_stream",
    "generate_pure_real_drift_stream",
]
