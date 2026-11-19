"""Kịch bản chạy thực nghiệm chuẩn hóa đầy đủ 10 phương pháp cho từng bộ dữ liệu Benchmark.

Tài liệu tham khảo:
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: a student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351–4378.
    (Bám sát mã nguồn gốc `github.com/vcerqueira/studd`: `studd_batch.py`, `tracker_output.py`,
    `tracker_covariates.py`, `workflows.py`).

Hỗ trợ đầy đủ 10 phương pháp trong Bảng 3 & Bảng 4 của bài báo:
    1. `BL-st`  (Static Baseline): Huấn luyện 1 lần trên cửa sổ W đầu, không cập nhật.
    2. `BL-ret` (Periodic Retraining): Tái huấn luyện định kỳ sau mỗi W mẫu mới.
    3. `STUDD`  (Proposed Unsupervised): Teacher–Student + kiểm định Page–Hinkley trên mimicry loss.
    4. `SS`     (Supervised Sliding): Giám sát lỗi thật tức thì (100% nhãn, trễ = 0) bằng Page–Hinkley.
    5. `DSS`    (Delayed Supervised): Giám sát lỗi thật bị trễ W/2 mẫu (100% nhãn) bằng Page–Hinkley.
    6. `WS`     (Weakly Supervised): Giám sát lỗi thật tức thì nhưng chỉ nhận 50% nhãn bằng Page–Hinkley.
    7. `DWS`    (Delayed Weakly Supervised): Kết hợp 50% nhãn và trễ W/2 mẫu bằng Page–Hinkley.
    8. `OS`     (Output Sliding KS-Test): Kiểm định KS trên hai cửa sổ trượt đầu ra kề nhau (step = 1).
    9. `OF`     (Output Fixed KS-Test): Kiểm định KS giữa cửa sổ trượt đầu ra với cửa sổ tham chiếu cố định (step = 1).
    10. `FF`    (Feature Fixed KS-Test): Kiểm định KS trên từng đặc trưng đầu vào so với tham chiếu cố định (step = 1).
"""

import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Đảm bảo import được package src/ khi chạy từ bất kỳ thư mục nào
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_stream
from src.models import get_table2_models, select_best_teacher_student

RESULTS_KHANH_DIR = os.path.join(PROJECT_ROOT, "results", "khanh")

ALL_10_METHODS: List[str] = [
    "BL-st",
    "BL-ret",
    "STUDD",
    "SS",
    "DSS",
    "WS",
    "DWS",
    "OS",
    "OF",
    "FF",
]


# ==============================================================================
# 1. SỐ LIỆU BẢNG 3 (KAPPA) & BẢNG 4 (COST) ĐẦY ĐỦ 10 PHƯƠNG PHÁP (7 BỘ KHÁNH)
# ==============================================================================

PAPER_BENCHMARK_NUMBERS: Dict[str, Dict[str, Dict[str, float]]] = {
    "bike": {
        "BL-st":  {"kappa": 0.31, "cost": 0.06},
        "BL-ret": {"kappa": 0.31, "cost": 0.98},
        "STUDD":  {"kappa": 0.32, "cost": 0.06},
        "SS":     {"kappa": 0.30, "cost": 0.46},
        "DSS":    {"kappa": 0.35, "cost": 0.35},
        "WS":     {"kappa": 0.32, "cost": 0.46},
        "DWS":    {"kappa": 0.32, "cost": 0.17},
        "OS":     {"kappa": 0.23, "cost": 0.06},
        "OF":     {"kappa": 0.32, "cost": 0.06},
        "FF":     {"kappa": 0.31, "cost": 0.98},
    },
    "luxembourg": {
        "BL-st":  {"kappa": 1.00, "cost": 0.53},
        "BL-ret": {"kappa": 1.00, "cost": 1.05},
        "STUDD":  {"kappa": 1.00, "cost": 0.53},
        "SS":     {"kappa": 1.00, "cost": 0.53},
        "DSS":    {"kappa": 1.00, "cost": 0.53},
        "WS":     {"kappa": 1.00, "cost": 0.53},
        "DWS":    {"kappa": 1.00, "cost": 0.53},
        "OS":     {"kappa": 1.00, "cost": 0.53},
        "OF":     {"kappa": 1.00, "cost": 0.53},
        "FF":     {"kappa": 1.00, "cost": 1.05},
    },
    "ozone": {
        "BL-st":  {"kappa": 0.13, "cost": 0.39},
        "BL-ret": {"kappa": 0.17, "cost": 1.18},
        "STUDD":  {"kappa": 0.13, "cost": 0.39},
        "SS":     {"kappa": 0.14, "cost": 0.39},
        "DSS":    {"kappa": 0.13, "cost": 0.39},
        "WS":     {"kappa": 0.13, "cost": 0.39},
        "DWS":    {"kappa": 0.13, "cost": 0.39},
        "OS":     {"kappa": 0.14, "cost": 0.39},
        "OF":     {"kappa": 0.17, "cost": 0.39},
        "FF":     {"kappa": 0.17, "cost": 0.99},
    },
    "gas": {
        "BL-st":  {"kappa": 0.32, "cost": 0.07},
        "BL-ret": {"kappa": 0.51, "cost": 0.93},
        "STUDD":  {"kappa": 0.48, "cost": 0.86},
        "SS":     {"kappa": 0.62, "cost": 0.89},
        "DSS":    {"kappa": 0.53, "cost": 0.78},
        "WS":     {"kappa": 0.58, "cost": 2.00},
        "DWS":    {"kappa": 0.51, "cost": 1.58},
        "OS":     {"kappa": 0.44, "cost": 0.50},
        "OF":     {"kappa": 0.57, "cost": 0.86},
        "FF":     {"kappa": 0.51, "cost": 0.93},
    },
    "outdoor": {
        "BL-st":  {"kappa": 0.37, "cost": 0.25},
        "BL-ret": {"kappa": 0.38, "cost": 1.00},
        "STUDD":  {"kappa": 0.36, "cost": 0.25},
        "SS":     {"kappa": 0.38, "cost": 0.50},
        "DSS":    {"kappa": 0.34, "cost": 0.62},
        "WS":     {"kappa": 0.32, "cost": 0.25},
        "DWS":    {"kappa": 0.36, "cost": 0.62},
        "OS":     {"kappa": 0.38, "cost": 0.62},
        "OF":     {"kappa": 0.37, "cost": 0.75},
        "FF":     {"kappa": 0.38, "cost": 1.00},
    },
    "rialto": {
        "BL-st":  {"kappa": 0.30, "cost": 0.01},
        "BL-ret": {"kappa": 0.34, "cost": 1.00},
        "STUDD":  {"kappa": 0.38, "cost": 0.21},
        "SS":     {"kappa": 0.47, "cost": 1.23},
        "DSS":    {"kappa": 0.37, "cost": 0.50},
        "WS":     {"kappa": 0.40, "cost": 1.16},
        "DWS":    {"kappa": 0.36, "cost": 0.79},
        "OS":     {"kappa": 0.40, "cost": 0.50},
        "OF":     {"kappa": 0.36, "cost": 0.75},
        "FF":     {"kappa": 0.34, "cost": 1.00},
    },
    "gmsc": {
        "BL-st":  {"kappa": 0.32, "cost": 0.01},
        "BL-ret": {"kappa": 0.06, "cost": 0.99},
        "STUDD":  {"kappa": 0.32, "cost": 0.01},
        "SS":     {"kappa": 0.32, "cost": 0.01},
        "DSS":    {"kappa": 0.32, "cost": 0.01},
        "WS":     {"kappa": 0.32, "cost": 0.01},
        "DWS":    {"kappa": 0.32, "cost": 0.01},
        "OS":     {"kappa": 0.32, "cost": 0.01},
        "OF":     {"kappa": 0.32, "cost": 0.01},
        "FF":     {"kappa": 0.11, "cost": 0.01},
    },
}


# ==============================================================================
# 2. BỘ PHÁT HIỆN PAGE-HINKLEY CHUẨN SCIKIT-MULTIFLOW (MỤC 5.7 BÀI BÁO)
# ==============================================================================

class SkMultiflowPageHinkley:
    """Kiểm định tuần tự Page-Hinkley khớp 100% mã nguồn `scikit-multiflow` mà bài báo sử dụng.

    Trong `skmultiflow.drift_detection.PageHinkley`:
        - `x_mean = x_mean + (x - x_mean) / sample_count`
        - `sum = max(0.0, alpha * sum + (x - x_mean - delta))`
        - Báo động khi `sample_count >= min_instances` và `sum > threshold`.
    """

    def __init__(
        self,
        delta: float = 0.001,
        threshold: float = 50.0,
        alpha: float = 1.0 - 0.0001,
        min_instances: int = 30,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances
        self.sample_count: int = 0
        self.x_mean: float = 0.0
        self.sum: float = 0.0
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.x_mean = 0.0
        self.sum = 0.0

    def update(self, x: float) -> Tuple[bool, float]:
        self.sample_count += 1
        self.x_mean += (x - self.x_mean) / float(self.sample_count)
        self.sum = max(0.0, self.alpha * self.sum + (x - self.x_mean - self.delta))

        alarm = (self.sample_count >= self.min_instances) and (self.sum > self.threshold)
        stat = float(self.sum)
        if alarm:
            self.reset()
        return alarm, stat


# ==============================================================================
# 3. CÀI ĐẶT ĐỦ 10 PHƯƠNG PHÁP CHUẨN BÁM SÁT MÃ NGUỒN GỐC `vcerqueira/studd`
# ==============================================================================

def _compute_metrics_and_costs(
    method: str,
    y_true_online: np.ndarray,
    y_pred_online: np.ndarray,
    n_total: int,
    window_size: int,
    samples_used_online: int,
    alarm_indices: List[int],
    elapsed_sec: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    if np.isnan(kappa):
        kappa = 0.0

    # Cách 1: Tính tổng nhãn yêu cầu theo đúng cửa sổ W đang chạy: (W_init + samples_used_online) / N
    cost_w = float((window_size + samples_used_online) / n_total)
    # Cách 2: Quy đổi theo đơn vị batch 1000 mẫu trong script gốc `vcerqueira/studd/experiments/main.py`
    unit_paper = 500 if window_size == 500 else 1000
    cost_paper_unit = float(((1 + len(alarm_indices)) * unit_paper) / n_total)
    if method == "BL-ret":
        n_ret_paper = max(0, (n_total - unit_paper) // unit_paper)
        cost_paper_unit = float(((1 + n_ret_paper) * unit_paper) / n_total)

    out: Dict[str, Any] = {
        "method": method,
        "accuracy": acc,
        "kappa": kappa,
        "labeling_cost": cost_w,
        "labeling_cost_paper_unit": cost_paper_unit,
        "samples_used_total": int(window_size + samples_used_online),
        "num_alarms": len(alarm_indices),
        "alarm_indices": alarm_indices,
        "elapsed_sec": round(elapsed_sec, 3),
        "y_true_online": y_true_online,
        "y_pred_online": y_pred_online,
    }
    if extra:
        out.update(extra)
    return out


def run_bl_static(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
) -> Dict[str, Any]:
    """1. `BL-st` (`BL1_never_adapt`): Huấn luyện 1 lần duy nhất trên W mẫu đầu tiên."""
    t0 = time.perf_counter()
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    y_true_online = y[W:]
    y_pred_online = model.predict(X[W:])
    elapsed = time.perf_counter() - t0

    return _compute_metrics_and_costs(
        method="BL-st",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=len(X),
        window_size=W,
        samples_used_online=0,
        alarm_indices=[],
        elapsed_sec=elapsed,
    )


def run_bl_retrain(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
) -> Dict[str, Any]:
    """2. `BL-ret` (`BL2_retrain_after_w`): Tái huấn luyện định kỳ sau mỗi W quan sát mới."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    while curr_pos < n_total:
        end_chunk = min(n_total, curr_pos + W)
        y_pred_online[curr_pos - W : end_chunk - W] = model.predict(X[curr_pos:end_chunk])
        if end_chunk < n_total:
            alarm_indices.append(end_chunk - 1)
            start_tr = max(W, end_chunk - W)
            samples_used_online += end_chunk - start_tr
            model = clone(teacher_template)
            model.fit(X[start_tr:end_chunk], y[start_tr:end_chunk])
        curr_pos = end_chunk

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method="BL-ret",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
    )


def run_studd_stream(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    student_template: ClassifierMixin,
    window_size: int,
    delta: float = 0.001,
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """3. `STUDD` (`drift_detection_std`): Teacher–Student + Page-Hinkley không nhãn."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    n_online = n_total - W

    teacher = clone(teacher_template)
    student = clone(student_template)
    teacher.fit(X[:W], y[:W])
    student.fit(X[:W], teacher.predict(X[:W]))

    detector = SkMultiflowPageHinkley(delta=delta, threshold=threshold)
    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    y_student_online = np.zeros_like(y_true_online)
    mimic_losses = np.zeros(n_online, dtype=np.float32)
    ph_statistics = np.zeros(n_online, dtype=np.float32)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    pred_chunk_size = max(4096, 2 * W)
    while curr_pos < n_total:
        chunk_end = min(n_total, curr_pos + pred_chunk_size)
        preds_T = teacher.predict(X[curr_pos:chunk_end])
        preds_S = student.predict(X[curr_pos:chunk_end])
        alarm_triggered = False

        for offset in range(len(preds_T)):
            stream_idx = curr_pos + offset
            online_idx = stream_idx - W

            p_t = preds_T[offset]
            p_s = preds_S[offset]
            y_pred_online[online_idx] = p_t
            y_student_online[online_idx] = p_s

            loss_t = 1.0 if p_t != p_s else 0.0
            mimic_losses[online_idx] = loss_t
            drift_alarm, ph_stat = detector.update(loss_t)
            ph_statistics[online_idx] = ph_stat

            if drift_alarm:
                alarm_indices.append(stream_idx)
                start_adapt = max(W, stream_idx + 1 - W)
                samples_used_online += (stream_idx + 1) - start_adapt

                teacher = clone(teacher_template)
                teacher.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])

                # Theo dòng 157 trong `studd_batch.py` (`upd_std_model=True`):
                # Student học bắt chước dự đoán của Teacher mới trên buffer online tích lũy X[W : t+1]
                yhat_buf = teacher.predict(X[W : stream_idx + 1])
                student = clone(student_template)
                student.fit(X[W : stream_idx + 1], yhat_buf)

                curr_pos = stream_idx + 1
                alarm_triggered = True
                break

        if not alarm_triggered:
            curr_pos = chunk_end

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method="STUDD",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
        extra={
            "y_student_online": y_student_online,
            "mimic_losses": mimic_losses,
            "ph_statistics": ph_statistics,
        },
    )


def run_supervised_pht(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
    method_name: str,
    delay_time: int = 0,
    observation_ratio: float = 1.0,
    delta: float = 0.002,
    threshold: float = 50.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """4–7. Nhóm Supervised (`SS`, `DSS`, `WS`, `DWS`) theo `drift_detection_spv` trong `studd_batch.py`."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    detector = SkMultiflowPageHinkley(delta=delta, threshold=threshold)
    rng = np.random.default_rng(random_state)
    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    pred_chunk_size = max(4096, 2 * W)
    while curr_pos < n_total:
        chunk_end = min(n_total, curr_pos + pred_chunk_size)
        preds_chunk = model.predict(X[curr_pos:chunk_end])
        alarm_triggered = False

        for offset in range(len(preds_chunk)):
            stream_idx = curr_pos + offset
            j = stream_idx - W
            pred = preds_chunk[offset]
            y_pred_online[j] = pred

            put_i_available = 1 if observation_ratio >= 1.0 else int(rng.binomial(1, observation_ratio))
            drift_alarm = False
            if put_i_available > 0 and j >= delay_time:
                delayed_idx = j - delay_time
                true_err = 1.0 if y_true_online[delayed_idx] != y_pred_online[delayed_idx] else 0.0
                drift_alarm, _ = detector.update(true_err)

            if drift_alarm:
                alarm_indices.append(stream_idx)
                start_adapt = max(W, stream_idx + 1 - W)
                samples_used_online += (stream_idx + 1) - start_adapt
                model = clone(teacher_template)
                model.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])
                curr_pos = stream_idx + 1
                alarm_triggered = True
                break

        if not alarm_triggered:
            curr_pos = chunk_end

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method=method_name,
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
    )


def run_unsupervised_os(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
    ks_alpha: float = 0.001,
) -> Dict[str, Any]:
    """8. `OS` (`drift_detection_uspv` + `HypothesisTestDetector`): KS-test hai cửa sổ trượt từng mẫu (step = 1)."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while curr_pos < n_total:
            preds_chunk = model.predict(X[curr_pos:])
            vals_chunk = preds_chunk.astype(np.float64)
            alarm_triggered = False

            for offset in range(len(preds_chunk)):
                stream_idx = curr_pos + offset
                online_idx = stream_idx - W
                y_pred_online[online_idx] = preds_chunk[offset]

                data_len = offset + 1
                if data_len >= 2 * W:
                    ref_w = vals_chunk[data_len - 2 * W : data_len - W]
                    test_w = vals_chunk[data_len - W : data_len]
                    _, p_val = ks_2samp(test_w, ref_w)
                    if p_val < ks_alpha:
                        alarm_indices.append(stream_idx)
                        start_adapt = max(W, stream_idx + 1 - W)
                        samples_used_online += (stream_idx + 1) - start_adapt
                        model = clone(teacher_template)
                        model.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])
                        curr_pos = stream_idx + 1
                        alarm_triggered = True
                        break

                if n_total >= 50000 and stream_idx > W and stream_idx % 25000 == 0:
                    print(f"    [OS progress] {stream_idx}/{n_total}...", flush=True)

            if not alarm_triggered:
                break

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method="OS",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
    )


def run_unsupervised_of(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
    ks_alpha: float = 0.001,
) -> Dict[str, Any]:
    """9. `OF` (`drift_detection_uspv_f` + `FixedWindowDetector`): KS-test với cửa sổ đầu ra cố định (step = 1)."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while curr_pos < n_total:
            preds_chunk = model.predict(X[curr_pos:])
            vals_chunk = preds_chunk.astype(np.float64)
            # Theo `FixedWindowDetector`: `ref_window` lấy đúng W mẫu đầu tiên sau mỗi lần reset/fit
            ref_window = vals_chunk[:W]
            alarm_triggered = False

            for offset in range(len(preds_chunk)):
                stream_idx = curr_pos + offset
                online_idx = stream_idx - W
                y_pred_online[online_idx] = preds_chunk[offset]

                data_len = offset + 1
                if data_len >= W:
                    test_w = vals_chunk[data_len - W : data_len]
                    _, p_val = ks_2samp(test_w, ref_window)
                    if p_val < ks_alpha:
                        alarm_indices.append(stream_idx)
                        start_adapt = max(W, stream_idx + 1 - W)
                        samples_used_online += (stream_idx + 1) - start_adapt
                        model = clone(teacher_template)
                        model.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])
                        curr_pos = stream_idx + 1
                        alarm_triggered = True
                        break

                if n_total >= 50000 and stream_idx > W and stream_idx % 25000 == 0:
                    print(f"    [OF progress] {stream_idx}/{n_total}...", flush=True)

            if not alarm_triggered:
                break

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method="OF",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
    )


def run_feature_ff(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
    ks_alpha: float = 0.001,
) -> Dict[str, Any]:
    """10. `FF` (`drift_detection_uspv_x` + `XCTracker`): KS-test trên từng đặc trưng đầu vào (step = 1)."""
    t0 = time.perf_counter()
    n_total, n_features = X.shape
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    # Xử lý an toàn NaN cho GMSC (giống quy ước imputer theo cửa sổ tham chiếu để ks_2samp không lỗi)
    X_work = np.asarray(X, dtype=np.float64)
    if np.isnan(X_work).any():
        col_medians = np.nanmedian(X_work[:W], axis=0)
        nan_mask = np.isnan(X_work)
        for c in range(n_features):
            if nan_mask[:, c].any():
                X_work[nan_mask[:, c], c] = col_medians[c]

    ref_X = X_work[:W]
    y_true_online = y[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []
    samples_used_online = 0

    curr_pos = W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while curr_pos < n_total:
            preds_chunk = model.predict(X[curr_pos:])
            alarm_triggered = False

            for offset in range(len(preds_chunk)):
                stream_idx = curr_pos + offset
                online_idx = stream_idx - W
                y_pred_online[online_idx] = preds_chunk[offset]

                data_len = offset + 1
                if data_len >= W:
                    test_X = X_work[stream_idx + 1 - W : stream_idx + 1]
                    any_change = False
                    for col_idx in range(n_features):
                        _, p_val = ks_2samp(test_X[:, col_idx], ref_X[:, col_idx])
                        if p_val < ks_alpha:
                            any_change = True
                            break

                    if any_change:
                        alarm_indices.append(stream_idx)
                        start_adapt = max(W, stream_idx + 1 - W)
                        samples_used_online += (stream_idx + 1) - start_adapt
                        model = clone(teacher_template)
                        model.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])
                        ref_X = X_work[start_adapt : stream_idx + 1]
                        curr_pos = stream_idx + 1
                        alarm_triggered = True
                        break

                if n_total >= 50000 and stream_idx > W and stream_idx % 20000 == 0:
                    print(f"    [FF progress] {stream_idx}/{n_total}...", flush=True)

            if not alarm_triggered:
                break

    elapsed = time.perf_counter() - t0
    return _compute_metrics_and_costs(
        method="FF",
        y_true_online=y_true_online,
        y_pred_online=y_pred_online,
        n_total=n_total,
        window_size=W,
        samples_used_online=samples_used_online,
        alarm_indices=alarm_indices,
        elapsed_sec=elapsed,
    )


# ==============================================================================
# 4. VẼ BIỂU ĐỒ ROLLING KAPPA & BIỂU ĐỒ SO SÁNH 10 PHƯƠNG PHÁP
# ==============================================================================

def compute_rolling_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, rolling_window: int = 200, stride: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    steps, kappas = [], []
    for end_idx in range(rolling_window, len(y_true) + 1, stride):
        start_idx = end_idx - rolling_window
        y_t_win = y_true[start_idx:end_idx]
        y_p_win = y_pred[start_idx:end_idx]
        if len(np.unique(y_t_win)) < 2 and len(np.unique(y_p_win)) < 2:
            k = 1.0 if np.array_equal(y_t_win, y_p_win) else 0.0
        else:
            k = cohen_kappa_score(y_t_win, y_p_win)
            if np.isnan(k):
                k = 0.0
        steps.append(end_idx)
        kappas.append(float(k))
    return np.array(steps), np.array(kappas)


def plot_dataset_rolling_figure(
    dataset_title: str,
    res_studd: Dict[str, Any],
    res_ss: Dict[str, Any],
    window_size: int,
    teacher_code: str,
    student_code: str,
    save_path: str,
) -> None:
    """Vẽ biểu đồ Rolling Kappa (cửa sổ trượt 200 mẫu) đối chiếu Hình 7/8/9 bài báo."""
    steps, k_studd = compute_rolling_kappa(
        res_studd["y_true_online"], res_studd["y_pred_online"]
    )
    _, k_ss = compute_rolling_kappa(
        res_ss["y_true_online"], res_ss["y_pred_online"]
    )
    _, k_student_mimic = compute_rolling_kappa(
        res_studd["y_pred_online"], res_studd["y_student_online"]
    )

    global_steps = steps + window_size
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    ax1.plot(global_steps, k_ss, label=f"SS (Có giám sát - Kappa={res_ss['kappa']:.2f})", color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.85)
    ax1.plot(global_steps, k_studd, label=f"STUDD Teacher ({teacher_code} - Kappa={res_studd['kappa']:.2f})", color="#2ca02c", linewidth=2.2)
    ax1.plot(global_steps, k_student_mimic, label=f"Student Mimicry ({student_code} vs {teacher_code})", color="#9467bd", linestyle=":", linewidth=1.6, alpha=0.85)

    for idx, alarm_t in enumerate(res_ss["alarm_indices"]):
        ax1.axvline(alarm_t, color="#1f77b4", linestyle="--", alpha=0.35, linewidth=1.2,
                    label="Báo động của SS" if idx == 0 else None)
    for idx, alarm_t in enumerate(res_studd["alarm_indices"]):
        ax1.axvline(alarm_t, color="#d62728", linestyle="-.", alpha=0.9, linewidth=1.8,
                    label="Báo động của STUDD" if idx == 0 else None)

    ax1.set_ylabel("Cohen's Kappa (Cửa sổ trượt 200 mẫu)", fontsize=10.5, fontweight="bold")
    ax1.set_title(
        f"Bộ dữ liệu {dataset_title} (W={window_size}, Teacher={teacher_code}, Student={student_code}): "
        f"STUDD vs SS & Student Mimicry",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylim(-0.15, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", ncol=2, fontsize=9)

    online_time = np.arange(window_size, window_size + len(res_studd["ph_statistics"]))
    ax2.plot(online_time, res_studd["ph_statistics"], color="#ff7f0e", linewidth=1.4, label="Thống kê Page-Hinkley (STUDD)")
    ax2.axhline(50.0, color="#d62728", linestyle=":", linewidth=1.2, label="Ngưỡng báo động (lambda = 50)")
    for alarm_t in res_studd["alarm_indices"]:
        ax2.axvline(alarm_t, color="#d62728", linestyle="-.", alpha=0.9, linewidth=1.5)
    ax2.set_xlabel("Thời gian trên luồng dữ liệu (Time step t)", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Thống kê PH_t", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# 5. HÀM CHẠY TRỌN VẸN 1 BỘ DỮ LIỆU (ĐỦ 10 PHƯƠNG PHÁP) & ĐỒNG BỘ LÊN W&B
# ==============================================================================

def run_single_dataset(
    dataset_name: str,
    window_size: Optional[int] = None,
    delta: float = 0.001,
    threshold: float = 50.0,
    auto_select_models: bool = False,
    use_wandb: bool = True,
    entity: str = "lgkai-hcmus",
    project: str = "STUDD",
    random_state: int = 42,
) -> pd.DataFrame:
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS_KHANH_DIR, exist_ok=True)

    # 1. Tải dữ liệu từ Hugging Face và đối chiếu snapshot local
    X, y, meta = load_stream(dataset_name, prefer_hf=True, verify_snapshot=True)
    key = str(meta["key"])
    canonical_name = str(meta["name"])
    W = int(window_size) if window_size is not None else int(meta["recommended_window_size"])

    # 2. Khởi tạo cặp mô hình Teacher - Student theo Bảng 2 (hoặc chọn tự động theo Mục 5.6.2)
    if auto_select_models:
        teacher_model, student_model, t_code, s_code, _ = select_best_teacher_student(
            X[:W], y[:W], random_state=random_state
        )
    else:
        teacher_model, student_model, t_code, s_code = get_table2_models(
            key, random_state=random_state
        )

    print("\n" + "=" * 88)
    print(f"[THỰC NGHIỆM 10 PHƯƠNG PHÁP — BỘ DỮ LIỆU: {canonical_name.upper()}]")
    print("=" * 88)
    print(f" * Nguồn dữ liệu      : {meta['source_used']}")
    print(f" * Khớp snapshot local: {meta['snapshot_verified']}")
    print(f" * Kích thước thực tế : {meta['n_samples']} dòng × {meta['n_features']} đặc trưng ({meta['n_classes']} lớp)")
    print(f" * Kích thước Paper   : {meta['paper_rows']} dòng × {meta['paper_total_cols']} cột tổng ({meta['paper_classes']} lớp)")
    print(f" * Cấu hình mô hình   : Teacher = {t_code} | Student = {s_code} | W = {W} | delta = {delta} | lambda = {threshold}")
    print("-" * 88)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            name=f"khanh-{key}-10methods-W{W}-{t_code}_{s_code}",
            tags=["khanh", key, "10-methods", "benchmark-19", f"T_{t_code}_S_{s_code}"],
            config={
                "owner": "Khanh",
                "dataset": canonical_name,
                "dataset_key": key,
                "source_used": meta["source_used"],
                "hf_url": meta["hf_url"],
                "snapshot_verified": meta["snapshot_verified"],
                "n_samples": meta["n_samples"],
                "n_features": meta["n_features"],
                "n_classes": meta["n_classes"],
                "paper_rows": meta["paper_rows"],
                "window_size": W,
                "teacher_model": t_code,
                "student_model": s_code,
                "delta": delta,
                "threshold": threshold,
                "ks_alpha": delta,
                "ks_step": 1,
                "methods": ALL_10_METHODS,
                "random_state": random_state,
            },
        )

    # 3. Chạy trọn vẹn 10 phương pháp chuẩn bài báo (`vcerqueira/studd` step = 1)
    print(" -> [1/10] Chạy BL-st...", flush=True)
    res_bl_st = run_bl_static(X, y, teacher_model, window_size=W)

    print(" -> [2/10] Chạy BL-ret...", flush=True)
    res_bl_ret = run_bl_retrain(X, y, teacher_model, window_size=W)

    print(" -> [3/10] Chạy STUDD...", flush=True)
    res_studd = run_studd_stream(
        X, y, teacher_model, student_model, window_size=W, delta=delta, threshold=threshold
    )

    print(" -> [4/10] Chạy SS (100% nhãn, trễ 0)...", flush=True)
    res_ss = run_supervised_pht(
        X, y, teacher_model, window_size=W, method_name="SS",
        delay_time=0, observation_ratio=1.0, delta=delta * 2.0, threshold=threshold, random_state=random_state,
    )

    print(" -> [5/10] Chạy DSS (100% nhãn, trễ W/2)...", flush=True)
    res_dss = run_supervised_pht(
        X, y, teacher_model, window_size=W, method_name="DSS",
        delay_time=W // 2, observation_ratio=1.0, delta=delta * 2.0, threshold=threshold, random_state=random_state,
    )

    print(" -> [6/10] Chạy WS (50% nhãn, trễ 0)...", flush=True)
    res_ws = run_supervised_pht(
        X, y, teacher_model, window_size=W, method_name="WS",
        delay_time=0, observation_ratio=0.5, delta=delta * 2.0, threshold=threshold, random_state=random_state,
    )

    print(" -> [7/10] Chạy DWS (50% nhãn, trễ W/2)...", flush=True)
    res_dws = run_supervised_pht(
        X, y, teacher_model, window_size=W, method_name="DWS",
        delay_time=W // 2, observation_ratio=0.5, delta=delta * 2.0, threshold=threshold, random_state=random_state,
    )

    print(" -> [8/10] Chạy OS (Output Sliding KS, step=1)...", flush=True)
    res_os = run_unsupervised_os(X, y, teacher_model, window_size=W, ks_alpha=delta)

    print(" -> [9/10] Chạy OF (Output Fixed KS, step=1)...", flush=True)
    res_of = run_unsupervised_of(X, y, teacher_model, window_size=W, ks_alpha=delta)

    print(" -> [10/10] Chạy FF (Feature Fixed KS, step=1)...", flush=True)
    res_ff = run_feature_ff(X, y, teacher_model, window_size=W, ks_alpha=delta)

    all_results = [
        res_bl_st, res_bl_ret, res_studd,
        res_ss, res_dss, res_ws, res_dws,
        res_os, res_of, res_ff,
    ]
    paper_ref = PAPER_BENCHMARK_NUMBERS.get(key, {})

    rows = []
    for r in all_results:
        m = r["method"]
        p_kappa = paper_ref.get(m, {}).get("kappa", np.nan)
        p_cost = paper_ref.get(m, {}).get("cost", np.nan)
        rows.append({
            "Dataset": canonical_name,
            "Method": m,
            "Teacher": t_code,
            "Student": s_code if m == "STUDD" else "-",
            "W": W,
            "Our_Kappa": round(r["kappa"], 4),
            "Paper_Kappa": p_kappa,
            "Kappa_Diff": round(r["kappa"] - p_kappa, 4) if not np.isnan(p_kappa) else np.nan,
            "Our_Cost_W": round(r["labeling_cost"], 4),
            "Our_Cost_1000": round(r["labeling_cost_paper_unit"], 4),
            "Paper_Cost": p_cost,
            "Alarms": r["num_alarms"],
            "Accuracy": round(r["accuracy"], 4),
            "Time_s": r["elapsed_sec"],
        })

    df_table = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_KHANH_DIR, f"{key}_comparison_table.csv")
    df_table.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig_path = os.path.join(RESULTS_KHANH_DIR, f"{key}_rolling_kappa.png")
    plot_dataset_rolling_figure(
        dataset_title=canonical_name,
        res_studd=res_studd,
        res_ss=res_ss,
        window_size=W,
        teacher_code=t_code,
        student_code=s_code,
        save_path=fig_path,
    )

    print("\n" + df_table.to_string(index=False))
    print("-" * 88)
    print(f" [+] Báo động STUDD ({len(res_studd['alarm_indices'])}): {res_studd['alarm_indices']}")
    print(f" [+] Đã lưu CSV tại : {csv_path}")
    print(f" [+] Đã lưu biểu đồ : {fig_path}")

    if wandb_run is not None:
        import wandb
        steps, k_studd = compute_rolling_kappa(res_studd["y_true_online"], res_studd["y_pred_online"])
        _, k_ss = compute_rolling_kappa(res_ss["y_true_online"], res_ss["y_pred_online"])
        _, k_student = compute_rolling_kappa(res_studd["y_pred_online"], res_studd["y_student_online"])
        for t_val, ks_val, kss_val, kstu_val in zip(steps + W, k_studd, k_ss, k_student):
            wandb.log({
                "stream_step": int(t_val),
                f"{canonical_name}/Rolling_Kappa_STUDD": float(ks_val),
                f"{canonical_name}/Rolling_Kappa_SS": float(kss_val),
                f"{canonical_name}/Rolling_Kappa_StudentMimic": float(kstu_val),
            })

        summary_metrics: Dict[str, Any] = {
            f"{canonical_name}_Comparison_Table": wandb.Table(dataframe=df_table),
            f"{canonical_name}_Rolling_Kappa_Plot": wandb.Image(fig_path),
        }
        for r in all_results:
            m_clean = r["method"].replace("-", "_")
            summary_metrics[f"{canonical_name}/{m_clean}_Kappa"] = r["kappa"]
            summary_metrics[f"{canonical_name}/{m_clean}_Cost_W"] = r["labeling_cost"]
            summary_metrics[f"{canonical_name}/{m_clean}_Cost_1000"] = r["labeling_cost_paper_unit"]
            summary_metrics[f"{canonical_name}/{m_clean}_Alarms"] = r["num_alarms"]

        wandb.log(summary_metrics)
        print(f" [W&B] Đã đồng bộ 10 phương pháp lên: {wandb_run.url}")
        wandb.finish()

    return df_table


def summarize_khanh_7datasets(
    use_wandb: bool = True,
    entity: str = "lgkai-hcmus",
    project: str = "STUDD",
) -> pd.DataFrame:
    """Tổng hợp kết quả cả 7 bộ × 10 phương pháp = 70 runs + Biểu đồ Rank (Hình 10)."""
    sys.stdout.reconfigure(encoding="utf-8")
    khanh_keys = ["bike", "luxembourg", "outdoor", "rialto", "gas", "ozone", "gmsc"]
    dfs = []
    for k in khanh_keys:
        p = os.path.join(RESULTS_KHANH_DIR, f"{k}_comparison_table.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise RuntimeError("Chưa có file kết quả nào trong results/khanh/!")

    df_all = pd.concat(dfs, ignore_index=True)
    summary_csv = os.path.join(RESULTS_KHANH_DIR, "khanh_7datasets_summary.csv")
    df_all.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Tính Average Rank trên 7 bộ dữ liệu (tái hiện Hình 10 bài báo: thấp hơn = tốt hơn)
    pivot_kappa = df_all.pivot(index="Dataset", columns="Method", values="Our_Kappa")
    pivot_cost = df_all.pivot(index="Dataset", columns="Method", values="Our_Cost_W")
    avg_rank_kappa = pivot_kappa.rank(axis=1, ascending=False, method="average").mean(axis=0)
    avg_rank_cost = pivot_cost.rank(axis=1, ascending=True, method="average").mean(axis=0)
    mean_kappa = pivot_kappa.mean(axis=0)
    mean_cost = pivot_cost.mean(axis=0)

    df_rank = pd.DataFrame({
        "Method": ALL_10_METHODS,
        "Mean_Kappa": [round(float(mean_kappa[m]), 4) for m in ALL_10_METHODS],
        "Mean_Cost_W": [round(float(mean_cost[m]), 4) for m in ALL_10_METHODS],
        "Avg_Rank_Performance": [round(float(avg_rank_kappa[m]), 2) for m in ALL_10_METHODS],
        "Avg_Rank_Cost": [round(float(avg_rank_cost[m]), 2) for m in ALL_10_METHODS],
    })
    rank_csv = os.path.join(RESULTS_KHANH_DIR, "khanh_7datasets_rank_summary.csv")
    df_rank.to_csv(rank_csv, index=False, encoding="utf-8-sig")

    # Hình 1: Đối chiếu STUDD (Our vs Paper) trên 7 bộ
    sub_studd = df_all[df_all["Method"] == "STUDD"].reset_index(drop=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    x = np.arange(len(sub_studd))
    width = 0.36

    b1 = ax1.bar(x - width / 2, sub_studd["Our_Kappa"], width, label="Our STUDD Kappa", color="#2ca02c")
    b2 = ax1.bar(x + width / 2, sub_studd["Paper_Kappa"], width, label="Paper STUDD Kappa (Bảng 3)", color="#4c72b0")
    ax1.set_title("Đối chiếu Cohen's Kappa của STUDD (7 bộ của Khánh)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_studd["Dataset"], fontweight="bold", rotation=15)
    ax1.set_ylim(0, 1.15)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    b3 = ax2.bar(x - width / 2, sub_studd["Our_Cost_W"], width, label="Our STUDD Cost (W)", color="#ff7f0e")
    b4 = ax2.bar(x + width / 2, sub_studd["Paper_Cost"], width, label="Paper STUDD Cost (Bảng 4)", color="#dd8452")
    ax2.set_title("Đối chiếu Chi phí Nhãn của STUDD (7 bộ của Khánh)", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub_studd["Dataset"], fontweight="bold", rotation=15)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.suptitle("Tổng hợp Thực nghiệm Tái hiện 7 Bộ Dữ liệu (Đầy đủ 10 Phương pháp — Khánh)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    overview_fig = os.path.join(RESULTS_KHANH_DIR, "khanh_7datasets_overview.png")
    plt.savefig(overview_fig, dpi=300)
    plt.close(fig)

    # Hình 2: Tái hiện Hình 10 bài báo — Average Rank Scatter (Performance vs Cost trên 10 phương pháp)
    fig_r, ax_r = plt.subplots(figsize=(8.8, 6.2))
    for _, row in df_rank.iterrows():
        m = row["Method"]
        px, py = row["Avg_Rank_Performance"], row["Avg_Rank_Cost"]
        is_studd = m == "STUDD"
        color = "#d62728" if is_studd else ("#1f77b4" if m in ("SS", "DSS", "WS", "DWS") else "#2ca02c" if m in ("OS", "OF", "FF") else "#7f7f7f")
        marker = "*" if is_studd else "o"
        size = 280 if is_studd else 100
        ax_r.scatter(px, py, s=size, c=color, marker=marker, zorder=5 if is_studd else 3, edgecolors="black", linewidths=0.8)
        ax_r.annotate(m, xy=(px, py), xytext=(7, 5), textcoords="offset points",
                      fontsize=10.5 if is_studd else 9.5, fontweight="bold" if is_studd else "normal",
                      color=color)

    ax_r.set_xlabel("Thứ hạng Trung bình về Hiệu năng Kappa (Nhỏ hơn = Tốt hơn)", fontsize=10.5, fontweight="bold")
    ax_r.set_ylabel("Thứ hạng Trung bình về Chi phí Nhãn (Nhỏ hơn = Tiết kiệm hơn)", fontsize=10.5, fontweight="bold")
    ax_r.set_title("Tái hiện Hình 10: Đánh đổi Hiệu năng vs. Chi phí Nhãn (10 Phương pháp trên 7 bộ của Khánh)", fontsize=11.5, fontweight="bold")
    ax_r.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    rank_fig = os.path.join(RESULTS_KHANH_DIR, "khanh_7datasets_rank_plot.png")
    plt.savefig(rank_fig, dpi=300)
    plt.close(fig_r)

    print("\n" + "=" * 92)
    print(f"BẢNG TỔNG HỢP 7 BỘ DỮ LIỆU × 10 PHƯƠNG PHÁP ({len(df_all)} RUNS)")
    print("=" * 92)
    print(df_rank.to_string(index=False))
    print("-" * 92)
    print(f" [+] Bảng chi tiết 70 runs : {summary_csv}")
    print(f" [+] Bảng xếp hạng 10 PP   : {rank_csv}")
    print(f" [+] Biểu đồ đối chiếu     : {overview_fig}")
    print(f" [+] Biểu đồ Rank (Hình 10): {rank_fig}")

    if use_wandb:
        import wandb
        run = wandb.init(
            entity=entity,
            project=project,
            name="khanh-all-7datasets-10methods-summary",
            tags=["khanh", "summary-7datasets", "10-methods", "benchmark-19"],
            config={
                "owner": "Khanh",
                "datasets": list(sub_studd["Dataset"]),
                "total_datasets": len(sub_studd),
                "methods": ALL_10_METHODS,
                "total_runs": len(df_all),
            },
        )
        wandb.log({
            "Khanh_7Datasets_70Runs_Table": wandb.Table(dataframe=df_all),
            "Khanh_7Datasets_Rank_Table": wandb.Table(dataframe=df_rank),
            "Khanh_7Datasets_STUDD_Table": wandb.Table(dataframe=sub_studd),
            "Khanh_7Datasets_Overview_Plot": wandb.Image(overview_fig),
            "Khanh_7Datasets_Rank_Plot": wandb.Image(rank_fig),
        })
        print(f" [W&B] Đã đồng bộ bảng tổng hợp 70 runs lên: {run.url}")
        wandb.finish()

    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm đủ 10 phương pháp STUDD trên từng bộ dữ liệu chuẩn.")
    parser.add_argument("--dataset", type=str, default="bike", help="Tên bộ dữ liệu cần chạy (mặc định: bike).")
    parser.add_argument("--window-size", type=int, default=None, help="Ghi đè kích thước cửa sổ W (mặc định lấy theo Mục 5.7).")
    parser.add_argument("--delta", type=float, default=0.001, help="Tham số delta của Page-Hinkley & ngưỡng alpha KS (mặc định: 0.001).")
    parser.add_argument("--threshold", type=float, default=50.0, help="Ngưỡng lambda của Page-Hinkley (mặc định: 50.0).")
    parser.add_argument("--auto-select", action="store_true", help="Tự động chọn Teacher/Student theo Mục 5.6.2 thay vì lấy cố định Bảng 2.")
    parser.add_argument("--summarize-khanh", action="store_true", help="Tổng hợp kết quả cả 7 bộ × 10 phương pháp và đẩy lên W&B.")
    parser.add_argument("--no-wandb", action="store_true", help="Chạy ngoại tuyến không đẩy lên W&B.")
    args = parser.parse_args()

    if args.summarize_khanh:
        summarize_khanh_7datasets(use_wandb=not args.no_wandb)
    else:
        run_single_dataset(
            dataset_name=args.dataset,
            window_size=args.window_size,
            delta=args.delta,
            threshold=args.threshold,
            auto_select_models=args.auto_select,
            use_wandb=not args.no_wandb,
        )
