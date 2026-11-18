"""Kịch bản chạy thực nghiệm chuẩn hóa cho từng bộ dữ liệu trong 19 Benchmark Streams.

Hỗ trợ:
    - Tải dữ liệu tự động từ Hugging Face (pinned commit) và đối chiếu snapshot local (`src/data_loader.py`).
    - Khởi tạo mô hình Teacher và Student đúng theo Bảng 2 (hoặc tự động chọn theo Mục 5.6.2) (`src/models.py`).
    - Đánh giá 5 phương pháp chuẩn của bài báo STUDD:
        1. `BL-st`  (Static Baseline): Huấn luyện 1 lần trên cửa sổ W đầu, không cập nhật.
        2. `BL-ret` (Periodic Retraining Baseline): Tái huấn luyện định kỳ sau mỗi W mẫu mới.
        3. `STUDD`  (Proposed Unsupervised): Học trò bắt chước Thầy giáo + Kiểm định Page-Hinkley.
        4. `SS`     (Supervised Sliding PHT): Giám sát lỗi thật I(y_hat != y_true) bằng Page-Hinkley.
        5. `OS`     (Output Sliding KS-Test): Kiểm định Kolmogorov-Smirnov trên cửa sổ trượt đầu ra.
    - Lưu bảng kết quả CSV + biểu đồ Rolling Kappa (300 DPI) vào `results/khanh/` và đồng bộ lên W&B.
"""

import argparse
import os
import sys
import time
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


# ==============================================================================
# 1. SỐ LIỆU BẢNG 3 (KAPPA) & BẢNG 4 (COST) TRONG BÀI BÁO CHO 7 BỘ CỦA KHÁNH
# ==============================================================================

PAPER_BENCHMARK_NUMBERS: Dict[str, Dict[str, Dict[str, float]]] = {
    "bike": {
        "BL-st":  {"kappa": 0.31, "cost": 0.06},
        "BL-ret": {"kappa": 0.31, "cost": 0.98},
        "STUDD":  {"kappa": 0.32, "cost": 0.06},
        "SS":     {"kappa": 0.30, "cost": 0.46},
        "OS":     {"kappa": 0.23, "cost": 0.06},
    },
    "luxembourg": {
        "BL-st":  {"kappa": 1.00, "cost": 0.53},
        "BL-ret": {"kappa": 1.00, "cost": 1.05},
        "STUDD":  {"kappa": 1.00, "cost": 0.53},
        "SS":     {"kappa": 1.00, "cost": 0.53},
        "OS":     {"kappa": 1.00, "cost": 0.53},
    },
    "ozone": {
        "BL-st":  {"kappa": 0.13, "cost": 0.39},
        "BL-ret": {"kappa": 0.17, "cost": 1.18},
        "STUDD":  {"kappa": 0.13, "cost": 0.39},
        "SS":     {"kappa": 0.14, "cost": 0.39},
        "OS":     {"kappa": 0.14, "cost": 0.39},
    },
    "gas": {
        "BL-st":  {"kappa": 0.32, "cost": 0.07},
        "BL-ret": {"kappa": 0.51, "cost": 0.93},
        "STUDD":  {"kappa": 0.48, "cost": 0.86},
        "SS":     {"kappa": 0.62, "cost": 0.89},
        "OS":     {"kappa": 0.44, "cost": 0.50},
    },
    "outdoor": {
        "BL-st":  {"kappa": 0.37, "cost": 0.25},
        "BL-ret": {"kappa": 0.38, "cost": 1.00},
        "STUDD":  {"kappa": 0.36, "cost": 0.25},
        "SS":     {"kappa": 0.38, "cost": 0.50},
        "OS":     {"kappa": 0.38, "cost": 0.62},
    },
    "rialto": {
        "BL-st":  {"kappa": 0.30, "cost": 0.01},
        "BL-ret": {"kappa": 0.34, "cost": 1.00},
        "STUDD":  {"kappa": 0.38, "cost": 0.21},
        "SS":     {"kappa": 0.47, "cost": 1.23},
        "OS":     {"kappa": 0.40, "cost": 0.50},
    },
    "gmsc": {
        "BL-st":  {"kappa": 0.32, "cost": 0.01},
        "BL-ret": {"kappa": 0.06, "cost": 0.99},
        "STUDD":  {"kappa": 0.32, "cost": 0.01},
        "SS":     {"kappa": 0.32, "cost": 0.01},
        "OS":     {"kappa": 0.32, "cost": 0.01},
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
# 3. CÀI ĐẶT 5 PHƯƠNG PHÁP ĐÁNH GIÁ TRÊN LUỒNG (CAUSAL ONLINE EVALUATION)
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
    """1. `BL-st`: Huấn luyện 1 lần duy nhất trên W mẫu đầu tiên."""
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
    """2. `BL-ret`: Tái huấn luyện định kỳ sau mỗi W quan sát mới (Mục 5.5)."""
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
    """3. `STUDD`: Phương pháp đề xuất (Teacher - Student + Page-Hinkley không nhãn)."""
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
                # Huấn luyện lại Teacher và Student trên cửa sổ W mẫu gần nhất (Mục 5.3 & Hình 6 bài báo)
                start_adapt = max(W, stream_idx + 1 - W)
                samples_used_online += (stream_idx + 1) - start_adapt

                teacher = clone(teacher_template)
                teacher.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])

                yhat_adapt = teacher.predict(X[start_adapt : stream_idx + 1])
                student = clone(student_template)
                student.fit(X[start_adapt : stream_idx + 1], yhat_adapt)

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


def run_supervised_ss(
    X: np.ndarray,
    y: np.ndarray,
    teacher_template: ClassifierMixin,
    window_size: int,
    delta: float = 0.002,
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """4. `SS`: Giám sát lỗi thật I(y_hat != y_true) bằng Page-Hinkley (Mục 5.5)."""
    t0 = time.perf_counter()
    n_total = len(X)
    W = window_size
    model = clone(teacher_template)
    model.fit(X[:W], y[:W])

    detector = SkMultiflowPageHinkley(delta=delta, threshold=threshold)
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
            online_idx = stream_idx - W
            pred = preds_chunk[offset]
            y_pred_online[online_idx] = pred

            true_err = 1.0 if pred != y[stream_idx] else 0.0
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
        method="SS",
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
    step_check: int = 50,
) -> Dict[str, Any]:
    """5. `OS` (Output Sliding KS-Test, Žliobaitė 2010): Kiểm định KS trên hai cửa sổ trượt đầu ra."""
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
        preds_chunk = model.predict(X[curr_pos:])
        probs_all = model.predict_proba(X[curr_pos:])
        # Lấy xác suất lớp dương (nhị phân) hoặc độ tự tin cực đại (đa lớp) theo đúng studd_batch.py
        if probs_all.shape[1] < 2:
            scores_chunk = probs_all[:, 0]
        elif probs_all.shape[1] == 2:
            scores_chunk = probs_all[:, 1]
        else:
            scores_chunk = np.max(probs_all, axis=1)

        alarm_triggered = False
        for offset in range(len(preds_chunk)):
            stream_idx = curr_pos + offset
            online_idx = stream_idx - W
            y_pred_online[online_idx] = preds_chunk[offset]

            # So sánh 2 cửa sổ trượt liền kề kích thước W (cần tối thiểu 2*W quan sát kể từ lần huấn luyện cuối)
            n_since_fit = offset + 1
            if n_since_fit >= 2 * W and n_since_fit % step_check == 0:
                ref_w = scores_chunk[n_since_fit - 2 * W : n_since_fit - W]
                test_w = scores_chunk[n_since_fit - W : n_since_fit]
                _, p_val = ks_2samp(ref_w, test_w)
                if p_val < ks_alpha:
                    alarm_indices.append(stream_idx)
                    start_adapt = max(W, stream_idx + 1 - W)
                    samples_used_online += (stream_idx + 1) - start_adapt
                    model = clone(teacher_template)
                    model.fit(X[start_adapt : stream_idx + 1], y[start_adapt : stream_idx + 1])
                    curr_pos = stream_idx + 1
                    alarm_triggered = True
                    break

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


# ==============================================================================
# 4. VẼ BIỂU ĐỒ ROLLING KAPPA (TÁI HIỆN HÌNH 9 TRONG BÀI BÁO CHO BIKE & CÁC BỘ KHÁC)
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
    res_bl_st: Dict[str, Any],
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
    # Hiệu năng bắt chước của Student so với dự đoán của Teacher (như đường Student trong Hình 7/8/9)
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
# 5. HÀM CHẠY TRỌN VẸN 1 BỘ DỮ LIỆU & ĐỒNG BỘ LÊN WEIGHTS & BIASES
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
        teacher_model, student_model, t_code, s_code, val_scores = select_best_teacher_student(
            X[:W], y[:W], random_state=random_state
        )
    else:
        teacher_model, student_model, t_code, s_code = get_table2_models(
            key, random_state=random_state
        )
        val_scores = {}

    print("\n" + "=" * 78)
    print(f"[THỰC NGHIỆM BỘ DỮ LIỆU: {canonical_name.upper()}]")
    print("=" * 78)
    print(f" * Nguồn dữ liệu      : {meta['source_used']}")
    print(f" * Khớp snapshot local: {meta['snapshot_verified']}")
    print(f" * Kích thước thực tế : {meta['n_samples']} dòng × {meta['n_features']} đặc trưng ({meta['n_classes']} lớp)")
    print(f" * Kích thước Paper   : {meta['paper_rows']} dòng × {meta['paper_total_cols']} cột tổng ({meta['paper_classes']} lớp)")
    print(f" * Phân bố nhãn       : {meta['class_counts']}")
    print(f" * Cấu hình mô hình   : Teacher = {t_code} | Student = {s_code} | Cửa sổ W = {W} | delta = {delta} | lambda = {threshold}")
    print("-" * 78)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            name=f"khanh-{key}-W{W}-{t_code}_{s_code}",
            tags=["khanh", key, "benchmark-19", f"T_{t_code}_S_{s_code}"],
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
                "random_state": random_state,
            },
        )

    # 3. Chạy 5 phương pháp chuẩn
    res_bl_st = run_bl_static(X, y, teacher_model, window_size=W)
    res_bl_ret = run_bl_retrain(X, y, teacher_model, window_size=W)
    res_studd = run_studd_stream(
        X, y, teacher_model, student_model, window_size=W, delta=delta, threshold=threshold
    )
    res_ss = run_supervised_ss(
        X, y, teacher_model, window_size=W, delta=delta * 2.0, threshold=threshold
    )
    res_os = run_unsupervised_os(
        X, y, teacher_model, window_size=W, ks_alpha=delta, step_check=50
    )

    all_results = [res_bl_st, res_bl_ret, res_studd, res_ss, res_os]
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
        res_bl_st=res_bl_st,
        window_size=W,
        teacher_code=t_code,
        student_code=s_code,
        save_path=fig_path,
    )

    print(df_table.to_string(index=False))
    print("-" * 78)
    print(f" [+] Các mốc báo động của STUDD : {res_studd['alarm_indices']}")
    print(f" [+] Các mốc báo động của SS    : {res_ss['alarm_indices']}")
    print(f" [+] Các mốc báo động của OS    : {res_os['alarm_indices']}")
    print(f" [+] Đã lưu bảng CSV tại        : {csv_path}")
    print(f" [+] Đã lưu biểu đồ tại         : {fig_path}")

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

        summary_metrics = {
            f"{canonical_name}/STUDD_Kappa": res_studd["kappa"],
            f"{canonical_name}/STUDD_Paper_Kappa": paper_ref.get("STUDD", {}).get("kappa", np.nan),
            f"{canonical_name}/STUDD_Cost_W": res_studd["labeling_cost"],
            f"{canonical_name}/STUDD_Cost_1000": res_studd["labeling_cost_paper_unit"],
            f"{canonical_name}/STUDD_Paper_Cost": paper_ref.get("STUDD", {}).get("cost", np.nan),
            f"{canonical_name}/STUDD_Alarms": res_studd["num_alarms"],
            f"{canonical_name}/BL_st_Kappa": res_bl_st["kappa"],
            f"{canonical_name}/SS_Kappa": res_ss["kappa"],
            f"{canonical_name}/OS_Kappa": res_os["kappa"],
            f"{canonical_name}_Comparison_Table": wandb.Table(dataframe=df_table),
            f"{canonical_name}_Rolling_Kappa_Plot": wandb.Image(fig_path),
        }
        wandb.log(summary_metrics)
        print(f" [W&B] Đã đồng bộ kết quả thực nghiệm lên: {wandb_run.url}")
        wandb.finish()

    return df_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm STUDD trên từng bộ dữ liệu chuẩn.")
    parser.add_argument("--dataset", type=str, default="bike", help="Tên bộ dữ liệu cần chạy (mặc định: bike).")
    parser.add_argument("--window-size", type=int, default=None, help="Ghi đè kích thước cửa sổ W (mặc định lấy theo Mục 5.7).")
    parser.add_argument("--delta", type=float, default=0.001, help="Tham số delta của Page-Hinkley (mặc định: 0.001).")
    parser.add_argument("--threshold", type=float, default=50.0, help="Ngưỡng lambda của Page-Hinkley (mặc định: 50.0).")
    parser.add_argument("--auto-select", action="store_true", help="Tự động chọn Teacher/Student theo Mục 5.6.2 thay vì lấy cố định Bảng 2.")
    parser.add_argument("--no-wandb", action="store_true", help="Chạy ngoại tuyến không đẩy lên W&B.")
    args = parser.parse_args()

    run_single_dataset(
        dataset_name=args.dataset,
        window_size=args.window_size,
        delta=args.delta,
        threshold=args.threshold,
        auto_select_models=args.auto_select,
        use_wandb=not args.no_wandb,
    )
