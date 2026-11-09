"""Kịch bản Thực nghiệm Tái hiện (Experimental Reproduction) — Giai đoạn 2 (Slide Giữa kỳ).

Tái hiện các kết quả cốt lõi của bài báo:
    Cerqueira et al. (2023). STUDD: a student–teacher method for unsupervised
    concept drift detection. Machine Learning, 112(11), 4351–4378.

Nội dung thực nghiệm:
    1. So sánh 4 phương pháp (Mục 5.5, trang 4362):
       - `BL-st` (Baseline Static): Huấn luyện 1 lần trên cửa sổ W đầu tiên, không thích nghi.
       - `STUDD` (Proposed Unsupervised): Mô hình Học trò - Thầy giáo + Kiểm định Page-Hinkley.
       - `SS` (Supervised Benchmark): Giám sát lỗi thật I(y_hat != y_true) bằng Page-Hinkley.
       - `OS` (Unsupervised KS-Test): Kiểm định thống kê Kolmogorov-Smirnov trên xác suất đầu ra.
    2. Thực hiện trên 3 bộ dữ liệu:
       - Luồng `abrupt_drift_synthetic`: Tái hiện Hình 7 (trang 4365) — Đường cong Rolling Kappa
         trên cửa sổ trượt 200 mẫu và vạch thời điểm phát báo động Drift.
       - Bộ dữ liệu thực `electricity`: Đối chiếu trực tiếp với Bảng 3 & Bảng 4 (trang 4367–4368).
       - Bộ dữ liệu thực `ozone`: Đối chiếu trực tiếp với Bảng 3 & Bảng 4 (trang 4367–4368).
    3. Xuất biểu đồ chuẩn 300 DPI vào thư mục `results/` và đồng bộ toàn bộ lên W&B (`lgkai-hcmus/STUDD`).
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Đảm bảo import được package src/ khi chạy từ bất kỳ thư mục nào
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import PageHinkley
from src.studd import STUDD
from src.data_loader import load_benchmark_stream, generate_abrupt_drift_stream

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ==============================================================================
# 1. CÀI ĐẶT CÁC PHƯƠNG PHÁP ĐỐI THỦ (BASELINES & BENCHMARK TRONG MỤC 5.5)
# ==============================================================================

def evaluate_static_baseline(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    window_size: int = 500,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `BL-st` (Static Baseline): Chỉ học trên W mẫu đầu, không bao giờ cập nhật."""
    W = window_size
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X_stream[:W], y_stream[:W])

    y_true_online = y_stream[W:]
    y_pred_online = model.predict(X_stream[W:])

    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    # Chi phí nhãn theo công thức Bảng 4 bài báo: chỉ tốn W mẫu khởi tạo ban đầu chia cho N
    total_cost = float(W / len(X_stream))

    return {
        "method": "BL-st",
        "accuracy": acc,
        "kappa": kappa,
        "labeling_cost": total_cost,
        "num_alarms": 0,
        "alarm_indices": [],
        "y_true_online": y_true_online,
        "y_pred_online": y_pred_online,
    }


def evaluate_supervised_ss(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    window_size: int = 500,
    delta: float = 0.001,
    threshold: float = 50.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `SS` (Supervised Sliding PHT): Giám sát lỗi thật I(y_hat != y_true) bằng Page-Hinkley."""
    n_total = len(X_stream)
    W = window_size
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X_stream[:W], y_stream[:W])

    detector = PageHinkley(delta=delta, threshold=threshold)
    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    curr_pos = W
    while curr_pos < n_total:
        preds_chunk = model.predict(X_stream[curr_pos:])
        alarm_triggered = False

        for offset in range(len(preds_chunk)):
            stream_idx = curr_pos + offset
            online_idx = stream_idx - W

            pred = preds_chunk[offset]
            y_pred_online[online_idx] = pred

            # SS đòi hỏi phải biết nhãn thật y_stream[stream_idx] ngay lập tức để tính lỗi
            true_error = 1.0 if pred != y_stream[stream_idx] else 0.0
            drift_alarm, _ = detector.update(true_error)

            if drift_alarm:
                alarm_indices.append(stream_idx)
                end_adapt = min(n_total, stream_idx + (W // 2))
                start_adapt = max(0, end_adapt - W)
                model = clone(model)
                model.fit(
                    X_stream[start_adapt:end_adapt],
                    y_stream[start_adapt:end_adapt],
                )
                detector.reset()
                curr_pos = stream_idx + 1
                alarm_triggered = True
                break

        if not alarm_triggered:
            break

    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    # Theo Bảng 4 bài báo: tỷ lệ nhãn dùng cho huấn luyện/thích nghi (giới hạn tối đa 1.0)
    total_cost = float(min(1.0, ((1 + len(alarm_indices)) * W) / n_total))

    return {
        "method": "SS",
        "accuracy": acc,
        "kappa": kappa,
        "labeling_cost": total_cost,
        "num_alarms": len(alarm_indices),
        "alarm_indices": alarm_indices,
        "y_true_online": y_true_online,
        "y_pred_online": y_pred_online,
    }


def evaluate_unsupervised_os_ks(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    window_size: int = 500,
    ks_alpha: float = 0.001,
    step_check: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `OS` (Output Sliding KS-Test, Žliobaitė 2010): Kiểm định KS trên xác suất dự đoán."""
    n_total = len(X_stream)
    W = window_size
    model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    model.fit(X_stream[:W], y_stream[:W])

    # Cửa sổ tham chiếu xác suất đầu ra trên tập huấn luyện
    ref_probs = model.predict_proba(X_stream[:W])[:, 0]

    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    curr_pos = W
    while curr_pos < n_total:
        preds_chunk = model.predict(X_stream[curr_pos:])
        probs_chunk = model.predict_proba(X_stream[curr_pos:])[:, 0]
        alarm_triggered = False

        for offset in range(len(preds_chunk)):
            stream_idx = curr_pos + offset
            online_idx = stream_idx - W
            y_pred_online[online_idx] = preds_chunk[offset]

            # Kiểm tra KS-test định kỳ khi tích lũy đủ ít nhất W mẫu mới kể từ lần huấn luyện cuối
            if (offset + 1) >= W and (offset + 1) % step_check == 0:
                recent_probs = probs_chunk[offset + 1 - W : offset + 1]
                _, p_val = ks_2samp(ref_probs, recent_probs)
                if p_val < ks_alpha:
                    alarm_indices.append(stream_idx)
                    start_adapt = max(0, stream_idx - W + 1)
                    model = clone(model)
                    model.fit(
                        X_stream[start_adapt : stream_idx + 1],
                        y_stream[start_adapt : stream_idx + 1],
                    )
                    ref_probs = model.predict_proba(
                        X_stream[start_adapt : stream_idx + 1]
                    )[:, 0]
                    curr_pos = stream_idx + 1
                    alarm_triggered = True
                    break

        if not alarm_triggered:
            break

    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    total_cost = float(min(1.0, ((1 + len(alarm_indices)) * W) / n_total))

    return {
        "method": "OS",
        "accuracy": acc,
        "kappa": kappa,
        "labeling_cost": total_cost,
        "num_alarms": len(alarm_indices),
        "alarm_indices": alarm_indices,
        "y_true_online": y_true_online,
        "y_pred_online": y_pred_online,
    }


# ==============================================================================
# 2. HÀM TÍNH ROLLING KAPPA & VẼ BIỂU ĐỒ CHUẨN BÀI BÁO (HÌNH 7 & HÌNH 10)
# ==============================================================================

def compute_rolling_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, rolling_window: int = 200, stride: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Tính hệ số Cohen's Kappa trên cửa sổ trượt 200 mẫu (đúng theo thiết lập Hình 7 trang 4366)."""
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


def plot_rolling_kappa_figure(
    res_studd: Dict[str, Any],
    res_ss: Dict[str, Any],
    res_static: Dict[str, Any],
    window_size: int,
    true_drift_points: List[int],
    save_path: str,
) -> None:
    """Vẽ biểu đồ tái hiện Hình 7 của bài báo: Rolling Kappa + Thời điểm báo động + Thống kê Page-Hinkley."""
    steps, kappa_studd = compute_rolling_kappa(
        res_studd["y_true_online"], res_studd["y_pred_online"]
    )
    _, kappa_ss = compute_rolling_kappa(
        res_ss["y_true_online"], res_ss["y_pred_online"]
    )
    _, kappa_static = compute_rolling_kappa(
        res_static["y_true_online"], res_static["y_pred_online"]
    )

    # Dịch chỉ số thời gian về trục thời gian toàn cục của luồng (cộng thêm W)
    global_steps = steps + window_size

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # --- Subplot 1: Đường cong Rolling Cohen's Kappa ---
    ax1.plot(global_steps, kappa_static, label="BL-st (Không thích nghi)", color="#9e9e9e", linestyle=":", linewidth=2)
    ax1.plot(global_steps, kappa_ss, label="SS (Có giám sát 100% nhãn)", color="#1f77b4", linestyle="--", linewidth=2)
    ax1.plot(global_steps, kappa_studd, label="STUDD (Đề xuất - Không nhãn)", color="#2ca02c", linewidth=2.5)

    # Vẽ vạch đứng tại điểm Drift thật và điểm STUDD phát báo động
    for idx, dp in enumerate(true_drift_points):
        ax1.axvline(dp, color="#d62728", linestyle="-", alpha=0.6, linewidth=1.8,
                    label="Điểm xảy ra Drift thật" if idx == 0 else None)

    for idx, alarm_t in enumerate(res_studd["alarm_indices"]):
        ax1.axvline(alarm_t, color="#2ca02c", linestyle="-.", alpha=0.85, linewidth=1.8,
                    label="Báo động của STUDD" if idx == 0 else None)

    ax1.set_ylabel("Cohen's Kappa (Cửa sổ trượt 200 mẫu)", fontsize=11, fontweight="bold")
    ax1.set_title("Tái hiện Hình 7: Hiệu năng thích nghi của STUDD so với SS và Baseline tĩnh", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", ncol=2, fontsize=9.5)

    # --- Subplot 2: Đại lượng thống kê Page-Hinkley của STUDD ---
    online_time = np.arange(window_size, window_size + len(res_studd["ph_statistics"]))
    ax2.plot(online_time, res_studd["ph_statistics"], color="#ff7f0e", linewidth=1.5, label="Thống kê Page-Hinkley (m_t - M_t)")
    for alarm_t in res_studd["alarm_indices"]:
        ax2.axvline(alarm_t, color="#2ca02c", linestyle="-.", alpha=0.85, linewidth=1.5)
    ax2.set_xlabel("Thời gian trên luồng dữ liệu (Time step t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Thống kê PH_t", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_summary_comparison_figure(df_summary: pd.DataFrame, save_path: str) -> None:
    """Vẽ biểu đồ cột kép so sánh Cohen's Kappa và Labeling Cost (Bám sát Bảng 3 & Bảng 4 bài báo)."""
    datasets = df_summary["Dataset"].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(13, 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        sub = df_summary[df_summary["Dataset"] == ds_name]
        x = np.arange(len(sub))
        width = 0.35

        bars1 = ax.bar(x - width / 2, sub["Our_Kappa"], width, label="Cohen's Kappa (Càng cao càng tốt)", color="#4c72b0")
        bars2 = ax.bar(x + width / 2, sub["Our_Cost"], width, label="Chi phí nhãn (Càng thấp càng tốt)", color="#dd8452")

        ax.set_title(f"Bộ dữ liệu: {ds_name.upper()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Method"], fontweight="bold")
        max_val = max(sub["Our_Kappa"].max(), sub["Our_Cost"].max(), 1.0)
        ax.set_ylim(0, max_val * 1.18)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

    axes[0].set_ylabel("Giá trị chỉ số (0.0 - 1.0)", fontsize=11, fontweight="bold")
    axes[-1].legend(loc="upper right", fontsize=9)
    plt.suptitle("Đối chiếu Hiệu năng (Kappa) và Chi phí Nhãn (Cost) giữa các Phương pháp", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


# ==============================================================================
# 3. CHƯƠNG TRÌNH CHẠY THỰC NGHIỆM CHÍNH & ĐỒNG BỘ LÊN WEIGHTS & BIASES
# ==============================================================================

# Số liệu công bố chính thức trong Bảng 3 (trang 4367) và Bảng 4 (trang 4368) của bài báo STUDD
PAPER_PUBLISHED_NUMBERS = {
    "abrupt_drift": {
        "BL-st": {"paper_kappa": 0.59, "paper_cost": 0.09},
        "STUDD": {"paper_kappa": 0.73, "paper_cost": 0.28},
        "SS":    {"paper_kappa": 0.78, "paper_cost": 0.19},
        "OS":    {"paper_kappa": 0.79, "paper_cost": 0.19},
    },
    "electricity": {
        "BL-st": {"paper_kappa": 0.13, "paper_cost": 0.02},
        "STUDD": {"paper_kappa": 0.29, "paper_cost": 0.43},
        "SS":    {"paper_kappa": 0.39, "paper_cost": 0.69},
        "OS":    {"paper_kappa": 0.33, "paper_cost": 0.09},
    },
    "ozone": {
        "BL-st": {"paper_kappa": 0.13, "paper_cost": 0.39},
        "STUDD": {"paper_kappa": 0.13, "paper_cost": 0.39},
        "SS":    {"paper_kappa": 0.14, "paper_cost": 0.39},
        "OS":    {"paper_kappa": 0.14, "paper_cost": 0.39},
    },
}


def run_all_reproductions(use_wandb: bool = True, entity: str = "lgkai-hcmus", project: str = "STUDD") -> None:
    """Chạy toàn bộ thực nghiệm tái hiện Giai đoạn 2 và đồng bộ lên Weights & Biases."""
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    wandb_run = None
    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            entity=entity,
            project=project,
            name="stage2-reproduction-suite",
            tags=["reproduction", "midterm-slide", "STUDD"],
            config={
                "paper": "Cerqueira et al. (2023) - STUDD",
                "delta": 0.001,
                "random_state": 42,
                "datasets": ["abrupt_drift", "electricity", "ozone"],
                "methods": ["BL-st", "STUDD", "SS", "OS"],
            },
        )

    summary_rows = []

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM 1: LUỒNG ABRUPT DRIFT (Tái hiện Hình 7 & Bảng 3/4 - AbruptInsects)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 1] TÁI HIỆN HÌNH 7 TRÊN LUỒNG ABRUPT CONCEPT DRIFT")
    print("=" * 70)
    X_ab, y_ab, meta_ab = generate_abrupt_drift_stream(
        n_samples=5500, n_features=8, drift_points=[1800, 3600], random_state=42
    )
    W_ab = 500

    res_ab_static = evaluate_static_baseline(X_ab, y_ab, window_size=W_ab, random_state=42)
    studd_ab = STUDD(window_size=W_ab, delta=0.001, threshold=25.0, random_state=42)
    res_ab_studd = studd_ab.evaluate_stream(X_ab, y_ab)
    # Chỉnh lại công thức tổng chi phí nhãn theo chuẩn Bảng 4: (1 + num_alarms) * W / N_total
    res_ab_studd["labeling_cost"] = float(min(1.0, ((1 + res_ab_studd["num_alarms"]) * W_ab) / len(X_ab)))
    res_ab_studd["method"] = "STUDD"

    res_ab_ss = evaluate_supervised_ss(X_ab, y_ab, window_size=W_ab, delta=0.001, threshold=25.0, random_state=42)
    res_ab_os = evaluate_unsupervised_os_ks(X_ab, y_ab, window_size=W_ab, ks_alpha=0.001, random_state=42)

    for r in [res_ab_static, res_ab_studd, res_ab_ss, res_ab_os]:
        m_name = r["method"]
        p_ref = PAPER_PUBLISHED_NUMBERS["abrupt_drift"][m_name]
        summary_rows.append({
            "Dataset": "abrupt_drift",
            "Method": m_name,
            "Our_Kappa": round(r["kappa"], 4),
            "Paper_Kappa": p_ref["paper_kappa"],
            "Kappa_Diff": round(r["kappa"] - p_ref["paper_kappa"], 4),
            "Our_Cost": round(r["labeling_cost"], 4),
            "Paper_Cost": p_ref["paper_cost"],
            "Alarms": r["num_alarms"],
        })
        print(f" -> {m_name:6s} | Kappa: {r['kappa']:.3f} (Bài báo: {p_ref['paper_kappa']:.2f}) | "
              f"Cost: {r['labeling_cost']:.2f} (Bài báo: {p_ref['paper_cost']:.2f}) | Alarms: {r['num_alarms']}")

    fig1_path = os.path.join(RESULTS_DIR, "fig1_rolling_kappa_abrupt_drift.png")
    plot_rolling_kappa_figure(
        res_studd=res_ab_studd,
        res_ss=res_ab_ss,
        res_static=res_ab_static,
        window_size=W_ab,
        true_drift_points=meta_ab["drift_points"],
        save_path=fig1_path,
    )
    print(f" [+] Đã xuất biểu đồ tái hiện Hình 7 tại: {fig1_path}")

    # Log chuỗi thời gian Rolling Kappa của Thí nghiệm 1 lên W&B
    if wandb_run is not None:
        import wandb
        steps, k_studd = compute_rolling_kappa(res_ab_studd["y_true_online"], res_ab_studd["y_pred_online"])
        _, k_ss = compute_rolling_kappa(res_ab_ss["y_true_online"], res_ab_ss["y_pred_online"])
        _, k_st = compute_rolling_kappa(res_ab_static["y_true_online"], res_ab_static["y_pred_online"])
        for s_idx, (t_val, ks_val, kss_val, kst_val) in enumerate(zip(steps + W_ab, k_studd, k_ss, k_st)):
            wandb.log({
                "stream_step": int(t_val),
                "AbruptDrift/Rolling_Kappa_STUDD": float(ks_val),
                "AbruptDrift/Rolling_Kappa_SS": float(kss_val),
                "AbruptDrift/Rolling_Kappa_Static": float(kst_val),
            })

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM 2: BỘ DỮ LIỆU THỰC ELECTRICITY (Tái hiện Bảng 3 & Bảng 4)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 2] THỰC NGHIỆM TRÊN BỘ DỮ LIỆU THỰC TẾ ELECTRICITY")
    print("=" * 70)
    X_el, y_el, _ = load_benchmark_stream("electricity", max_samples=15000)
    W_el = 2000

    res_el_static = evaluate_static_baseline(X_el, y_el, window_size=W_el, random_state=42)
    studd_el = STUDD(window_size=W_el, delta=0.001, threshold=30.0, random_state=42)
    res_el_studd = studd_el.evaluate_stream(X_el, y_el)
    res_el_studd["labeling_cost"] = float(((1 + res_el_studd["num_alarms"]) * W_el) / len(X_el))
    res_el_studd["method"] = "STUDD"

    res_el_ss = evaluate_supervised_ss(X_el, y_el, window_size=W_el, delta=0.001, threshold=30.0, random_state=42)
    res_el_os = evaluate_unsupervised_os_ks(X_el, y_el, window_size=W_el, ks_alpha=0.0001, step_check=200, random_state=42)

    for r in [res_el_static, res_el_studd, res_el_ss, res_el_os]:
        m_name = r["method"]
        p_ref = PAPER_PUBLISHED_NUMBERS["electricity"][m_name]
        summary_rows.append({
            "Dataset": "electricity",
            "Method": m_name,
            "Our_Kappa": round(r["kappa"], 4),
            "Paper_Kappa": p_ref["paper_kappa"],
            "Kappa_Diff": round(r["kappa"] - p_ref["paper_kappa"], 4),
            "Our_Cost": round(r["labeling_cost"], 4),
            "Paper_Cost": p_ref["paper_cost"],
            "Alarms": r["num_alarms"],
        })
        print(f" -> {m_name:6s} | Kappa: {r['kappa']:.3f} (Bài báo: {p_ref['paper_kappa']:.2f}) | "
              f"Cost: {r['labeling_cost']:.2f} (Bài báo: {p_ref['paper_cost']:.2f}) | Alarms: {r['num_alarms']}")

    # --------------------------------------------------------------------------
    # THÍ NGHIỆM 3: BỘ DỮ LIỆU THỰC OZONE (Tái hiện Bảng 3 & Bảng 4)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[THÍ NGHIỆM 3] THỰC NGHIỆM TRÊN BỘ DỮ LIỆU THỰC TẾ OZONE")
    print("=" * 70)
    X_oz, y_oz, _ = load_benchmark_stream("ozone", max_samples=None)
    W_oz = 500

    res_oz_static = evaluate_static_baseline(X_oz, y_oz, window_size=W_oz, random_state=42)
    studd_oz = STUDD(window_size=W_oz, delta=0.001, threshold=50.0, random_state=42)
    res_oz_studd = studd_oz.evaluate_stream(X_oz, y_oz)
    res_oz_studd["labeling_cost"] = float(((1 + res_oz_studd["num_alarms"]) * W_oz) / len(X_oz))
    res_oz_studd["method"] = "STUDD"

    res_oz_ss = evaluate_supervised_ss(X_oz, y_oz, window_size=W_oz, delta=0.001, threshold=50.0, random_state=42)
    res_oz_os = evaluate_unsupervised_os_ks(X_oz, y_oz, window_size=W_oz, ks_alpha=0.001, step_check=100, random_state=42)

    for r in [res_oz_static, res_oz_studd, res_oz_ss, res_oz_os]:
        m_name = r["method"]
        p_ref = PAPER_PUBLISHED_NUMBERS["ozone"][m_name]
        summary_rows.append({
            "Dataset": "ozone",
            "Method": m_name,
            "Our_Kappa": round(r["kappa"], 4),
            "Paper_Kappa": p_ref["paper_kappa"],
            "Kappa_Diff": round(r["kappa"] - p_ref["paper_kappa"], 4),
            "Our_Cost": round(r["labeling_cost"], 4),
            "Paper_Cost": p_ref["paper_cost"],
            "Alarms": r["num_alarms"],
        })
        print(f" -> {m_name:6s} | Kappa: {r['kappa']:.3f} (Bài báo: {p_ref['paper_kappa']:.2f}) | "
              f"Cost: {r['labeling_cost']:.2f} (Bài báo: {p_ref['paper_cost']:.2f}) | Alarms: {r['num_alarms']}")

    # --------------------------------------------------------------------------
    # LƯU BẢNG TỔNG HỢP CSV & BIỂU ĐỒ SO SÁNH TỔNG QUAN
    # --------------------------------------------------------------------------
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "reproduction_comparison_table.csv")
    df_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    fig2_path = os.path.join(RESULTS_DIR, "fig2_performance_vs_cost_comparison.png")
    plot_summary_comparison_figure(df_summary, save_path=fig2_path)

    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP KẾT QUẢ TÁI HIỆN (SO SÁNH VỚI BÀI BÁO GỐC)")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\n [+] Đã lưu bảng số liệu CSV tại : {csv_path}")
    print(f" [+] Đã lưu biểu đồ tổng hợp tại : {fig2_path}")

    # Đẩy bảng kết quả và hình ảnh biểu đồ lên W&B
    if wandb_run is not None:
        import wandb
        wandb.log({
            "Reproduction_Comparison_Table": wandb.Table(dataframe=df_summary),
            "Fig1_Rolling_Kappa_AbruptDrift": wandb.Image(fig1_path),
            "Fig2_Kappa_vs_Cost_Summary": wandb.Image(fig2_path),
        })
        print(f"\n [W&B] Đã đồng bộ toàn bộ bảng biểu và đồ thị lên: {wandb_run.url}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm tái hiện bài báo STUDD.")
    parser.add_argument("--no-wandb", action="store_true", help="Chạy ngoại tuyến không đẩy lên W&B.")
    args = parser.parse_args()

    run_all_reproductions(use_wandb=not args.no_wandb)
