"""Commit 6: Đối chiếu có kiểm soát giữa Paper-Text (Protocol A) và Author-Code Audit (Protocol B).

Mục tiêu: Kiểm chứng định lượng 3 nguyên nhân gốc rễ giải thích sai lệch số liệu với bài báo:
    1. **Protocol A vs Protocol B (W=2000 vs W=1000, SS delta=0.001 vs 0.002, cumulative Student buffer)**
       trên 8 bộ dữ liệu lệch mạnh: CoverType, Poker, Powersupply, Electricity, Sensors, Posture,
       Insects, InsectsAbrupt.
    2. **Seed Separation Ablation (42/42 vs 42/142)** trên InsectsAbrupt (RF/RF), Insects (XGB/XGB),
       Arabic (XGB/XGB), Poker (DT/DT) để định lượng bẫy Idempotent Mimicry.
    3. **Detector Bug Ablation (Legacy `sum - min_sum` vs Fixed `max(0, sum)`)** trên luồng dài
       (Sensors, CoverType, Posture, Electricity) để định lượng báo động dư do trôi thống kê âm.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fullscope_wandb import (
    RESULTS_DIR,
    SEED_STUDENT,
    SEED_TEACHER,
    log_fullscope_run,
)
from src.baselines import evaluate_supervised_pht
from src.data_loader import load_all_19_stream
from src.models import PAPER_TABLE2_MODELS, SafeStreamClassifier, build_model
from src.studd import STUDD

EIGHT_DIVERGENT_DATASETS: Tuple[str, ...] = (
    "CoverType",
    "Poker",
    "Powersupply",
    "Electricity",
    "Sensors",
    "Posture",
    "Insects",
    "InsectsAbrupt",
)

LEGACY_DETECTOR_DATASETS: Tuple[str, ...] = ("Sensors", "CoverType", "Posture", "Electricity")
MIMICRY_SEED_DATASETS: Tuple[str, ...] = ("InsectsAbrupt", "Insects", "Arabic", "Poker")


class LegacyDriftPageHinkley:
    """Bản tái lập lỗi Page-Hinkley cũ (không chặn dưới 0.0) dùng để ablation nguyên nhân sai lệch."""

    def __init__(self, delta: float = 0.001, threshold: float = 50.0, alpha: float = 0.9999, min_instances: int = 30) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.mean = 0.0
        self.cum_sum = 0.0
        self.min_sum = 0.0

    def update(self, x: float) -> Tuple[bool, float]:
        self.sample_count += 1
        self.mean += (x - self.mean) / self.sample_count
        self.cum_sum = self.alpha * self.cum_sum + (x - self.mean - self.delta)
        self.min_sum = min(self.min_sum, self.cum_sum)
        stat = self.cum_sum - self.min_sum
        alarm = self.sample_count >= self.min_instances and stat > self.threshold
        if alarm:
            self.reset()
        return alarm, float(stat)


def _make_pair(canon: str, *, t_seed: int, s_seed: int) -> Tuple[SafeStreamClassifier, SafeStreamClassifier, str, str]:
    key = canon.lower().replace("-", "").replace("_", "")
    t_code, s_code = PAPER_TABLE2_MODELS[key]
    teacher = (
        SafeStreamClassifier(SVC(random_state=t_seed), scale_features=True)
        if t_code.upper() in ("SVC", "SVM")
        else build_model(t_code, random_state=t_seed, n_jobs=2)
    )
    student = (
        SafeStreamClassifier(SVC(random_state=s_seed), scale_features=True)
        if s_code.upper() in ("SVC", "SVM")
        else build_model(s_code, random_state=s_seed, n_jobs=2)
    )
    return teacher, student, t_code, s_code


def _run_studd_cumulative_buffer(
    X: np.ndarray,
    y: np.ndarray,
    teacher_model: SafeStreamClassifier,
    student_model: SafeStreamClassifier,
    *,
    window_size: int,
    delta: float = 0.001,
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """Biến thể STUDD của Protocol B (author code): Student dùng buffer online tích lũy sau W."""
    studd = STUDD(
        teacher_model=clone(teacher_model),
        student_model=clone(student_model),
        window_size=window_size,
        delta=delta,
        threshold=threshold,
        random_state=SEED_TEACHER,
    )
    studd.fit_offline(X[:window_size], y[:window_size])
    n_total = len(X)
    y_true_online = y[window_size:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    chunk_size = max(window_size, 2000)
    curr_pos = window_size
    while curr_pos < n_total:
        scan_pos = curr_pos
        alarm_triggered = False
        while scan_pos < n_total:
            chunk_end = min(n_total, scan_pos + chunk_size)
            preds_t = studd._predict_teacher_batch(X[scan_pos:chunk_end])
            preds_s = studd._predict_student_batch(X[scan_pos:chunk_end])
            for offset in range(len(preds_t)):
                s_idx = scan_pos + offset
                o_idx = s_idx - window_size
                pt, ps = preds_t[offset], preds_s[offset]
                y_pred_online[o_idx] = pt
                loss_t = 1.0 if pt != ps else 0.0
                alarm, _ = studd.detector.update(loss_t)
                if alarm:
                    alarm_indices.append(s_idx)
                    # Teacher học từ cửa sổ W mới nhất; Student học tích lũy từ đầu giai đoạn online (tính chất Protocol B)
                    start_t = max(0, s_idx + 1 - window_size)
                    studd.teacher_model = clone(teacher_model).fit(X[start_t : s_idx + 1], y[start_t : s_idx + 1])
                    # Buffer tích lũy (giới hạn 15,000 mẫu gần nhất cho SVC/LR tránh O(N^2) cực đoan nhưng vẫn giữ ngữ cảnh đa concept)
                    buf_start = max(window_size, s_idx + 1 - max(window_size * 5, 10000))
                    X_buf = np.vstack([X[:window_size], X[buf_start : s_idx + 1]])
                    yhat_buf = studd._predict_teacher_batch(X_buf)
                    if len(np.unique(yhat_buf)) >= 2:
                        studd._single_student_class_fallback = None
                        studd.student_model = clone(student_model).fit(X_buf, yhat_buf)
                    else:
                        studd._single_student_class_fallback = yhat_buf[0]
                    studd.detector.reset()
                    curr_pos = s_idx + 1
                    alarm_triggered = True
                    break
            if alarm_triggered:
                break
            scan_pos = chunk_end
        if not alarm_triggered:
            break

    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    if np.isnan(kappa):
        kappa = 0.0
    cost = float(((1 + len(alarm_indices)) * window_size) / n_total)
    return {"kappa": kappa, "labeling_cost": cost, "num_alarms": len(alarm_indices), "alarm_indices": alarm_indices}


def _run_studd_legacy_ph(
    X: np.ndarray,
    y: np.ndarray,
    teacher_model: SafeStreamClassifier,
    student_model: SafeStreamClassifier,
    *,
    window_size: int,
) -> Dict[str, Any]:
    """Chạy STUDD với detector cũ (`sum - min_sum`) để định lượng nguyên nhân báo động dư."""
    studd = STUDD(
        teacher_model=teacher_model,
        student_model=student_model,
        window_size=window_size,
        delta=0.001,
        threshold=50.0,
        random_state=SEED_TEACHER,
    )
    studd.detector = LegacyDriftPageHinkley(delta=0.001, threshold=50.0)
    return studd.evaluate_stream(X, y)


def run_protocol_comparison(summary_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    archived = pd.read_csv(summary_csv)

    proto_rows: List[Dict[str, object]] = []
    for ds_name in EIGHT_DIVERGENT_DATASETS:
        X, y, meta = load_all_19_stream(ds_name)
        canon = str(meta["canonical_name"])
        W_text = int(meta["recommended_window_size"])
        W_code = 1000 if (len(X) >= 10_000 or canon in ("Insects", "InsectsAbrupt")) else 500
        # Theo source code tác giả (main.py), các stream lớn dùng n_train_obs = 1000; 2 bộ Insects nhỏ < 10k nhưng Paper Table 4 có W=500 -> Insects 500
        if canon in ("Insects", "InsectsAbrupt"):
            W_code = 500

        teacher, student, t_code, s_code = _make_pair(canon, t_seed=SEED_TEACHER, s_seed=SEED_STUDENT)

        # Protocol A: Paper-Text Canonical (W_text, PH max(0), SS delta=0.001, sliding W, T=42/S=142)
        res_a_studd = STUDD(teacher_model=teacher, student_model=student, window_size=W_text, delta=0.001, threshold=50.0, random_state=SEED_TEACHER).evaluate_stream(X, y)
        res_a_ss = evaluate_supervised_pht(X, y, base_model=teacher, window_size=W_text, delta=0.001, threshold=50.0, l_access=1.0, l_delay=0, method_name="SS", random_state=SEED_TEACHER)

        # Protocol B: Author-Code Audit (W_code, PH max(0), SS delta=0.002, cumulative student buffer, T=42/S=142)
        res_b_studd = _run_studd_cumulative_buffer(X, y, teacher, student, window_size=W_code, delta=0.001, threshold=50.0)
        res_b_ss = evaluate_supervised_pht(X, y, base_model=teacher, window_size=W_code, delta=0.002, threshold=50.0, l_access=1.0, l_delay=0, method_name="SS", random_state=SEED_TEACHER)

        arch_studd = archived[(archived["Dataset"] == canon) & (archived["Method"] == "STUDD")].iloc[0]
        arch_ss = archived[(archived["Dataset"] == canon) & (archived["Method"] == "SS")].iloc[0]

        proto_rows.append(
            {
                "Dataset": canon,
                "Teacher": t_code,
                "Student": s_code,
                "W_ProtoA_Text": W_text,
                "W_ProtoB_Code": W_code,
                "Paper_STUDD_Kappa": float(arch_studd["Paper_Kappa"]),
                "Paper_STUDD_Cost": float(arch_studd["Paper_Cost"]),
                "Archived_STUDD_Kappa": round(float(arch_studd["Our_Kappa"]), 4),
                "Archived_STUDD_Cost": round(float(arch_studd["Our_Cost"]), 4),
                "Archived_STUDD_Alarms": int(arch_studd["Alarms"]),
                "ProtoA_STUDD_Kappa": round(float(res_a_studd["kappa"]), 4),
                "ProtoA_STUDD_Cost": round(float(res_a_studd["labeling_cost"]), 4),
                "ProtoA_STUDD_Alarms": int(res_a_studd["num_alarms"]),
                "ProtoB_STUDD_Kappa": round(float(res_b_studd["kappa"]), 4),
                "ProtoB_STUDD_Cost": round(float(res_b_studd["labeling_cost"]), 4),
                "ProtoB_STUDD_Alarms": int(res_b_studd["num_alarms"]),
                "Paper_SS_Kappa": float(arch_ss["Paper_Kappa"]),
                "ProtoA_SS_Kappa": round(float(res_a_ss["kappa"]), 4),
                "ProtoA_SS_Alarms": int(res_a_ss["num_alarms"]),
                "ProtoB_SS_Kappa": round(float(res_b_ss["kappa"]), 4),
                "ProtoB_SS_Alarms": int(res_b_ss["num_alarms"]),
            }
        )
        print(
            f"[Proto A/B] {canon:13s} | Paper={arch_studd['Paper_Kappa']:.2f}/{arch_studd['Paper_Cost']:.2f} | "
            f"Arch={arch_studd['Our_Kappa']:.3f}/{arch_studd['Our_Cost']:.2f} | "
            f"A={res_a_studd['kappa']:.3f}/{res_a_studd['labeling_cost']:.2f}({res_a_studd['num_alarms']}) | "
            f"B={res_b_studd['kappa']:.3f}/{res_b_studd['labeling_cost']:.2f}({res_b_studd['num_alarms']})",
            flush=True,
        )

    # Ablation nguyên nhân gốc rễ: Seed (42/42 vs 42/142) & Detector (Legacy vs Fixed)
    ablation_rows: List[Dict[str, object]] = []
    for ds_name in MIMICRY_SEED_DATASETS:
        X, y, meta = load_all_19_stream(ds_name)
        canon = str(meta["canonical_name"])
        W = int(meta["recommended_window_size"])
        t42, s42, t_code, s_code = _make_pair(canon, t_seed=42, s_seed=42)
        t42b, s142, _, _ = _make_pair(canon, t_seed=42, s_seed=142)

        res_same = STUDD(teacher_model=t42, student_model=s42, window_size=W, delta=0.001, threshold=50.0, random_state=42).evaluate_stream(X, y)
        res_split = STUDD(teacher_model=t42b, student_model=s142, window_size=W, delta=0.001, threshold=50.0, random_state=42).evaluate_stream(X, y)
        ablation_rows.append(
            {
                "Ablation_Type": "Seed_Separation_42_vs_142",
                "Dataset": canon,
                "Pair": f"{t_code}/{s_code}",
                "W": W,
                "Baseline_Variant": "Seed_42_42",
                "Baseline_Kappa": round(float(res_same["kappa"]), 4),
                "Baseline_Cost": round(float(res_same["labeling_cost"]), 4),
                "Baseline_Alarms": int(res_same["num_alarms"]),
                "Treatment_Variant": "Seed_42_142",
                "Treatment_Kappa": round(float(res_split["kappa"]), 4),
                "Treatment_Cost": round(float(res_split["labeling_cost"]), 4),
                "Treatment_Alarms": int(res_split["num_alarms"]),
                "Kappa_Delta": round(float(res_split["kappa"] - res_same["kappa"]), 4),
                "Alarms_Delta": int(res_split["num_alarms"] - res_same["num_alarms"]),
            }
        )

    for ds_name in LEGACY_DETECTOR_DATASETS:
        X, y, meta = load_all_19_stream(ds_name)
        canon = str(meta["canonical_name"])
        W = int(meta["recommended_window_size"])
        t_mod, s_mod, t_code, s_code = _make_pair(canon, t_seed=42, s_seed=142)
        res_leg = _run_studd_legacy_ph(X, y, t_mod, s_mod, window_size=W)
        res_fix = STUDD(teacher_model=t_mod, student_model=s_mod, window_size=W, delta=0.001, threshold=50.0, random_state=42).evaluate_stream(X, y)
        ablation_rows.append(
            {
                "Ablation_Type": "Detector_Legacy_vs_Fixed",
                "Dataset": canon,
                "Pair": f"{t_code}/{s_code}",
                "W": W,
                "Baseline_Variant": "Legacy_PH_Unclamped",
                "Baseline_Kappa": round(float(res_leg["kappa"]), 4),
                "Baseline_Cost": round(float(res_leg["labeling_cost"]), 4),
                "Baseline_Alarms": int(res_leg["num_alarms"]),
                "Treatment_Variant": "Fixed_PH_MaxZero",
                "Treatment_Kappa": round(float(res_fix["kappa"]), 4),
                "Treatment_Cost": round(float(res_fix["labeling_cost"]), 4),
                "Treatment_Alarms": int(res_fix["num_alarms"]),
                "Kappa_Delta": round(float(res_fix["kappa"] - res_leg["kappa"]), 4),
                "Alarms_Delta": int(res_fix["num_alarms"] - res_leg["num_alarms"]),
            }
        )

    return pd.DataFrame(proto_rows), pd.DataFrame(ablation_rows)


def plot_ablation_figure(proto_df: pd.DataFrame, ab_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8), dpi=160)

    x = np.arange(len(proto_df))
    w = 0.25
    ax1.bar(x - w, proto_df["Paper_STUDD_Kappa"], width=w, color="#16a34a", label="Paper (Table 3)")
    ax1.bar(x, proto_df["ProtoA_STUDD_Kappa"], width=w, color="#2563eb", label="Protocol A (Paper-Text Fixed, T=42/S=142)")
    ax1.bar(x + w, proto_df["ProtoB_STUDD_Kappa"], width=w, color="#f59e0b", label="Protocol B (Author-Code Audit: W=1000/SS 0.002)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(proto_df["Dataset"], rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("STUDD Cohen's Kappa", fontsize=9.5, fontweight="bold")
    ax1.set_title("Protocol A (Paper-Text) vs Protocol B (Author-Code Audit) Across 8 Datasets", fontsize=10.5, fontweight="bold")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.0)

    # Panel 2: Cost / Alarm impact of Seed Separation and Fixed PH
    labels = [f"{r.Dataset}\n({'Seed' if 'Seed' in r.Ablation_Type else 'PH'})" for r in ab_df.itertuples(index=False)]
    y2 = np.arange(len(ab_df))
    base_al = ab_df["Baseline_Alarms"].to_numpy()
    trt_al = ab_df["Treatment_Alarms"].to_numpy()
    ax2.barh(y2 - 0.18, base_al, height=0.35, color="#94a3b8", label="Before Fix (Seed 42/42 or Legacy PH)")
    ax2.barh(y2 + 0.18, trt_al, height=0.35, color="#dc2626", label="After Fix (Seed 42/142 or Fixed PH max(0))")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("STUDD Drift Alarms Count", fontsize=9.5, fontweight="bold")
    ax2.set_title("Root-Cause Isolation: Seed Trap & Unclamped PH Alarms", fontsize=10.5, fontweight="bold")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.0)

    fig.suptitle(
        "Controlled Protocol Comparison & Deviation Root-Cause Ablations",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wandb", action="store_true", help="Tắt đồng bộ W&B.")
    args = parser.parse_args()

    summary_csv = RESULTS_DIR / "studd_19datasets_summary.csv"
    proto_df, ab_df = run_protocol_comparison(summary_csv)

    proto_csv = RESULTS_DIR / "original_code_comparison_table.csv"
    ab_csv = RESULTS_DIR / "seed_and_detector_ablation.csv"
    fig_path = RESULTS_DIR / "fig_protocol_deviation_ablation.png"

    proto_df.to_csv(proto_csv, index=False)
    ab_df.to_csv(ab_csv, index=False)
    plot_ablation_figure(proto_df, ab_df, fig_path)

    print("\n=== SEED & DETECTOR ABLATION SUMMARY ===")
    print(ab_df[["Ablation_Type", "Dataset", "Baseline_Alarms", "Treatment_Alarms", "Baseline_Kappa", "Treatment_Kappa", "Baseline_Cost", "Treatment_Cost"]].to_string(index=False))

    mean_a_k = float(proto_df["ProtoA_STUDD_Kappa"].mean())
    mean_b_k = float(proto_df["ProtoB_STUDD_Kappa"].mean())
    mean_p_k = float(proto_df["Paper_STUDD_Kappa"].mean())
    log_fullscope_run(
        run_key="protocol_ablation",
        run_name="fullscope-protocol-ablation",
        job_type="protocol_comparison_ablation",
        config={
            "datasets": list(EIGHT_DIVERGENT_DATASETS),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        summary_metrics={
            "protoA_mean_kappa": round(mean_a_k, 4),
            "protoB_mean_kappa": round(mean_b_k, 4),
            "paper_mean_kappa": round(mean_p_k, 4),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        tables={
            "original_code_comparison_table": proto_df,
            "seed_and_detector_ablation": ab_df,
        },
        images={"fig_protocol_deviation_ablation": fig_path},
        enabled=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
