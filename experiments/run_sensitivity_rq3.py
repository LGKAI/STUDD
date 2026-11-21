"""Commit 5: Mục 5.8.5 Sensitivity Analysis (`l_access × l_delay`, Hình 14, RQ3) trên 19 bộ dữ liệu.

Khảo sát độ nhạy của bộ phát hiện giám sát theo:
    - `l_access` in {1%, 10%, 25%, 50%} (tỷ lệ truy cập nhãn thật trực tuyến)
    - `l_delay` in {250, 500, 1000, 1500, 2000, 4000} (độ trễ nhãn theo số mẫu)
Đối chiếu trực tiếp với `SS (100%, delay=0)` và `STUDD (0% nhãn khi giám sát, seed T=42/S=142)`
trên toàn bộ 19 bộ dữ liệu để trả lời RQ3: *"How does STUDD compare to supervised methods
across varying levels of label availability and delay?"*
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.svm import SVC
from experiments.analyze_19datasets_full import bayesian_sign_test_dirichlet
from experiments.fullscope_wandb import (
    RESULTS_DIR,
    SEED_RNG,
    SEED_STUDENT,
    SEED_TEACHER,
    log_fullscope_run,
)
from src.baselines import _fit_safe, _predict_safe
from src.data_loader import ALL_19_DATASETS, load_all_19_stream
from src.detector import PageHinkley
from src.models import PAPER_TABLE2_MODELS, SafeStreamClassifier, build_model
from src.studd import STUDD

L_ACCESS_GRID: Tuple[float, ...] = (0.01, 0.10, 0.25, 0.50)
L_DELAY_GRID: Tuple[int, ...] = (250, 500, 1000, 1500, 2000, 4000)
DELTA = 0.001
PH_THRESHOLD = 50.0


def _build_models(canon: str) -> Tuple[SafeStreamClassifier, SafeStreamClassifier, str, str]:
    key = canon.lower().replace("-", "").replace("_", "")
    t_code, s_code = PAPER_TABLE2_MODELS[key]
    if t_code.upper() in ("SVC", "SVM"):
        teacher = SafeStreamClassifier(SVC(random_state=SEED_TEACHER), scale_features=True)
    else:
        teacher = build_model(t_code, random_state=SEED_TEACHER, n_jobs=2)
    if s_code.upper() in ("SVC", "SVM"):
        student = SafeStreamClassifier(SVC(random_state=SEED_STUDENT), scale_features=True)
    else:
        student = build_model(s_code, random_state=SEED_STUDENT, n_jobs=2)
    return teacher, student, t_code, s_code


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = float(cohen_kappa_score(y_true, y_pred))
    return 0.0 if np.isnan(k) else k


def evaluate_delayed_sparse_supervised_pht(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    teacher_template: SafeStreamClassifier,
    *,
    window_size: int,
    l_access: float,
    l_delay: int,
    label_mask: Optional[np.ndarray] = None,
    init_online_preds: Optional[np.ndarray] = None,
    delta: float = DELTA,
    threshold: float = PH_THRESHOLD,
    random_state: int = SEED_TEACHER,
) -> Dict[str, Any]:
    """Đánh giá giám sát PH dưới chế độ nhãn thưa (`l_access`) & nhãn trễ (`l_delay`) đúng Mục 5.8.5.

    - Mặt nạ `label_mask[stream_idx]` quyết định mẫu trực tuyến `stream_idx >= W` có được gán nhãn không.
    - Nhãn của mẫu `obs_idx = stream_idx - l_delay` chỉ tới detector tại bước `stream_idx`.
    - Khi có báo động tại `stream_idx`, Teacher được huấn luyện lại từ tối đa `W` mẫu gần nhất
      đã nhận xong nhãn tính đến `stream_idx` (`[:W]` ban đầu + các mẫu `j <= stream_idx - l_delay`
      có `label_mask[j] == True`).
    """
    n_total = len(X_stream)
    W = window_size
    if label_mask is None:
        rng = np.random.default_rng(random_state)
        label_mask = np.ones(n_total, dtype=bool)
        if l_access < 1.0:
            label_mask[W:] = rng.random(n_total - W) < l_access

    if init_online_preds is None:
        model, single_cls = _fit_safe(teacher_template, X_stream[:W], y_stream[:W])
        init_online_preds = _predict_safe(model, single_cls, X_stream[W:])
    else:
        model, single_cls = None, None

    y_true_online = y_stream[W:]
    y_pred_online = np.array(init_online_preds, copy=True)
    detector = PageHinkley(delta=delta, threshold=threshold)
    alarm_indices: List[int] = []

    chunk_size = max(W, 2000)
    curr_pos = W
    last_reset_online_idx = 0
    using_init = True

    while curr_pos < n_total:
        scan_pos = curr_pos
        alarm_triggered = False

        while scan_pos < n_total:
            chunk_end = min(n_total, scan_pos + chunk_size)
            if using_init:
                preds_chunk = init_online_preds[scan_pos - W : chunk_end - W]
            else:
                preds_chunk = _predict_safe(model, single_cls, X_stream[scan_pos:chunk_end])
                y_pred_online[scan_pos - W : chunk_end - W] = preds_chunk

            for offset in range(len(preds_chunk)):
                stream_idx = scan_pos + offset
                online_idx = stream_idx - W
                obs_online_idx = online_idx - l_delay
                obs_stream_idx = stream_idx - l_delay

                if obs_online_idx >= last_reset_online_idx and label_mask[obs_stream_idx]:
                    err = 1.0 if y_pred_online[obs_online_idx] != y_true_online[obs_online_idx] else 0.0
                    drift_alarm, _ = detector.update(err)
                    if drift_alarm:
                        alarm_indices.append(stream_idx)
                        cutoff_arrived = max(W - 1, stream_idx - l_delay)
                        # Thu thập các chỉ số đã có nhãn tính đến thời điểm báo động
                        online_known = np.flatnonzero(label_mask[W : cutoff_arrived + 1]) + W
                        if len(online_known) >= W:
                            adapt_idx = online_known[-W:]
                        elif len(online_known) > 0:
                            need_init = W - len(online_known)
                            init_idx = np.arange(max(0, W - need_init), W)
                            adapt_idx = np.concatenate([init_idx, online_known])
                        else:
                            adapt_idx = np.arange(0, W)

                        model, single_cls = _fit_safe(
                            teacher_template, X_stream[adapt_idx], y_stream[adapt_idx]
                        )
                        using_init = False
                        detector.reset()
                        last_reset_online_idx = online_idx + 1
                        curr_pos = stream_idx + 1
                        alarm_triggered = True
                        break

            if alarm_triggered:
                break
            scan_pos = chunk_end

        if not alarm_triggered:
            break

    n_online_labels = int(np.sum(label_mask[W:]))
    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = _safe_kappa(y_true_online, y_pred_online)
    return {
        "kappa": kappa,
        "accuracy": acc,
        "num_alarms": len(alarm_indices),
        "alarm_indices": alarm_indices,
        "online_labeled_count": n_online_labels,
        "total_labeling_cost_actual": float((W + n_online_labels) / n_total),
    }


def run_dataset_sensitivity(dataset_name: str) -> List[Dict[str, object]]:
    """Chạy đầy đủ lưới 24 cấu hình (l_access x l_delay) + SS + STUDD cho 1 bộ dữ liệu."""
    t0 = time.perf_counter()
    X, y, meta = load_all_19_stream(dataset_name)
    canon = str(meta["canonical_name"])
    W = int(meta["recommended_window_size"])
    teacher, student, t_code, s_code = _build_models(canon)

    # Tiền tính dự đoán mô hình khởi tạo (dùng chung cho tất cả các cấu hình trước báo động 1)
    init_model, init_single = _fit_safe(teacher, X[:W], y[:W])
    init_online_preds = _predict_safe(init_model, init_single, X[W:])
    y_true_online = y[W:]

    bl_kappa = _safe_kappa(y_true_online, init_online_preds)
    res_studd = STUDD(
        teacher_model=teacher,
        student_model=student,
        window_size=W,
        delta=DELTA,
        threshold=PH_THRESHOLD,
        random_state=SEED_TEACHER,
    ).evaluate_stream(X, y)
    res_ss = evaluate_delayed_sparse_supervised_pht(
        X,
        y,
        teacher,
        window_size=W,
        l_access=1.0,
        l_delay=0,
        init_online_preds=init_online_preds,
        random_state=SEED_TEACHER,
    )

    rows: List[Dict[str, object]] = [
        {
            "Dataset": canon,
            "Teacher": t_code,
            "Student": s_code,
            "W": W,
            "N": len(X),
            "Config": "STUDD",
            "Kind": "STUDD",
            "l_access": 0.0,
            "l_access_pct": 0,
            "l_delay": 0,
            "Kappa": round(float(res_studd["kappa"]), 4),
            "Accuracy": round(float(res_studd["accuracy"]), 4),
            "Alarms": int(res_studd["num_alarms"]),
            "Effective_Label_Cost": round(float(res_studd["labeling_cost"]), 4),
            "STUDD_Kappa_Ref": round(float(res_studd["kappa"]), 4),
            "STUDD_Minus_Config": 0.0,
            "BL_st_Kappa": round(bl_kappa, 4),
        },
        {
            "Dataset": canon,
            "Teacher": t_code,
            "Student": s_code,
            "W": W,
            "N": len(X),
            "Config": "SS (100%, d=0)",
            "Kind": "SS_Ideal",
            "l_access": 1.0,
            "l_access_pct": 100,
            "l_delay": 0,
            "Kappa": round(float(res_ss["kappa"]), 4),
            "Accuracy": round(float(res_ss["accuracy"]), 4),
            "Alarms": int(res_ss["num_alarms"]),
            "Effective_Label_Cost": round(float(res_ss["total_labeling_cost_actual"]), 4),
            "STUDD_Kappa_Ref": round(float(res_studd["kappa"]), 4),
            "STUDD_Minus_Config": round(float(res_studd["kappa"]) - float(res_ss["kappa"]), 4),
            "BL_st_Kappa": round(bl_kappa, 4),
        },
    ]

    # Tạo mặt nạ Bernoulli cố định theo từng mức l_access (SEED_RNG=42) để so sánh ghép cặp công bằng across delays
    n_total = len(X)
    masks: Dict[float, np.ndarray] = {}
    for l_acc in L_ACCESS_GRID:
        rng = np.random.default_rng(SEED_RNG + int(round(l_acc * 1000)))
        mask = np.ones(n_total, dtype=bool)
        mask[W:] = rng.random(n_total - W) < l_acc
        masks[l_acc] = mask

    for l_acc in L_ACCESS_GRID:
        pct = int(round(l_acc * 100))
        for l_del in L_DELAY_GRID:
            res_cfg = evaluate_delayed_sparse_supervised_pht(
                X,
                y,
                teacher,
                window_size=W,
                l_access=l_acc,
                l_delay=l_del,
                label_mask=masks[l_acc],
                init_online_preds=init_online_preds,
                random_state=SEED_TEACHER,
            )
            k_val = float(res_cfg["kappa"])
            diff = float(res_studd["kappa"]) - k_val
            rows.append(
                {
                    "Dataset": canon,
                    "Teacher": t_code,
                    "Student": s_code,
                    "W": W,
                    "N": n_total,
                    "Config": f"S_{pct}%_d{l_del}",
                    "Kind": "Grid",
                    "l_access": l_acc,
                    "l_access_pct": pct,
                    "l_delay": l_del,
                    "Kappa": round(k_val, 4),
                    "Accuracy": round(float(res_cfg["accuracy"]), 4),
                    "Alarms": int(res_cfg["num_alarms"]),
                    "Effective_Label_Cost": round(float(res_cfg["total_labeling_cost_actual"]), 4),
                    "STUDD_Kappa_Ref": round(float(res_studd["kappa"]), 4),
                    "STUDD_Minus_Config": round(diff, 4),
                    "BL_st_Kappa": round(bl_kappa, 4),
                }
            )

    dt_sec = time.perf_counter() - t0
    print(
        f"[OK] {canon:14s} ({dt_sec:5.1f}s) | STUDD={res_studd['kappa']:.4f} | "
        f"SS={res_ss['kappa']:.4f} | Grid_Mean={np.mean([r['Kappa'] for r in rows if r['Kind']=='Grid']):.4f}",
        flush=True,
    )
    return rows


def aggregate_sensitivity_summary(per_ds_df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp Mean Kappa, Mean Advantage vs STUDD và Bayesian P(STUDD >= S) trên 19 bộ."""
    grid_rows = per_ds_df[per_ds_df["Kind"] == "Grid"].copy()
    studd_map = (
        per_ds_df[per_ds_df["Kind"] == "STUDD"]
        .set_index("Dataset")["Kappa"]
        .to_dict()
    )

    summary_rows: List[Dict[str, object]] = []
    # Thêm dòng tham chiếu STUDD và SS
    for kind_name in ("STUDD", "SS_Ideal"):
        sub = per_ds_df[per_ds_df["Kind"] == kind_name]
        summary_rows.append(
            {
                "Config": sub["Config"].iloc[0],
                "Kind": kind_name,
                "l_access_pct": int(sub["l_access_pct"].iloc[0]),
                "l_delay": int(sub["l_delay"].iloc[0]),
                "Datasets": len(sub),
                "Mean_Kappa": round(float(sub["Kappa"].mean()), 4),
                "Median_Kappa": round(float(sub["Kappa"].median()), 4),
                "Mean_Diff_STUDD_Minus_Cfg": round(float(sub["STUDD_Minus_Config"].mean()), 4),
                "Mean_Alarms": round(float(sub["Alarms"].mean()), 2),
                "Mean_Label_Cost": round(float(sub["Effective_Label_Cost"].mean()), 4),
                "STUDD_Win_Count": 0,
                "STUDD_ROPE_Count": len(sub) if kind_name == "STUDD" else 0,
                "STUDD_Lose_Count": 0,
                "Bayes_P_STUDD_Win": np.nan,
                "Bayes_P_STUDD_Win_Or_ROPE": np.nan,
            }
        )

    for l_acc in L_ACCESS_GRID:
        pct = int(round(l_acc * 100))
        for l_del in L_DELAY_GRID:
            sub = grid_rows[(grid_rows["l_access_pct"] == pct) & (grid_rows["l_delay"] == l_del)].sort_values("Dataset")
            cfg_kappas = sub["Kappa"].to_numpy(dtype=np.float64)
            studd_kappas = np.array([studd_map[ds] for ds in sub["Dataset"]], dtype=np.float64)
            bstats = bayesian_sign_test_dirichlet(
                studd_kappas, cfg_kappas, higher_is_better=True, rope=0.05, seed=SEED_RNG
            )
            summary_rows.append(
                {
                    "Config": f"S_{pct}%_d{l_del}",
                    "Kind": "Grid",
                    "l_access_pct": pct,
                    "l_delay": l_del,
                    "Datasets": len(sub),
                    "Mean_Kappa": round(float(np.mean(cfg_kappas)), 4),
                    "Median_Kappa": round(float(np.median(cfg_kappas)), 4),
                    "Mean_Diff_STUDD_Minus_Cfg": round(float(np.mean(studd_kappas - cfg_kappas)), 4),
                    "Mean_Alarms": round(float(sub["Alarms"].mean()), 2),
                    "Mean_Label_Cost": round(float(sub["Effective_Label_Cost"].mean()), 4),
                    "STUDD_Win_Count": int(bstats["n_win"]),
                    "STUDD_ROPE_Count": int(bstats["n_rope"]),
                    "STUDD_Lose_Count": int(bstats["n_lose"]),
                    "Bayes_P_STUDD_Win": round(float(bstats["Simplex_P_Win"]), 4),
                    "Bayes_P_STUDD_Win_Or_ROPE": round(
                        float(bstats["Simplex_P_Win"] + bstats["Simplex_P_ROPE"]), 4
                    ),
                }
            )
    return pd.DataFrame(summary_rows)


def plot_sensitivity_fig14(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ Fig. 14 (RQ3): Độ nhạy Mean Kappa và Bayesian P(STUDD >= Supervised) theo l_access × l_delay."""
    grid = summary_df[summary_df["Kind"] == "Grid"].copy()
    studd_k = float(summary_df[summary_df["Kind"] == "STUDD"]["Mean_Kappa"].iloc[0])
    ss_k = float(summary_df[summary_df["Kind"] == "SS_Ideal"]["Mean_Kappa"].iloc[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.5), dpi=160)

    colors = {1: "#94a3b8", 10: "#f59e0b", 25: "#3b82f6", 50: "#1d4ed8"}
    for pct in (1, 10, 25, 50):
        sub = grid[grid["l_access_pct"] == pct].sort_values("l_delay")
        ax1.plot(
            sub["l_delay"],
            sub["Mean_Kappa"],
            marker="o",
            linewidth=2.0,
            color=colors[pct],
            label=f"Supervised l_access={pct}%",
        )
    ax1.axhline(studd_k, color="#dc2626", linestyle="--", linewidth=2.2, label=f"STUDD unsupervised ({studd_k:.4f})")
    ax1.axhline(ss_k, color="#16a34a", linestyle=":", linewidth=1.8, label=f"SS ideal 100%, d=0 ({ss_k:.4f})")
    ax1.set_xticks(list(L_DELAY_GRID))
    ax1.set_xlabel("Label Delay l_delay (instances)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Mean Cohen's Kappa Across 19 Datasets", fontsize=9.5, fontweight="bold")
    ax1.set_title("Fig. 14A · Sensitivity Curve: Supervised Kappa vs STUDD", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=8.2)

    # Heatmap chênh lệch STUDD - Supervised
    piv_diff = grid.pivot(index="l_access_pct", columns="l_delay", values="Mean_Diff_STUDD_Minus_Cfg")
    piv_prob = grid.pivot(index="l_access_pct", columns="l_delay", values="Bayes_P_STUDD_Win_Or_ROPE")
    mat = piv_diff.to_numpy(dtype=float)
    vmax = max(0.05, float(np.max(np.abs(mat))))
    im = ax2.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax2.set_xticks(np.arange(len(L_DELAY_GRID)))
    ax2.set_xticklabels([str(d) for d in L_DELAY_GRID], fontsize=9)
    ax2.set_yticks(np.arange(4))
    ax2.set_yticklabels([f"{p}%" for p in piv_diff.index], fontsize=9.5, fontweight="bold")
    for i, pct in enumerate(piv_diff.index):
        for j, dly in enumerate(piv_diff.columns):
            dv = float(piv_diff.loc[pct, dly])
            pv = float(piv_prob.loc[pct, dly])
            ax2.text(
                j,
                i,
                f"{dv:+.4f}\n(P>={pv:.2f})",
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold" if dv > 0 else "normal",
                color="#111827",
            )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.03)
    cbar.set_label("Mean Kappa Advantage (STUDD - S)", fontsize=8.5)
    ax2.set_xlabel("Label Delay l_delay (instances)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Online Label Access l_access", fontsize=9.5, fontweight="bold")
    ax2.set_title("Fig. 14B · Delta Kappa (STUDD - S) & P(STUDD >= S in Bayes)", fontsize=10.5, fontweight="bold")

    fig.suptitle(
        f"Section 5.8.5 / RQ3 Sensitivity Analysis Across 19 Datasets (Seed T={SEED_TEACHER}/S={SEED_STUDENT})",
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
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint per-dataset đã lưu.")
    args = parser.parse_args()

    per_ds_csv = RESULTS_DIR / "sensitivity_rq3_per_dataset.csv"
    summary_csv = RESULTS_DIR / "sensitivity_rq3_summary.csv"
    fig_path = RESULTS_DIR / "fig14_sensitivity_rq3.png"

    all_rows: List[Dict[str, object]] = []
    done_datasets: set = set()
    if args.resume and per_ds_csv.exists():
        existing_df = pd.read_csv(per_ds_csv)
        all_rows = existing_df.to_dict("records")
        done_datasets = set(existing_df["Dataset"].unique())

    for ds_name in ALL_19_DATASETS:
        _, _, meta = load_all_19_stream(ds_name)
        canon = str(meta["canonical_name"])
        if canon in done_datasets:
            print(f"[SKIP] {canon} đã có trong checkpoint.")
            continue
        ds_rows = run_dataset_sensitivity(ds_name)
        all_rows.extend(ds_rows)
        pd.DataFrame(all_rows).to_csv(per_ds_csv, index=False)

    per_ds_df = pd.DataFrame(all_rows)
    summary_df = aggregate_sensitivity_summary(per_ds_df)
    summary_df.to_csv(summary_csv, index=False)
    plot_sensitivity_fig14(summary_df, fig_path)

    print("\n=== TỔNG HỢP RQ3 SENSITIVITY (19 DATASETS) ===")
    print(summary_df[["Config", "Mean_Kappa", "Mean_Diff_STUDD_Minus_Cfg", "STUDD_Win_Count", "STUDD_ROPE_Count", "STUDD_Lose_Count", "Bayes_P_STUDD_Win_Or_ROPE"]].to_string(index=False))

    grid = summary_df[summary_df["Kind"] == "Grid"]
    studd_wins_or_ties_cells = int((grid["Mean_Diff_STUDD_Minus_Cfg"] >= 0.0).sum())
    log_fullscope_run(
        run_key="sensitivity_rq3",
        run_name="fullscope-sensitivity-rq3",
        job_type="sensitivity_analysis_rq3",
        config={
            "scope": "19_datasets_section_5_8_5",
            "l_access_grid": list(L_ACCESS_GRID),
            "l_delay_grid": list(L_DELAY_GRID),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
            "seed_rng": SEED_RNG,
        },
        summary_metrics={
            "grid_cells": len(grid),
            "studd_mean_kappa": float(summary_df[summary_df["Kind"] == "STUDD"]["Mean_Kappa"].iloc[0]),
            "ss_mean_kappa": float(summary_df[summary_df["Kind"] == "SS_Ideal"]["Mean_Kappa"].iloc[0]),
            "studd_outperforms_mean_cells": studd_wins_or_ties_cells,
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        tables={
            "sensitivity_rq3_summary": summary_df,
            "sensitivity_rq3_per_dataset": per_ds_df,
        },
        images={"fig14_sensitivity_rq3": fig_path},
        enabled=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
