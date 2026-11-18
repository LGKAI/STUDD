"""Kịch bản Tái hiện Thực nghiệm 6 Bộ Dữ liệu (Đạt — 10 Phương pháp, Trần 150.000 mẫu).

Tái hiện thực nghiệm tuần tự trên 6 bộ dữ liệu phân công:
    1. `electricity` (45,312 mẫu — chạy trước)
    2. `powersupply` (29,928 mẫu)
    3. `noaa`        (18,159 mẫu)
    4. `covertype`   (cắt 150,000 mẫu đầu)
    5. `poker`       (cắt 150,000 mẫu đầu)
    6. `sensors`     (cắt 150,000 mẫu đầu)

Hỗ trợ chạy thử nghiệm ngoại tuyến (`--no-wandb`), đồng bộ trực tiếp lên W&B
trong lúc chạy, hoặc đồng bộ lại từ các file CSV/PNG đã lưu (`--sync-wandb`).
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baselines import (
    ALL_METHODS,
    compute_alarm_metrics,
    compute_labeling_efficiency,
    evaluate_all_10_methods,
)
from src.data_loader import DAT_6_DATASETS, PAPER_MAX_SAMPLES, load_paper_stream
from src.studd import build_paper_models

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Số liệu công bố chính thức trong Bảng 3 (Cohen's Kappa, trang 4367) và Bảng 4 (Cost, trang 4368)
PAPER_TABLE3_KAPPA: Dict[str, Dict[str, float]] = {
    "electricity": {"STUDD": 0.29, "BL-st": 0.13, "BL-ret": 0.39, "SS": 0.39, "DSS": 0.39, "WS": 0.39, "DWS": 0.34, "OS": 0.33, "OF": 0.38, "FF": 0.39},
    "powersupply": {"STUDD": 0.10, "BL-st": 0.09, "BL-ret": 0.11, "SS": 0.11, "DSS": 0.09, "WS": 0.11, "DWS": 0.11, "OS": 0.11, "OF": 0.10, "FF": 0.11},
    "noaa":        {"STUDD": 0.39, "BL-st": 0.39, "BL-ret": 0.45, "SS": 0.44, "DSS": 0.43, "WS": 0.46, "DWS": 0.41, "OS": 0.46, "OF": 0.40, "FF": 0.46},
    "covertype":   {"STUDD": 0.52, "BL-st": 0.50, "BL-ret": 0.67, "SS": 0.63, "DSS": 0.58, "WS": 0.53, "DWS": 0.53, "OS": 0.33, "OF": 0.54, "FF": 0.66},
    "poker":       {"STUDD": 0.48, "BL-st": 0.28, "BL-ret": 0.46, "SS": 0.65, "DSS": 0.61, "WS": 0.68, "DWS": 0.58, "OS": 0.20, "OF": 0.18, "FF": 0.45},
    "sensors":     {"STUDD": 0.59, "BL-st": 0.11, "BL-ret": 0.67, "SS": 0.83, "DSS": 0.63, "WS": 0.77, "DWS": 0.56, "OS": 0.53, "OF": 0.60, "FF": 0.67},
}

PAPER_TABLE4_COST: Dict[str, Dict[str, float]] = {
    "electricity": {"STUDD": 0.43, "BL-st": 0.02, "BL-ret": 0.99, "SS": 0.69, "DSS": 0.51, "WS": 0.82, "DWS": 0.29, "OS": 0.09, "OF": 0.38, "FF": 0.99},
    "powersupply": {"STUDD": 0.07, "BL-st": 0.03, "BL-ret": 0.97, "SS": 0.30, "DSS": 0.10, "WS": 0.33, "DWS": 0.20, "OS": 0.43, "OF": 0.57, "FF": 0.94},
    "noaa":        {"STUDD": 0.06, "BL-st": 0.06, "BL-ret": 0.99, "SS": 0.22, "DSS": 0.22, "WS": 0.28, "DWS": 0.17, "OS": 0.28, "OF": 0.22, "FF": 0.99},
    "covertype":   {"STUDD": 0.09, "BL-st": 0.01, "BL-ret": 0.99, "SS": 0.49, "DSS": 0.42, "WS": 0.51, "DWS": 0.16, "OS": 0.09, "OF": 0.29, "FF": 0.99},
    "poker":       {"STUDD": 0.01, "BL-st": 0.01, "BL-ret": 0.97, "SS": 2.69, "DSS": 2.51, "WS": 4.51, "DWS": 3.45, "OS": 0.43, "OF": 0.62, "FF": 0.97},
    "sensors":     {"STUDD": 0.16, "BL-st": 0.01, "BL-ret": 0.99, "SS": 1.43, "DSS": 0.51, "WS": 2.05, "DWS": 0.71, "OS": 0.37, "OF": 0.47, "FF": 0.99},
}


def _relative_diff_pct(ours: float, paper: float) -> Optional[float]:
    """Tính độ lệch tương đối (%) so với giá trị bài báo."""
    if paper == 0.0:
        return None
    return round(100.0 * (ours - paper) / abs(paper), 2)


def run_single_dataset(
    dataset_name: str,
    window_size_override: Optional[int] = None,
    max_samples: int = PAPER_MAX_SAMPLES,
    seed: int = 42,
    source_mode: str = "auto",
    use_wandb: bool = False,
    entity: str = "lgkai-hcmus",
    project: str = "STUDD",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy tái hiện đầy đủ 10 phương pháp trên một bộ dữ liệu và lưu bảng kết quả."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t_start = time.perf_counter()

    X, y, meta = load_paper_stream(
        dataset_name=dataset_name,
        max_samples=max_samples,
        normalize=False,
        source_mode=source_mode,
    )
    W = int(window_size_override or meta["recommended_window_size"])
    teacher_model, student_model, t_code, s_code = build_paper_models(dataset_name, random_state=seed)

    print("\n" + "=" * 78)
    print(
        f"[DATASET: {meta['canonical_name']}] samples={meta['n_samples']}, features={meta['n_features']}, "
        f"classes={meta['n_classes']}, W={W}, Teacher={t_code}, Student={s_code}"
    )
    print(f"Nguồn nạp: {meta['source']}")
    print("=" * 78)

    results_by_method = evaluate_all_10_methods(
        X_stream=X,
        y_stream=y,
        teacher_model=teacher_model,
        student_model=student_model,
        window_size=W,
        delta=0.001,
        ph_threshold=50.0,
        ks_alpha=0.001,
        random_state=seed,
        verbose=True,
    )
    elapsed_sec = time.perf_counter() - t_start

    alarm_stats = compute_alarm_metrics(results_by_method, window_size=W, n_total=len(X))
    eff_stats = compute_labeling_efficiency(results_by_method)

    ds_key = dataset_name.lower().strip()
    rows: List[Dict[str, Any]] = []
    for method in ALL_METHODS:
        res = results_by_method[method]
        p_kappa = PAPER_TABLE3_KAPPA[ds_key][method]
        p_cost = PAPER_TABLE4_COST[ds_key][method]
        rows.append({
            "Dataset": meta["canonical_name"],
            "Method": method,
            "Teacher": t_code,
            "Student": s_code if method == "STUDD" else t_code,
            "Samples": len(X),
            "Window_Size": W,
            "Our_Accuracy": round(res["accuracy"], 4),
            "Our_Kappa": round(res["kappa"], 4),
            "Paper_Kappa": p_kappa,
            "Kappa_Diff": round(res["kappa"] - p_kappa, 4),
            "Kappa_Rel_Diff_Pct": _relative_diff_pct(res["kappa"], p_kappa),
            "Our_Cost": round(res["labeling_cost"], 4),
            "Paper_Cost": p_cost,
            "Cost_Diff": round(res["labeling_cost"] - p_cost, 4),
            "Cost_Rel_Diff_Pct": _relative_diff_pct(res["labeling_cost"], p_cost),
            "Alarms": res["num_alarms"],
            "Retrains": res.get("num_retrains", res["num_alarms"]),
            "Label_Efficiency": round(eff_stats[method], 4) if eff_stats.get(method) is not None else None,
            "Runtime_Sec": round(elapsed_sec, 2),
        })

    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, f"{ds_key}_reproduction.csv")
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    alarm_rows: List[Dict[str, Any]] = []
    for method, m_dict in alarm_stats.items():
        alarm_rows.append({
            "Dataset": meta["canonical_name"],
            "Method": method,
            "SS_First_Alarm": results_by_method["SS"]["alarm_indices"][0] if results_by_method["SS"]["alarm_indices"] else None,
            "MDR": m_dict["MDR"],
            "FAR": m_dict["FAR"],
            "MTD": round(m_dict["MTD"], 4) if m_dict["MTD"] is not None else None,
            "MTFA": round(m_dict["MTFA"], 4) if m_dict["MTFA"] is not None else None,
        })
    df_alarm = pd.DataFrame(alarm_rows)
    alarm_csv_path = os.path.join(RESULTS_DIR, f"{ds_key}_alarm_metrics.csv")
    df_alarm.to_csv(alarm_csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[+] Hoàn tất {meta['canonical_name']} sau {elapsed_sec:.2f}s -> {csv_path}")
    print(df_res[["Method", "Our_Kappa", "Paper_Kappa", "Kappa_Diff", "Our_Cost", "Paper_Cost", "Alarms", "Label_Efficiency"]].to_string(index=False))

    if use_wandb:
        _log_dataset_run_to_wandb(ds_key, df_res, df_alarm, entity=entity, project=project)

    return df_res, df_alarm


def _log_dataset_run_to_wandb(
    ds_key: str,
    df_res: pd.DataFrame,
    df_alarm: pd.DataFrame,
    entity: str,
    project: str,
) -> str:
    """Đẩy kết quả của một bộ dữ liệu lên Weights & Biases và trả về URL của run."""
    import wandb

    first_row = df_res.iloc[0]
    run = wandb.init(
        entity=entity,
        project=project,
        name=f"reproduce-{ds_key}-w{int(first_row['Window_Size'])}",
        tags=["reproduction", "6-datasets", "dat", ds_key],
        config={
            "dataset": str(first_row["Dataset"]),
            "samples": int(first_row["Samples"]),
            "window_size": int(first_row["Window_Size"]),
            "teacher": str(first_row["Teacher"]),
            "student": str(first_row["Student"]),
            "delta": 0.001,
            "ph_threshold": 50.0,
            "ks_alpha": 0.001,
            "random_state": 42,
        },
        reinit=True,
    )
    metrics_payload: Dict[str, Any] = {
        "comparison_table": wandb.Table(dataframe=df_res),
        "runtime_sec": float(first_row["Runtime_Sec"]),
    }
    if not df_alarm.empty:
        metrics_payload["alarm_metrics_table"] = wandb.Table(dataframe=df_alarm)

    for _, row in df_res.iterrows():
        m = str(row["Method"])
        metrics_payload[f"{m}/kappa"] = float(row["Our_Kappa"])
        metrics_payload[f"{m}/paper_kappa"] = float(row["Paper_Kappa"])
        metrics_payload[f"{m}/cost"] = float(row["Our_Cost"])
        metrics_payload[f"{m}/paper_cost"] = float(row["Paper_Cost"])
        metrics_payload[f"{m}/alarms"] = int(row["Alarms"])

    wandb.log(metrics_payload)
    run_url = str(run.url)
    wandb.finish()
    print(f"[W&B] Đã đồng bộ '{ds_key}' -> {run_url}")
    return run_url


def aggregate_6datasets_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gộp kết quả của 6 bộ dữ liệu đã chạy và vẽ biểu đồ tổng hợp chuẩn 300 DPI."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comp_frames, alarm_frames = [], []

    for ds_key in DAT_6_DATASETS:
        comp_path = os.path.join(RESULTS_DIR, f"{ds_key}_reproduction.csv")
        alarm_path = os.path.join(RESULTS_DIR, f"{ds_key}_alarm_metrics.csv")
        if os.path.exists(comp_path):
            comp_frames.append(pd.read_csv(comp_path))
        if os.path.exists(alarm_path):
            alarm_frames.append(pd.read_csv(alarm_path))

    if not comp_frames:
        raise FileNotFoundError("Chưa tìm thấy file *_reproduction.csv nào trong thư mục results/.")

    df_all = pd.concat(comp_frames, ignore_index=True)
    summary_csv = os.path.join(RESULTS_DIR, "dat_6datasets_summary.csv")
    df_all.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    df_alarm_all = pd.concat(alarm_frames, ignore_index=True) if alarm_frames else pd.DataFrame()
    if not df_alarm_all.empty:
        alarm_summary = (
            df_alarm_all.groupby("Method", as_index=False)[["MDR", "FAR", "MTD", "MTFA"]]
            .mean(numeric_only=True)
            .round(4)
        )
        alarm_summary_csv = os.path.join(RESULTS_DIR, "dat_6datasets_alarm_summary.csv")
        alarm_summary.to_csv(alarm_summary_csv, index=False, encoding="utf-8-sig")
    else:
        alarm_summary = pd.DataFrame()

    _plot_6datasets_kappa_vs_cost(
        df_all, os.path.join(RESULTS_DIR, "fig_dat_6datasets_kappa_vs_cost.png")
    )
    _plot_6datasets_studd_vs_paper(
        df_all, os.path.join(RESULTS_DIR, "fig_dat_6datasets_studd_vs_paper.png")
    )
    print(f"[+] Đã tổng hợp bảng 6 bộ tại: {summary_csv}")
    return df_all, alarm_summary


def _plot_6datasets_kappa_vs_cost(df_all: pd.DataFrame, save_path: str) -> None:
    """Vẽ lưới 2x3 đối chiếu Cohen's Kappa và Labeling Cost trên 6 bộ dữ liệu (300 DPI)."""
    datasets = df_all["Dataset"].unique()
    n_ds = len(datasets)
    ncols = 3 if n_ds > 2 else max(1, n_ds)
    nrows = int(np.ceil(n_ds / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, ds_name in enumerate(datasets):
        ax = flat_axes[idx]
        sub = df_all[df_all["Dataset"] == ds_name]
        x = np.arange(len(sub))
        w = 0.38
        ax.bar(x - w / 2, sub["Our_Kappa"], w, label="Our Kappa", color="#2b5c8f")
        ax.bar(x + w / 2, sub["Our_Cost"], w, label="Our Cost", color="#d97a36")
        ax.set_title(f"{ds_name} (N={int(sub['Samples'].iloc[0]):,})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Method"], rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, max(1.05, float(max(sub["Our_Kappa"].max(), sub["Our_Cost"].max())) * 1.15))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(fontsize=8.5, loc="upper left")

    for j in range(n_ds, len(flat_axes)):
        flat_axes[j].axis("off")

    plt.suptitle("6-Dataset Reproduction: Cohen's Kappa vs. Labeling Cost across 10 Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_6datasets_studd_vs_paper(df_all: pd.DataFrame, save_path: str) -> None:
    """Vẽ biểu đồ đối chiếu trực tiếp STUDD của nhóm vs. Paper (Kappa & Cost)."""
    sub = df_all[df_all["Method"] == "STUDD"].copy()
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    b1 = ax1.bar(x - w / 2, sub["Our_Kappa"], w, label="Our Rebuild", color="#2ca02c")
    b2 = ax1.bar(x + w / 2, sub["Paper_Kappa"], w, label="Paper (Table 3)", color="#7f7f7f")
    ax1.set_title("STUDD Predictive Performance (Cohen's Kappa)", fontsize=11.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["Dataset"], rotation=20, ha="right")
    ax1.set_ylim(0, max(0.85, float(max(sub["Our_Kappa"].max(), sub["Paper_Kappa"].max())) * 1.18))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax1.legend()
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    b3 = ax2.bar(x - w / 2, sub["Our_Cost"], w, label="Our Rebuild", color="#1f77b4")
    b4 = ax2.bar(x + w / 2, sub["Paper_Cost"], w, label="Paper (Table 4)", color="#ff7f0e")
    ax2.set_title("STUDD Labeling Cost Ratio", fontsize=11.5, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["Dataset"], rotation=20, ha="right")
    ax2.set_ylim(0, max(0.6, float(max(sub["Our_Cost"].max(), sub["Paper_Cost"].max())) * 1.22))
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend()
    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def sync_saved_results_to_wandb(entity: str = "lgkai-hcmus", project: str = "STUDD") -> None:
    """Đẩy toàn bộ kết quả CSV và biểu đồ 300 DPI đã lưu lên W&B (dùng sau khi `wandb login`)."""
    import wandb

    urls: Dict[str, str] = {}
    for ds_key in DAT_6_DATASETS:
        comp_path = os.path.join(RESULTS_DIR, f"{ds_key}_reproduction.csv")
        alarm_path = os.path.join(RESULTS_DIR, f"{ds_key}_alarm_metrics.csv")
        if os.path.exists(comp_path):
            df_res = pd.read_csv(comp_path)
            df_alarm = pd.read_csv(alarm_path) if os.path.exists(alarm_path) else pd.DataFrame()
            urls[ds_key] = _log_dataset_run_to_wandb(ds_key, df_res, df_alarm, entity=entity, project=project)

    df_all, alarm_summary = aggregate_6datasets_results()
    summary_run = wandb.init(
        entity=entity,
        project=project,
        name="reproduce-dat-6datasets-summary",
        tags=["reproduction", "6-datasets", "dat", "summary"],
        config={"datasets": list(urls.keys()), "max_samples": PAPER_MAX_SAMPLES},
        reinit=True,
    )
    payload: Dict[str, Any] = {
        "dat_6datasets_summary": wandb.Table(dataframe=df_all),
        "fig_kappa_vs_cost": wandb.Image(os.path.join(RESULTS_DIR, "fig_dat_6datasets_kappa_vs_cost.png")),
        "fig_studd_vs_paper": wandb.Image(os.path.join(RESULTS_DIR, "fig_dat_6datasets_studd_vs_paper.png")),
    }
    if not alarm_summary.empty:
        payload["dat_6datasets_alarm_summary"] = wandb.Table(dataframe=alarm_summary)
    wandb.log(payload)
    print(f"[W&B] Đã đồng bộ Summary Run -> {summary_run.url}")
    wandb.finish()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chạy tái hiện 6 bộ dữ liệu STUDD (Đạt).")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DAT_6_DATASETS),
        choices=list(DAT_6_DATASETS),
        help="Danh sách bộ dữ liệu cần chạy tuần tự.",
    )
    parser.add_argument("--window-size", type=int, default=None, help="Ghi đè kích thước cửa sổ W (mặc định 2000).")
    parser.add_argument("--max-samples", type=int, default=PAPER_MAX_SAMPLES, help="Giới hạn số mẫu tối đa (mặc định 150000).")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên cố định.")
    parser.add_argument("--source-mode", default="auto", choices=["auto", "hosted", "local"], help="Nguồn nạp dữ liệu.")
    parser.add_argument("--no-wandb", action="store_true", help="Chạy ngoại tuyến, không đẩy log lên W&B.")
    parser.add_argument("--aggregate", action="store_true", help="Gộp kết quả hiện có và vẽ biểu đồ tổng hợp.")
    parser.add_argument("--sync-wandb", action="store_true", help="Đẩy toàn bộ kết quả CSV/PNG đã lưu lên W&B.")
    parser.add_argument("--entity", default="lgkai-hcmus", help="W&B entity.")
    parser.add_argument("--project", default="STUDD", help="W&B project.")
    args = parser.parse_args()

    if args.sync_wandb:
        sync_saved_results_to_wandb(entity=args.entity, project=args.project)
    elif args.aggregate and "--datasets" not in sys.argv:
        aggregate_6datasets_results()
    else:
        for ds in args.datasets:
            run_single_dataset(
                dataset_name=ds,
                window_size_override=args.window_size,
                max_samples=args.max_samples,
                seed=args.seed,
                source_mode=args.source_mode,
                use_wandb=not args.no_wandb,
                entity=args.entity,
                project=args.project,
            )
        if args.aggregate or len(args.datasets) > 1:
            aggregate_6datasets_results()
