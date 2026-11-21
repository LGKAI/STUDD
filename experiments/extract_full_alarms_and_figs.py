"""Commit 4: Hoàn tất phân tích báo động Table 5 trên 19 bộ dữ liệu & chuẩn hóa Hình 7-8.

Thực hiện:
    1. Trích xuất đầy đủ `alarm_indices` cho 7 bộ của Khánh (`results/khanh/<slug>_alarms.csv`)
       và 6 bộ của Đạt (`results/dat/<slug>_alarms.csv`) trên 8 phương pháp phát báo động
       {SS, STUDD, DSS, WS, DWS, OS, OF, FF} theo cấu hình thống nhất:
       - Teacher SEED = 42, Student SEED = 142
       - Fixed PageHinkley (`max(0, ...)`, `delta = 0.001`, `threshold = 50.0`), `ks_alpha = 0.001`
       - Safe imputation trên `GMSC` khi chạy `FF` để `ks_2samp` không bị `NaN`.
    2. Kết hợp với 6 tệp `results/khoa/*_alarms.csv` có sẵn để tính **Bảng 5** đầy đủ trên toàn bộ
       nhóm bộ dữ liệu có báo động `SS` (MDR, FAR, MTD, MTFA theo Paper Section 5.8.3 +
       Precision / Recall / F1 đối chiếu mọi báo động `SS`).
    3. Chạy chuẩn hóa `InsectsAbrupt` & `Posture` (`Teacher=42, Student=142`, fixed `PageHinkley`)
       để vẽ **Fig. 7** và **Fig. 8** (Rolling Kappa 200 mẫu + vạch báo động).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fullscope_wandb import (
    RESULTS_DIR,
    SEED_STUDENT,
    SEED_TEACHER,
    log_fullscope_run,
)
from src.baselines import (
    compute_alarm_metrics,
    evaluate_feature_fixed_ks,
    evaluate_output_ks,
    evaluate_static_baseline,
    evaluate_supervised_pht,
)
from src.data_loader import DAT_6_DATASETS, KHANH_7_DATASETS, ALL_19_DATASETS, load_all_19_stream
from src.models import PAPER_TABLE2_MODELS, SafeStreamClassifier, build_model
from src.studd import STUDD

ALARM_METHODS: Tuple[str, ...] = ("SS", "STUDD", "DSS", "WS", "DWS", "OS", "OF", "FF")
TABLE5_METHODS: Tuple[str, ...] = ("STUDD", "DSS", "WS", "DWS", "OS", "OF", "FF")

DELTA = 0.001
PH_THRESHOLD = 50.0
KS_ALPHA = 0.001

PAPER_TABLE5: Dict[str, Dict[str, Optional[float]]] = {
    "STUDD": {"MDR": 0.19, "FAR": 0.00, "MTD": 0.09, "MTFA": None},
    "DSS": {"MDR": 0.06, "FAR": 0.00, "MTD": 0.14, "MTFA": None},
    "WS": {"MDR": 0.12, "FAR": 0.00, "MTD": 0.14, "MTFA": None},
    "DWS": {"MDR": 0.19, "FAR": 0.00, "MTD": 0.16, "MTFA": None},
    "OS": {"MDR": 0.06, "FAR": 0.12, "MTD": 0.12, "MTFA": 0.44},
    "OF": {"MDR": 0.06, "FAR": 0.12, "MTD": 0.11, "MTFA": 0.30},
    "FF": {"MDR": 0.00, "FAR": 0.25, "MTD": 0.07, "MTFA": 0.15},
}

KHOA_SLUG_MAP: Dict[str, str] = {
    "InsectsAbrupt": "insects-abrupt",
    "Insects": "insects",
    "Keystroke": "keystroke",
    "Arabic": "arabic",
    "ArabicShuffled": "arabic-shuffled",
    "Posture": "posture",
}


def _build_stream_models(canon: str) -> Tuple[SafeStreamClassifier, SafeStreamClassifier, str, str]:
    key = canon.lower().replace("-", "").replace("_", "")
    t_code, s_code = PAPER_TABLE2_MODELS[key]
    teacher = build_model(t_code, random_state=SEED_TEACHER, n_jobs=2)
    student = build_model(s_code, random_state=SEED_STUDENT, n_jobs=2)
    # Các bộ đánh giá ở đây chỉ dùng .predict(), không dùng .predict_proba()
    if t_code.upper() in ("SVC", "SVM"):
        teacher.base_estimator.set_params(probability=False)
    if s_code.upper() in ("SVC", "SVM"):
        student.base_estimator.set_params(probability=False)
    return teacher, student, t_code, s_code


def _impute_for_ff(X: np.ndarray, W: int) -> np.ndarray:
    """Điền khuyết NaN theo trung vị cửa sổ đầu W (cho GMSC) để KS-test đặc trưng hoạt động an toàn."""
    if not np.isnan(X).any():
        return X
    X_out = np.array(X, dtype=np.float32, copy=True)
    col_meds = np.nanmedian(X_out[:W], axis=0)
    col_meds = np.where(np.isnan(col_meds), 0.0, col_meds)
    for col in range(X_out.shape[1]):
        mask = np.isnan(X_out[:, col])
        if mask.any():
            X_out[mask, col] = col_meds[col]
    return X_out


def extract_alarms_for_dataset(dataset_name: str, out_csv: Path) -> pd.DataFrame:
    """Chạy 8 bộ phát hiện cảnh báo trên 1 bộ dữ liệu và xuất `Source_Row_Zero_Based`."""
    X, y, meta = load_all_19_stream(dataset_name)
    canon = str(meta["canonical_name"])
    W = int(meta["recommended_window_size"])
    teacher, student, t_code, s_code = _build_stream_models(canon)

    print(f" -> Trích xuất báo động: {canon:13s} (N={len(X):6d}, W={W:4d}, T={t_code}, S={s_code})...", flush=True)
    res_map: Dict[str, Dict[str, Any]] = {
        "SS": evaluate_supervised_pht(X, y, base_model=teacher, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, l_access=1.0, l_delay=0, method_name="SS", random_state=SEED_TEACHER),
        "STUDD": STUDD(teacher_model=teacher, student_model=student, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, random_state=SEED_TEACHER).evaluate_stream(X, y),
        "DSS": evaluate_supervised_pht(X, y, base_model=teacher, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, l_access=1.0, l_delay=W // 2, method_name="DSS", random_state=SEED_TEACHER),
        "WS": evaluate_supervised_pht(X, y, base_model=teacher, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, l_access=0.5, l_delay=0, method_name="WS", random_state=SEED_TEACHER),
        "DWS": evaluate_supervised_pht(X, y, base_model=teacher, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, l_access=0.5, l_delay=W // 2, method_name="DWS", random_state=SEED_TEACHER),
        "OS": evaluate_output_ks(X, y, base_model=teacher, window_size=W, ks_alpha=KS_ALPHA, mode="sliding", random_state=SEED_TEACHER),
        "OF": evaluate_output_ks(X, y, base_model=teacher, window_size=W, ks_alpha=KS_ALPHA, mode="fixed", random_state=SEED_TEACHER),
        "FF": evaluate_feature_fixed_ks(_impute_for_ff(X, W), y, base_model=teacher, window_size=W, ks_alpha=KS_ALPHA, random_state=SEED_TEACHER),
    }

    rows: List[Dict[str, object]] = []
    for method in ALARM_METHODS:
        for idx in res_map[method]["alarm_indices"]:
            rows.append({"Method": method, "Source_Row_Zero_Based": int(idx)})

    df_out = pd.DataFrame(rows, columns=["Method", "Source_Row_Zero_Based"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    counts_str = ", ".join(f"{m}:{len(res_map[m]['alarm_indices'])}" for m in ALARM_METHODS)
    print(f"    [OK] {canon}: {counts_str}")
    return df_out


def match_alarms_greedy(
    ss_alarms: Sequence[int],
    cand_alarms: Sequence[int],
    window_size: int,
) -> Tuple[int, int, int, List[int]]:
    """Khớp tham lam 1-1 mỗi báo động SS `t_ss` với báo động sớm nhất của phương pháp trong `[t_ss, t_ss + W]`."""
    used = [False] * len(cand_alarms)
    tp = 0
    delays: List[int] = []
    for t_ss in sorted(ss_alarms):
        hi = t_ss + window_size
        for j, t_c in enumerate(cand_alarms):
            if not used[j] and t_ss <= t_c <= hi:
                used[j] = True
                tp += 1
                delays.append(int(t_c - t_ss))
                break
    fp = len(cand_alarms) - tp
    fn = len(ss_alarms) - tp
    return tp, fp, fn, delays


def _rolling_kappa(y_true: np.ndarray, y_pred: np.ndarray, win: int = 200) -> np.ndarray:
    n = len(y_true)
    vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(win - 1, n):
        yt = y_true[i - win + 1 : i + 1]
        yp = y_pred[i - win + 1 : i + 1]
        if len(np.unique(yt)) < 2 and len(np.unique(yp)) < 2 and np.array_equal(yt, yp):
            vals[i] = 1.0
        else:
            k = float(cohen_kappa_score(yt, yp))
            vals[i] = 0.0 if np.isnan(k) else k
    return vals


def plot_standardized_alarm_figure(dataset_name: str, fig_label: str, out_path: Path) -> Dict[str, Any]:
    """Vẽ Fig. 7 (InsectsAbrupt) hoặc Fig. 8 (Posture) chuẩn hóa với Seed T=42, S=142 & fixed PH."""
    X, y, meta = load_all_19_stream(dataset_name)
    canon = str(meta["canonical_name"])
    W = int(meta["recommended_window_size"])
    teacher, student, t_code, s_code = _build_stream_models(canon)

    res_bl = evaluate_static_baseline(X, y, base_model=teacher, window_size=W, random_state=SEED_TEACHER)
    res_ss = evaluate_supervised_pht(X, y, base_model=teacher, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, l_access=1.0, l_delay=0, method_name="SS", random_state=SEED_TEACHER)
    res_studd = STUDD(teacher_model=teacher, student_model=student, window_size=W, delta=DELTA, threshold=PH_THRESHOLD, random_state=SEED_TEACHER).evaluate_stream(X, y)

    steps = np.arange(W, len(X))
    rk_bl = _rolling_kappa(res_bl["y_true_online"], res_bl["y_pred_online"], win=200)
    rk_ss = _rolling_kappa(res_ss["y_true_online"], res_ss["y_pred_online"], win=200)
    rk_studd = _rolling_kappa(res_studd["y_true_online"], res_studd["y_pred_online"], win=200)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.4), dpi=160, sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax1.plot(steps, rk_bl, color="#64748b", linestyle=":", linewidth=1.4, label=f"BL-st (Kappa={res_bl['kappa']:.3f})")
    ax1.plot(steps, rk_ss, color="#16a34a", linestyle="-.", linewidth=1.5, label=f"SS (Kappa={res_ss['kappa']:.3f}, alarms={res_ss['num_alarms']})")
    ax1.plot(steps, rk_studd, color="#dc2626", linewidth=1.8, label=f"STUDD T={SEED_TEACHER}/S={SEED_STUDENT} (Kappa={res_studd['kappa']:.3f}, alarms={res_studd['num_alarms']})")

    for i, t_al in enumerate(res_ss["alarm_indices"]):
        ax1.axvline(t_al, color="#16a34a", linestyle=":", alpha=0.55, linewidth=1.2, label="SS Alarm" if i == 0 else None)
    for i, t_al in enumerate(res_studd["alarm_indices"]):
        ax1.axvline(t_al, color="#dc2626", linestyle="--", alpha=0.85, linewidth=1.5, label="STUDD Alarm" if i == 0 else None)
        ax2.axvline(t_al, color="#dc2626", linestyle="--", alpha=0.85, linewidth=1.5)

    ax1.set_ylabel("Rolling Cohen's Kappa (200 samples)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=8.5, ncol=2)
    ax1.set_title(
        f"{fig_label}: Visualizing Drift Alarms on {canon} (W={W}, {t_code}/{s_code}, Seed T={SEED_TEACHER}/S={SEED_STUDENT}, fixed PH)",
        fontsize=11,
        fontweight="bold",
    )

    ax2.plot(steps, res_studd["ph_statistics"], color="#f59e0b", linewidth=1.2, label="Page-Hinkley Statistic (STUDD Mimicry Loss)")
    ax2.axhline(PH_THRESHOLD, color="#dc2626", linestyle=":", linewidth=1.1, label=f"Threshold lambda={PH_THRESHOLD:g}")
    ax2.set_xlabel("Stream Time Step t", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("PH Statistic", fontsize=9.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "dataset": canon,
        "studd_kappa": round(float(res_studd["kappa"]), 4),
        "ss_kappa": round(float(res_ss["kappa"]), 4),
        "bl_st_kappa": round(float(res_bl["kappa"]), 4),
        "studd_alarms": res_studd["alarm_indices"],
        "ss_alarms": res_ss["alarm_indices"],
    }


def _locate_alarm_csv(canon: str) -> Path:
    if canon in KHOA_SLUG_MAP:
        return RESULTS_DIR / "khoa" / f"{KHOA_SLUG_MAP[canon]}_alarms.csv"
    slug = canon.lower()
    if slug in KHANH_7_DATASETS:
        return RESULTS_DIR / "khanh" / f"{slug}_alarms.csv"
    return RESULTS_DIR / "dat" / f"{slug}_alarms.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wandb", action="store_true", help="Tắt đồng bộ W&B.")
    parser.add_argument("--skip-existing", action="store_true", help="Bỏ qua tệp alarms CSV đã trích xuất.")
    args = parser.parse_args()

    # 1. Trích xuất alarm_indices cho 6 bộ Đạt + 7 bộ Khánh
    for ds_name in list(DAT_6_DATASETS) + list(KHANH_7_DATASETS):
        _, _, meta = load_all_19_stream(ds_name)
        canon = str(meta["canonical_name"])
        out_csv = _locate_alarm_csv(canon)
        if args.skip_existing and out_csv.exists():
            continue
        extract_alarms_for_dataset(ds_name, out_csv)

    # 2. Tổng hợp Bảng 5 trên toàn bộ 19 bộ dữ liệu
    per_ds_rows: List[Dict[str, object]] = []
    ss_alarmed_datasets: List[str] = []

    for ds in ALL_19_DATASETS:
        X, _, meta = load_all_19_stream(ds)
        canon = str(meta["canonical_name"])
        W = int(meta["recommended_window_size"])
        n_total = len(X)
        alarm_csv = _locate_alarm_csv(canon)
        df_al = pd.read_csv(alarm_csv)

        results_by_method: Dict[str, Dict[str, Any]] = {}
        for m in ALARM_METHODS:
            pts = sorted(int(v) for v in df_al.loc[df_al["Method"] == m, "Source_Row_Zero_Based"].tolist())
            results_by_method[m] = {"alarm_indices": pts}

        ss_alarms = results_by_method["SS"]["alarm_indices"]
        has_ss = len(ss_alarms) > 0
        if has_ss:
            ss_alarmed_datasets.append(canon)
        t5_metrics = compute_alarm_metrics(results_by_method, window_size=W, n_total=n_total) if has_ss else {}

        for m in TABLE5_METHODS:
            cand_alarms = results_by_method[m]["alarm_indices"]
            tp, fp, fn, delays = match_alarms_greedy(ss_alarms, cand_alarms, W) if has_ss else (0, len(cand_alarms), 0, [])
            m_stats = t5_metrics.get(m, {})
            per_ds_rows.append(
                {
                    "Dataset": canon,
                    "W": W,
                    "N": n_total,
                    "SS_Has_Alarm": has_ss,
                    "SS_Alarms_Count": len(ss_alarms),
                    "Method": m,
                    "Alarms_Count": len(cand_alarms),
                    "MDR": m_stats.get("MDR"),
                    "FAR": m_stats.get("FAR"),
                    "MTD": m_stats.get("MTD"),
                    "MTFA": m_stats.get("MTFA"),
                    "TP_Windowed": tp,
                    "FP_Windowed": fp,
                    "FN_Windowed": fn,
                    "Mean_Delay_Steps": round(float(np.mean(delays)), 1) if delays else None,
                }
            )

    per_ds_df = pd.DataFrame(per_ds_rows)
    per_ds_csv = RESULTS_DIR / "table5_per_dataset_alarms.csv"
    per_ds_df.to_csv(per_ds_csv, index=False)

    # Tổng hợp Table 5 trên các bộ có báo động SS (giống Section 5.8.3 bài báo)
    sub_ss = per_ds_df[per_ds_df["SS_Has_Alarm"]].copy()
    summary_rows: List[Dict[str, object]] = []
    for m in TABLE5_METHODS:
        m_df = sub_ss[sub_ss["Method"] == m]
        tot_tp = int(m_df["TP_Windowed"].sum())
        tot_fp = int(m_df["FP_Windowed"].sum())
        tot_fn = int(m_df["FN_Windowed"].sum())
        tot_al = int(m_df["Alarms_Count"].sum())
        prec = tot_tp / tot_al if tot_al > 0 else 0.0
        rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        our_mdr = float(m_df["MDR"].mean())
        our_far = float(m_df["FAR"].mean())
        our_mtd = float(m_df["MTD"].dropna().mean()) if m_df["MTD"].notna().any() else np.nan
        our_mtfa = float(m_df["MTFA"].dropna().mean()) if m_df["MTFA"].notna().any() else np.nan
        mean_delay_steps = float(m_df["Mean_Delay_Steps"].dropna().mean()) if m_df["Mean_Delay_Steps"].notna().any() else np.nan

        p_ref = PAPER_TABLE5[m]
        p_mdr = float(p_ref["MDR"]) if p_ref["MDR"] is not None else np.nan
        p_far = float(p_ref["FAR"]) if p_ref["FAR"] is not None else np.nan
        p_mtd = float(p_ref["MTD"]) if p_ref["MTD"] is not None else np.nan
        p_mtfa = float(p_ref["MTFA"]) if p_ref["MTFA"] is not None else np.nan

        summary_rows.append(
            {
                "Method": m,
                "SS_Alarmed_Datasets": len(ss_alarmed_datasets),
                "Our_MDR": round(our_mdr, 4),
                "Paper_MDR": p_mdr,
                "Our_FAR": round(our_far, 4),
                "Paper_FAR": p_far,
                "Our_MTD": round(our_mtd, 4) if not np.isnan(our_mtd) else np.nan,
                "Paper_MTD": p_mtd,
                "Our_MTFA": round(our_mtfa, 4) if not np.isnan(our_mtfa) else np.nan,
                "Paper_MTFA": p_mtfa,
                "Total_Alarms_SS_Datasets": tot_al,
                "Total_Alarms_All19": int(per_ds_df[per_ds_df["Method"] == m]["Alarms_Count"].sum()),
                "TP_Windowed": tot_tp,
                "FP_Windowed": tot_fp,
                "FN_Windowed": tot_fn,
                "Precision_Windowed": round(prec, 4),
                "Recall_Windowed": round(rec, 4),
                "F1_Windowed": round(f1, 4),
                "Mean_Delay_Steps": round(mean_delay_steps, 1) if not np.isnan(mean_delay_steps) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "table5_alarm_analysis_19datasets.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 3. Vẽ chuẩn hóa Fig. 7 (InsectsAbrupt) & Fig. 8 (Posture)
    fig7_path = RESULTS_DIR / "fig7_insects_abrupt_alarms.png"
    fig8_path = RESULTS_DIR / "fig8_posture_alarms.png"
    fig7_info = plot_standardized_alarm_figure("InsectsAbrupt", "Fig. 7", fig7_path)
    fig8_info = plot_standardized_alarm_figure("Posture", "Fig. 8", fig8_path)

    print(f"\n=== TABLE 5 ALARM ANALYSIS ({len(ss_alarmed_datasets)}/19 DATASETS WITH SS ALARMS) ===")
    print(f"SS-alarmed datasets: {ss_alarmed_datasets}")
    print(summary_df[["Method", "Our_MDR", "Paper_MDR", "Our_FAR", "Paper_FAR", "Our_MTD", "Paper_MTD", "Our_MTFA", "Paper_MTFA", "Total_Alarms_All19", "F1_Windowed"]].to_string(index=False))
    print("\n=== STANDARDIZED FIGS 7-8 (Seed T=42, S=142) ===")
    print("Fig. 7 InsectsAbrupt:", fig7_info)
    print("Fig. 8 Posture:", fig8_info)

    studd_row = summary_df[summary_df["Method"] == "STUDD"].iloc[0]
    log_fullscope_run(
        run_key="table5_alarms",
        run_name="fullscope-table5-alarms-and-figs7-8",
        job_type="table5_alarm_analysis",
        config={
            "scope": "19_datasets_alarm_analysis",
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
            "delta": DELTA,
            "ph_threshold": PH_THRESHOLD,
            "ks_alpha": KS_ALPHA,
        },
        summary_metrics={
            "ss_alarmed_datasets_count": len(ss_alarmed_datasets),
            "studd_our_mdr": float(studd_row["Our_MDR"]),
            "studd_our_far": float(studd_row["Our_FAR"]),
            "studd_our_mtd": float(studd_row["Our_MTD"]) if pd.notna(studd_row["Our_MTD"]) else None,
            "studd_total_alarms_all19": int(studd_row["Total_Alarms_All19"]),
            "fig7_insects_abrupt_studd_kappa": fig7_info["studd_kappa"],
            "fig7_insects_abrupt_studd_alarms": len(fig7_info["studd_alarms"]),
            "fig8_posture_studd_kappa": fig8_info["studd_kappa"],
            "fig8_posture_studd_alarms": len(fig8_info["studd_alarms"]),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        tables={
            "table5_alarm_analysis_19datasets": summary_df,
            "table5_per_dataset_alarms": per_ds_df,
        },
        images={
            "fig7_insects_abrupt_alarms": fig7_path,
            "fig8_posture_alarms": fig8_path,
        },
        enabled=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
