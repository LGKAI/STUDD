"""Commit 2: Kiểm chứng quy trình chọn thuật toán Teacher–Student 70/30 (Mục 5.6.2, Bảng 2).

Chạy trên cửa sổ khởi tạo W của toàn bộ 19 bộ dữ liệu với hạt giống đồng nhất:
    - Teacher seed = 42
    - Student seed = 142
Chia theo thứ tự thời gian 70% đầu (Train) / 30% sau (Validation) và chấm Cohen's Kappa
cho cả 7 họ mô hình {RF, XGB, RC, DT, LR, NB, SVC} trên tập Validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

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
from src.data_loader import ALL_19_DATASETS, load_all_19_stream
from src.models import PAPER_TABLE2_MODELS, SafeStreamClassifier, build_model

CANDIDATE_POOL: Tuple[str, ...] = ("RF", "XGB", "RC", "DT", "LR", "NB", "SVC")


def _build_fast_selector_model(code: str, random_state: int) -> SafeStreamClassifier:
    """Khởi tạo bộ phân loại theo pool 7 thuật toán; tắt Platt scaling cho SVC khi chỉ cần .predict()."""
    model = build_model(code, random_state=random_state, n_jobs=2)
    if code.upper().strip() in ("SVC", "SVM"):
        model.base_estimator.set_params(probability=False)
    return model


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    val = float(cohen_kappa_score(y_true, y_pred))
    return 0.0 if np.isnan(val) else val


def verify_single_dataset(dataset_name: str) -> Dict[str, object]:
    """Đánh giá chọn thuật toán 70/30 trên cửa sổ đầu W của một bộ dữ liệu."""
    X, y, meta = load_all_19_stream(dataset_name)
    canon = str(meta["canonical_name"])
    w = int(meta["recommended_window_size"])

    X_init, y_init = X[:w], y[:w]
    split_idx = int(w * 0.7)
    X_tr, X_val = X_init[:split_idx], X_init[split_idx:]
    y_tr, y_val = y_init[:split_idx], y_init[split_idx:]

    # 1. Chọn Teacher trên 70/30 với Teacher SEED = 42
    t_scores: Dict[str, float] = {}
    for code in CANDIDATE_POOL:
        clf = _build_fast_selector_model(code, random_state=SEED_TEACHER)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        t_scores[code] = _safe_kappa(y_val, preds)

    best_t_code = max(CANDIDATE_POOL, key=lambda c: (t_scores[c], -CANDIDATE_POOL.index(c)))

    key = canon.lower().replace("-", "").replace("_", "")
    paper_t_code, paper_s_code = PAPER_TABLE2_MODELS[key]

    # 2. Chọn Student bắt chước Our_Best_Teacher (SEED_STUDENT = 142)
    best_teacher = _build_fast_selector_model(best_t_code, random_state=SEED_TEACHER)
    best_teacher.fit(X_tr, y_tr)
    yhat_tr = best_teacher.predict(X_tr)
    yhat_val = best_teacher.predict(X_val)

    s_scores: Dict[str, float] = {}
    for code in CANDIDATE_POOL:
        clf = _build_fast_selector_model(code, random_state=SEED_STUDENT)
        clf.fit(X_tr, yhat_tr)
        s_preds = clf.predict(X_val)
        s_scores[code] = _safe_kappa(yhat_val, s_preds)
    best_s_code = max(CANDIDATE_POOL, key=lambda c: (s_scores[c], -CANDIDATE_POOL.index(c)))

    # 3. Kiểm tra thêm chọn Student khi cố định Paper Teacher (để lý giải Bảng 2 công bằng hơn)
    if paper_t_code == best_t_code:
        s_scores_given_paper_t = s_scores
    else:
        p_teacher = _build_fast_selector_model(paper_t_code, random_state=SEED_TEACHER)
        p_teacher.fit(X_tr, y_tr)
        p_yhat_tr = p_teacher.predict(X_tr)
        p_yhat_val = p_teacher.predict(X_val)
        s_scores_given_paper_t = {}
        for code in CANDIDATE_POOL:
            clf = _build_fast_selector_model(code, random_state=SEED_STUDENT)
            clf.fit(X_tr, p_yhat_tr)
            s_scores_given_paper_t[code] = _safe_kappa(p_yhat_val, clf.predict(X_val))
    best_s_given_paper_t = max(
        CANDIDATE_POOL, key=lambda c: (s_scores_given_paper_t[c], -CANDIDATE_POOL.index(c))
    )

    row: Dict[str, object] = {
        "Dataset": canon,
        "W": w,
        "Train_Size": split_idx,
        "Val_Size": w - split_idx,
        "Seed_Teacher": SEED_TEACHER,
        "Seed_Student": SEED_STUDENT,
        "Paper_Teacher": paper_t_code,
        "Our_Best_Teacher": best_t_code,
        "Teacher_Exact_Match": bool(best_t_code == paper_t_code),
        "Paper_Teacher_Val_Kappa": round(t_scores[paper_t_code], 4),
        "Our_Best_Teacher_Val_Kappa": round(t_scores[best_t_code], 4),
        "Teacher_Kappa_Gap": round(t_scores[best_t_code] - t_scores[paper_t_code], 4),
        "Paper_Student": paper_s_code,
        "Our_Best_Student": best_s_code,
        "Best_Student_Given_Paper_T": best_s_given_paper_t,
        "Student_Exact_Match": bool(best_s_code == paper_s_code),
        "Student_Match_Given_Paper_T": bool(best_s_given_paper_t == paper_s_code),
        "Paper_Student_Val_Kappa_Given_Paper_T": round(s_scores_given_paper_t[paper_s_code], 4),
        "Best_Student_Val_Kappa_Given_Paper_T": round(s_scores_given_paper_t[best_s_given_paper_t], 4),
    }
    for code in CANDIDATE_POOL:
        row[f"T_Kappa_{code}"] = round(t_scores[code], 4)
    for code in CANDIDATE_POOL:
        row[f"S_Kappa_{code}"] = round(s_scores[code], 4)
    return row


def plot_selection_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ heatmap Validation Kappa của 7 họ mô hình Teacher trên 19 bộ dữ liệu."""
    cols = [f"T_Kappa_{c}" for c in CANDIDATE_POOL]
    mat = df[cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 8.2), dpi=160)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(CANDIDATE_POOL)))
    ax.set_xticklabels(CANDIDATE_POOL, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(df)))
    ylabels = [
        f"{r.Dataset} (Paper: {r.Paper_Teacher}/{r.Paper_Student} | Our: {r.Our_Best_Teacher}/{r.Our_Best_Student})"
        for r in df.itertuples(index=False)
    ]
    ax.set_yticklabels(ylabels, fontsize=8.5)

    for i, r in enumerate(df.itertuples(index=False)):
        for j, code in enumerate(CANDIDATE_POOL):
            val = mat[i, j]
            txt_color = "white" if val >= 0.58 else "#111827"
            suffix = ""
            if code == r.Paper_Teacher and code == r.Our_Best_Teacher:
                suffix = "\n[P=O]"
            elif code == r.Paper_Teacher:
                suffix = "\n[Paper]"
            elif code == r.Our_Best_Teacher:
                suffix = "\n[Our]"
            ax.text(
                j,
                i,
                f"{val:.3f}{suffix}",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=7.2,
                fontweight="bold" if suffix else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Validation Kappa (70/30 Chronological Split on Initial Window W)", fontsize=9)
    ax.set_title(
        "Section 5.6.2 Verification: 7-Family Validation Kappa Across 19 Datasets\n"
        f"(Teacher Seed={SEED_TEACHER}, Student Seed={SEED_STUDENT}; [P]=Paper Table 2, [O]=Our Max Val Kappa)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wandb", action="store_true", help="Tắt đồng bộ W&B nếu cần chạy offline.")
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for idx, ds in enumerate(ALL_19_DATASETS, 1):
        row = verify_single_dataset(ds)
        rows.append(row)
        print(
            f"[{idx:2d}/19] {row['Dataset']:15s} W={row['W']:4d} | "
            f"Teacher Paper={row['Paper_Teacher']:3s} ({row['Paper_Teacher_Val_Kappa']:.4f}) vs "
            f"Our={row['Our_Best_Teacher']:3s} ({row['Our_Best_Teacher_Val_Kappa']:.4f}) | "
            f"Student Paper={row['Paper_Student']:3s} vs Our={row['Our_Best_Student']:3s}"
        )

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "table2_selection_verification.csv"
    out_png = RESULTS_DIR / "fig_table2_selection_heatmap.png"
    df.to_csv(out_csv, index=False)
    plot_selection_heatmap(df, out_png)

    t_matches = int(df["Teacher_Exact_Match"].sum())
    s_matches = int(df["Student_Exact_Match"].sum())
    s_given_p_matches = int(df["Student_Match_Given_Paper_T"].sum())
    mean_t_gap = float(df["Teacher_Kappa_Gap"].mean())
    median_t_gap = float(df["Teacher_Kappa_Gap"].median())

    print("\n=== TỔNG HỢP KIỂM CHỨNG MỤC 5.6.2 (TABLE 2) ===")
    print(f"Teacher exact match: {t_matches}/19")
    print(f"Student exact match: {s_matches}/19 (Given Paper Teacher: {s_given_p_matches}/19)")
    print(f"Mean Validation Kappa Gap (Our_Best - Paper_Teacher): {mean_t_gap:.4f} (median={median_t_gap:.4f})")

    log_fullscope_run(
        run_key="table2_selection",
        run_name="fullscope-table2-selection-verification",
        job_type="table2_selection_verification",
        config={
            "scope": "19_datasets_table2_selection",
            "split": "70_30_chronological",
            "candidate_pool": list(CANDIDATE_POOL),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        summary_metrics={
            "datasets_count": len(df),
            "teacher_exact_match_count": t_matches,
            "student_exact_match_count": s_matches,
            "student_match_given_paper_teacher_count": s_given_p_matches,
            "mean_teacher_kappa_gap": round(mean_t_gap, 4),
            "median_teacher_kappa_gap": round(median_t_gap, 4),
            "seed_teacher": SEED_TEACHER,
            "seed_student": SEED_STUDENT,
        },
        tables={"table2_selection_verification": df},
        images={"fig_table2_selection_heatmap": out_png},
        enabled=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
