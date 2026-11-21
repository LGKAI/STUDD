"""Commit 3: Bayesian Sign Test (Figs. 11-12) và Hiệu suất sử dụng nhãn (Fig. 13) trên 19 bộ dữ liệu.

Sử dụng dữ liệu hợp nhất từ `results/studd_19datasets_summary.csv` (190 dòng = 19 bộ x 10 phương pháp)
để thực hiện:
    1. Benavoli et al. (2014, 2017) Bayesian Signed-Rank / Sign Test với ROPE = +/-5%, s = 1 tại z0 = 0
       và 50,000 mẫu Monte Carlo từ phân phối Dirichlet (seed = 42).
    2. Tính toán Labeling Efficiency (Kappa trung bình, Cost trung bình, tỷ lệ Kappa/Cost, biên Pareto).
    3. Sinh `results/fig11_12_bayesian_sign_test.png`, `results/fig13_label_efficiency.png` và log lên W&B.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fullscope_wandb import RESULTS_DIR, SEED_RNG, log_fullscope_run

METHOD_ORDER: Tuple[str, ...] = (
    "STUDD",
    "BL-st",
    "BL-ret",
    "SS",
    "DSS",
    "WS",
    "DWS",
    "OS",
    "OF",
    "FF",
)
BASELINES: Tuple[str, ...] = ("BL-st", "BL-ret", "SS", "DSS", "WS", "DWS", "OS", "OF", "FF")


def bayesian_sign_test_dirichlet(
    studd_vals: np.ndarray,
    other_vals: np.ndarray,
    *,
    higher_is_better: bool = True,
    rope: float = 0.05,
    prior_s: float = 1.0,
    n_samples: int = 50_000,
    seed: int = SEED_RNG,
) -> Dict[str, float]:
    """Thực hiện Bayesian Sign Test (Benavoli et al., 2014/2017) với ROPE đối xứng.

    Với độ đo 'higher_is_better=True' (Kappa): rel_diff = (studd - other) / max(|other|, 1e-9).
    Với độ đo 'higher_is_better=False' (Cost): rel_diff = (other - studd) / max(|other|, 1e-9)
    nên chiều `Win` luôn mang nghĩa `STUDD tốt hơn đối phương > 5%`.
    """
    a = np.asarray(studd_vals, dtype=np.float64)
    b = np.asarray(other_vals, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-9)
    raw_diff = (a - b) if higher_is_better else (b - a)
    rel_diff = np.where((a == 0.0) & (b == 0.0), 0.0, raw_diff / denom)

    n_lose = int(np.sum(rel_diff < -rope))
    n_rope = int(np.sum(np.abs(rel_diff) <= rope))
    n_win = int(np.sum(rel_diff > rope))

    alpha = np.array([n_lose + 1e-6, n_rope + prior_s, n_win + 1e-6], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(alpha, size=n_samples)

    mean_probs = draws.mean(axis=0)
    argmax_idx = np.argmax(draws, axis=1)
    simplex_win = float(np.mean(argmax_idx == 2))
    simplex_rope = float(np.mean(argmax_idx == 1))
    simplex_lose = float(np.mean(argmax_idx == 0))

    return {
        "n_datasets": len(a),
        "n_win": n_win,
        "n_rope": n_rope,
        "n_lose": n_lose,
        "mean_rel_diff": float(np.mean(rel_diff)),
        "median_rel_diff": float(np.median(rel_diff)),
        "P_Mean_Win": float(mean_probs[2]),
        "P_Mean_ROPE": float(mean_probs[1]),
        "P_Mean_Lose": float(mean_probs[0]),
        "Simplex_P_Win": simplex_win,
        "Simplex_P_ROPE": simplex_rope,
        "Simplex_P_Lose": simplex_lose,
    }


def compute_pareto_flags(mean_kappa: np.ndarray, mean_cost: np.ndarray) -> np.ndarray:
    """Xác định các điểm nằm trên biên Pareto (tối đa Kappa, tối thiểu Cost)."""
    n = len(mean_kappa)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (
                mean_kappa[j] >= mean_kappa[i]
                and mean_cost[j] <= mean_cost[i]
                and (mean_kappa[j] > mean_kappa[i] or mean_cost[j] < mean_cost[i])
            ):
                is_pareto[i] = False
                break
    return is_pareto


def analyze_bayesian_and_efficiency(summary_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy toàn bộ tính toán Bayesian Sign Test & Labeling Efficiency từ CSV 190 dòng."""
    df = pd.read_csv(summary_csv)
    if len(df) != 190:
        raise ValueError(f"Kỳ vọng 190 dòng (19 datasets x 10 methods), nhận được {len(df)}.")

    bayes_rows: List[Dict[str, object]] = []
    specs = [
        ("Our", "Kappa", "Our_Kappa", True),
        ("Paper", "Kappa", "Paper_Kappa", True),
        ("Our", "Cost", "Our_Cost", False),
        ("Paper", "Cost", "Paper_Cost", False),
    ]

    for source_label, metric_label, col_name, higher_is_better in specs:
        piv = df.pivot(index="Dataset", columns="Method", values=col_name)
        studd_vec = piv["STUDD"].to_numpy(dtype=np.float64)
        for baseline in BASELINES:
            other_vec = piv[baseline].to_numpy(dtype=np.float64)
            stats = bayesian_sign_test_dirichlet(
                studd_vec,
                other_vec,
                higher_is_better=higher_is_better,
                rope=0.05,
                prior_s=1.0,
                n_samples=50_000,
                seed=SEED_RNG,
            )
            bayes_rows.append(
                {
                    "Source": source_label,
                    "Metric": metric_label,
                    "Comparison": f"STUDD vs {baseline}",
                    "Baseline": baseline,
                    "Seed_RNG": SEED_RNG,
                    "ROPE": 0.05,
                    **stats,
                }
            )
    bayes_df = pd.DataFrame(bayes_rows)

    # Ranking & Labeling Efficiency
    piv_our_k = df.pivot(index="Dataset", columns="Method", values="Our_Kappa")
    piv_paper_k = df.pivot(index="Dataset", columns="Method", values="Paper_Kappa")
    piv_our_c = df.pivot(index="Dataset", columns="Method", values="Our_Cost")
    piv_paper_c = df.pivot(index="Dataset", columns="Method", values="Paper_Cost")

    our_k_rank = piv_our_k.rank(axis=1, ascending=False, method="average").mean(axis=0)
    paper_k_rank = piv_paper_k.rank(axis=1, ascending=False, method="average").mean(axis=0)
    our_c_rank = piv_our_c.rank(axis=1, ascending=True, method="average").mean(axis=0)
    paper_c_rank = piv_paper_c.rank(axis=1, ascending=True, method="average").mean(axis=0)

    eff_rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        ok = piv_our_k[method].to_numpy(dtype=np.float64)
        pk = piv_paper_k[method].to_numpy(dtype=np.float64)
        oc = piv_our_c[method].to_numpy(dtype=np.float64)
        pc = piv_paper_c[method].to_numpy(dtype=np.float64)

        our_mean_k, our_mean_c = float(np.mean(ok)), float(np.mean(oc))
        paper_mean_k, paper_mean_c = float(np.mean(pk)), float(np.mean(pc))
        eff_rows.append(
            {
                "Method": method,
                "Our_Mean_Kappa": round(our_mean_k, 4),
                "Paper_Mean_Kappa": round(paper_mean_k, 4),
                "Kappa_Rel_Diff_Pct": round(100.0 * (our_mean_k - paper_mean_k) / max(paper_mean_k, 1e-9), 2),
                "Our_Median_Kappa": round(float(np.median(ok)), 4),
                "Paper_Median_Kappa": round(float(np.median(pk)), 4),
                "Our_Mean_Cost": round(our_mean_c, 4),
                "Paper_Mean_Cost": round(paper_mean_c, 4),
                "Cost_Rel_Diff_Pct": round(100.0 * (our_mean_c - paper_mean_c) / max(paper_mean_c, 1e-9), 2),
                "Our_Median_Cost": round(float(np.median(oc)), 4),
                "Paper_Median_Cost": round(float(np.median(pc)), 4),
                "Our_Kappa_per_Cost": round(our_mean_k / max(our_mean_c, 1e-9), 4),
                "Paper_Kappa_per_Cost": round(paper_mean_k / max(paper_mean_c, 1e-9), 4),
                "Our_Mean_Pairwise_Kappa_per_Cost": round(float(np.mean(ok / np.maximum(oc, 1e-9))), 4),
                "Our_Mean_Rank_Kappa": round(float(our_k_rank[method]), 2),
                "Paper_Mean_Rank_Kappa": round(float(paper_k_rank[method]), 2),
                "Our_Mean_Rank_Cost": round(float(our_c_rank[method]), 2),
                "Paper_Mean_Rank_Cost": round(float(paper_c_rank[method]), 2),
            }
        )

    eff_df = pd.DataFrame(eff_rows)
    eff_df["Our_Pareto_Optimal"] = compute_pareto_flags(
        eff_df["Our_Mean_Kappa"].to_numpy(), eff_df["Our_Mean_Cost"].to_numpy()
    )
    eff_df["Paper_Pareto_Optimal"] = compute_pareto_flags(
        eff_df["Paper_Mean_Kappa"].to_numpy(), eff_df["Paper_Mean_Cost"].to_numpy()
    )
    return bayes_df, eff_df


def plot_bayesian_figs_11_12(bayes_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ Fig. 11 (Kappa) & Fig. 12 (Labeling Cost) so sánh Bayesian posterior Our vs Paper."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=160, sharex=True)
    panels = [
        (0, 0, "Our", "Kappa", "Fig. 11A · Our Reproduction: Predictive Kappa (ROPE=5%)"),
        (0, 1, "Paper", "Kappa", "Fig. 11B · Paper Reference: Predictive Kappa (ROPE=5%)"),
        (1, 0, "Our", "Cost", "Fig. 12A · Our Reproduction: Labeling Cost Reduction (ROPE=5%)"),
        (1, 1, "Paper", "Cost", "Fig. 12B · Paper Reference: Labeling Cost Reduction (ROPE=5%)"),
    ]

    for r_idx, c_idx, src, metric, title in panels:
        ax = axes[r_idx, c_idx]
        sub = bayes_df[(bayes_df["Source"] == src) & (bayes_df["Metric"] == metric)].set_index("Baseline")
        sub = sub.loc[list(BASELINES)]

        y_pos = np.arange(len(BASELINES))
        p_win = sub["Simplex_P_Win"].to_numpy()
        p_rope = sub["Simplex_P_ROPE"].to_numpy()
        p_lose = sub["Simplex_P_Lose"].to_numpy()

        ax.barh(y_pos, p_win, color="#16a34a", alpha=0.88, label="P(STUDD > Baseline)")
        ax.barh(y_pos, p_rope, left=p_win, color="#cbd5e1", alpha=0.95, label="P(ROPE |diff|<=5%)")
        ax.barh(y_pos, p_lose, left=p_win + p_rope, color="#dc2626", alpha=0.85, label="P(STUDD < Baseline)")

        for i, base in enumerate(BASELINES):
            w_cnt = int(sub.loc[base, "n_win"])
            r_cnt = int(sub.loc[base, "n_rope"])
            l_cnt = int(sub.loc[base, "n_lose"])
            ax.text(
                1.02,
                i,
                f"W/T/L={w_cnt}/{r_cnt}/{l_cnt} ({p_win[i]:.2f})",
                va="center",
                ha="left",
                fontsize=8.0,
                color="#1f2937",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"vs {b}" for b in BASELINES], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.38)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.axvline(0.5, color="#334155", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        if r_idx == 1:
            ax.set_xlabel("Bayesian Sign Test Simplex Probability (50,000 Dirichlet draws)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Bayesian Sign Test Across 19 Datasets (Benavoli et al. ROPE=+/-5%, RNG Seed={SEED_RNG})",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_label_efficiency_fig13(eff_df: pd.DataFrame, out_path: Path) -> None:
    """Vẽ Fig. 13: Hiệu suất sử dụng nhãn (Mean Labeling Cost vs Mean Kappa) + đường biên Pareto."""
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=160, sharex=True, sharey=True)
    specs = [
        (0, "Our_Mean_Cost", "Our_Mean_Kappa", "Our_Pareto_Optimal", "Fig. 13A · Our 19-Dataset Reproduction"),
        (1, "Paper_Mean_Cost", "Paper_Mean_Kappa", "Paper_Pareto_Optimal", "Fig. 13B · Paper Reference (Tables 3-4)"),
    ]

    for idx, cost_col, kappa_col, pareto_col, title in specs:
        ax = axes[idx]
        pareto_pts = eff_df[eff_df[pareto_col]].sort_values(cost_col)
        ax.plot(
            pareto_pts[cost_col],
            pareto_pts[kappa_col],
            linestyle="--",
            color="#2563eb",
            alpha=0.65,
            linewidth=1.5,
            label="Pareto Frontier",
        )

        for r in eff_df.itertuples(index=False):
            c_val = float(getattr(r, cost_col))
            k_val = float(getattr(r, kappa_col))
            is_studd = r.Method == "STUDD"
            is_par = bool(getattr(r, pareto_col))
            color = "#dc2626" if is_studd else ("#1d4ed8" if is_par else "#475569")
            size = 150 if is_studd else (95 if is_par else 70)
            marker = "*" if is_studd else ("D" if is_par else "o")
            ax.scatter([c_val], [k_val], s=size, color=color, marker=marker, zorder=5, edgecolors="white", linewidths=0.8)
            dy = 0.008 if r.Method not in ("DWS", "WS") else -0.012
            dx = 0.012 if c_val < 0.8 else -0.045
            ax.annotate(
                f"{r.Method}\n({k_val:.3f}, {c_val:.2f})",
                xy=(c_val, k_val),
                xytext=(c_val + dx, k_val + dy),
                fontsize=7.8,
                fontweight="bold" if is_studd else "normal",
                color=color,
            )

        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Labeling Cost (lower is better)", fontsize=9.5)
        if idx == 0:
            ax.set_ylabel("Mean Cohen's Kappa (higher is better)", fontsize=9.5)
        ax.set_xlim(-0.02, 1.05)
        ax.set_ylim(0.45, 0.72)

    fig.suptitle(
        "Section 5.7.1 / Fig. 13: Labeling Efficiency Across All 19 Benchmarks (Top-Left = Pareto Ideal)",
        fontsize=12.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wandb", action="store_true", help="Tắt đồng bộ W&B.")
    args = parser.parse_args()

    summary_csv = RESULTS_DIR / "studd_19datasets_summary.csv"
    bayes_df, eff_df = analyze_bayesian_and_efficiency(summary_csv)

    bayes_csv = RESULTS_DIR / "bayesian_sign_test_summary.csv"
    eff_csv = RESULTS_DIR / "label_efficiency_19datasets.csv"
    fig_bayes = RESULTS_DIR / "fig11_12_bayesian_sign_test.png"
    fig_eff = RESULTS_DIR / "fig13_label_efficiency.png"

    bayes_df.to_csv(bayes_csv, index=False)
    eff_df.to_csv(eff_csv, index=False)
    plot_bayesian_figs_11_12(bayes_df, fig_bayes)
    plot_label_efficiency_fig13(eff_df, fig_eff)

    print("=== LABELING EFFICIENCY (19 DATASETS) ===")
    print(
        eff_df[
            [
                "Method",
                "Our_Mean_Kappa",
                "Paper_Mean_Kappa",
                "Our_Mean_Cost",
                "Paper_Mean_Cost",
                "Our_Kappa_per_Cost",
                "Paper_Kappa_per_Cost",
                "Our_Pareto_Optimal",
            ]
        ].to_string(index=False)
    )
    print("\n=== BAYESIAN SIGN TEST HIGHLIGHTS (OUR REPRODUCTION) ===")
    our_b = bayes_df[bayes_df["Source"] == "Our"]
    print(
        our_b[
            ["Metric", "Baseline", "n_win", "n_rope", "n_lose", "Simplex_P_Win", "Simplex_P_ROPE", "Simplex_P_Lose"]
        ].to_string(index=False)
    )

    studd_row = eff_df[eff_df["Method"] == "STUDD"].iloc[0]
    ss_k_our = float(
        bayes_df[(bayes_df["Source"] == "Our") & (bayes_df["Metric"] == "Kappa") & (bayes_df["Baseline"] == "SS")]["Simplex_P_Win"].iloc[0]
    )
    ss_c_our = float(
        bayes_df[(bayes_df["Source"] == "Our") & (bayes_df["Metric"] == "Cost") & (bayes_df["Baseline"] == "SS")]["Simplex_P_Win"].iloc[0]
    )

    log_fullscope_run(
        run_key="bayesian_efficiency",
        run_name="fullscope-bayesian-and-efficiency",
        job_type="bayesian_and_efficiency",
        config={
            "scope": "19_datasets_10_methods",
            "rope": 0.05,
            "dirichlet_samples": 50_000,
            "prior_s": 1.0,
            "seed_rng": SEED_RNG,
        },
        summary_metrics={
            "studd_our_mean_kappa": float(studd_row["Our_Mean_Kappa"]),
            "studd_paper_mean_kappa": float(studd_row["Paper_Mean_Kappa"]),
            "studd_our_mean_cost": float(studd_row["Our_Mean_Cost"]),
            "studd_paper_mean_cost": float(studd_row["Paper_Mean_Cost"]),
            "studd_our_kappa_per_cost": float(studd_row["Our_Kappa_per_Cost"]),
            "studd_paper_kappa_per_cost": float(studd_row["Paper_Kappa_per_Cost"]),
            "bayes_our_kappa_p_win_vs_ss": round(ss_k_our, 4),
            "bayes_our_cost_p_win_vs_ss": round(ss_c_our, 4),
            "seed_rng": SEED_RNG,
        },
        tables={
            "bayesian_sign_test_summary": bayes_df,
            "label_efficiency_19datasets": eff_df,
        },
        images={
            "fig11_12_bayesian_sign_test": fig_bayes,
            "fig13_label_efficiency": fig_eff,
        },
        enabled=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
