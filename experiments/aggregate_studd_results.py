"""Build the main report and CSV from the three archived STUDD result sets."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ("BL-st", "BL-ret", "STUDD", "SS", "DSS", "WS", "DWS", "OS", "OF", "FF")
SOURCES = {
    "Khoa": (
        "khoa/khoa_6datasets_summary.csv",
        "0262de5",
        "Page–Hinkley và baseline của Đạt, Kaggle CPU, seed 42",
        "(W + W × số lần huấn luyện lại) / N",
    ),
    "Khánh": (
        "khanh/khanh_7datasets_summary.csv",
        "cfb5db6",
        "Bộ chạy của Khánh, kiểm tra từng mẫu",
        "(W + số lượt dùng nhãn trực tuyến) / N",
    ),
    "Đạt": (
        "dat_6datasets_summary.csv",
        "00834c3",
        "Page–Hinkley và baseline của Đạt",
        "(W + W × số lần huấn luyện lại) / N",
    ),
}
DATASETS = {
    "Khoa": {
        "InsectsAbrupt": "insects-abrupt",
        "Insects": "insects",
        "Keystroke": "keystroke",
        "Arabic": "arabic",
        "ArabicShuffled": "arabic-shuffled",
        "Posture": "posture",
    },
    "Khánh": {
        "Bike": 17_379,
        "Luxembourg": 1_901,
        "Outdoor": 4_000,
        "Rialto": 82_250,
        "Gas": 13_910,
        "Ozone": 1_847,
        "GMSC": 150_000,
    },
    "Đạt": {
        "Electricity": "electricity",
        "Powersupply": "powersupply",
        "NOAA": "noaa",
        "CoverType": "covertype",
        "Poker": "poker",
        "Sensors": "sensors",
    },
}


# Run URLs are documented in the Dat branch README at e6b7dbd.
DAT_WANDB_ROOT = "https://wandb.ai/lgkai-hcmus/STUDD/runs"
DAT_RUN_IDS = {
    "Electricity": "vlj7zc8h",
    "Powersupply": "pkbdf0np",
    "NOAA": "lyhf11v9",
    "CoverType": "r6s8kd07",
    "Poker": "pub91li5",
    "Sensors": "kd3shf5o",
}
DAT_SUMMARY_RUN_ID = "iuigzb01"
KHANH_SUMMARY_RUN_URL = "https://wandb.ai/lgkai-hcmus/STUDD/runs/tekuwx8d"


def normalize_wandb_url(url: str) -> str:
    """Normalize CoreWeave-mirrored W&B links to canonical wandb.ai run URLs."""
    if not url:
        return ""
    cleaned = url.strip().replace("https://forge.coreweave.com/wandb/", "https://wandb.ai/")
    if cleaned.endswith("/overview"):
        cleaned = cleaned[: -len("/overview")]
    return cleaned


def compute_rank_summary(records):
    """Compute mean Kappa/Cost and standard average ranks across all 19 datasets."""
    by_dataset = defaultdict(dict)
    for row in records:
        by_dataset[row["Dataset"]][row["Method"]] = row

    datasets = sorted(by_dataset.keys())
    n_datasets = len(datasets)
    if n_datasets == 0:
        raise ValueError("Không có dữ liệu để tính xếp hạng trung bình.")

    def _avg_ranks(metric_key: str, descending: bool) -> dict[str, float]:
        totals = defaultdict(float)
        for dataset in datasets:
            m_dict = by_dataset[dataset]
            values = [(method, float(m_dict[method][metric_key])) for method in METHODS]
            values.sort(key=lambda item: item[1], reverse=descending)
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[j + 1][1] == values[i][1]:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2.0
                for k in range(i, j + 1):
                    totals[values[k][0]] += avg_rank
                i = j + 1
        return {method: totals[method] / n_datasets for method in METHODS}

    our_perf_rank = _avg_ranks("Our_Kappa", descending=True)
    paper_perf_rank = _avg_ranks("Paper_Kappa", descending=True)
    our_cost_rank = _avg_ranks("Our_Cost", descending=False)
    paper_cost_rank = _avg_ranks("Paper_Cost", descending=False)

    summary = []
    for method in METHODS:
        our_mean_kappa = sum(float(by_dataset[ds][method]["Our_Kappa"]) for ds in datasets) / n_datasets
        paper_mean_kappa = sum(float(by_dataset[ds][method]["Paper_Kappa"]) for ds in datasets) / n_datasets
        our_mean_cost = sum(float(by_dataset[ds][method]["Our_Cost"]) for ds in datasets) / n_datasets
        paper_mean_cost = sum(float(by_dataset[ds][method]["Paper_Cost"]) for ds in datasets) / n_datasets
        summary.append({
            "Method": method,
            "Our_Mean_Kappa": our_mean_kappa,
            "Paper_Mean_Kappa": paper_mean_kappa,
            "Our_Mean_Cost": our_mean_cost,
            "Paper_Mean_Cost": paper_mean_cost,
            "Our_Rank_Perf": our_perf_rank[method],
            "Paper_Rank_Perf": paper_perf_rank[method],
            "Our_Rank_Cost": our_cost_rank[method],
            "Paper_Rank_Cost": paper_cost_rank[method],
        })
    return summary


def read_results():
    khanh_report = (RESULTS / "khanh/khanh-7datasets.md").read_text(encoding="utf-8")
    khanh_links = dict(re.findall(
        r"^\| \*\*([^*]+)\*\* \|[^\n]*\[\w+\]\((https://[^ )]+/runs/[^ )]+)\) \|$",
        khanh_report,
        flags=re.MULTILINE,
    ))
    if khanh_links.keys() != DATASETS["Khánh"].keys():
        raise ValueError("Các link run theo dataset của Khánh không đầy đủ.")

    if DAT_RUN_IDS.keys() != DATASETS["Đạt"].keys():
        raise ValueError("Các link run theo dataset của Đạt không đầy đủ.")

    records = []
    for owner, (source, revision, protocol, cost_basis) in SOURCES.items():
        with (RESULTS / source).open(encoding="utf-8-sig", newline="") as stream:
            data = list(csv.DictReader(stream))

        expected = {(dataset, method) for dataset in DATASETS[owner] for method in METHODS}
        actual = {(row["Dataset"], row["Method"]) for row in data}
        if actual != expected or len(data) != len(expected):
            raise ValueError(f"{source}: sai tập dataset–phương pháp hoặc có dòng trùng.")

        for row in sorted(data, key=lambda r: (r["Dataset"].casefold(), METHODS.index(r["Method"]))):
            dataset = row["Dataset"]
            if owner == "Khoa":
                slug = DATASETS[owner][dataset]
                link = json.loads((RESULTS / f"khoa/{slug}_wandb.json").read_text(encoding="utf-8"))["url"]
                details = f"khoa/{slug}_reproduction.csv"
            elif owner == "Khánh":
                link = khanh_links[dataset]
                details = f"khanh/{dataset.lower()}_comparison_table.csv"
            else:
                link = f"{DAT_WANDB_ROOT}/{DAT_RUN_IDS[dataset]}"
                details = f"{DATASETS[owner][dataset]}_reproduction.csv"
            if not (RESULTS / details).is_file():
                raise FileNotFoundError(RESULTS / details)

            # Normalize Alarms & Retrains schema across all 3 branches:
            # In Khanh's runner (experiments/run_dataset.py), BL-ret records periodic retrains
            # in the Alarms column (0 drift alarms), while all other 9 methods perform 1 retrain
            # per drift alarm (BL-st = 0). Aligning with Khoa & Dat yields zero empty Retrains.
            if owner == "Khánh":
                if row["Method"] == "BL-ret":
                    alarms = "0"
                    retrains = row["Alarms"]
                else:
                    alarms = row["Alarms"]
                    retrains = row["Alarms"]
            else:
                alarms = row["Alarms"]
                retrains = row.get("Retrains", "")

            normalized_link = normalize_wandb_url(link)
            records.append({
                "Dataset": dataset,
                "Method": row["Method"],
                "Owner": owner,
                "Samples": row.get("Samples") or DATASETS[owner][dataset],
                "Window_Size": row.get("Window_Size") or row["W"],
                "Teacher": row["Teacher"],
                "Student": row["Student"] if row["Method"] == "STUDD" else "",
                "Our_Accuracy": row.get("Our_Accuracy") or row["Accuracy"],
                "Our_Kappa": row["Our_Kappa"],
                "Paper_Kappa": row["Paper_Kappa"],
                "Kappa_Diff": row["Kappa_Diff"],
                "Our_Cost": row.get("Our_Cost") or row["Our_Cost_W"],
                "Paper_Cost": row["Paper_Cost"],
                "Our_Cost_1000_Reported": row.get("Our_Cost_1000", ""),
                "Alarms": alarms,
                "Retrains": retrains,
                "Protocol": protocol,
                "Cost_Basis": cost_basis,
                "Source_Revision": revision,
                "Summary_Source": f"results/{source}",
                "Dataset_Source": f"results/{details}",
                "WandB_Run": normalized_link,
            })

    if len({(row["Dataset"], row["Method"]) for row in records}) != 190:
        raise ValueError("Bảng hợp nhất cần 190 cặp dataset–phương pháp duy nhất.")
    if any(row["Retrains"] == "" or row["Alarms"] == "" for row in records):
        raise ValueError("Phát hiện dòng thiếu Alarms hoặc Retrains sau chuẩn hóa.")
    if any("forge.coreweave.com" in row["WandB_Run"] for row in records):
        raise ValueError("Phát hiện URL W&B chưa được chuẩn hóa về wandb.ai.")
    return records


def write_report(records):
    rank_summary = compute_rank_summary(records)
    studd_rank = next(r for r in rank_summary if r["Method"] == "STUDD")
    bl_st_rank = next(r for r in rank_summary if r["Method"] == "BL-st")
    bl_ret_rank = next(r for r in rank_summary if r["Method"] == "BL-ret")

    lines = [
        "# Kết quả tái lập STUDD (19 bộ dữ liệu × 10 phương pháp)",
        "",
        "Bảng tổng hợp lưu trữ **190 kết quả**, gồm **19 bộ dữ liệu × 10 phương pháp** từ ba nhánh thực nghiệm của nhóm. "
        "Toàn bộ số liệu đo lường gốc (`Our_Kappa`, `Our_Accuracy`, `Our_Cost`, `Paper_Kappa`, `Paper_Cost`) được bảo toàn nguyên vẹn theo đúng nguồn lưu trữ. "
        "[CSV toàn bộ 190 kết quả](studd_19datasets_summary.csv) · "
        "[Bài báo STUDD, Bảng 3–4](https://doi.org/10.1007/s10994-022-06188-7)",
        "",
        "- **Khoa (6 bộ dữ liệu):** [Report và 60 kết quả](khoa/README.md), "
        "[CSV 60 kết quả](khoa/khoa_6datasets_summary.csv), W&B [`lgkai-hcmus/Testing`](https://wandb.ai/lgkai-hcmus/Testing).",
        "- **Khánh (7 bộ dữ liệu):** [Report và 70 kết quả](khanh/khanh-7datasets.md), "
        "[CSV 70 kết quả](khanh/khanh_7datasets_summary.csv), "
        f"[run tổng hợp W&B]({KHANH_SUMMARY_RUN_URL}).",
        "- **Đạt (6 bộ dữ liệu):** [CSV 60 kết quả](dat_6datasets_summary.csv), "
        "[thống kê cảnh báo MDR/FAR/MTD/MTFA](dat_6datasets_alarm_summary.csv) và "
        f"[run tổng hợp W&B]({DAT_WANDB_ROOT}/{DAT_SUMMARY_RUN_ID}).",
        "",
        "## Đối chiếu phương pháp STUDD trên 19 bộ dữ liệu",
        "",
        "| Dataset | Người chạy | Teacher / Student | N | W | Kappa nhóm / paper | Chi phí nhãn nhóm / paper | Cảnh báo | Nguồn |",
        "| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        if row["Method"] != "STUDD":
            continue
        source = "[CSV](" + row["Dataset_Source"].removeprefix("results/") + ")"
        if row["WandB_Run"]:
            source += f" · [W&B]({row['WandB_Run']})"
        pair = f"{row['Teacher']} / {row['Student']}"
        lines.append(
            f"| {row['Dataset']} | {row['Owner']} | {pair} | {row['Samples']} | {row['Window_Size']} | "
            f"{float(row['Our_Kappa']):.4f} / {float(row['Paper_Kappa']):.2f} | "
            f"{float(row['Our_Cost']):.4f} / {float(row['Paper_Cost']):.2f} | {row['Alarms']} | {source} |"
        )

    lines += [
        "",
        "## Xếp hạng trung bình (Average Rank) của 10 phương pháp trên cả 19 bộ dữ liệu — Tái hiện Hình 10",
        "",
        "Xếp hạng trung bình (average tie-breaking theo chuẩn Friedman; **nhỏ hơn = tốt hơn** trên cả hai trục Hiệu năng và Chi phí nhãn) "
        "được tính trực tiếp từ toàn bộ 190 dòng kết quả và đối chiếu song song với Bảng 3–4 của bài báo gốc:",
        "",
        "| Method | Mean Kappa (nhóm / paper) | Avg Rank Hiệu năng (nhóm / paper) | Avg Rank Chi phí (nhóm / paper) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for item in rank_summary:
        method = f"**`{item['Method']}`**" if item["Method"] == "STUDD" else f"`{item['Method']}`"
        lines.append(
            f"| {method} | {item['Our_Mean_Kappa']:.4f} / {item['Paper_Mean_Kappa']:.4f} | "
            f"{item['Our_Rank_Perf']:.2f} / {item['Paper_Rank_Perf']:.2f} | "
            f"{item['Our_Rank_Cost']:.2f} / {item['Paper_Rank_Cost']:.2f} |"
        )

    lines += [
        "",
        f"Trên cả 19 bộ dữ liệu, **`STUDD` đạt thứ hạng chi phí nhãn thấp thứ 2 toàn bảng** "
        f"(`{studd_rank['Our_Rank_Cost']:.2f}` so với `{studd_rank['Paper_Rank_Cost']:.2f}` trong bài báo), "
        f"chỉ đứng sau đường cơ sở tĩnh không cập nhật `BL-st` (`{bl_st_rank['Our_Rank_Cost']:.2f}` / `{bl_st_rank['Paper_Rank_Cost']:.2f}`), "
        f"tiết kiệm nhãn vượt trội so với `BL-ret` (`{bl_ret_rank['Our_Rank_Cost']:.2f}` / `{bl_ret_rank['Paper_Rank_Cost']:.2f}`), `FF` và `SS`, "
        f"đồng thời giữ Mean Kappa `{studd_rank['Our_Mean_Kappa']:.4f}` (paper: `{studd_rank['Paper_Mean_Kappa']:.4f}`) "
        f"và cấu trúc đánh đổi Hiệu năng–Chi phí bám sát Hình 10 của bài báo.",
        "",
        "![Average Rank 19 datasets](fig_19datasets_rank_plot.png)",
        "",
        "## So sánh protocol giữa 3 nhánh thành viên",
        "",
        "Bảng sau minh bạch hóa các khác biệt kỹ thuật giữa 3 nhánh thực nghiệm đã lưu trữ (không chỉnh sửa hồi tố số liệu đo cũ):",
        "",
        "| Thành viên (Số bộ) | Biến thể Page–Hinkley | `delta` cho nhóm giám sát (`SS/DSS/WS/DWS`) | Buffer tái huấn luyện Student (`STUDD`) | Công thức `Our_Cost` | Dự án W&B |",
        "| :--- | :--- | :---: | :--- | :--- | :--- |",
        "| **Khoa** (6 bộ) | `src/detector.py` (nhánh Đạt): `PH = sum - min_sum`, `alpha = 0.9999` (bản lưu cũ chưa chặn dưới `0.0`) | `0.001` | Cửa sổ `W` mẫu gần nhất `[t-W+1 : t+1]` | `(W + W × Retrains) / N` | `lgkai-hcmus/Testing` |",
        "| **Khánh** (7 bộ) | `SkMultiflowPageHinkley`: `sum = max(0.0, alpha * sum + (x - mean - delta))`, `alpha = 0.9999` | `0.002` (`STUDD`: `0.001`) | Buffer online tích lũy `X[W : t+1]` (`upd_std_model=True`) | `(W + samples_used_online) / N` *(kèm `Our_Cost_1000_Reported`)* | `lgkai-hcmus/STUDD` |",
        "| **Đạt** (6 bộ) | `src/detector.py`: `PH = sum - min_sum`, `alpha = 0.9999` (bản lưu cũ chưa chặn dưới `0.0`) | `0.001` | Cửa sổ `W` mẫu gần nhất `[t-W+1 : t+1]` | `(W + W × Retrains) / N` | `lgkai-hcmus/STUDD` |",
        "",
        "- **Chuẩn hóa schema trong `studd_19datasets_summary.csv`:** Toàn bộ 190 dòng có đủ hai cột `Alarms` và `Retrains` (không để trống). "
        "Cụ thể, với `BL-ret` trên 7 bộ của Khánh, số lần tái huấn luyện định kỳ (`7, 5, 73, 2, 6, 2, 40`) được chuyển sang `Retrains` và đặt `Alarms = 0` "
        "để thống nhất với Khoa và Đạt; 9 phương pháp còn lại của Khánh đều tái huấn luyện 1 lần cho mỗi cảnh báo nên `Retrains = Alarms`. "
        "Mọi đường dẫn W&B đều được chuẩn hóa về tên miền chính thức `https://wandb.ai/...`.",
        "",
        "## Phân tích sai lệch tổng hợp (19 bộ dữ liệu)",
        "",
        "1. **Khác biệt `W = 2000` (văn bản Mục 5.7) so với `W = 1000` (script gốc `vcerqueira/studd/experiments/main.py` & Bảng 4):** "
        "Văn bản Mục 5.7 quy định `W = 2000` cho các luồng lớn và `W = 500` cho 6 luồng nhỏ, nhưng trong mã nguồn `experiments/main.py` của tác giả, "
        "nhánh luồng lớn đặt `n_train_obs = 1000` và danh sách `small_data_streams` viết thường (`'ozone'`, `'outdoor'`, `'luxembourg'`) khiến 3 bộ viết hoa cũng rơi về `W = 1000`. "
        "Chính sự khác biệt kích thước cửa sổ này giải thích trực tiếp độ chênh cột `Cost` trên `Arabic`, `ArabicShuffled`, `Bike`, `NOAA`, `Electricity`, `Gas`, `Powersupply`, `Rialto` "
        "(chạy `W = 2000` cho chi phí cao gấp ~2× so với `W = 1000`) và trên `Luxembourg`, `Ozone`, `Outdoor` (chạy `W = 500` cho chi phí khởi tạo bằng ~1/2 Bảng 4). "
        "Cột `Our_Cost_1000_Reported` trong báo cáo của Khánh xác nhận khi quy đổi về đơn vị batch gốc, số liệu khớp sát Bảng 4.",
        "2. **Hiện tượng 0 cảnh báo ở các cặp mô hình đồng nhất (Teacher = Student):** "
        "Trên `InsectsAbrupt` (`RF/RF`, Kappa `0.5770` vs `0.73`, −20.96%) và `Arabic` (`XGB/XGB`, Kappa `0.6441` vs `0.71`, −9.28%) ở đợt chạy lưu trữ của Khoa, "
        "Teacher và Student dùng chung `random_state = 42` trên cùng họ mô hình nên hội tụ gần như trùng khớp ($L_t \\approx 0$). "
        "Hiện tượng tương tự xảy ra trên `Poker` (`DT/DT`, Kappa `0.2960` vs `0.48`, 0 cảnh báo) của Đạt, "
        "cũng như *Idempotent Mimicry Trap* mà Khánh ghi nhận trên các cặp tất định `XGB/XGB` (`Rialto`, `Ozone`, `GMSC`): "
        "sau lần huấn luyện lại đầu tiên, Student sao chép gần như nguyên vẹn hàm quyết định của Teacher ($S(x) \\equiv T(x)$), làm sai số bắt chước triệt tiêu và vô hiệu hóa Page–Hinkley.",
        "3. **Báo động dư trên các luồng dài (`Sensors`, `CoverType`, `Posture`, `Electricity`):** "
        "Ở bản `src/detector.py` dùng trong đợt chạy lưu trữ của Khoa và Đạt, thống kê `sum - min_sum` với hệ số quên `alpha = 0.9999` chưa chặn dưới `0.0` như `scikit-multiflow`. "
        "Khi `sum` âm sâu rồi suy giảm về `0` trên luồng dài (`45k–150k` mẫu), hiệu `sum - min_sum` trôi dương nhân tạo và sinh thêm cảnh báo trên "
        "`Sensors` (35 alarms, Cost `0.48` vs `0.16`), `CoverType` (20 alarms, Cost `0.28` vs `0.09`), `Posture` (11 alarms, Cost `0.16` vs `0.02`; so với 8 alarms khi Khoa chạy bằng `scikit-multiflow 0.5.3`) "
        "và `Electricity` (12 alarms, Cost `0.57` vs `0.43`). Lỗi này đã được sửa trong nhánh mã nguồn cho các lần tái chạy về sau, đồng thời số liệu đo cũ được giữ nguyên để đảm bảo minh bạch nguồn gốc (provenance).",
        "4. **Sai khác phiên bản dữ liệu nguồn và tiền xử lý:** "
        "`Gas` từ kho nguồn công khai có `128` đặc trưng cảm biến thay vì `16` đặc trưng rút gọn (`17` cột trong Bảng 1), giúp Kappa cao hơn bài báo nhưng vẫn khớp Cost (`0.8627` vs `0.86`); "
        "`Ozone` bản `mstz/ozone` (`8hr`) có `1,847 × 72` thay vì `2,574 × 32`; "
        "`GMSC` bản công khai (`cs-training.csv`) mất cân bằng nặng (`93.3% / 6.7%`) kèm `33,655` giá trị `NaN` (được điền median cục bộ theo cửa sổ `W` để chống rò rỉ dữ liệu); "
        "`Keystroke` có `10` đặc trưng thay vì `9`; "
        "`Arabic` và `ArabicShuffled` có `26` đặc trưng thay vì `27` và hiện lưu ở Kaggle Dataset riêng tư của Khoa (`phmchnkhoa/studd-arabic-benchmark-inputs`), cần công khai hóa hoặc mirror sang Hugging Face để tái lập hoàn toàn mở.",
        "",
        "## Phạm vi giữa kỳ & Phần mở rộng cho Giai đoạn 3",
        "",
        "- **Đã hoàn thành cho báo cáo giữa kỳ:**",
        "  - Tái lập đầy đủ **19 bộ dữ liệu × 10 phương pháp = 190 kết quả** đối chiếu Bảng 3 (Cohen's Kappa) và Bảng 4 (Labeling Cost), đồng bộ log thực nghiệm trên W&B.",
        "  - Tái hiện **Hình 10** (biểu đồ Average Rank đánh đổi Hiệu năng vs. Chi phí nhãn trên toàn bộ 19 bộ dữ liệu).",
        "  - Đánh giá các độ đo cảnh báo (`MDR`, `FAR`, `MTD`, `MTFA`) và `Labeling_Efficiency` cho 6 bộ dữ liệu của Đạt (`dat_6datasets_alarm_summary.csv`), cùng chuỗi đồ thị Rolling Kappa (đối chiếu Hình 7–9) cho 7 bộ của Khánh.",
        "- **Kế hoạch mở rộng cho Giai đoạn 3 (Báo cáo cuối kỳ / Poster phản biện):**",
        "  - Tính đầy đủ **Bảng 5 (`MDR`, `FAR`, `MTD`, `MTFA`)** và **Labeling Efficiency** thống nhất trên toàn bộ cả 19 bộ dữ liệu.",
        "  - Thực hiện kiểm định thống kê **Bayesian Sign Test** (tái hiện Hình 11–12 của bài báo) trên cả 19 bộ dữ liệu.",
        "  - Thực hiện **Sensitivity Analysis** theo lưới tham số `l_access × l_delay` cho câu hỏi nghiên cứu **RQ3**.",
        "  - Chạy đa hạt giống (**multi-seed runs**) và tái chạy đồng nhất cả 19 bộ trên bộ phát hiện `PageHinkley` đã sửa và tách seed Teacher/Student nhằm kiểm chứng sâu *Idempotent Mimicry Trap* (Stress Testing).",
        "",
        "---",
        "",
        "Tạo lại bảng tổng hợp, báo cáo và biểu đồ xếp hạng từ các tệp đã lưu bằng:",
        "",
        "```bash",
        "python3 experiments/aggregate_studd_results.py",
        "python3 experiments/plot_19datasets_rank.py",
        "```",
    ]
    (RESULTS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    records = read_results()
    with (RESULTS / "studd_19datasets_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    write_report(records)
    print(f"Wrote {len(records)} results across {len({r['Dataset'] for r in records})} datasets.")


if __name__ == "__main__":
    main()
