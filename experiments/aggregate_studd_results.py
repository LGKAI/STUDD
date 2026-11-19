"""Build the main report and CSV from the three archived STUDD result sets."""

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


def read_results():
    khanh_report = (RESULTS / "khanh/khanh-7datasets.md").read_text(encoding="utf-8")
    khanh_links = dict(re.findall(
        r"^\| \*\*([^*]+)\*\* \|[^\n]*\[\w+\]\((https://[^ )]+/runs/[^ )]+)\) \|$",
        khanh_report,
        flags=re.MULTILINE,
    ))
    if khanh_links.keys() != DATASETS["Khánh"].keys():
        raise ValueError("Các link run theo dataset của Khánh không đầy đủ.")

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
            link = ""
            if owner == "Khoa":
                slug = DATASETS[owner][dataset]
                link = json.loads((RESULTS / f"khoa/{slug}_wandb.json").read_text())["url"]
                details = f"khoa/{slug}_reproduction.csv"
            elif owner == "Khánh":
                link = khanh_links[dataset]
                details = f"khanh/{dataset.lower()}_comparison_table.csv"
            else:
                details = f"{DATASETS[owner][dataset]}_reproduction.csv"
            if not (RESULTS / details).is_file():
                raise FileNotFoundError(RESULTS / details)
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
                "Alarms": row["Alarms"],
                "Retrains": row.get("Retrains", ""),
                "Protocol": protocol,
                "Cost_Basis": cost_basis,
                "Source_Revision": revision,
                "Summary_Source": f"results/{source}",
                "Dataset_Source": f"results/{details}",
                "WandB_Run": link,
            })
    if len({(row["Dataset"], row["Method"]) for row in records}) != 190:
        raise ValueError("Bảng hợp nhất cần 190 cặp dataset–phương pháp duy nhất.")
    return records


def write_report(records):
    lines = [
        "# Kết quả tái lập STUDD",
        "",
        "Bảng tổng hợp có **190 kết quả**, gồm **19 bộ dữ liệu × 10 phương pháp**. "
        "Lấy các đợt chạy đủ 10 phương pháp của ba nhánh. "
        "[CSV toàn bộ kết quả](studd_19datasets_summary.csv) · "
        "[Bài báo STUDD, Bảng 3–4](https://doi.org/10.1007/s10994-022-06188-7)",
        "",
        "- **Khoa:** [Report và 60 kết quả](khoa/README.md), W&B `lgkai-hcmus/Testing`.",
        "- **Khánh:** [Report và 70 kết quả](khanh/khanh-7datasets.md), "
        "[run tổng hợp](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/tekuwx8d).",
        "- **Đạt:** [60 kết quả](dat_6datasets_summary.csv), "
        "[thống kê cảnh báo](dat_6datasets_alarm_summary.csv). "
        "Nhánh nguồn dẫn tới CSV theo bộ. Bảng này chỉ điền URL run đã được ghi nhận.",
        "",
        "## Đối chiếu phương pháp STUDD",
        "",
        "| Dataset | Người chạy | N | W | Kappa nhóm / paper | Chi phí nhãn nhóm / paper | Cảnh báo | Nguồn |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        if row["Method"] != "STUDD":
            continue
        source = "[CSV](" + row["Dataset_Source"].removeprefix("results/") + ")"
        if row["WandB_Run"]:
            source += f" · [W&B]({row['WandB_Run']})"
        lines.append(
            f"| {row['Dataset']} | {row['Owner']} | {row['Samples']} | {row['Window_Size']} | "
            f"{float(row['Our_Kappa']):.4f} / {float(row['Paper_Kappa']):.2f} | "
            f"{float(row['Our_Cost']):.4f} / {float(row['Paper_Cost']):.2f} | {row['Alarms']} | {source} |"
        )
    lines += [
        "",
        "## Cách đọc số liệu",
        "",
        "- Cột `Our_Cost` giữ cách tính của lượt chạy gốc. "
        "Khánh đếm số lượt dùng nhãn trực tuyến, còn Khoa và Đạt nhân số lần huấn luyện lại với W. "
        "Cả hai cách cộng W mẫu ban đầu và chia cho N. "
        "`Our_Cost_1000_Reported` là cột quy đổi từ báo cáo của Khánh.",
        "- Cột `Retrains` để trống ở các dòng nguồn chưa cung cấp số lần huấn luyện lại. "
        "Các phép đo cảnh báo như MDR và FAR vẫn nằm trong báo cáo nguồn để giữ quy ước đo của từng nhóm.",
        "- Arabic và ArabicShuffled cần quyền truy cập input Kaggle riêng tư của Khoa. "
        "Xem các sai khác với paper về phiên bản dữ liệu và tiền xử lý trong báo cáo của từng người.",
        "",
        "Tạo lại bảng và report từ các tệp đã lưu bằng `python3 experiments/aggregate_studd_results.py`.",
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
