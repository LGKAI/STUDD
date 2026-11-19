# Kết quả tái lập STUDD

Bảng tổng hợp có **190 kết quả**, gồm **19 bộ dữ liệu × 10 phương pháp**. Lấy các đợt chạy đủ 10 phương pháp của ba nhánh. [CSV toàn bộ kết quả](studd_19datasets_summary.csv) · [Bài báo STUDD, Bảng 3–4](https://doi.org/10.1007/s10994-022-06188-7)

- **Khoa:** [Report và 60 kết quả](khoa/README.md), W&B `lgkai-hcmus/Testing`.
- **Khánh:** [Report và 70 kết quả](khanh/khanh-7datasets.md), [run tổng hợp](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/tekuwx8d).
- **Đạt:** [60 kết quả](dat_6datasets_summary.csv), [thống kê cảnh báo](dat_6datasets_alarm_summary.csv). Nhánh nguồn dẫn tới CSV theo bộ. Bảng này chỉ điền URL run đã được ghi nhận.

## Đối chiếu phương pháp STUDD

| Dataset | Người chạy | N | W | Kappa nhóm / paper | Chi phí nhãn nhóm / paper | Cảnh báo | Nguồn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Arabic | Khoa | 8800 | 2000 | 0.6441 / 0.71 | 0.2273 / 0.23 | 0 | [CSV](khoa/arabic_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/50og8cwl) |
| ArabicShuffled | Khoa | 8800 | 2000 | 0.8905 / 0.88 | 0.2273 / 0.11 | 0 | [CSV](khoa/arabic-shuffled_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/zuq3ehd6) |
| Insects | Khoa | 5325 | 500 | 0.8191 / 0.81 | 0.0939 / 0.09 | 0 | [CSV](khoa/insects_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/dn36pykx) |
| InsectsAbrupt | Khoa | 5325 | 500 | 0.5770 / 0.73 | 0.0939 / 0.28 | 0 | [CSV](khoa/insects-abrupt_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/9ov2duqt) |
| Keystroke | Khoa | 1600 | 500 | 0.8861 / 0.84 | 0.3125 / 0.31 | 0 | [CSV](khoa/keystroke_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/07yv50m5) |
| Posture | Khoa | 150000 | 2000 | 0.4724 / 0.50 | 0.1600 / 0.02 | 11 | [CSV](khoa/posture_reproduction.csv) · [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/a3c5djlv) |
| Bike | Khánh | 17379 | 2000 | 0.3175 / 0.32 | 0.2086 / 0.06 | 1 | [CSV](khanh/bike_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/z7gwi104) |
| Gas | Khánh | 13910 | 2000 | 0.7173 / 0.48 | 0.8627 / 0.86 | 5 | [CSV](khanh/gas_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/7jfppc33) |
| GMSC | Khánh | 150000 | 2000 | 0.1940 / 0.32 | 0.0133 / 0.01 | 0 | [CSV](khanh/gmsc_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/o3l912e1) |
| Luxembourg | Khánh | 1901 | 500 | 1.0000 / 1.00 | 0.2630 / 0.53 | 0 | [CSV](khanh/luxembourg_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/270g3uw7) |
| Outdoor | Khánh | 4000 | 500 | 0.3736 / 0.36 | 0.2500 / 0.25 | 1 | [CSV](khanh/outdoor_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/0putgi2v) |
| Ozone | Khánh | 1847 | 500 | 0.2202 / 0.13 | 0.2707 / 0.39 | 0 | [CSV](khanh/ozone_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/lngaqacb) |
| Rialto | Khánh | 82250 | 2000 | 0.3953 / 0.38 | 0.0434 / 0.21 | 1 | [CSV](khanh/rialto_comparison_table.csv) · [W&B](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/sufuaw3n) |
| CoverType | Đạt | 150000 | 2000 | 0.5046 / 0.52 | 0.2800 / 0.09 | 20 | [CSV](covertype_reproduction.csv) |
| Electricity | Đạt | 45312 | 2000 | 0.4166 / 0.29 | 0.5738 / 0.43 | 12 | [CSV](electricity_reproduction.csv) |
| NOAA | Đạt | 18159 | 2000 | 0.4036 / 0.39 | 0.2203 / 0.06 | 1 | [CSV](noaa_reproduction.csv) |
| Poker | Đạt | 150000 | 2000 | 0.2960 / 0.48 | 0.0133 / 0.01 | 0 | [CSV](poker_reproduction.csv) |
| Powersupply | Đạt | 29928 | 2000 | 0.1067 / 0.10 | 0.2005 / 0.07 | 2 | [CSV](powersupply_reproduction.csv) |
| Sensors | Đạt | 150000 | 2000 | 0.2684 / 0.59 | 0.4800 / 0.16 | 35 | [CSV](sensors_reproduction.csv) |

## Cách đọc số liệu

- Cột `Our_Cost` giữ cách tính của lượt chạy gốc. Khánh đếm số lượt dùng nhãn trực tuyến, còn Khoa và Đạt nhân số lần huấn luyện lại với W. Cả hai cách cộng W mẫu ban đầu và chia cho N. `Our_Cost_1000_Reported` là cột quy đổi từ báo cáo của Khánh.
- Cột `Retrains` để trống ở các dòng nguồn chưa cung cấp số lần huấn luyện lại. Các phép đo cảnh báo như MDR và FAR vẫn nằm trong báo cáo nguồn để giữ quy ước đo của từng nhóm.
- Arabic và ArabicShuffled cần quyền truy cập input Kaggle riêng tư của Khoa. Xem các sai khác với paper về phiên bản dữ liệu và tiền xử lý trong báo cáo của từng người.

Tạo lại bảng và report từ các tệp đã lưu bằng `python3 experiments/aggregate_studd_results.py`.
