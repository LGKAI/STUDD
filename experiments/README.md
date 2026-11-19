# Kịch bản chạy thực nghiệm

Thư mục `experiments/` chứa các kịch bản tái hiện thực nghiệm, tổng hợp kết quả và trực quan hóa trên toàn bộ 19 bộ dữ liệu của STUDD. Dữ liệu luồng được nạp tự động trực tiếp khi chạy thông qua `src/data_loader.py` (và các nguồn đã ghim trong `notebooks/khoa/`).

## Danh sách kịch bản

| Kịch bản | Mục đích |
| --- | --- |
| `reproduce_dat_6datasets.py` | Chạy đầy đủ 10 phương pháp trên 6 bộ dữ liệu của Đạt (`Electricity`, `CoverType`, `Poker`, `Powersupply`, `NOAA`, `Sensors`), hỗ trợ `--no-wandb`, `--aggregate`, `--sync-wandb`. |
| `run_dataset.py` | Chạy thực nghiệm 10 phương pháp trên 7 bộ dữ liệu của Khánh (`Bike`, `Luxembourg`, `Outdoor`, `Rialto`, `Gas`, `Ozone`, `GMSC`), hỗ trợ `--dataset <tên>` và `--no-wandb`. |
| `build_khoa_kaggle_notebooks.py` | Đóng gói 6 bộ dữ liệu của Khoa thành các job để đẩy và chạy trên Kaggle CLI. |
| `reproduce_khoa_6datasets.py` | Chạy và đồng bộ kết quả 10 phương pháp cho 6 bộ dữ liệu của Khoa lên Weights & Biases (`--publish`). |
| `aggregate_studd_results.py` | Tổng hợp toàn bộ 190 kết quả (19 bộ × 10 phương pháp) thành `results/studd_19datasets_summary.csv` và tạo báo cáo đối chiếu `results/README.md`. |
| `plot_19datasets_rank.py` | Vẽ biểu đồ Hình 10 (Average Rank Performance vs. Cost) trên cả 19 bộ dữ liệu, xuất `results/fig_19datasets_rank_plot.png`. |
| `reproduce.py` | Kịch bản tái hiện cơ bản trên 3 luồng chuẩn để kiểm tra nhanh pipeline. |
| `stress_test.py` | Kịch bản kiểm thử chịu tải và phản biện lý thuyết (Lab 2). |

## Lệnh chạy nhanh

```bash
# 6 bộ của Đạt (chạy offline và tổng hợp)
python3 experiments/reproduce_dat_6datasets.py --no-wandb --aggregate

# 7 bộ của Khánh (ví dụ bộ Bike)
python3 experiments/run_dataset.py --dataset bike --no-wandb

# Tổng hợp 190 kết quả & vẽ biểu đồ Average Rank (Hình 10)
python3 experiments/aggregate_studd_results.py
python3 experiments/plot_19datasets_rank.py

# Tái hiện nhanh 3 luồng & kiểm thử chịu tải
python3 experiments/reproduce.py
python3 experiments/stress_test.py
```

## Sáu dataset của Khoa

`build_khoa_kaggle_notebooks.py` đóng gói các notebook thành job cho Kaggle CLI. Trình chạy `reproduce_khoa_6datasets.py` dùng các module trong `src/khoa/` từ [nhánh của Đạt tại revision cố định](https://github.com/LGKAI/STUDD/tree/00834c380c32ff41ed7746d41e949babe3aca658).

```bash
python3 experiments/build_khoa_kaggle_notebooks.py --output /tmp/khoa-kaggle-ten
kaggle kernels push -p /tmp/khoa-kaggle-ten/insects-abrupt
```

Thay `insects-abrupt` bằng `insects`, `keystroke`, `arabic`, `arabic-shuffled` hoặc `posture` cho các luồng còn lại. Kaggle lưu CSV so sánh, chuỗi Accuracy theo cửa sổ 200 mẫu và vị trí từng cảnh báo. Hai job Arabic dùng dataset Kaggle riêng tư đã gắn trong metadata.

Để đồng bộ trên máy, cài `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost` và `wandb`, rồi đăng nhập bằng `wandb login`. Tải các CSV và tệp cấu hình vào cùng thư mục trước khi chạy lệnh đồng bộ. Mỗi run W&B lưu cả 10 phương pháp của một dataset.

```bash
python3 experiments/reproduce_khoa_6datasets.py --publish insects-abrupt --output results/khoa
```

[Report kết quả](../results/README.md)
