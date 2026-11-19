# Kịch bản chạy

`prepare_data.py` chuẩn bị dữ liệu qua `src/data_loader.py`.

```bash
python3 experiments/prepare_data.py --dataset all --output-dir data
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
