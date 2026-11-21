# Kết quả thực nghiệm STUDD

## Đợt chạy 10 phương pháp · detector của Đạt

Kaggle CPU, seed 42, Page–Hinkley từ nhánh Đạt với `delta=0.001` và KS ở mức `0.001`. Mỗi run W&B chứa đủ 10 phương pháp. [Bộ chạy của Đạt](https://github.com/LGKAI/STUDD/tree/dat-6datasets) · [Bảng 3–4 của paper](https://doi.org/10.1007/s10994-022-06188-7)

| Dataset | Mẫu | W | Kappa STUDD nhóm / paper | Chi phí nhãn nhóm / paper | Cảnh báo STUDD | CSV chi tiết | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| InsectsAbrupt | 5325 | 500 | 0,5770 / 0,73 | 0,0939 / 0,28 | 0 | [CSV](insects-abrupt_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-insects-abrupt-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/tujkdqpz) |
| Insects | 5325 | 500 | 0,8191 / 0,81 | 0,0939 / 0,09 | 0 | [CSV](insects_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-insects-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/s6lcevkf) |
| Keystroke | 1600 | 500 | 0,8861 / 0,84 | 0,3125 / 0,31 | 0 | [CSV](keystroke_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-keystroke-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/p7k4ehtn) |
| Arabic | 8800 | 2000 | 0,6441 / 0,71 | 0,2273 / 0,23 | 0 | [CSV](arabic_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/bj3jcne9) |
| ArabicShuffled | 8800 | 2000 | 0,8905 / 0,88 | 0,2273 / 0,11 | 0 | [CSV](arabic-shuffled_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-shuffled-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/dshyx0hx) |
| Posture | 150000 | 2000 | 0,4724 / 0,50 | 0,1600 / 0,02 | 11 | [CSV](posture_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-posture-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/STUDD/runs/of55ovih) |

[CSV tổng hợp tất cả phương pháp](khoa_6datasets_summary.csv). Chi phí nhãn của nhóm tính theo bộ chạy của Đạt, gồm cửa sổ huấn luyện đầu và các cửa sổ huấn luyện lại, chia cho tổng số mẫu. Các tệp `*_alarms.csv` lưu vị trí cảnh báo theo chỉ số dòng bắt đầu từ 0.
