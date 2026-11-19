# Kết quả thực nghiệm STUDD

## Đợt chạy 10 phương pháp · detector của Đạt

Kaggle CPU, seed 42, Page–Hinkley từ nhánh Đạt với `delta=0.001` và KS ở mức `0.001`. Mỗi run W&B chứa đủ 10 phương pháp. [Bộ chạy của Đạt](https://github.com/LGKAI/STUDD/tree/dat-6datasets) · [Bảng 3–4 của paper](https://doi.org/10.1007/s10994-022-06188-7)

| Dataset | Mẫu | W | Kappa STUDD nhóm / paper | Chi phí nhãn nhóm / paper | Cảnh báo STUDD | CSV chi tiết | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| InsectsAbrupt | 5325 | 500 | 0,5770 / 0,73 | 0,0939 / 0,28 | 0 | [CSV](insects-abrupt_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-insects-abrupt-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/9ov2duqt) |
| Insects | 5325 | 500 | 0,8191 / 0,81 | 0,0939 / 0,09 | 0 | [CSV](insects_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-insects-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/dn36pykx) |
| Keystroke | 1600 | 500 | 0,8861 / 0,84 | 0,3125 / 0,31 | 0 | [CSV](keystroke_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-keystroke-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/07yv50m5) |
| Arabic | 8800 | 2000 | 0,6441 / 0,71 | 0,2273 / 0,23 | 0 | [CSV](arabic_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/50og8cwl) |
| ArabicShuffled | 8800 | 2000 | 0,8905 / 0,88 | 0,2273 / 0,11 | 0 | [CSV](arabic-shuffled_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-shuffled-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/zuq3ehd6) |
| Posture | 150000 | 2000 | 0,4724 / 0,50 | 0,1600 / 0,02 | 11 | [CSV](posture_reproduction.csv) · [Kaggle](https://www.kaggle.com/code/phmchnkhoa/studd-posture-10-methods-khoa/output) | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/a3c5djlv) |

[CSV tổng hợp tất cả phương pháp](khoa_6datasets_summary.csv). Chi phí nhãn của nhóm tính theo bộ chạy của Đạt, gồm cửa sổ huấn luyện đầu và các cửa sổ huấn luyện lại, chia cho tổng số mẫu. Các tệp `*_alarms.csv` lưu vị trí cảnh báo theo chỉ số dòng bắt đầu từ 0.

## Đợt chạy trước · BL-st và STUDD

Mỗi phương pháp có một dòng. Bảng được bổ sung theo từng dataset khi Kaggle chạy xong. Cột nguồn dẫn đến W&B hoặc đầu ra Kaggle.

| Dataset | Phương pháp | Kappa nhóm | Kappa paper | Lệch tương đối | Accuracy | Cảnh báo drift | Lần huấn luyện lại | Nguồn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| InsectsAbrupt | BL-st | 0,5770 | 0,59 | −2,21% | 66,22% | 0 | 0 | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/emgyw3ar) |
| InsectsAbrupt | STUDD | 0,5770 | 0,73 | −20,96% | 66,22% | 0 | 0 | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/2vvw2ish) |
| Keystroke | BL-st | 0,8861 | 0,84 | +5,48% | 91,45% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/lxtwyvp2) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-keystroke-khoa/output) |
| Keystroke | STUDD | 0,8861 | 0,84 | +5,48% | 91,45% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/g4vi25st) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-keystroke-khoa/output) |
| Posture | BL-st | 0,3372 | 0,34 | −0,81% | 47,80% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/a0twz6ao) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-posture-khoa/output) |
| Posture | STUDD | 0,4727 | 0,50 | −5,46% | 58,56% | 8 | 8 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/6d9safl5) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-posture-khoa/output) |
| ArabicShuffled | BL-st | 0,8905 | 0,88 | +1,20% | 90,15% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/vuqoqhtr) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-arabicshuffled-khoa/output) |
| ArabicShuffled | STUDD | 0,8905 | 0,88 | +1,20% | 90,15% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/zexhvs3n) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-arabicshuffled-khoa/output) |
| Arabic | BL-st | 0,6441 | 0,63 | +2,24% | 67,96% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/fnrtbott) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-khoa/output) |
| Arabic | STUDD | 0,6441 | 0,71 | −9,28% | 67,96% | 0 | 0 | [W&B](https://wandb.ai/lgkai-hcmus/Testing/runs/4d1il32r) · [Kaggle Output](https://www.kaggle.com/code/phmchnkhoa/studd-arabic-khoa/output) |

InsectsAbrupt dùng seed 42, 500 mẫu huấn luyện ban đầu và 4.825 mẫu đánh giá. Độ lệch tương đối tính bằng `100 × (Kappa nhóm − Kappa paper) / Kappa paper`. Kappa tham chiếu lấy từ Bảng 3, dòng **AbruptInsects**. Accuracy, cảnh báo và số lần huấn luyện lại là số liệu của nhóm. [Bài báo STUDD](https://doi.org/10.1007/s10994-022-06188-7)

Keystroke, Posture, Arabic và ArabicShuffled đã đồng bộ lên W&B từ bản chạy Kaggle v2. Chỉ số khớp CSV của v1.

## Lượt chạy dùng River

| Dataset | Phương pháp | Kappa nhóm | Kappa paper | Lệch tương đối | Accuracy | Cảnh báo drift | Lần huấn luyện lại | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| InsectsAbrupt | BL-st | 0,5770 | 0,59 | −2,21% | 66,22% | 0 | 0 | [Run](https://forge.coreweave.com/wandb/lgkai-hcmus/Testing/runs/qg3a3q3u/overview) |
| InsectsAbrupt | STUDD | 0,5770 | 0,73 | −20,96% | 66,22% | 0 | 0 | [Run](https://forge.coreweave.com/wandb/lgkai-hcmus/Testing/runs/gqyn5zn3/overview) |
| Insects | BL-st | 0,8191 | 0,81 | +1,12% | 85,62% | 0 | 0 | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/00ch1zea) |
| Insects | STUDD | 0,8191 | 0,81 | +1,12% | 85,62% | 0 | 0 | [Run](https://wandb.ai/lgkai-hcmus/Testing/runs/bjk0ktn4) |
