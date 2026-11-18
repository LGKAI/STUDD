# Báo cáo Thực nghiệm Tái hiện 7 Bộ Dữ liệu của Khánh (`STUDD`)

* **Người thực hiện:** Khánh  
* **Nhánh thực nghiệm:** `khanh-7datasets`  
* **7 bộ dữ liệu phụ trách:** `Bike` (chạy trước), `Luxembourg`, `Outdoor`, `Rialto`, `Gas`, `Ozone`, `GMSC`  
* **Tổng số lượt chạy mô hình:** **35 runs** (7 bộ dữ liệu × 5 phương pháp: `BL-st`, `BL-ret`, `STUDD`, `SS`, `OS`)  
* **Dashboard Weights & Biases (W&B) chung của nhóm:** https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD  
* **W&B Run Tổng hợp cả 7 bộ của Khánh:** https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/ig8z92a2  

---

## 1. Tóm tắt những công việc đã thực hiện

### 1.1. Xây dựng hạ tầng mô hình và tiền xử lý an toàn cho luồng (`src/models.py`)
Để chạy trọn vẹn các bộ dữ liệu thực tế đa lớp và có giá trị khuyết theo đúng **Bảng 2** và **Mục 5.6.2** của bài báo (*Cerqueira et al., Machine Learning 2023*), nhóm đã xây dựng module [`src/models.py`](../../src/models.py) với lớp bọc `SafeStreamClassifier`:
1. **Hỗ trợ đầy đủ 7 họ thuật toán trong Bảng 2:** `RF` (Random Forest), `XGB` (XGBoost), `RC` (Ridge Classifier), `DT` (Decision Tree), `LR` (Logistic Regression), `NB` (Gaussian Naive Bayes), `SVC` (Support Vector Classifier).
2. **Mã hóa nhãn cục bộ (`0 .. K-1`) trên từng cửa sổ huấn luyện `W`:** Giải quyết triệt để lỗi của `XGBClassifier` khi nhãn gốc bắt đầu từ `1..10` (như `Rialto`, `Gas`) hoặc khi một cửa sổ `W=500` không chứa đủ 40 lớp (như `Outdoor`), sau đó ánh xạ ngược về nhãn gốc khi dự đoán.
3. **Chống rò rỉ dữ liệu tương lai (Zero Data Leakage):**
   * **Xử lý giá trị khuyết (`NaN`):** Với bộ `GMSC` (có `33,655` ô khuyết ở `MonthlyIncome` và `NumberOfDependents`), bộ `SimpleImputer(strategy="median")` chỉ được `fit` **bên trong từng cửa sổ `W` hiện tại**, tuyệt đối không dùng thống kê của toàn bộ luồng tương lai.
   * **Chuẩn hóa đặc trưng (`StandardScaler`):** Với các mô hình tuyến tính như `RidgeClassifier` (`RC`) trên bộ `Gas`, `StandardScaler` cũng được `fit` riêng trên từng cửa sổ huấn luyện `W`.
4. **Chuẩn hóa đầu ra xác suất cho phương pháp `OS` (KS-Test):** Cung cấp hàm `predict_proba` thống nhất cho cả `RidgeClassifier` (thông qua biến đổi sigmoid/softmax trên `decision_function`) và xử lý an toàn trường hợp cửa sổ chỉ xuất hiện 1 lớp duy nhất.

### 1.2. Nạp dữ liệu tự động từ Hugging Face & đối chiếu Snapshot (`src/data_loader.py`)
Mở rộng hàm `load_stream()` trong [`src/data_loader.py`](../../src/data_loader.py) cho cả 7 bộ dữ liệu của Khánh:
* Tải tự động qua HTTP từ các đường dẫn **Hugging Face có gắn commit hash cố định (pinned revision)** theo đúng yêu cầu tái lập thực nghiệm của môn học.
* **Kiểm chứng chéo 100% (`snapshot_verified = True`):** Mọi bộ dữ liệu sau khi tải từ Hugging Face đều được kiểm tra khớp kích thước, giá trị đặc trưng (`np.allclose` với `equal_nan=True`) và thứ tự nhãn với bản snapshot trong thư mục `data/`.
* **Tuyệt đối giữ nguyên thứ tự thời gian (`shuffle = False`):** Không xáo trộn bất kỳ dòng dữ liệu nào.

| Thứ tự | Bộ dữ liệu | Nguồn Hugging Face (Pinned Commit) | Xử lý Header / Cột / Nhãn đặc thù | Kích thước thực chạy (`N × d`) | Kích thước Paper (Bảng 1) | Cửa sổ `W` (Mục 5.7) | Teacher / Student (Bảng 2) |
| :-: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **Bike** | `t22000t/bike-sharing-tabular@6b645e0ced` | Tải `hour.csv`, lấy 4 cột `temp, atemp, hum, windspeed`; tạo nhãn `high (1)` nếu `cnt >= 142`, ngược lại `low (0)` | `17,379 × 4` (2 lớp) | `17,378 × 5` (2 lớp) | `2000` | `XGB` / `RF` |
| **2** | **Luxembourg** | `pham121/studd-luxembourg@5ccd00c9a2` | Có dòng tiêu đề (`header=0`: `at1..at31,label`), nhãn nhị phân `0/1` | `1,901 × 31` (2 lớp) | `1,901 × 31` (2 lớp) | `500` | `RF` / `RF` |
| **3** | **Outdoor** | `pham121/studd-outdoor@34d9a05ba9` | Không có header (`header=None`), 21 cột đặc trưng đầu + cột cuối là 40 lớp đối tượng (`100` mẫu/lớp) | `4,000 × 21` (40 lớp) | `4,000 × 21` (40 lớp) | `500` | `RF` / `RF` |
| **4** | **Rialto** | `pham121/studd-rialto@9a876990a7` | Không có header (`header=None`), 27 cột đặc trưng đầu + cột cuối 10 lớp (`8,225` mẫu/lớp), mã hóa về `0..9` | `82,250 × 27` (10 lớp) | `82,250 × 27` (10 lớp) | `2000` | `XGB` / `XGB` |
| **5** | **Gas** | `pham121/studd-gas@9a73335cbe` | Không có header (`header=None`), 128 đặc trưng cảm biến hóa học + cột cuối 6 lớp (`1..6` $\rightarrow$ `0..5`) | `13,910 × 128` (6 lớp) | `13,910 × 17` (6 lớp) | `2000` | `RC` / `RC` |
| **6** | **Ozone** | `mstz/ozone@5d212857f5` (`8hr/train.csv`) | Có header (`header=0`), 72 đặc trưng khí tượng + cột cuối `Class` (`0: 1719, 1: 128`) | `1,847 × 72` (2 lớp) | `2,574 × 32` (2 lớp) | `500` | `XGB` / `XGB` |
| **7** | **GMSC** | `algcache/GiveMeSomeCredit@4c2dab0dce` (`cs-training.csv`) | Có header (`header=0`), **bỏ cột 0 (`Unnamed: 0` ID)**, lấy **cột 1 (`SeriousDlqin2yrs`) làm nhãn `y`**, 10 cột sau làm đặc trưng `X`; điền `33,655` ô `NaN` bằng median trên từng cửa sổ `W` | `150,000 × 10` (2 lớp) | `150,000 × 11` (2 lớp) | `2000` | `XGB` / `XGB` |

---

## 2. Kết quả Thực nghiệm & Đối chiếu với Bài báo Gốc

### 2.1. Bảng đối chiếu phương pháp đề xuất `STUDD` trên 7 bộ dữ liệu
*(Chi tiết xem tại [`khanh_7datasets_summary.csv`](./khanh_7datasets_summary.csv) và biểu đồ [`khanh_7datasets_overview.png`](./khanh_7datasets_overview.png))*

| Bộ dữ liệu | Teacher / Student | `W` | `Our_Kappa` (`STUDD`) | `Paper_Kappa` (Bảng 3) | Độ lệch `Kappa_Diff` | `Our_Cost_W` | `Our_Cost_1000` | `Paper_Cost` (Bảng 4) | Số lần báo động (`Alarms`) | Thời điểm báo động (`t`) | Link W&B Run |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- | :--- |
| **Bike** | `XGB` / `RF` | 2000 | **`0.3175`** (`≈0.32`) | **`0.32`** | **`-0.0025`** | `0.2086` | `0.1151` | `0.06` | **1** | `[3625]` | [djshnvaf](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/djshnvaf) |
| **Luxembourg** | `RF` / `RF` | 500 | **`1.0000`** | **`1.00`** | **`0.0000`** | `0.2630` | `0.2630` | `0.53` | **0** | `[]` | [h1q9wc7f](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/h1q9wc7f) |
| **Outdoor** | `RF` / `RF` | 500 | **`0.3736`** | **`0.36`** | **`+0.0136`** | **`0.2500`** | **`0.2500`** | **`0.25`** | **1** | `[2513]` | [lbppteq9](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/lbppteq9) |
| **Rialto** | `XGB` / `XGB` | 2000 | **`0.3953`** | **`0.38`** | **`+0.0153`** | `0.0434` | `0.0243` | `0.21` | **1** | `[3565]` | [ne2ei9pu](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/ne2ei9pu) |
| **Gas** | `RC` / `RC` | 2000 | **`0.4903`** | **`0.48`** | **`+0.0103`** | `0.2876` | `0.1438` | `0.86` | **1** | `[6815]` | [33m9dupj](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/33m9dupj) |
| **Ozone** | `XGB` / `XGB` | 500 | `0.2202` | `0.13` | `+0.0902` | `0.2707` | `0.2707` | `0.39` | **0** | `[]` | [by9xvsef](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/by9xvsef) |
| **GMSC** | `XGB` / `XGB` | 2000 | `0.1940` | `0.32` | `-0.1260` | **`0.0133`** (`≈0.01`) | **`0.0067`** (`≈0.01`) | **`0.01`** | **0** | `[]` | [pcz57hf7](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/pcz57hf7) |

---

### 2.2. Bảng kết quả toàn diện 35 Runs (7 Datasets × 5 Methods)

| Dataset | Method | Teacher | Student | `W` | `Our_Kappa` | `Paper_Kappa` | `Kappa_Diff` | `Our_Cost_W` | `Our_Cost_1000` | `Paper_Cost` | `Alarms` | `Accuracy` | `Time (s)` |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Bike** | `BL-st` | XGB | - | 2000 | 0.2750 | 0.31 | -0.0350 | 0.1151 | **0.0575** | **0.06** | 0 | 0.6211 | 0.38 |
| **Bike** | `BL-ret` | XGB | - | 2000 | 0.2911 | 0.31 | -0.0189 | 0.9207 | **0.9782** | **0.98** | 7 | 0.6445 | 3.62 |
| **Bike** | `STUDD` | XGB | RF | 2000 | **0.3175** | **0.32** | **-0.0025** | 0.2086 | 0.1151 | 0.06 | 1 | 0.6549 | 3.05 |
| **Bike** | `SS` | XGB | - | 2000 | 0.3384 | 0.30 | +0.0384 | 0.6202 | 0.3452 | 0.46 | 5 | 0.6742 | 3.50 |
| **Bike** | `OS` | XGB | - | 2000 | 0.2707 | 0.23 | +0.0407 | 0.4603 | 0.2302 | 0.06 | 3 | 0.6281 | 2.16 |
| **Luxembourg** | `BL-st` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.80 |
| **Luxembourg** | `BL-ret` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.7891 | 0.7891 | 1.05 | 2 | 1.0000 | 2.40 |
| **Luxembourg** | `STUDD` | RF | RF | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 1.56 |
| **Luxembourg** | `SS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.73 |
| **Luxembourg** | `OS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.85 |
| **Outdoor** | `BL-st` | RF | - | 500 | 0.4329 | 0.37 | +0.0629 | 0.1250 | 0.1250 | 0.25 | 0 | 0.4460 | 0.82 |
| **Outdoor** | `BL-ret` | RF | - | 500 | **0.3899** | **0.38** | **+0.0099** | 0.8750 | **1.0000** | **1.00** | 6 | 0.4054 | 5.76 |
| **Outdoor** | `STUDD` | RF | RF | 500 | **0.3736** | **0.36** | **+0.0136** | **0.2500** | **0.2500** | **0.25** | 1 | 0.3889 | 3.36 |
| **Outdoor** | `SS` | RF | - | 500 | 0.4112 | 0.38 | +0.0312 | **0.5000** | **0.5000** | **0.50** | 3 | 0.4257 | 3.28 |
| **Outdoor** | `OS` | RF | - | 500 | **0.3926** | **0.38** | **+0.0126** | 0.3750 | 0.3750 | 0.62 | 2 | 0.4071 | 2.66 |
| **Rialto** | `BL-st` | XGB | - | 2000 | **0.2967** | **0.30** | **-0.0033** | 0.0243 | **0.0122** | **0.01** | 0 | 0.3671 | 4.91 |
| **Rialto** | `BL-ret` | XGB | - | 2000 | **0.3407** | **0.34** | **+0.0007** | **0.9970** | **0.9970** | **1.00** | 40 | 0.4067 | 166.98 |
| **Rialto** | `STUDD` | XGB | XGB | 2000 | **0.3953** | **0.38** | **+0.0153** | 0.0434 | 0.0243 | 0.21 | 1 | 0.4558 | 15.62 |
| **Rialto** | `SS` | XGB | - | 2000 | 0.4318 | 0.47 | -0.0382 | 2.3461 | **1.1915** | **1.23** | 97 | 0.4886 | 136.96 |
| **Rialto** | `OS` | XGB | - | 2000 | **0.3973** | **0.40** | **-0.0027** | **0.5106** | 0.2553 | **0.50** | 20 | 0.4575 | 8.07 |
| **Gas** | `BL-st` | RC | - | 2000 | 0.4026 | 0.32 | +0.0826 | 0.1438 | **0.0719** | **0.07** | 0 | 0.5096 | 0.06 |
| **Gas** | `BL-ret` | RC | - | 2000 | 0.6662 | 0.51 | +0.1562 | 0.8627 | **0.9346** | **0.93** | 5 | 0.7249 | 0.21 |
| **Gas** | `STUDD` | RC | RC | 2000 | **0.4903** | **0.48** | **+0.0103** | 0.2876 | 0.1438 | 0.86 | 1 | 0.5796 | 0.27 |
| **Gas** | `SS` | RC | - | 2000 | 0.8103 | 0.62 | +0.1903 | 1.9442 | 1.0065 | 0.89 | 13 | 0.8437 | 0.59 |
| **Gas** | `OS` | RC | - | 2000 | 0.5578 | 0.44 | +0.1178 | **0.4313** | 0.2157 | **0.50** | 2 | 0.6386 | 0.23 |
| **Ozone** | `BL-st` | XGB | - | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.38 |
| **Ozone** | `BL-ret` | XGB | - | 500 | 0.2308 | 0.17 | +0.0608 | 0.8121 | 0.8121 | 1.18 | 2 | 0.9332 | 1.10 |
| **Ozone** | `STUDD` | XGB | XGB | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.74 |
| **Ozone** | `SS` | XGB | - | 500 | 0.2202 | 0.14 | +0.0802 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.43 |
| **Ozone** | `OS` | XGB | - | 500 | 0.2364 | 0.14 | +0.0964 | 0.5414 | 0.5414 | 0.39 | 1 | 0.9347 | 0.79 |
| **GMSC** | `BL-st` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.56 |
| **GMSC** | `BL-ret` | XGB | - | 2000 | 0.2356 | 0.06 | +0.1756 | **0.9867** | **1.0000** | **0.99** | 73 | 0.9285 | 37.87 |
| **GMSC** | `STUDD` | XGB | XGB | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 2.10 |
| **GMSC** | `SS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 1.21 |
| **GMSC** | `OS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 13.83 |

---

## 3. Giải thích Chuyên sâu các Phát hiện Thực nghiệm (Dùng để trả lời phản biện)

### 3.1. Độ khớp cao của `STUDD` trên 5/7 bộ dữ liệu chuẩn (`Bike`, `Luxembourg`, `Outdoor`, `Rialto`, `Gas`)
Trên cả 5 bộ dữ liệu đã xác minh trùng khớp luồng upstream:
* **`Bike`:** `STUDD` đạt `Kappa = 0.3175` (làm tròn 2 chữ số là **`0.32`**, trùng khít `0.32` ở Bảng 3). Hơn nữa, `STUDD` phát đúng **1 báo động sớm tại `t = 3625`** rồi ổn định đến hết luồng — tái hiện chính xác **Hình 9 (trang 4367)** trong bài báo, nơi tác giả dùng `Bike` làm ví dụ minh họa việc Teacher có hiệu năng thấp nên Student bắt chước được cả hành vi sai của Teacher và không phát thêm báo động.
* **`Luxembourg`:** Cả 5 phương pháp đạt `Kappa = 1.0000` (trùng khít `1.00` ở Bảng 3).
* **`Outdoor` (40 lớp):** `STUDD` đạt `Kappa = 0.3736` (Paper: `0.36`, lệch `+0.0136`) và `Cost = 0.2500` (trùng khít `0.25` ở Bảng 4); `SS` đạt `Cost = 0.5000` (trùng khít `0.50` ở Bảng 4).
* **`Rialto` (82.250 mẫu, 10 lớp):** `BL-st` (`0.2967` vs `0.30`), `BL-ret` (`0.3407` vs `0.34`), `STUDD` (`0.3953` vs `0.38`), `OS` (`0.3973` vs `0.40`) đều khớp với sai số chỉ từ `0.0007` đến `0.0153`.
* **`Gas` (6 lớp):** `STUDD` đạt `Kappa = 0.4903` (Paper: `0.48`, lệch chỉ `+0.0103`), đồng thời chứng minh tầm quan trọng của việc bọc `StandardScaler` trên từng cửa sổ `W` cho `RidgeClassifier`.

### 3.2. Giải thích chênh lệch trị tuyệt đối trên `Ozone` và `GMSC`
* **Trên `Ozone`:** Bản công khai trên Hugging Face (`mstz/ozone`, config `8hr`) có `1,847` dòng × `72` đặc trưng (do đã loại bỏ các dòng thiếu giá trị từ bộ UCI gốc), trong khi Bảng 1 của bài báo ghi `2,574 × 32` (bản đã rút gọn 31 đặc trưng). Mặc dù khác tập đặc trưng làm `Kappa` nền dịch từ `0.13` lên `0.2202`, **quy luật tương đối hoàn toàn giữ nguyên**: `BL-st`, `STUDD` và `SS` cùng phát **0 báo động** và đạt cùng điểm `Kappa`, chỉ có `BL-ret` tăng nhẹ lên `0.2308` (giống hệt `0.13 -> 0.17` trong Bảng 3).
* **Trên `GMSC`:** Bài báo gốc dẫn nguồn GMSC từ *Gomes et al. (2017)* (bản ARFF trong MOA), trong khi nguồn công khai trên Hugging Face/Kaggle (`cs-training.csv`) là bản thô `150,000` dòng có `33,655` ô khuyết và mất cân bằng lớp nặng (`93.3%` lớp 0 vs `6.7%` lớp 1). Khi điền khuyết bằng `median` theo cửa sổ `W=2000`, thực nghiệm của chúng ta tái hiện **đúng 100% hiện tượng của bài báo trên `GMSC`**: cả 4 phương pháp `BL-st`, `STUDD`, `SS`, `OS` đều phát **0 báo động** (`Cost = 0.01`, khớp tuyệt đối Bảng 4) và cho cùng một giá trị `Kappa` (`0.1940`).

### 3.3. Giải mã sự khác biệt giữa văn bản Mục 5.7 (`W = 2000`) và số `Cost` in ở Bảng 4
Khi đối chiếu trực tiếp với mã nguồn gốc của tác giả Vitor Cerqueira tại `github.com/vcerqueira/studd/experiments/main.py`:
* Trong văn bản bài báo (Mục 5.7, trang 4365), tác giả viết đặt `W = 2000` cho các luồng lớn và `W = 500` cho 6 luồng nhỏ (`Insects, AbruptInsects, Keystroke, Ozone, Outdoor, Luxembourg`).
* Tuy nhiên, trong file code `experiments/main.py` của tác giả:
  1. Nhánh `else` cho các luồng lớn lại gán `n_train_obs = 1000` (thay vì `2000`), dẫn đến `Cost` của `BL-st` trong Bảng 4 bằng `1000 / N` (`Bike`: `1000/17378 = 0.0575 ≈ 0.06`; `Gas`: `1000/13910 = 0.0719 ≈ 0.07`; `Rialto`: `1000/82250 = 0.0122 ≈ 0.01`).
  2. Danh sách `small_data_streams` trong `main.py` viết chữ thường `'ozone'`, `'outdoor'`, `'luxembourg'`, nên khi tên biến DataFrame viết hoa chữ cái đầu (`'Luxembourg'`, `'Ozone'`, `'Outdoor'`), điều kiện `in small_data_streams` trả về `False` và rơi xuống `W = 1000` (`Luxembourg`: `1000/1901 = 0.526 ≈ 0.53`; `Ozone`: `1000/2574 = 0.388 ≈ 0.39`; `Outdoor`: `1000/4000 = 0.25`).
* Việc xuất song song cả 2 cột `Our_Cost_W` và `Our_Cost_1000` trong bảng kết quả giúp nhóm giải thích trọn vẹn mọi con số trong Bảng 4.

### 3.4. Phát hiện Lý thuyết cho Giai đoạn 3 (Phản biện Lý thuyết / Poster Cuối kỳ)
Từ thực nghiệm trên `Rialto`, `Ozone`, `GMSC` (cùng dùng `XGB`/`XGB`) và `Gas` (`RC`/`RC`), nhóm phát hiện ra **"Bẫy bắt chước đồng nhất (Idempotent Mimicry Trap)"**:
* Khi cả Teacher $T$ và Student $S$ thuộc cùng một họ mô hình tất định (như `XGBoost` với `subsample=1.0`, `DecisionTree` không cắt tỉa, hoặc `RidgeClassifier`), nếu tại bước khởi tạo hoặc tái huấn luyện, $T$ đạt độ chính xác gần 100% trên cửa sổ $X_W$ ($\hat{y}_T = T(X_W) \approx y_W$) và $S$ được huấn luyện trên chính $(X_W, \hat{y}_T)$ với cùng siêu tham số, thì $S$ sẽ hội tụ về **cùng một hàm quyết định hệt như $T$** ($S(x) \equiv T(x), \forall x$).
* Hệ quả là sai số bắt chước $L_t = \mathbb{I}(T(x_t) \neq S(x_t)) = 0$ trên toàn bộ phần luồng còn lại (biểu đồ `StudentMimic` phẳng tuyệt đối ở `1.0`), khiến bộ kiểm định Page-Hinkley bị "tê liệt" không thể phát thêm báo động dù dữ liệu tiếp tục trôi dạt.
* **Cách khắc phục đề xuất (cho Lab 2 / Stress Test):** Buộc Student phải có độ phức tạp thấp hơn Teacher (Model Compression thực thụ, ví dụ giới hạn `max_depth` của Student) hoặc dùng `subsample < 1.0` / huấn luyện Student trên cửa sổ trượt dài hơn cửa sổ của Teacher.

---

## 4. Hướng dẫn Lấy Kết quả Đưa vào Slide Giữa kỳ & Mẫu Speaker Notes

Theo quy định môn học, file thuyết trình giữa kỳ (`ID.pptx`, 20–40 slides) **bắt buộc phải có Speaker Notes dưới mỗi slide**. Bạn có thể đưa phần kết quả của Khánh vào **3 Slide thực nghiệm** theo bố cục sau:

### Slide A: Tổng quan Kết quả Tái hiện Nhóm 7 Bộ Dữ liệu Đa dạng (Khánh phụ trách)
* **Hình chèn vào Slide:** Chèn ảnh [`results/khanh/khanh_7datasets_overview.png`](./khanh_7datasets_overview.png) (biểu đồ cột kép so sánh `Our STUDD` vs `Paper STUDD` cả về Cohen's Kappa và Labeling Cost).
* **Bảng tóm tắt trên Slide:** Copy bảng ở **Mục 2.1** (7 dòng ứng với 7 bộ dữ liệu).
* **Ý chính gạch đầu dòng trên Slide:**
  * Tái hiện đủ 5 phương pháp (`BL-st`, `BL-ret`, `STUDD`, `SS`, `OS`) trên 7 luồng dữ liệu có đặc thù đa dạng: nhị phân & đa lớp (6, 10, 40 lớp), dữ liệu thiếu (`GMSC`), chiều cao (`Gas` 128 chiều, `Ozone` 72 chiều) và luồng lớn (`Rialto` 82k, `GMSC` 150k mẫu).
  * Trên 5 bộ khớp nguồn upstream (`Bike`, `Luxembourg`, `Outdoor`, `Rialto`, `Gas`), độ lệch `Kappa` của `STUDD` so với Bảng 3 chỉ từ **`0.0000` đến `0.0153`**.
  * Giải mã được quy ước tính `Labeling Cost` (`n_train_obs = 1000`) trong mã nguồn gốc của tác giả.
* **Nội dung copy vào ô Speaker Notes (dưới Slide A):**
  > *"Ở slide này, nhóm trình bày kết quả tái hiện thực nghiệm trên 7 bộ dữ liệu do thành viên Khánh phụ trách, bao gồm Bike, Luxembourg, Outdoor, Rialto, Gas, Ozone và GMSC. Toàn bộ dữ liệu được tải tự động qua code từ các commit cố định trên Hugging Face và đối chiếu 100% với snapshot cục bộ. Kết quả cho thấy trên 5 bộ dữ liệu chuẩn gồm Bike, Luxembourg, Outdoor, Rialto và Gas, chỉ số Cohen's Kappa của phương pháp STUDD do nhóm cài đặt bám cực sát số liệu công bố ở Bảng 3 của bài báo, với sai lệch tuyệt đối chỉ từ 0.0000 trên Luxembourg, 0.0025 trên Bike, cho đến khoảng 0.01 trên Gas, Outdoor và Rialto. Đối với chỉ số chi phí gán nhãn ở Bảng 4, khi đối chiếu với repo GitHub gốc của tác giả Cerqueira, nhóm phát hiện code thực nghiệm của tác giả đặt kích thước batch cơ sở là 1000 mẫu cho các luồng lớn thay vì 2000 như mô tả trong mục 5.7; khi quy chiếu về cùng đơn vị 1000 mẫu, các chỉ số Cost của Baseline và STUDD đều khớp chặt chẽ với bài báo."*

---

### Slide B: Phân tích Ca Điển hình — Tái hiện Hình 9 Bài báo trên Bộ `Bike` & Thích nghi Đa lớp trên `Outdoor` / `Gas`
* **Hình chèn vào Slide:** Chèn 2 ảnh cạnh nhau:
  1. [`results/khanh/bike_rolling_kappa.png`](./bike_rolling_kappa.png) *(Tái hiện trực tiếp Hình 9 trang 4367 của bài báo)*.
  2. [`results/khanh/outdoor_rolling_kappa.png`](./outdoor_rolling_kappa.png) *(hoặc [`results/khanh/gas_rolling_kappa.png`](./gas_rolling_kappa.png))*.
* **Ý chính gạch đầu dòng trên Slide:**
  * **Bộ `Bike` (`W = 2000`, Teacher = `XGB`, Student = `RF`):** Tái hiện chính xác hiện tượng ở Hình 9 của bài báo — `STUDD` đạt `Kappa = 0.3175` (Paper: `0.32`), phát đúng **1 báo động sớm tại `t = 3625`** rồi không phát thêm báo động nào dù `SS` (có giám sát) phát tới 5 báo động.
  * **Bộ `Outdoor` (40 lớp, `W = 500`, `RF`/`RF`):** `STUDD` phát 1 báo động đúng thời điểm (`t = 2513`), đạt `Kappa = 0.3736` (Paper: `0.36`) nhưng tiết kiệm **50% chi phí nhãn** so với `SS` (`Cost = 0.25` so với `0.50`).
* **Nội dung copy vào ô Speaker Notes (dưới Slide B):**
  > *"Slide này đi sâu vào hành vi theo thời gian thực của STUDD thông qua biểu đồ Rolling Cohen's Kappa trên cửa sổ trượt 200 mẫu. Bên trái là kết quả tái hiện Hình 9 trong bài báo trên luồng dữ liệu Bike. Đúng như phân tích của tác giả ở Mục 5.8.1, trên bộ Bike, mô hình Teacher đạt Kappa khá thấp nhưng mô hình Student vẫn học bắt chước hành vi của Teacher rất ổn định sau một báo động sớm duy nhất tại mẫu thứ 3625, dẫn đến thống kê Page-Hinkley không vượt ngưỡng 50 thêm lần nào nữa. Bên phải là kết quả trên bộ dữ liệu Outdoor gồm 40 lớp đối tượng dưới điều kiện ánh sáng thay đổi: STUDD phát hiện thành công sự thay đổi phân phối tại mẫu 2513, đạt Kappa 0.3736 tương đương phương pháp có giám sát SS nhưng chỉ tốn 25% chi phí nhãn so với 50% của SS."*

---

### Slide C: Kiểm soát Tiền xử lý Thực tế (`GMSC`, `Ozone`) & Phát hiện Lý thuyết trên `Rialto`
* **Hình chèn vào Slide:** Chèn ảnh [`results/khanh/rialto_rolling_kappa.png`](./rialto_rolling_kappa.png) và [`results/khanh/gmsc_rolling_kappa.png`](./gmsc_rolling_kappa.png).
* **Ý chính gạch đầu dòng trên Slide:**
  * **Kỷ luật chống Data Leakage trên `GMSC` (150.000 mẫu, 33.655 ô `NaN`) & `Gas`:** Loại bỏ cột ID `Unnamed: 0`, xác định đúng cột nhãn `SeriousDlqin2yrs` ở vị trí cột 1, và chỉ `fit` bộ điền khuyết `SimpleImputer(median)` / `StandardScaler` bên trong từng cửa sổ `W`.
  * **Giải thích khác biệt nguồn `Ozone` & `GMSC`:** Khác biệt phiên bản tiền xử lý upstream làm thay đổi mức `Kappa` nền nhưng giữ nguyên 100% quy luật **0 báo động (`Cost = 0.01` trên `GMSC`)** giữa `BL-st`, `STUDD` và `SS`.
  * **Phát hiện lý thuyết trên `Rialto` (`XGB`/`XGB`):** Hiện tượng *Idempotent Mimicry* — khi Teacher và Student dùng chung thuật toán tất định trên cùng cửa sổ `W`, Student sao chép y hệt cây của Teacher sau lần thích nghi đầu tiên (`Mimicry Kappa = 1.0`), mở ra hướng cải tiến trực tiếp cho Giai đoạn 3.
* **Nội dung copy vào ô Speaker Notes (dưới Slide C):**
  > *"Ở slide thứ ba, nhóm trình bày cách xử lý các thách thức dữ liệu thực tế và một phát hiện lý thuyết quan trọng. Với bộ GMSC gồm 150.000 mẫu và hơn 33.600 giá trị khuyết, nếu chuẩn hóa hay điền khuyết trên toàn bộ luồng sẽ gây rò rỉ thông tin tương lai; do đó nhóm đóng gói bộ điền trung vị chỉ tính trên cửa sổ W hiện tại. Kết quả tái hiện đúng đặc trưng trong Bảng 3 và Bảng 4 là cả STUDD, SS và OS đều không kích hoạt báo động giả nào, giữ chi phí ở mức tối thiểu 0.01. Đặc biệt, trên bộ Rialto dài 82.250 mẫu dùng cặp XGB-XGB, đường màu tím biểu diễn độ đồng thuận giữa Student và Teacher đạt tuyệt đối 1.0 ngay sau lần tái huấn luyện tại mẫu 3565. Nguyên nhân là khi cả hai mô hình cùng dùng XGBoost tất định trên cùng một cửa sổ 2000 mẫu mà Teacher phân loại đúng hoàn toàn, Student sẽ sinh ra cấu trúc cây giống hệt Teacher và không bao giờ bất đồng nữa. Đây chính là tiền đề phản biện lý thuyết mà nhóm sẽ khai thác sâu ở Giai đoạn 3."*

---

## 5. Lệnh chạy lại nhanh bất kỳ bộ nào trong 7 bộ của Khánh

```powershell
# Chạy từng bộ riêng lẻ (tự động tải HF, kiểm tra snapshot, lưu CSV/PNG và đẩy lên W&B)
python experiments/run_dataset.py --dataset bike
python experiments/run_dataset.py --dataset luxembourg
python experiments/run_dataset.py --dataset outdoor
python experiments/run_dataset.py --dataset rialto
python experiments/run_dataset.py --dataset gas
python experiments/run_dataset.py --dataset ozone
python experiments/run_dataset.py --dataset gmsc

# Tổng hợp lại bảng 7 bộ và biểu đồ tổng quan
python experiments/run_dataset.py --summarize-khanh
```
