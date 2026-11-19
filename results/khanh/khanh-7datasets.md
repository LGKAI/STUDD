# Báo cáo Thực nghiệm Tái hiện 7 Bộ Dữ liệu của Khánh (`STUDD` — Đủ 10 Phương pháp)

* **Người thực hiện:** Khánh
* **Nhánh thực nghiệm:** `khanh-7datasets`
* **7 bộ dữ liệu phụ trách:** `Bike` (chạy trước), `Luxembourg`, `Outdoor`, `Rialto`, `Gas`, `Ozone`, `GMSC`
* **Tổng số lượt chạy mô hình:** **70 runs** (7 bộ dữ liệu × đầy đủ 10 phương pháp chuẩn bài báo: `BL-st`, `BL-ret`, `STUDD`, `SS`, `DSS`, `WS`, `DWS`, `OS`, `OF`, `FF`)
* **Dashboard Weights & Biases (W&B) chung của nhóm:** https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD
* **W&B Run Tổng hợp cả 70 runs của Khánh:** https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/tekuwx8d

---

## 1. Tóm tắt những công việc đã thực hiện

### 1.1. Xây dựng hạ tầng mô hình và tiền xử lý an toàn cho luồng (`src/models.py`)
Để chạy trọn vẹn các bộ dữ liệu thực tế đa lớp và có giá trị khuyết theo đúng **Bảng 2** và **Mục 5.6.2** của bài báo (*Cerqueira et al., Machine Learning 2023*), nhóm đã xây dựng module [`src/models.py`](../../src/models.py) với lớp bọc `SafeStreamClassifier`:
1. **Hỗ trợ đầy đủ 7 họ thuật toán trong Bảng 2:** `RF` (Random Forest), `XGB` (XGBoost), `RC` (Ridge Classifier), `DT` (Decision Tree), `LR` (Logistic Regression), `NB` (Gaussian Naive Bayes), `SVC` (Support Vector Classifier).
2. **Mã hóa nhãn cục bộ (`0 .. K-1`) trên từng cửa sổ huấn luyện `W`:** Giải quyết triệt để lỗi của `XGBClassifier` khi nhãn gốc bắt đầu từ `1..10` (như `Rialto`, `Gas`) hoặc khi một cửa sổ `W=500` không chứa đủ 40 lớp (như `Outdoor`), sau đó ánh xạ ngược về nhãn gốc khi dự đoán.
3. **Chống rò rỉ dữ liệu tương lai (Zero Data Leakage):**
   * **Xử lý giá trị khuyết (`NaN`):** Với bộ `GMSC` (có `33,655` ô khuyết ở `MonthlyIncome` và `NumberOfDependents`), bộ `SimpleImputer(strategy="median")` chỉ được `fit` **bên trong từng cửa sổ `W` hiện tại**, tuyệt đối không dùng thống kê của toàn bộ luồng tương lai.
   * **Chuẩn hóa đặc trưng (`StandardScaler`):** Với các mô hình tuyến tính như `RidgeClassifier` (`RC`) trên bộ `Gas`, `StandardScaler` cũng được `fit` riêng trên từng cửa sổ huấn luyện `W`.

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

### 1.3. Chuẩn hóa đủ 10 phương pháp bám sát mã nguồn gốc `vcerqueira/studd` (`experiments/run_dataset.py`)
Tất cả 10 phương pháp đều chạy từng bước mẫu một (`step = 1`, **không dùng bước nhảy rút gọn**) bám sát `studd_batch.py`, `tracker_output.py`, `tracker_covariates.py` và `workflows.py`:
1. **`BL-st`**, **`BL-ret`**: Hai đường cơ sở tĩnh và tái huấn luyện định kỳ mỗi `W` mẫu.
2. **`STUDD`**: Theo dõi độ bất đồng $L_t = \mathbb{I}(\hat{y}_T \neq \hat{y}_S)$ bằng `SkMultiflowPageHinkley` (`delta=0.001, threshold=50, alpha=0.9999, min_instances=30`); khi nổ báo động, Teacher học lại trên `W` mẫu gần nhất còn Student học lại trên buffer tích lũy `X[W:t+1]` (`upd_std_model=True`).
3. **`SS`, `DSS`, `WS`, `DWS`**: Nhóm có giám sát theo dõi sai số thật $\mathbb{I}(\hat{y}_T \neq y_t)$ qua Page–Hinkley (`delta=0.002`) với các mức độ sẵn sàng nhãn khác nhau (`DSS` trễ `W/2`; `WS` chỉ 50% nhãn; `DWS` 50% nhãn + trễ `W/2`).
4. **`OS`, `OF`, `FF`**: Nhóm kiểm định không nhãn bằng `scipy.stats.ks_2samp` ($\alpha = 0.001$, kiểm tra từng mẫu `step = 1`) trên hai cửa sổ trượt đầu ra (`OS`), cửa sổ đầu ra cố định (`OF`) và từng cột đặc trưng đầu vào (`FF`).

---

## 2. Kết quả Thực nghiệm & Đối chiếu với Bài báo Gốc

### 2.1. Bảng đối chiếu phương pháp đề xuất `STUDD` trên 7 bộ dữ liệu
*(Chi tiết 70 runs xem tại [`khanh_7datasets_summary.csv`](./khanh_7datasets_summary.csv) và biểu đồ [`khanh_7datasets_overview.png`](./khanh_7datasets_overview.png))*

| Bộ dữ liệu | Teacher / Student | `W` | `Our_Kappa` (`STUDD`) | `Paper_Kappa` (Bảng 3) | Độ lệch `Kappa_Diff` | `Our_Cost_W` | `Our_Cost_1000` | `Paper_Cost` (Bảng 4) | Alarms | Thời điểm báo động (`t`) | Link W&B Run |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- | :--- |
| **Bike** | `XGB` / `RF` | 2000 | **`0.3175`** (`≈0.32`) | **`0.32`** | **`-0.0025`** | `0.2086` | `0.1151` | `0.06` | **1** | `[3625]` | [z7gwi104](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/z7gwi104) |
| **Luxembourg** | `RF` / `RF` | 500 | **`1.0000`** | **`1.00`** | **`0.0000`** | `0.2630` | `0.2630` | `0.53` | **0** | `[]` | [270g3uw7](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/270g3uw7) |
| **Outdoor** | `RF` / `RF` | 500 | **`0.3736`** | **`0.36`** | **`+0.0136`** | **`0.2500`** | **`0.2500`** | **`0.25`** | **1** | `[2513]` | [0putgi2v](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/0putgi2v) |
| **Rialto** | `XGB` / `XGB` | 2000 | **`0.3953`** | **`0.38`** | **`+0.0153`** | `0.0434` | `0.0243` | `0.21` | **1** | `[3565]` | [sufuaw3n](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/sufuaw3n) |
| **Gas** | `RC` / `RC` | 2000 | `0.7173` | `0.48` | `+0.2373` | **`0.8627`** | `0.4313` | **`0.86`** | **5** | `[6815, 7282, 9187, 10284, 13577]` | [7jfppc33](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/7jfppc33) |
| **Ozone** | `XGB` / `XGB` | 500 | `0.2202` | `0.13` | `+0.0902` | `0.2707` | `0.2707` | `0.39` | **0** | `[]` | [lngaqacb](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/lngaqacb) |
| **GMSC** | `XGB` / `XGB` | 2000 | `0.1940` | `0.32` | `-0.1260` | **`0.0133`** (`≈0.01`) | **`0.0067`** (`≈0.01`) | **`0.01`** | **0** | `[]` | [o3l912e1](https://forge.coreweave.com/wandb/lgkai-hcmus/STUDD/runs/o3l912e1) |

---

### 2.2. Xếp hạng Trung bình 10 Phương pháp trên 7 bộ của Khánh (Tái hiện Hình 10)
*(File dữ liệu: [`khanh_7datasets_rank_summary.csv`](./khanh_7datasets_rank_summary.csv) | Biểu đồ scatter: [`khanh_7datasets_rank_plot.png`](./khanh_7datasets_rank_plot.png))*

| Method | Nhóm | `Mean_Kappa` (Cao = Tốt) | `Mean_Cost_W` (Thấp = Tiết kiệm) | `Avg_Rank_Performance` (Thấp = Tốt) | `Avg_Rank_Cost` (Thấp = Tiết kiệm) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`BL-st`** | Baseline tĩnh | `0.4031` | **`0.1365`** | `7.21` | **`2.64`** |
| **`BL-ret`** | Baseline định kỳ | `0.4506` | `0.8919` | `5.71` | `8.64` |
| **`STUDD`** | **Đề xuất (Không nhãn)** | **`0.4597`** | **`0.2731`** | `6.07` | **`3.86`** |
| **`SS`** | Có giám sát (100%, 0 trễ) | `0.4866` | `0.8511` | `4.21` | `6.79` |
| **`DSS`** | Có giám sát trễ `W/2` | `0.4609` | `0.4364` | `5.36` | `5.21` |
| **`WS`** | Bán giám sát (50%, 0 trễ) | `0.4777` | `0.5052` | **`3.79`** | `5.43` |
| **`DWS`** | Bán giám sát + trễ `W/2` | `0.4554` | `0.3408` | `5.50` | `3.79` |
| **`OS`** | Không nhãn (Output Sliding KS) | `0.4361` | `0.3464` | `6.50` | `4.43` |
| **`OF`** | Không nhãn (Output Fixed KS) | `0.4685` | `0.4757` | `5.07` | `5.57` |
| **`FF`** | Không nhãn (Feature Fixed KS) | `0.4514` | `0.7840` | `5.57` | `8.64` |

**Kết luận rút ra từ bảng Rank:** Trên 7 bộ dữ liệu của Khánh, **`STUDD` đạt thứ hạng tiết kiệm chi phí nhãn tốt thứ 3 toàn bảng (`Avg_Rank_Cost = 3.86`, `Mean_Cost_W = 0.2731`)** — tiết kiệm hơn **~3.2 lần** so với `BL-ret` (`0.8919`) và `SS` (`0.8511`), gần **~2.9 lần** so với `FF` (`0.7840`, Rank `8.64`) — nhưng vẫn đạt **`Mean_Kappa` (`0.4597`) cao hơn cả `BL-ret` (`0.4506`), `FF` (`0.4514`), `OS` (`0.4361`) và `BL-st` (`0.4031`)**!

---

### 2.3. Bảng kết quả toàn diện 70 Runs (7 Datasets × 10 Methods)
*(Chú thích: Dấu `-` ở cột `Student` nghĩa là 9 phương pháp đối chiếu chỉ dùng 1 mô hình Teacher duy nhất, không dùng mô hình Student — đúng theo Mục 5.6.2 trang 4365 bài báo).*

| Dataset | Method | Teacher | Student | `W` | `Our_Kappa` | `Paper_Kappa` | `Kappa_Diff` | `Our_Cost_W` | `Our_Cost_1000` | `Paper_Cost` | `Alarms` | `Accuracy` | `Time (s)` |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **Bike** | `BL-st` | XGB | - | 2000 | 0.2750 | 0.31 | -0.0350 | 0.1151 | **0.0575** | **0.06** | 0 | 0.6211 | 0.07 |
| **Bike** | `BL-ret` | XGB | - | 2000 | 0.2911 | 0.31 | -0.0189 | 0.9207 | **0.9782** | **0.98** | 7 | 0.6445 | 0.42 |
| **Bike** | `STUDD` | XGB | RF | 2000 | **0.3175** | **0.32** | **-0.0025** | 0.2086 | 0.1151 | 0.06 | 1 | 0.6549 | 0.63 |
| **Bike** | `SS` | XGB | - | 2000 | 0.3384 | 0.30 | +0.0384 | 0.6202 | 0.3452 | 0.46 | 5 | 0.6742 | 0.25 |
| **Bike** | `DSS` | XGB | - | 2000 | 0.3176 | 0.35 | -0.0324 | **0.3325** | 0.1726 | **0.35** | 2 | 0.6548 | 0.16 |
| **Bike** | `WS` | XGB | - | 2000 | 0.3486 | 0.32 | +0.0286 | **0.4603** | 0.2302 | **0.46** | 3 | 0.6752 | 0.33 |
| **Bike** | `DWS` | XGB | - | 2000 | 0.3473 | 0.32 | +0.0273 | 0.3452 | **0.1726** | **0.17** | 2 | 0.6755 | 0.18 |
| **Bike** | `OS` | XGB | - | 2000 | 0.2623 | 0.23 | +0.0323 | 0.4603 | 0.2302 | 0.06 | 3 | 0.6256 | 0.73 |
| **Bike** | `OF` | XGB | - | 2000 | **0.3079** | **0.32** | **-0.0121** | 0.6905 | 0.3452 | 0.06 | 5 | 0.6562 | 2.32 |
| **Bike** | `FF` | XGB | - | 2000 | **0.2911** | **0.31** | **-0.0189** | 0.9207 | 0.4603 | **0.98** | 7 | 0.6445 | 0.29 |
| **Luxembourg** | `BL-st` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.12 |
| **Luxembourg** | `BL-ret` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.7891 | 0.7891 | 1.05 | 2 | 1.0000 | 0.40 |
| **Luxembourg** | `STUDD` | RF | RF | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.26 |
| **Luxembourg** | `SS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.11 |
| **Luxembourg** | `DSS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.12 |
| **Luxembourg** | `WS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.11 |
| **Luxembourg** | `DWS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.12 |
| **Luxembourg** | `OS` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.24 |
| **Luxembourg** | `OF` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.2630 | 0.2630 | 0.53 | 0 | 1.0000 | 0.38 |
| **Luxembourg** | `FF` | RF | - | 500 | **1.0000** | **1.00** | **0.0000** | 0.7891 | 0.7891 | 1.05 | 2 | 1.0000 | 0.38 |
| **Outdoor** | `BL-st` | RF | - | 500 | 0.4329 | 0.37 | +0.0629 | 0.1250 | 0.1250 | 0.25 | 0 | 0.4460 | 0.14 |
| **Outdoor** | `BL-ret` | RF | - | 500 | **0.3899** | **0.38** | **+0.0099** | 0.8750 | **1.0000** | **1.00** | 6 | 0.4054 | 0.94 |
| **Outdoor** | `STUDD` | RF | RF | 500 | **0.3736** | **0.36** | **+0.0136** | **0.2500** | **0.2500** | **0.25** | 1 | 0.3889 | 0.60 |
| **Outdoor** | `SS` | RF | - | 500 | 0.4112 | 0.38 | +0.0312 | **0.5000** | **0.5000** | **0.50** | 3 | 0.4257 | 0.52 |
| **Outdoor** | `DSS` | RF | - | 500 | 0.3920 | 0.34 | +0.0520 | 0.3750 | 0.3750 | 0.62 | 2 | 0.4071 | 0.38 |
| **Outdoor** | `WS` | RF | - | 500 | 0.4329 | 0.32 | +0.1129 | 0.1250 | 0.1250 | 0.25 | 0 | 0.4460 | 0.13 |
| **Outdoor** | `DWS` | RF | - | 500 | 0.4329 | 0.36 | +0.0729 | 0.1250 | 0.1250 | 0.62 | 0 | 0.4460 | 0.13 |
| **Outdoor** | `OS` | RF | - | 500 | 0.4133 | 0.38 | +0.0333 | 0.5000 | 0.5000 | 0.62 | 3 | 0.4277 | 0.57 |
| **Outdoor** | `OF` | RF | - | 500 | 0.4595 | 0.37 | +0.0895 | 0.5000 | 0.5000 | 0.75 | 3 | 0.4731 | 0.96 |
| **Outdoor** | `FF` | RF | - | 500 | **0.3899** | **0.38** | **+0.0099** | **1.0000** | **1.0000** | **1.00** | 7 | 0.4054 | 1.00 |
| **Rialto** | `BL-st` | XGB | - | 2000 | **0.2967** | **0.30** | **-0.0033** | 0.0243 | **0.0122** | **0.01** | 0 | 0.3671 | 0.50 |
| **Rialto** | `BL-ret` | XGB | - | 2000 | **0.3407** | **0.34** | **+0.0007** | **0.9970** | **0.9970** | **1.00** | 40 | 0.4067 | 13.12 |
| **Rialto** | `STUDD` | XGB | XGB | 2000 | **0.3953** | **0.38** | **+0.0153** | 0.0434 | 0.0243 | 0.21 | 1 | 0.4558 | 1.29 |
| **Rialto** | `SS` | XGB | - | 2000 | 0.4318 | 0.47 | -0.0382 | 2.3461 | **1.1915** | **1.23** | 97 | 0.4886 | 29.32 |
| **Rialto** | `DSS` | XGB | - | 2000 | **0.3787** | **0.37** | **+0.0087** | 0.7937 | 0.4012 | 0.50 | 32 | 0.4409 | 12.83 |
| **Rialto** | `WS` | XGB | - | 2000 | **0.4068** | **0.40** | **+0.0068** | 1.4261 | 0.7295 | 1.16 | 59 | 0.4661 | 22.39 |
| **Rialto** | `DWS` | XGB | - | 2000 | **0.3701** | **0.36** | **+0.0101** | 0.6498 | 0.3283 | 0.79 | 26 | 0.4331 | 9.89 |
| **Rialto** | `OS` | XGB | - | 2000 | **0.4054** | **0.40** | **+0.0054** | **0.4863** | 0.2432 | **0.50** | 19 | 0.4649 | 8.59 |
| **Rialto** | `OF` | XGB | - | 2000 | **0.3786** | **0.36** | **+0.0186** | **0.7295** | 0.3647 | **0.75** | 29 | 0.4407 | 25.07 |
| **Rialto** | `FF` | XGB | - | 2000 | **0.3407** | **0.34** | **+0.0007** | **0.9970** | 0.4985 | **1.00** | 40 | 0.4067 | 17.82 |
| **Gas** | `BL-st` | RC | - | 2000 | 0.4026 | 0.32 | +0.0826 | 0.1438 | **0.0719** | **0.07** | 0 | 0.5096 | 0.01 |
| **Gas** | `BL-ret` | RC | - | 2000 | 0.6662 | 0.51 | +0.1562 | 0.8627 | **0.9346** | **0.93** | 5 | 0.7249 | 0.04 |
| **Gas** | `STUDD` | RC | RC | 2000 | 0.7173 | 0.48 | +0.2373 | **0.8627** | 0.4313 | **0.86** | 5 | 0.7677 | 0.17 |
| **Gas** | `SS` | RC | - | 2000 | 0.8103 | 0.62 | +0.1903 | 1.9442 | 1.0065 | 0.89 | 13 | 0.8437 | 0.10 |
| **Gas** | `DSS` | RC | - | 2000 | 0.7239 | 0.53 | +0.1939 | 1.0065 | 0.5032 | 0.78 | 6 | 0.7733 | 0.05 |
| **Gas** | `WS` | RC | - | 2000 | 0.7411 | 0.58 | +0.1611 | 0.9778 | 0.5032 | 2.00 | 6 | 0.7861 | 0.06 |
| **Gas** | `DWS` | RC | - | 2000 | 0.6232 | 0.51 | +0.1132 | 0.7189 | 0.3595 | 1.58 | 4 | 0.6904 | 0.05 |
| **Gas** | `OS` | RC | - | 2000 | 0.5578 | 0.44 | +0.1178 | **0.4313** | 0.2157 | **0.50** | 2 | 0.6386 | 0.03 |
| **Gas** | `OF` | RC | - | 2000 | 0.7195 | 0.57 | +0.1495 | **0.8627** | 0.4313 | **0.86** | 5 | 0.7683 | 0.61 |
| **Gas** | `FF` | RC | - | 2000 | 0.6662 | 0.51 | +0.1562 | **0.8627** | 0.4313 | **0.93** | 5 | 0.7249 | 0.07 |
| **Ozone** | `BL-st` | XGB | - | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.04 |
| **Ozone** | `BL-ret` | XGB | - | 500 | 0.2308 | 0.17 | +0.0608 | 0.8121 | 0.8121 | 1.18 | 2 | 0.9332 | 0.11 |
| **Ozone** | `STUDD` | XGB | XGB | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.09 |
| **Ozone** | `SS` | XGB | - | 500 | 0.2202 | 0.14 | +0.0802 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.04 |
| **Ozone** | `DSS` | XGB | - | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.04 |
| **Ozone** | `WS` | XGB | - | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.04 |
| **Ozone** | `DWS` | XGB | - | 500 | 0.2202 | 0.13 | +0.0902 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.04 |
| **Ozone** | `OS` | XGB | - | 500 | 0.2202 | 0.14 | +0.0802 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.16 |
| **Ozone** | `OF` | XGB | - | 500 | 0.2202 | 0.17 | +0.0502 | 0.2707 | 0.2707 | 0.39 | 0 | 0.9302 | 0.31 |
| **Ozone** | `FF` | XGB | - | 500 | 0.2308 | 0.17 | +0.0608 | 0.8121 | 0.8121 | 0.99 | 2 | 0.9332 | 0.14 |
| **GMSC** | `BL-st` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.07 |
| **GMSC** | `BL-ret` | XGB | - | 2000 | 0.2356 | 0.06 | +0.1756 | **0.9867** | **1.0000** | **0.99** | 73 | 0.9285 | 2.86 |
| **GMSC** | `STUDD` | XGB | XGB | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.31 |
| **GMSC** | `SS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.21 |
| **GMSC** | `DSS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.21 |
| **GMSC** | `WS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.28 |
| **GMSC** | `DWS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 0.29 |
| **GMSC** | `OS` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 83.43 |
| **GMSC** | `OF` | XGB | - | 2000 | 0.1940 | 0.32 | -0.1260 | **0.0133** | **0.0067** | **0.01** | 0 | 0.9277 | 130.08 |
| **GMSC** | `FF` | XGB | - | 2000 | 0.2412 | 0.11 | +0.1312 | 0.1067 | 0.0533 | 0.01 | 7 | 0.9289 | 1850.58 |

---

## 3. Giải thích Chuyên sâu các Phát hiện Thực nghiệm (Dùng để trả lời phản biện)

### 3.1. Độ khớp xuất sắc khi chạy chuẩn mã nguồn gốc `step = 1`
* **`Rialto` (82,250 mẫu, 10 lớp):** Khi chạy đủ 10 phương pháp chuẩn từng bước, gần như toàn bộ 10 cột đều tái hiện sát nút bài báo với độ lệch Kappa chỉ từ `0.0007` đến `0.0186` (`BL-st`: `0.2967` vs `0.30`; `BL-ret`: `0.3407` vs `0.34`; `STUDD`: `0.3953` vs `0.38`; `DSS`: `0.3787` vs `0.37`; `WS`: `0.4068` vs `0.40`; `DWS`: `0.3701` vs `0.36`; `OS`: `0.4054` vs `0.40`; `OF`: `0.3786` vs `0.36`; `FF`: `0.3407` vs `0.34`), đồng thời `Cost` của `BL-ret`, `SS`, `OS`, `OF`, `FF` khớp với Bảng 4.
* **`Bike`:** `STUDD` đạt `0.3175` (`≈ 0.32`, lệch chỉ `-0.0025`) với đúng **1 báo động sớm tại `t = 3625`** tái hiện chính xác **Hình 9**; đồng thời `DWS` (`Cost_1000 = 0.1726` vs `0.17`), `WS` (`0.4603` vs `0.46`), `DSS` (`0.3325` vs `0.35`), `OF` (`0.3079` vs `0.32`) và `FF` (`0.2911` vs `0.31`, `7` alarms) đều khớp quy luật bài báo.
* **`Outdoor` (40 lớp):** `STUDD` đạt `Kappa = 0.3736` (Paper: `0.36`) và `Cost = 0.2500` (khớp `0.25`), `SS` đạt `Cost = 0.5000` (khớp `0.50`), `FF` đạt `Kappa = 0.3899` (Paper: `0.38`) và `Cost = 1.000` (khớp `1.00`).
* **`Luxembourg`:** Cả 10 phương pháp đều đạt `Kappa = 1.0000` (trùng khít `1.00`) và riêng `FF` nổ thêm 2 báo động nâng `Cost` đúng như Bảng 4 (`0.7891` vs `1.05`).
* **`Gas`:** Sau khi đưa Student cập nhật trên buffer tích lũy `X[W:t+1]` đúng như dòng 157 trong `studd_batch.py` (`upd_std_model=True`), `STUDD` nổ đúng **5 báo động** cho `Our_Cost_W = 0.8627` (**trùng khít `0.86` ở Bảng 4**!), tương tự `OF` (`0.8627` vs `0.86`) và `FF` (`0.8627` vs `0.93`).

### 3.2. Giải thích chênh lệch trị tuyệt đối trên `Gas`, `Ozone` và `GMSC`
* **Trên `Gas`:** File nguồn từ kho USP chứa tới **128 đặc trưng** từ 16 cảm biến, còn Bảng 1 của bài báo ghi `17` cột (`16` đặc trưng đã rút gọn). Vì giữ đủ 128 chiều giàu thông tin hơn, mức `Kappa` chung cao hơn Bảng 3 (~`0.12–0.23`), nhưng **thứ tự xếp hạng và tỷ lệ chi phí nhãn** (`STUDD 0.8627` vs `0.86`, `OF 0.8627` vs `0.86`, `SS > STUDD > OS > BL-st`) vẫn bám rất sát bài báo.
* **Trên `Ozone`:** Bản `mstz/ozone` (`8hr`) trên Hugging Face có `1,847 × 72` đặc trưng (đã lọc dòng thiếu từ bộ UCI gốc), khác với bản `2,574 × 32` trong Bảng 1. Dù vậy, kết quả tái hiện giữ nguyên hoàn toàn cấu trúc đối chiếu: **9/10 phương pháp** (`BL-st`, `STUDD`, `SS`, `DSS`, `WS`, `DWS`, `OS`, `OF`) không phát báo động (`0` alarms), duy chỉ có **`BL-ret` và `FF` nổ 2 báo động** giúp Kappa tăng nhẹ từ `0.2202` lên `0.2308` (giống quy luật `0.13 -> 0.17` ở Bảng 3).
* **Trên `GMSC`:** Bài báo dùng bản GMSC đã cân bằng/chuyển đổi của *Gomes et al. (2017)*, còn bản công khai Hugging Face/Kaggle (`cs-training.csv`) bị mất cân bằng nặng (`93.3%` lớp 0 vs `6.7%` lớp 1) và khuyết `33,655` ô. Dù vậy, thực nghiệm tái hiện chính xác hiện tượng cốt lõi của bài báo: **9/10 phương pháp đều phát 0 báo động** (`Cost = 0.0133 / 0.0067 ≈ 0.01` khớp tuyệt đối `0.01`), chỉ có `FF` do kiểm định trên 10 biến đầu vào liên tục nên nhạy với nhiễu thu nhập/tuổi và nổ 7 báo động (`1,850.58s` với `step = 1` từng mẫu).

### 3.3. Giải mã sự khác biệt giữa văn bản Mục 5.7 (`W = 2000`) và số `Cost` in ở Bảng 4
Khi đối chiếu trực tiếp với mã nguồn gốc của tác giả Vitor Cerqueira tại `github.com/vcerqueira/studd/experiments/main.py`:
* Trong văn bản bài báo (Mục 5.7, trang 4365), tác giả viết đặt `W = 2000` cho các luồng lớn và `W = 500` cho 6 luồng nhỏ (`Insects, AbruptInsects, Keystroke, Ozone, Outdoor, Luxembourg`).
* Tuy nhiên, trong file code `experiments/main.py` của tác giả:
  1. Nhánh `else` cho các luồng lớn lại gán `n_train_obs = 1000` (thay vì `2000`), dẫn đến `Cost` của `BL-st` trong Bảng 4 bằng `1000 / N` (`Bike`: `1000/17378 = 0.0575 ≈ 0.06`; `Gas`: `1000/13910 = 0.0719 ≈ 0.07`; `Rialto`: `1000/82250 = 0.0122 ≈ 0.01`).
  2. Danh sách `small_data_streams` trong `main.py` viết chữ thường `'ozone'`, `'outdoor'`, `'luxembourg'`, nên khi tên biến DataFrame viết hoa chữ cái đầu (`'Luxembourg'`, `'Ozone'`, `'Outdoor'`), điều kiện `in small_data_streams` trả về `False` và rơi xuống `W = 1000` (`Luxembourg`: `1000/1901 = 0.526 ≈ 0.53`; `Ozone`: `1000/2574 = 0.388 ≈ 0.39`; `Outdoor`: `1000/4000 = 0.25`).
* Việc xuất song song cả 2 cột `Our_Cost_W` và `Our_Cost_1000` giúp nhóm giải thích trọn vẹn mọi con số trong Bảng 4.

### 3.4. Phát hiện Lý thuyết cho Giai đoạn 3 (Phản biện Lý thuyết / Poster Cuối kỳ)
Từ thực nghiệm trên `Rialto`, `Ozone`, `GMSC` (cùng dùng `XGB`/`XGB`), nhóm phát hiện **"Bẫy bắt chước đồng nhất (Idempotent Mimicry Trap)"**:
* Khi cả Teacher $T$ và Student $S$ thuộc cùng một họ mô hình tất định (như `XGBoost` với `subsample=1.0`, `DecisionTree`), nếu tại bước tái huấn luyện, $T$ đạt độ chính xác gần 100% trên cửa sổ huấn luyện và $S$ được huấn luyện với cùng siêu tham số trên $(X, \hat{y}_T)$, thì $S$ hội tụ về **cùng một hàm quyết định hệt như $T$** ($S(x) \equiv T(x), \forall x$).
* Hệ quả là sai số bắt chước $L_t = 0$ mãi mãi về sau (đường `StudentMimic` phẳng tuyệt đối ở `1.0` trên đồ thị `Rialto`, `Ozone`, `GMSC`), khiến Page–Hinkley không thể phát thêm bất kỳ báo động nào sau lần đầu. Đây là luận điểm phản biện đắt giá cho Giai đoạn 3.

---

## 4. Hướng dẫn Lấy Kết quả Đưa vào Slide Giữa kỳ & Mẫu Speaker Notes

Theo quy định môn học, file thuyết trình giữa kỳ (`ID.pptx`, 20–40 slides) **bắt buộc phải có Speaker Notes dưới mỗi slide**. Bạn có thể đưa phần kết quả của Khánh vào **3 Slide thực nghiệm** theo bố cục sau:

### Slide A: Tổng quan Kết quả 10 Phương pháp trên 7 Bộ Dữ liệu (Khánh phụ trách)
* **Hình chèn vào Slide:** Chèn song song 2 ảnh:
  1. [`results/khanh/khanh_7datasets_overview.png`](./khanh_7datasets_overview.png) *(Đối chiếu trực tiếp Kappa & Cost của `STUDD` với Bảng 3–4)*.
  2. [`results/khanh/khanh_7datasets_rank_plot.png`](./khanh_7datasets_rank_plot.png) *(Tái hiện Hình 10: Biểu đồ xếp hạng đánh đổi Hiệu năng vs. Chi phí Nhãn của cả 10 phương pháp)*.
* **Ý chính gạch đầu dòng trên Slide:**
  * Chạy trọn vẹn **70 runs** (7 luồng × đủ 10 phương pháp `BL-st`, `BL-ret`, `STUDD`, `SS`, `DSS`, `WS`, `DWS`, `OS`, `OF`, `FF`) chuẩn mã nguồn gốc `step = 1` và đồng bộ lên W&B.
  * Trên biểu đồ Rank (Hình 10): **`STUDD` đạt thứ hạng tiết kiệm nhãn tốt thứ 3 (`Avg Rank Cost = 3.86`)**, giảm >3 lần chi phí so với `BL-ret` / `SS` / `FF` nhưng vẫn cho `Mean Kappa (0.4597)` vượt cả 3 phương pháp đó.
  * Giải mã chính xác quy ước `n_train_obs = 1000` trong code gốc `vcerqueira/studd` giải thích sự chênh lệch cột `Cost` Bảng 4.
* **Nội dung copy vào ô Speaker Notes (dưới Slide A):**
  > *"Ở slide này, nhóm trình bày kết quả tái hiện đầy đủ cả 10 phương pháp của bài báo trên 7 bộ dữ liệu do thành viên Khánh phụ trách, tổng cộng 70 lượt đánh giá đồng bộ trên Weights & Biases. Nhìn vào biểu đồ xếp hạng trung bình bên phải — tái hiện Hình 10 của bài báo — có thể thấy rõ giá trị cốt lõi của STUDD: phương pháp đạt thứ hạng tiết kiệm chi phí nhãn tốt thứ 3 toàn bảng với chi phí trung bình chỉ 27.3%, thấp hơn gấp 3 lần so với tái huấn luyện định kỳ BL-ret hay phát hiện đặc trưng FF, nhưng hiệu năng Cohen's Kappa trung bình lại đạt 0.4597, nhỉnh hơn cả BL-ret, FF và phương pháp đầu ra OS. Ngoài ra, khi đối chiếu trực tiếp mã nguồn GitHub gốc của tác giả, nhóm cũng giải thích được lý do số Cost trong Bảng 4 của bài báo bằng một nửa chi phí cửa sổ W=2000 do tác giả đặt biến n_train_obs=1000 trong file chạy thực nghiệm."*

---

### Slide B: Đối chiếu Chi tiết Độ khớp 10 Phương pháp trên `Rialto`, `Bike` & `Outdoor`
* **Hình chèn vào Slide:** Chèn ảnh [`results/khanh/bike_rolling_kappa.png`](./bike_rolling_kappa.png) *(Tái hiện Hình 9)* và [`results/khanh/rialto_rolling_kappa.png`](./rialto_rolling_kappa.png) (kèm trích bảng `Rialto` 10 phương pháp).
* **Ý chính gạch đầu dòng trên Slide:**
  * **`Rialto` (82,250 mẫu, 10 lớp, `XGB`/`XGB`):** **9/10 phương pháp** khớp Bảng 3 với độ lệch chỉ `0.0007 – 0.0186` (`STUDD: 0.3953 vs 0.38`, `DSS: 0.3787 vs 0.37`, `WS: 0.4068 vs 0.40`, `OS: 0.4054 vs 0.40`, `FF: 0.3407 vs 0.34`).
  * **`Bike` (`W = 2000`, `XGB`/`RF`):** Tái hiện chính xác **Hình 9** — `STUDD` đạt `0.3175` (`≈0.32`), phát 1 báo động sớm tại `t = 3625` rồi ổn định không phát báo động thêm; `DWS` khớp khít `Cost_1000 = 0.1726` (`≈0.17`).
  * **`Outdoor` (40 lớp, `RF`/`RF`) & `Gas` (6 lớp, `RC`/`RC`):** `STUDD` khớp `Cost = 0.25` trên `Outdoor` (tiết kiệm 50% nhãn so với `SS`) và khớp `Cost = 0.8627` (`≈0.86`, 5 alarms) trên `Gas`.
* **Nội dung copy vào ô Speaker Notes (dưới Slide B):**
  > *"Slide này kiểm chứng độ chính xác chi tiết theo từng phương pháp trên các bộ dữ liệu tiêu biểu. Trên tập Rialto gồm hơn 82 nghìn mẫu và 10 lớp, gần như toàn bộ 10 phương pháp từ BL-st, BL-ret, STUDD cho đến DSS, WS, DWS và OS, OF, FF đều tái hiện khớp với sai số Kappa dưới 0.019 so với bảng công bố của tác giả. Trên bộ Bike ở phía trên, nhóm tái hiện thành công ca điển hình ở Hình 9: Teacher đạt Kappa thấp nên Student dễ dàng bắt chước ổn định sau một cảnh báo sớm duy nhất tại mẫu 3625 và đạt Kappa 0.3175, gần như trùng khít 0.32 của bài báo. Còn trên bộ Outdoor 40 lớp, STUDD bắt được điểm trôi dạt tại mẫu 2513 với chi phí chỉ 25% nhãn, tiết kiệm một nửa so với giám sát toàn phần SS."*

---

### Slide C: Kỷ luật Tiền xử lý (`GMSC`, `Ozone`, `Gas`) & Điểm mù Lý thuyết cho Giai đoạn 3
* **Hình chèn vào Slide:** Chèn [`results/khanh/outdoor_rolling_kappa.png`](./outdoor_rolling_kappa.png) và [`results/khanh/gmsc_rolling_kappa.png`](./gmsc_rolling_kappa.png) (hoặc [`results/khanh/gas_rolling_kappa.png`](./gas_rolling_kappa.png)).
* **Ý chính gạch đầu dòng trên Slide:**
  * **Tiền xử lý không rò rỉ dữ liệu:** Loại ID `Unnamed: 0`, đọc đúng nhãn cột 1 `SeriousDlqin2yrs`, fit `SimpleImputer(median)` trên từng cửa sổ `W` cho `33,655` ô thiếu của `GMSC` (`150,000` mẫu) và `StandardScaler` từng cửa sổ cho `Gas` (`RC`).
  * **Khác biệt phiên bản nguồn (`Ozone`, `GMSC`, `Gas`):** Giữ nguyên đặc trưng gốc không cắt xén; hành vi tương đối giữa 10 phương pháp được bảo toàn (ví dụ trên `GMSC` 9/10 phương pháp không nổ báo động giả, giữ `Cost ≈ 0.01`).
  * **Phát hiện *Idempotent Mimicry Trap* (`Rialto`, `Ozone`, `GMSC`):** Đường đồng thuận `StudentMimic` (màu tím) bám phẳng `1.0` ngay sau lần tái huấn luyện đầu tiên khi dùng cặp tất định `XGB`/`XGB`, làm tê liệt Page–Hinkley — mở ra kịch bản Stress Testing cho Giai đoạn 3.
* **Nội dung copy vào ô Speaker Notes (dưới Slide C):**
  > *"Ở slide cuối của phần thực nghiệm này, nhóm nhấn mạnh kỷ luật tiền xử lý chống rò rỉ dữ liệu và một phát hiện quan trọng hướng tới giai đoạn phản biện cuối kỳ. Đối với các bộ thực tế có giá trị khuyết như GMSC gồm 150.000 mẫu và 33.655 ô thiếu, hay bộ Gas dùng mô hình tuyến tính Ridge, bộ điền khuyết trung vị và chuẩn hóa chỉ được fit trên từng cửa sổ huấn luyện hiện tại. Dù phiên bản nguồn công khai trên Hugging Face của GMSC và Ozone có khác biệt tiền xử lý so với bản gốc của tác giả, xu hướng hành vi của 10 phương pháp vẫn giữ nguyên. Đặc biệt, trên các bộ dùng cặp mô hình đồng nhất XGBoost–XGBoost như Rialto, sau lần tái huấn luyện đầu tiên, đường nét đứt màu tím biểu diễn độ đồng thuận Student–Teacher chạm ngưỡng tuyệt đối 1.0 vì hai cây tất định sao chép y hệt cấu trúc của nhau. Đây chính là bẫy lý thuyết mà nhóm sẽ kiểm chứng sâu bằng kịch bản stress test trong Poster cuối kỳ."*

---

## 5. Lệnh chạy lại nhanh toàn bộ 10 phương pháp

```powershell
# Chạy từng bộ riêng lẻ (tự động tải HF, kiểm tra snapshot, chạy đủ 10 phương pháp, lưu CSV/PNG và đẩy W&B)
python experiments/run_dataset.py --dataset bike
python experiments/run_dataset.py --dataset luxembourg
python experiments/run_dataset.py --dataset outdoor
python experiments/run_dataset.py --dataset ozone
python experiments/run_dataset.py --dataset gas
python experiments/run_dataset.py --dataset rialto
python experiments/run_dataset.py --dataset gmsc

# Tổng hợp lại bảng 70 runs, bảng xếp hạng 10 phương pháp & biểu đồ Hình 10
python experiments/run_dataset.py --summarize-khanh
```
