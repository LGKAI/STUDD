# STUDD

Đồ án Nhập môn học máy

Dự án nghiên cứu, thực nghiệm và cài đặt lại (reproduce) mô hình **STUDD** dựa trên bài báo khoa học:

> **STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection**  
> *Vitor Cerqueira, Heitor Murilo Gomes, Albert Bifet, Luis Torgo* (Dalhousie University, University of Waikato, Télécom ParisTech).  
> Công bố tại: **Machine Learning (Springer 2023)**, pages 4351–4378.  
> Xếp hạng (Rank): **Q1** (SCIE / Scopus).

---

## 1. Tổng quan về Bài nghiên cứu STUDD

### 1.1. Bối cảnh & Bài toán Concept Drift
Trong các hệ thống học máy triển khai trên luồng dữ liệu thời gian thực (Data Streams) như IoT, phát hiện gian lận tài chính hay dự báo phụ tải điện năng, môi trường luôn biến động khiến phân phối dữ liệu $P(X, Y)$ thay đổi theo thời gian. Hiện tượng này được gọi là **Trôi dạt khái niệm (Concept Drift)**, làm cho mô hình học máy huấn luyện trên dữ liệu cũ bị suy giảm độ chính xác nghiêm trọng.

### 1.2. Hạn chế của các phương pháp truyền thống
Phần lớn các thuật toán phát hiện Concept Drift kinh điển (như DDM, EDDM, ADWIN, Page-Hinkley giám sát) đều thuộc nhóm **có giám sát (supervised)**. Chúng đòi hỏi hệ thống phải có **nhãn thực tế tức thì ($y_t$)** ngay sau mỗi dự đoán để đo lường sai số. Tuy nhiên, trong thực tế sản xuất:
* Việc gán nhãn thực tế tốn kém chi phí chuyên gia rất lớn.
* Nhãn thường bị trễ thời gian rất dài (delayed feedback) hoặc hoàn toàn không có nhãn (unlabeled stream).

### 1.3. Giải pháp của STUDD (Student-Teacher Drift Detection)
Bài báo nghiên cứu đề xuất phương pháp **STUDD** (*A Student-Teacher Method for Unsupervised Concept Drift Detection* - Cerqueira et al., *Machine Learning* 2023 Springer) nhằm phát hiện trôi dạt mà **không cần nhãn thực tế** theo cơ chế sau:
1. **Kiến trúc Thầy - Trò (Teacher - Student Paradigm):**
   * **Teacher ($T$):** Mô hình tĩnh ban đầu, đóng vai trò chuyên gia tham chiếu đưa ra nhãn dự đoán $\hat{y}_{t, T}$.
   * **Student ($S$):** Mô hình học trực tuyến (online learner), được huấn luyện để **học bắt chước** hành vi của Teacher. Nhãn mà Student học theo chính là dự đoán của Teacher $\hat{y}_{t, T}$, hoàn toàn không cần nhãn thật $y_t$.
2. **Cơ chế Phát hiện Trôi dạt (Unsupervised Detection):**
   * Với mỗi mẫu dữ liệu mới $x_t$, cả Teacher và Student cùng đưa ra dự đoán. Sai số bắt chước (Mimicry Loss / Discrepancy) $L_t = \mathbb{I}(\hat{y}_{t, T} \neq \hat{y}_{t, S})$ được theo dõi liên tục thông qua thuật toán kiểm định tuần tự **Page-Hinkley (PH)**.
   * Khi dữ liệu ổn định, Student bắt chước Teacher rất tốt ($L_t \approx 0$). Khi Concept Drift xảy ra, phân phối dữ liệu biến đổi khiến Student và Teacher phát sinh bất đồng lớn $\rightarrow$ Giá trị tích lũy Page-Hinkley tăng vọt vượt ngưỡng cho phép ($\lambda$), kích hoạt cảnh báo phát hiện Concept Drift.
3. **Thích nghi Tối ưu Chi phí (Active Adaptive Retraining):**
   * Chỉ khi có tín hiệu cảnh báo Drift từ STUDD, hệ thống mới yêu cầu nhãn thực tế (True Labels) cho một tập mẫu cục bộ để tái huấn luyện lại Teacher và Student. Nhờ đó, STUDD giúp cân bằng tối ưu giữa **hiệu năng mô hình** và **tiết kiệm tối đa chi phí gán nhãn**.

---

## 2. Các Cột mốc & Bài nộp Đồ án

* **Giai đoạn 1:** Milestone 1 (Chọn bài báo) ──► Nộp form đăng ký (Pass/Fail)
* **Giai đoạn 2:** Milestone 2 (Đọc hiểu) + Lab 1 (Tái hiện thực nghiệm) ──► Nộp **SLIDE GIỮA KỲ (45% điểm)**
  * **Hạn nộp:** 21/11
  * **Sản phẩm nộp:** File `ID.pptx` (20 – 40 slide, dung lượng $\le$ 20MB, bắt buộc kèm Speaker Notes dưới mỗi slide)
  * **Minh chứng đi kèm:** Nộp link công khai repository GitHub mã nguồn
* **Giai đoạn 3:** Milestone 3 (Phản biện lý thuyết) + Lab 2 (Stress test thực nghiệm) ──► Nộp **POSTER CUỐI KỲ (50% điểm)**
  * **Sản phẩm nộp:** File `ID.pptx` (Poster workshop)
* **Kỷ luật thực nghiệm (5% điểm):** Chất lượng repository GitHub và theo dõi thực nghiệm đầy đủ trên Weights & Biases (W&B)
  * **W&B Project Dashboard:** [https://wandb.ai/lgkai-hcmus/STUDD](https://wandb.ai/lgkai-hcmus/STUDD)

---

## 3. Cấu trúc Thư mục Dự án

```text
Project1_STUDD/
├── data/             # Dữ liệu chuẩn (OpenML cache) và dữ liệu giả lập (synthetic)
├── src/              # Mã nguồn cài đặt mô hình STUDD, Page-Hinkley và bộ nạp dữ liệu
│   ├── __init__.py   # Đóng gói package src
│   ├── detector.py   # Module 1: Thuật toán kiểm định tuần tự Page-Hinkley
│   ├── studd.py      # Module 2: Kiến trúc Teacher - Student phát hiện Concept Drift
│   └── data_loader.py# Module 3: Bộ nạp dữ liệu luồng tự động (OpenML & Synthetic)
├── experiments/      # Kịch bản chạy thực nghiệm (Reproduction & Stress Testing)
│   ├── reproduce.py  # Chạy thực nghiệm tái hiện so sánh đối đầu với bài báo
│   └── stress_test.py# Kịch bản kiểm thử chịu tải và phản biện lý thuyết
├── notebooks/        # Jupyter Notebooks phân tích và trực quan hóa kết quả
├── results/          # Biểu đồ trượt xuất ra (figures) và bảng số liệu (tables)
├── docs/             # Bài báo gốc STUDD, Slide báo cáo giữa kỳ và Poster cuối kỳ
├── .gitignore        # Cấu hình bỏ qua file rác, môi trường ảo và dữ liệu nặng
├── requirements.txt  # Danh sách các thư viện phụ thuộc chuẩn Python 3.9+
└── README.md         # Tài liệu giới thiệu đề tài, các cột mốc và cấu trúc dự án
```

---

## 4. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
Khuyến nghị sử dụng Python 3.9 đến 3.11:

* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 2: Cài đặt các thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 3: Đăng nhập tài khoản Weights & Biases (W&B)
Để đồng bộ kết quả thực nghiệm lên dashboard của nhóm:
```powershell
python -m wandb login
```
*(Lấy API key tại [https://wandb.ai/authorize](https://wandb.ai/authorize) và dán vào terminal khi được nhắc)*

### Bước 4: Kiểm thử các module độc lập (Unit Test)
Chạy kiểm tra từng thành phần cốt lõi của hệ thống:
```powershell
# 1. Kiểm thử thuật toán kiểm định tuần tự Page-Hinkley (Module 1)
python src/detector.py

# 2. Kiểm thử khung làm việc Teacher - Student và phát hiện trôi dạt (Module 2)
python src/studd.py

# 3. Kiểm thử bộ nạp dữ liệu luồng tổng hợp và benchmark OpenML (Module 3)
python src/data_loader.py
```

### Bước 5: Chạy thực nghiệm tái hiện (Reproduction Experiments)
Chạy toàn bộ quy trình so sánh 4 chiến lược (`BL-st`, `STUDD`, `SS`, `OS`) trên cả 3 luồng dữ liệu (`abrupt_drift`, `electricity`, `ozone`):
```powershell
python experiments/reproduce.py
```

Sau khi chạy xong:
* Toàn bộ chỉ số, tham số và biểu đồ phân tích được đẩy trực tiếp lên dashboard: [W&B lgkai-hcmus/STUDD](https://wandb.ai/lgkai-hcmus/STUDD).
* Các file trực quan hóa và bảng kết quả chất lượng cao (300 DPI) được lưu tại thư mục `results/`:
  * `results/fig1_rolling_kappa_abrupt_drift.png`: Biểu đồ Kappa score trượt theo thời gian.
  * `results/fig2_performance_vs_cost_comparison.png`: Biểu đồ so sánh hiệu năng (Kappa) và chi phí gán nhãn giữa 4 phương pháp.
  * `results/reproduction_comparison_table.csv`: Bảng tổng hợp số liệu chi tiết.


---

## 4. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
Khuyến nghị sử dụng Python 3.9 đến 3.11:

* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 2: Cài đặt các thư viện phụ thuộc
```powershell
pip install -r requirements.txt
```

### Bước 3: Đăng nhập tài khoản Weights & Biases (W&B)
Để đồng bộ kết quả thực nghiệm lên dashboard của nhóm:
```powershell
python -m wandb login
```
*(Lấy API key tại [https://wandb.ai/authorize](https://wandb.ai/authorize) và dán vào terminal khi được nhắc)*

### Bước 4: Kiểm thử các module độc lập (Unit Test)
Chạy kiểm tra từng thành phần cốt lõi của hệ thống:
```powershell
# 1. Kiểm thử thuật toán kiểm định tuần tự Page-Hinkley (Module 1)
python src/detector.py

# 2. Kiểm thử khung làm việc Teacher - Student và phát hiện trôi dạt (Module 2)
python src/studd.py

# 3. Kiểm thử bộ nạp dữ liệu luồng tổng hợp và benchmark OpenML (Module 3)
python src/data_loader.py
```

### Bước 5: Chạy thực nghiệm tái hiện (Reproduction Experiments)
Chạy toàn bộ quy trình so sánh 4 chiến lược (`BL-st`, `STUDD`, `SS`, `OS`) trên cả 3 luồng dữ liệu (`abrupt_drift`, `electricity`, `ozone`):
```powershell
python experiments/reproduce.py
```

Sau khi chạy xong:
* Toàn bộ chỉ số, tham số và biểu đồ phân tích được đẩy trực tiếp lên dashboard: [W&B lgkai-hcmus/STUDD](https://wandb.ai/lgkai-hcmus/STUDD).
* Các file trực quan hóa và bảng kết quả chất lượng cao (300 DPI) được lưu tại thư mục `results/`:
  * `results/fig1_rolling_kappa_abrupt_drift.png`: Biểu đồ Kappa score trượt theo thời gian.
  * `results/fig2_performance_vs_cost_comparison.png`: Biểu đồ so sánh hiệu năng (Kappa) và chi phí gán nhãn giữa 4 phương pháp.
  * `results/reproduction_comparison_table.csv`: Bảng tổng hợp số liệu chi tiết.