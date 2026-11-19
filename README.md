# STUDD

## Thực nghiệm của Khoa

Khoa chạy **10 phương pháp trên 6 bộ dữ liệu** bằng Kaggle CLI. Trình chạy dùng cùng phiên bản Page–Hinkley và baseline với nhánh `dat-6datasets`.

- [Notebook và hướng dẫn Kaggle](notebooks/README.md) · [Script chạy thực nghiệm](experiments/reproduce_khoa_6datasets.py)
- [Report đối chiếu với Bảng 3–4 và link từng run](results/khoa/README.md) · [CSV tổng hợp 60 kết quả](results/khoa/khoa_6datasets_summary.csv)
- [W&B Testing](https://wandb.ai/lgkai-hcmus/Testing). Mỗi run tổng hợp 10 phương pháp của một dataset.

Arabic và ArabicShuffled dùng input Kaggle riêng tư. Các file cấu hình trong `results/khoa/` lưu nguồn và checksum của các bộ đã chạy.


Đồ án Nhập môn học máy

Nghiên cứu, thực nghiệm và cài đặt lại (reproduce) mô hình **STUDD** dựa trên bài báo khoa học:

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
├── data/                             # Dữ liệu chuẩn (hosted HF/USP cache) và manifest nguồn
│   └── hosted_sources.json           # Cấu hình nguồn tải tự động & SHA-256
├── src/                              # Mã nguồn tự cài đặt STUDD, Page-Hinkley, baselines & data loader
│   ├── __init__.py                   # Đóng gói package src
│   ├── detector.py                   # Module 1: Thuật toán kiểm định tuần tự Page-Hinkley
│   ├── studd.py                      # Module 2: Kiến trúc Teacher - Student & mô hình Bảng 2
│   ├── data_loader.py                # Module 3: Bộ nạp luồng tự động từ HF/USP/OpenML
│   └── baselines.py                  # Module 4: 10 phương pháp đối sánh (Mục 5.5) & độ đo cảnh báo
├── experiments/                      # Kịch bản chạy thực nghiệm (Reproduction & Stress Testing)
│   ├── reproduce.py                  # Kịch bản tái hiện cơ bản (3 luồng)
│   ├── reproduce_dat_6datasets.py    # Kịch bản tái hiện đầy đủ 10 phương pháp trên 6 bộ (cap 150k)
│   └── stress_test.py                # Kịch bản kiểm thử chịu tải và phản biện lý thuyết
├── notebooks/                        # Jupyter Notebooks phân tích và trực quan hóa kết quả
├── results/                          # Bảng CSV kết quả và biểu đồ 300 DPI
├── docs/                             # Bài báo gốc STUDD và báo cáo kiểm tra 19 bộ dữ liệu
├── .gitignore                        # Cấu hình bỏ qua môi trường ảo, cache nặng và tài liệu nháp
├── requirements.txt                  # Danh sách thư viện phụ thuộc Python 3.9+
└── README.md                         # Tài liệu giới thiệu đề tài, hướng dẫn và bảng ánh xạ W&B
```

---

## 4. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
Khuyến nghị sử dụng Python 3.9 trở lên (đã kiểm thử trên Python 3.13):

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
```powershell
# 1. Kiểm thử thuật toán kiểm định tuần tự Page-Hinkley (Module 1)
python src/detector.py

# 2. Kiểm thử khung làm việc Teacher - Student & mô hình Bảng 2 (Module 2)
python src/studd.py

# 3. Kiểm thử bộ nạp tự động 6 bộ dữ liệu (Module 3)
python src/data_loader.py

# 4. Kiểm thử đầy đủ 10 phương pháp đối sánh & độ đo cảnh báo (Module 4)
python src/baselines.py
```

### Bước 5: Chạy thực nghiệm tái hiện (Reproduction Experiments)

* **Chạy thử nghiệm cơ bản (3 luồng):**
  ```powershell
  python experiments/reproduce.py
  ```
* **Chạy tái hiện đầy đủ 10 phương pháp trên 6 bộ dữ liệu (tuần tự, giới hạn 150.000 mẫu/luồng):**
  ```powershell
  # Chạy offline toàn bộ 6 bộ (hoặc chọn riêng với --datasets electricity)
  python experiments/reproduce_dat_6datasets.py --no-wandb --aggregate

  # Chạy và đẩy trực tiếp lên W&B (sau khi đã wandb login)
  python experiments/reproduce_dat_6datasets.py --aggregate

  # Hoặc đồng bộ các kết quả CSV/PNG đã chạy từ results/ lên W&B
  python experiments/reproduce_dat_6datasets.py --sync-wandb
  ```

---

## 5. Thư viện ngoài Stack chuẩn & Bảng Ánh xạ Thực nghiệm (6 Bộ dữ liệu)

### 5.1. Giải trình thư viện ngoài stack `numpy` / `scipy` / `scikit-learn`
* `xgboost`: Bắt buộc để tái hiện đúng bộ phân lớp `XGB` (Extreme Gradient Boosting) được bài báo dùng làm Student/Teacher trong Bảng 2 (ví dụ trên `Electricity`).
* `requests`, `huggingface-hub`, `pyarrow`, `openml`: Hỗ trợ nạp dữ liệu tự động qua mã nguồn từ Hugging Face, USP repository và OpenML kèm kiểm tra định dạng CSV/Parquet.
* `pandas`, `matplotlib`, `seaborn`: Xử lý dữ liệu bảng luồng và xuất biểu đồ phân tích chuẩn 300 DPI.
* `wandb`: Theo dõi và lưu trữ tập trung cấu hình, chỉ số, bảng biểu và đồ thị tại [W&B lgkai-hcmus/STUDD](https://wandb.ai/lgkai-hcmus/STUDD).

### 5.2. Tổng hợp kết quả tái hiện 10 phương pháp trên 6 bộ dữ liệu (Đạt, $W = 2000$, cap $150,000$)

| Bộ dữ liệu | Số mẫu ($N$) | Cặp T / S | STUDD Kappa (Ours / Paper) | STUDD Cost (Ours / Paper) | Script tạo kết quả | Kết quả lưu trữ & W&B |
|---|---|---|---|---|---|---|
| **Electricity** | 45,312 | `NB` / `XGB` | `0.4166` / `0.29` | `0.5738` / `0.43` | `experiments/reproduce_dat_6datasets.py --datasets electricity` | [electricity_reproduction.csv](results/electricity_reproduction.csv) |
| **Powersupply** | 29,928 | `SVC` / `RF` | `0.1067` / `0.10` | `0.2005` / `0.07` | `experiments/reproduce_dat_6datasets.py --datasets powersupply` | [powersupply_reproduction.csv](results/powersupply_reproduction.csv) |
| **NOAA** | 18,159 | `RF` / `RF` | `0.4036` / `0.39` | `0.2203` / `0.06` | `experiments/reproduce_dat_6datasets.py --datasets noaa` | [noaa_reproduction.csv](results/noaa_reproduction.csv) |
| **CoverType** | 150,000 | `LR` / `LR` | `0.5046` / `0.52` | `0.2800` / `0.09` | `experiments/reproduce_dat_6datasets.py --datasets covertype` | [covertype_reproduction.csv](results/covertype_reproduction.csv) |
| **Poker** | 150,000 | `DT` / `DT` | `0.2960` / `0.48` | `0.0133` / `0.01` | `experiments/reproduce_dat_6datasets.py --datasets poker` | [poker_reproduction.csv](results/poker_reproduction.csv) |
| **Sensors** | 150,000 | `DT` / `DT` | `0.2684` / `0.59` | `0.4800` / `0.16` | `experiments/reproduce_dat_6datasets.py --datasets sensors` | [sensors_reproduction.csv](results/sensors_reproduction.csv) |

> **Lưu ý về độ lệch chính (Deviation Analysis):** Một số khác biệt giữa bản tái hiện (`W = 2000` theo Mục 5.7 văn bản bài báo) và Bảng 3–4 của bài báo bắt nguồn từ: (1) mã nguồn gốc `vcerqueira/studd` sử dụng `W = 1000` cho các luồng lớn (khớp tỷ lệ `BL-st = 1000 / N` trong Bảng 4); (2) trên `Poker` (`DT/DT`), do `DT` ghi nhớ tốt cửa sổ $W=2000$, thống kê Page-Hinkley cực đại chưa vượt ngưỡng mặc định $\lambda=50$ nên không kích hoạt báo động; (3) khác biệt xử lý reset cửa sổ của `OF`/`FF` và phiên bản `scikit-learn` / `xgboost` mới. Chi tiết đầy đủ được lưu trong [results/dat_6datasets_summary.csv](results/dat_6datasets_summary.csv) và [results/dat_6datasets_alarm_summary.csv](results/dat_6datasets_alarm_summary.csv).

## Dataset branch snapshot

The `data` branch adds the dataset files and [19-dataset source report](docs/dataset_reports/STUDD_19_DATASETS_REPORT.md). Original HF links and external upstream links are both retained. See [data/README.md](data/README.md) for folder meanings and verification limitations. These files do not constitute new model results.
