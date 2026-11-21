# STUDD

Đồ án Nhập môn học máy

Nghiên cứu, thực nghiệm và cài đặt lại (reproduce) mô hình **STUDD** dựa trên bài báo khoa học:

> **STUDD: A Student–Teacher Method for Unsupervised Concept Drift Detection**  
> *Vitor Cerqueira, Heitor Murilo Gomes, Albert Bifet, Luis Torgo* (Dalhousie University, University of Waikato, Télécom ParisTech).  
> Công bố tại: **Machine Learning (Springer 2023)**, pages 4351–4378.  
> Xếp hạng (Rank): **Q1** (SCIE / Scopus).

---

**Kết quả tích hợp:** 19 bộ dữ liệu × 10 phương pháp, gồm 60 kết quả của Khoa, 70 của Khánh và 60 của Đạt. [Report đối chiếu](results/README.md) · [CSV 190 kết quả](results/studd_19datasets_summary.csv)

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
   * Theo Cerqueira et al. (2023, Mục 4.1, 5.3 và 6.4), STUDD sử dụng các bộ phân lớp học theo lô (**batch classifiers** như `RF`, `XGB`, `DT`, `LR`, `RC`, ...):
   * **Teacher ($T$):** Được huấn luyện theo lô trên tập dữ liệu khởi tạo $(X_{\text{tr}}, y_{\text{tr}})$, đóng vai trò mô hình tham chiếu đưa ra dự đoán $\hat{y}_{t, T}$.
   * **Student ($S$):** Được huấn luyện ngoại tuyến theo lô (**batch learner**) trên tập $(X_{\text{tr}}, \hat{y}_T)$ để **học bắt chước** hành vi dự đoán của Teacher mà không cần nhãn thật $y_t$ ở bước bắt chước.
   * Trong suốt giai đoạn giám sát trực tuyến trên luồng dữ liệu, cả Teacher ($T$) và Student ($S$) đều được **giữ cố định** để đo độ bất đồng dự đoán $L_t = \mathbb{I}(\hat{y}_{t, T} \neq \hat{y}_{t, S})$; cả hai chỉ được tái huấn luyện theo lô trên cửa sổ $W$ quan sát gần nhất khi kiểm định Page-Hinkley kích hoạt cảnh báo trôi dạt.
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
  * **W&B:** [STUDD](https://wandb.ai/lgkai-hcmus/STUDD) và [Testing](https://wandb.ai/lgkai-hcmus/Testing)

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
│   ├── baselines.py                  # 10 phương pháp đối sánh và độ đo cảnh báo
│   ├── models.py                     # Mô hình và cấu hình của Khánh
│   └── khoa/                         # Phiên bản mã chạy đã dùng cho kết quả của Khoa
├── experiments/                      # Kịch bản chạy thực nghiệm (Reproduction & Stress Testing)
│   ├── reproduce.py                  # Kịch bản tái hiện cơ bản (3 luồng)
│   ├── reproduce_dat_6datasets.py    # Kịch bản tái hiện đầy đủ 10 phương pháp trên 6 bộ (cap 150k)
│   ├── run_dataset.py                # Chạy 7 bộ của Khánh
│   ├── aggregate_studd_results.py    # Tạo CSV và report chung
│   ├── plot_19datasets_rank.py       # Vẽ biểu đồ Average Rank (Hình 10) trên 19 bộ dữ liệu
│   ├── build_khoa_kaggle_notebooks.py # Đóng gói notebook Kaggle của Khoa
│   ├── reproduce_khoa_6datasets.py   # Chạy và đồng bộ 6 bộ của Khoa
│   ├── fullscope_wandb.py            # Chuẩn hóa W&B URL & manifest cho bộ Full Scope
│   ├── verify_table2_selection.py    # Kiểm chứng chọn thuật toán 70/30 Bảng 2 trên 19 bộ
│   ├── analyze_19datasets_full.py    # Bayesian Sign Test (Hình 11-12) & Efficiency (Hình 13)
│   ├── extract_full_alarms_and_figs.py # Tổng hợp Bảng 5 cảnh báo 19 bộ & Hình 7-8
│   ├── run_sensitivity_rq3.py        # Quét lưới l_access × l_delay (Hình 14, RQ3)
│   ├── compare_original_code_protocols.py # Đối chiếu Protocol A/B & ablation seed/detector
│   └── stress_test.py                # Kịch bản kiểm thử chịu tải và phản biện lý thuyết
├── notebooks/                        # Notebook tổng hợp Full Scope & các notebook của Khoa
├── results/                          # Bảng CSV kết quả, report chung và biểu đồ 300 DPI
│   ├── README.md                     # Report tổng hợp & phân tích độ lệch 19 bộ dữ liệu
│   ├── studd_19datasets_summary.csv  # Bảng tổng hợp 190 kết quả (19 bộ × 10 phương pháp)
│   └── fig_19datasets_rank_plot.png  # Biểu đồ Average Rank (Hình 10) trên 19 bộ dữ liệu
├── tests/                            # Kiểm thử tự động xác minh các bản vá cốt lõi & Full Scope
│   ├── test_studd_main_fixes.py      # Bộ kiểm thử tích hợp cho nhánh main (7 tests)
│   └── test_full_scope_analytics.py  # Bộ kiểm thử Full Scope Analytics, Bảng 5 & RQ3 (5 tests)
├── docs/                             # Bài báo gốc STUDD và báo cáo kiểm tra 19 bộ dữ liệu
├── .gitignore                        # Cấu hình bỏ qua môi trường ảo, cache nặng và tài liệu nháp
├── requirements.txt                  # Thư viện cho các script thực nghiệm
└── README.md                         # Tài liệu giới thiệu đề tài, hướng dẫn và bảng ánh xạ W&B
```

---

## 4. Hướng dẫn Cài đặt & Chạy Thực nghiệm

### Bước 1: Khởi tạo và kích hoạt môi trường ảo
Khuyến nghị dùng Python 3.13:

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

* **Chạy từng bộ của Khánh:**
  ```bash
  python experiments/run_dataset.py --dataset bike --no-wandb
  ```
  [Report của Khánh](results/khanh/khanh-7datasets.md) ghi cấu hình, sai khác và các link W&B.

* **Đóng gói 6 bộ của Khoa để chạy trên Kaggle:**
  ```bash
  python experiments/build_khoa_kaggle_notebooks.py --output /tmp/khoa-kaggle-ten
  kaggle kernels push -p /tmp/khoa-kaggle-ten/insects-abrupt
  ```
  Xem [hướng dẫn Kaggle](notebooks/README.md), [đồng bộ W&B](experiments/README.md) và [report của Khoa](results/khoa/README.md). Arabic và ArabicShuffled cần quyền truy cập Kaggle Dataset riêng tư.

* **Chạy bộ phân tích Full Scope trên đủ 19 bộ dữ liệu (Seed `Teacher=42`, `Student=142`, `RNG=42`):**
  ```powershell
  python experiments/verify_table2_selection.py
  python experiments/analyze_19datasets_full.py
  python experiments/extract_full_alarms_and_figs.py
  python experiments/run_sensitivity_rq3.py --resume
  python experiments/compare_original_code_protocols.py
  python -m unittest discover -s tests -v
  ```
  Xem notebook tổng hợp đã thực thi tại [notebooks/STUDD_Full_Scope_Reproduction.ipynb](notebooks/STUDD_Full_Scope_Reproduction.ipynb).

---

## 5. Thư viện và đối chiếu kết quả

### 5.1. Giải trình thư viện ngoài stack `numpy` / `scipy` / `scikit-learn`
* `xgboost`: Bắt buộc để tái hiện đúng bộ phân lớp `XGB` (Extreme Gradient Boosting) được bài báo dùng làm Student/Teacher trong Bảng 2 (ví dụ trên `Electricity`).
* `requests`, `huggingface-hub`, `pyarrow`, `openml`: Hỗ trợ nạp dữ liệu tự động qua mã nguồn từ Hugging Face, USP repository và OpenML kèm kiểm tra định dạng CSV/Parquet.
* `pandas`, `matplotlib`, `seaborn`: Xử lý dữ liệu bảng luồng và xuất biểu đồ phân tích chuẩn 300 DPI.
* `wandb`: Theo dõi và lưu trữ tập trung cấu hình, chỉ số, bảng biểu và đồ thị tại [W&B lgkai-hcmus/STUDD](https://wandb.ai/lgkai-hcmus/STUDD).

### 5.2. Kết quả theo dataset

[Report chung](results/README.md) đối chiếu Kappa và chi phí nhãn của STUDD trên 19 bộ với Bảng 3–4. [CSV tổng hợp](results/studd_19datasets_summary.csv) lưu đủ 10 phương pháp mỗi bộ cùng nguồn và cấu hình tương ứng, đi kèm [Biểu đồ Average Rank 19 bộ](results/fig_19datasets_rank_plot.png) tái hiện Hình 10 của bài báo. Tái tạo báo cáo tích hợp và biểu đồ Hình 10 từ các kết quả đã ghi nhận bằng `python experiments/aggregate_studd_results.py` và `python experiments/plot_19datasets_rank.py`.

> **Lưu ý về độ lệch chính (Deviation Analysis):** Phân tích độ lệch chi tiết trên cả 19 bộ dữ liệu của cả 3 thành viên (**Khoa, Khánh, Đạt**) được tổng hợp đầy đủ tại [results/README.md](results/README.md) (cùng báo cáo chi tiết của [Khoa](results/khoa/README.md) và [Khánh](results/khanh/khanh-7datasets.md)). Một số nguyên nhân sai khác tiêu biểu so với Bảng 3–4 của bài báo gồm: (1) khác biệt cỡ cửa sổ giữa văn bản Mục 5.7 (`W = 2000`) và mã nguồn gốc `vcerqueira/studd` (`W = 1000` cho các luồng lớn, khớp tỷ lệ `BL-st = 1000 / N` trong Bảng 4); (2) trên `Poker` (`DT/DT`), do `DT` ghi nhớ tốt cửa sổ $W=2000$, thống kê Page-Hinkley cực đại chưa vượt ngưỡng mặc định $\lambda=50$ nên không kích hoạt báo động; (3) khác biệt số đặc trưng upstream (như `Keystroke`, `Arabic`), xử lý reset cửa sổ của `OF`/`FF` và phiên bản `scikit-learn` / `xgboost` mới. Dữ liệu riêng 6 bộ của Đạt cũng được lưu trong [results/dat_6datasets_summary.csv](results/dat_6datasets_summary.csv) và [results/dat_6datasets_alarm_summary.csv](results/dat_6datasets_alarm_summary.csv).

## Dataset branch snapshot

The `data` branch adds the dataset files and [19-dataset source report](docs/dataset_reports/STUDD_19_DATASETS_REPORT.md). Original HF links and external upstream links are both retained. See [data/README.md](data/README.md) for folder meanings and verification limitations. These files do not constitute new model results.
