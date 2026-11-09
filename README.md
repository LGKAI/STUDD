# STUDD

Đồ án Nhập môn học máy

---

## Các Cột mốc & Bài nộp Đồ án

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

## Cấu trúc Thư mục Dự án

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
