# Notebook thực nghiệm STUDD

## Notebook tổng hợp Full Scope (Khuyến nghị xem trước)

* [`STUDD_Full_Scope_Reproduction.ipynb`](STUDD_Full_Scope_Reproduction.ipynb): Notebook tổng hợp toàn trình (kèm đầy đủ output & biểu đồ nhúng sẵn) gọi trực tiếp các module trong `src/` và `experiments/`, bao quát 6 hạng mục Full Scope với quy chuẩn hạt giống đồng nhất (`Teacher=42`, `Student=142`, `RNG=42`):
  1. Bộ nạp 19 luồng dữ liệu & fallback 3 tầng cho `Arabic` / `ArabicShuffled`
  2. Kiểm chứng quy trình chọn thuật toán 70/30 (Mục 5.6.2 & Bảng 2)
  3. Tổng hợp 19 bộ (Bảng 3–4), Bayesian Sign Test (Hình 11–12) & Labeling Efficiency (Hình 13)
  4. Phân tích cảnh báo Bảng 5 trên 19 bộ & đồ thị chuẩn hóa Hình 7–8
  5. Phân tích độ nhạy `l_access × l_delay` trên 19 bộ (Mục 5.8.5, Hình 14, RQ3)
  6. Đối chiếu có kiểm soát Paper-Text vs Author-Code & Ablation nguyên nhân sai lệch

## Phần của Khoa

Mỗi notebook chạy **BL-st** và **STUDD** trên một bộ dữ liệu. Cặp mô hình lấy từ Bảng 2 và số tham chiếu Kappa lấy từ Bảng 3 của paper.

| Notebook | Teacher / Student | Mẫu train đầu | Mẫu đánh giá | Nguồn chạy |
| --- | --- | ---: | ---: | --- |
| [01 InsectsAbrupt](khoa/01-insects-abrupt.ipynb) | RF / RF | 500 | 4.825 | Hugging Face |
| [02 Insects](khoa/02-insects.ipynb) | XGB / XGB | 500 | 4.825 | Hugging Face |
| [03 Keystroke](khoa/03-keystroke.ipynb) | XGB / RF | 500 | 1.100 | Hugging Face |
| [04 Arabic](khoa/04-arabic.ipynb) | XGB / XGB | 2.000 | 6.800 | Kaggle Dataset riêng tư |
| [05 ArabicShuffled](khoa/05-arabic-shuffled.ipynb) | RF / RF | 2.000 | 6.800 | Kaggle Dataset riêng tư |
| [06 Posture](khoa/06-posture.ipynb) | XGB / RF | 2.000 | 148.000 | Hugging Face |

### Chạy trên Kaggle

1. Tạo Kaggle Notebook rồi dùng **File → Import Notebook** để mở file `.ipynb`. Bật **Internet** và để **Accelerator = None**.
2. Chạy cell cài thư viện đầu tiên trong một phiên mới. Notebook có sẵn lớp Page–Hinkley từ scikit-multiflow 0.5.3, nên không cần cài toàn bộ gói scikit-multiflow trên Kaggle.
3. Cấp quyền cho secret `WANDB_API_KEY` và đặt `LOG_TO_WANDB = True` ở cell 2 nếu muốn gửi kết quả vào [STUDD](https://wandb.ai/lgkai-hcmus/STUDD).
4. Chạy các cell theo thứ tự. Cell 5 in bảng Kappa và lưu CSV. Cell 6 in URL của hai run. Chép URL đã xác nhận **Finished** vào README chính. Chạy lại cell 6 sẽ tạo run mới.

Chạy lại **01 InsectsAbrupt** để lấy kết quả với cấu hình detector trong paper. Các run mới có hậu tố `paper-seed42`. Với Arabic và ArabicShuffled, gắn [dataset Kaggle riêng tư của Khoa](https://www.kaggle.com/datasets/phmchnkhoa/studd-arabic-benchmark-inputs). Kaggle giải nén hai ZIP upstream khi nhập. Cell 3 kiểm tra SHA-256 của tệp đã giải nén và giữ nguyên thứ tự luồng.

> **Lưu ý trước khi nộp bài / chấm điểm:** Trước khi nộp bản cuối cùng để giảng viên đánh giá, cần chuyển dataset Kaggle `phmchnkhoa/studd-arabic-benchmark-inputs` sang chế độ **Public** trên Kaggle (hoặc mirror công khai lên Hugging Face) để người đánh giá có thể tái hiện trực tiếp `04-arabic.ipynb` và `05-arabic-shuffled.ipynb`.

**Sai khác cần ghi trong báo cáo.** Keystroke có 10 đặc trưng và Arabic có 26 đặc trưng ở bản upstream. Bảng 1 tương ứng với 9 và 27 đặc trưng. Posture có 164.860 dòng ở nguồn và notebook dùng 150.000 dòng đầu theo giới hạn trong paper. Page–Hinkley có `delta=0.001` và các tham số còn lại theo mặc định của scikit-multiflow. Notebook dùng trực tiếp cặp mô hình Bảng 2. Lượt chạy toàn bộ mã tác giả là phần đối chiếu riêng, vì script công khai dùng các giá trị khác mục 5.7.

[Nguồn dữ liệu của nhóm](../docs/dataset_reports/STUDD_19_DATASETS_REPORT.md) · [Bài báo](https://doi.org/10.1007/s10994-022-06188-7)

### Chạy 10 phương pháp qua Kaggle CLI

Đợt chạy này dùng baseline, mô hình và Page–Hinkley của nhánh `dat-6datasets` tại revision `00834c380c32ff41ed7746d41e949babe3aca658`. Các notebook hai phương pháp ở trên giữ detector của lượt chạy trước.

```bash
python3 experiments/build_khoa_kaggle_notebooks.py --output /tmp/khoa-kaggle-ten
kaggle kernels push -p /tmp/khoa-kaggle-ten/insects-abrupt
```

Mỗi job xuất CSV kết quả, lịch sử accuracy trượt và vị trí cảnh báo. [Report tổng hợp](../results/README.md)
