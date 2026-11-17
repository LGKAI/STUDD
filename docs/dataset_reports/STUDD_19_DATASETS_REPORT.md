# Report nguồn Hugging Face cho 19 dataset STUDD

Ngày kiểm tra: **17/11/2026**. Nguồn danh sách/kích thước: Bảng 1, mục 5.2 và 5.7 trong [paper](../Paper%20STUDD%20-%202022.pdf).

## Kết luận

- **13/19 có nguồn Hugging Face đối chiếu được với file upstream:** ba bộ đã chuẩn bị trước và 10 mirror vừa kiểm tra.
- **10 mirror mới đều Public**, có file dữ liệu, README và provenance; hash file trên HF khớp file nguồn trên máy. Commit và link download cố định được ghi ở từng mục.
- **6/19 vẫn là ứng viên/dữ liệu nền:** CoverType, Poker, Ozone, Arabic, ArabicShuffled, GMSC; việc upload 10 bộ không giải quyết khác biệt preprocessing của sáu mục này.
- Khớp hash upstream không chứng minh khớp dữ liệu paper. Những khác biệt shape, nhãn, header, Powersupply và điều khoản Gas vẫn cần theo dõi.
- **Giữ nguyên các link ngoài HF** (USP Drive, incremental-ks) để truy vết nguồn. Dữ liệu local do người dùng tổ chức trong `data/dataset_huggingface/` và `data/dataset_not_huggingface/`.
- Chi tiết API public, commit và hash được lưu trong `hf_published_verification.json`. Kiểm tra không dùng API token.

Dung lượng dưới đây dùng **MB = 1.000.000 byte**. Size HF là file theo API tree tại commit ghi trong audit; không dùng nhãn size_categories vì có card ghi sai. Size nguồn đầy đủ có thể lớn hơn dữ liệu thực chạy sau cắt 150.000 mẫu.

## Danh mục từng dataset

### 1. Electricity

- Paper: 45,312 mẫu × 9 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 45,312 mẫu.
- Trạng thái: **Đã xác minh với upstream**.
- Hugging Face: [mstz/electricity](https://huggingface.co/datasets/mstz/electricity). Revision: `5f26b8dd2ecdd56f56694fe4f260a1adaa6d77a9`.
- [File tải cố định](https://huggingface.co/datasets/mstz/electricity/resolve/5f26b8dd2ecdd56f56694fe4f260a1adaa6d77a9/electricity/train.csv).
- Dung lượng nguồn đã đo: **3.00 MB**.
- CSV làm việc ước lượng theo shape paper: **2.45–7.34 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Khớp toàn bộ giá trị, nhãn và thứ tự với ARFF tham chiếu sau mapping is_up 1/0 → UP/DOWN.

### 2. CoverType

- Nguồn ngoài HF: [USP — ForestCoverType.csv](https://drive.google.com/file/d/1V40LhiMLJfnwlY5keoFiLEOoU6zmIZrx/view); [link download](https://drive.google.com/uc?export=download&id=1V40LhiMLJfnwlY5keoFiLEOoU6zmIZrx). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 581,012 mẫu × 55 cột tổng (gồm nhãn), 5 lớp; chạy tối đa 150,000 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [mstz/covertype](https://huggingface.co/datasets/mstz/covertype). Revision: `72ae206f21f416f3e7981183537e9ab071aff006`.
- Dung lượng nguồn ứng viên theo metadata: **75.17 MB**.
- CSV làm việc ước lượng theo shape paper: **49.50–148.50 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Card mô tả 7 lớp; paper Bảng 1 ghi 5. Chưa kiểm tra nội dung/luồng và cách lọc lớp. Bản jxie/covtype chia train/val/test, không ghép tùy ý để làm luồng.

### 3. Poker

- Nguồn ngoài HF: [USP — PokerHand.csv](https://drive.google.com/file/d/1cGcAHeSYUoyN_JCo9SYKneG9pn4VeQ2k/view); [link download](https://drive.google.com/uc?export=download&id=1cGcAHeSYUoyN_JCo9SYKneG9pn4VeQ2k). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 1,025,010 mẫu × 12 cột tổng (gồm nhãn), 10 lớp; chạy tối đa 150,000 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [yzhuang/metatree_pokerhand](https://huggingface.co/datasets/yzhuang/metatree_pokerhand). Revision: `1b04c2227c63fdc4dff43d71f81fca4fb894c949`.
- Dung lượng nguồn ứng viên theo metadata: **7.12 MB**.
- CSV làm việc ước lượng theo shape paper: **10.80–32.40 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Metadata có 580.432 train + 248.769 validation = 829.201 mẫu, không khớp 1.025.010 trong paper. X là sequence uint8, có id; chưa chứng minh phục hồi được thứ tự và đủ mẫu. Chỉ là ứng viên họ dữ liệu.

### 4. Gas

- Hugging Face của nhóm: [pham121/studd-gas](https://huggingface.co/datasets/pham121/studd-gas).
- Commit cố định: `9a73335cbe0368e5ae0a57245b1c9dabcbed5e6b`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-gas/resolve/9a73335cbe0368e5ae0a57245b1c9dabcbed5e6b/GasSensorArray.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (sha256); dung lượng **17.28 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — GasSensorArray.csv](https://drive.google.com/file/d/11igK96DPWn8cgIwHfp2_w8JdpmTYsHVz/view); [link download](https://drive.google.com/uc?export=download&id=11igK96DPWn8cgIwHfp2_w8JdpmTYsHVz). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Danh mục USP ghi Gas có 128 feature, khác 16 feature suy ra từ Bảng 1 paper; cần xác minh bước chọn/tổng hợp đặc trưng.
- Paper: 13,910 mẫu × 17 cột tổng (gồm nhãn), 6 lớp; chạy tối đa 13,910 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **1.42–4.26 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Chưa xác định được đúng bản benchmark. Các bản gas-chamber năm 2025 và synthetic MOX không tương đương.

### 5. Luxembourg

- Hugging Face của nhóm: [pham121/studd-luxembourg](https://huggingface.co/datasets/pham121/studd-luxembourg).
- Commit cố định: `5ccd00c9a23ed7f10123a0903b64d0e3db3f01d8`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-luxembourg/resolve/5ccd00c9a23ed7f10123a0903b64d0e3db3f01d8/Luxembourg.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **0.22 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — Luxembourg.csv](https://drive.google.com/file/d/1BwiUIE97rZbZbMHGODbWaGpSJ7DmGwMB/view); [link download](https://drive.google.com/uc?export=download&id=1BwiUIE97rZbZbMHGODbWaGpSJ7DmGwMB). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 1,901 mẫu × 31 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 1,901 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **0.35–1.06 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần bộ khảo sát Internet 2002–2007; các corpus ngôn ngữ Luxembourg không đúng bài toán.

### 6. Ozone

- Nguồn ngoài HF: [USP — Ozone.csv](https://drive.google.com/file/d/1WNsYV4-kc51S9E6x1JuniRZ0LM6EEZjB/view); [link download](https://drive.google.com/uc?export=download&id=1WNsYV4-kc51S9E6x1JuniRZ0LM6EEZjB). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- USP catalog ghi 2.534 mẫu, 72 feature; cũng không khớp shape Bảng 1 paper.
- Paper: 2,574 mẫu × 32 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 2,574 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [mstz/ozone](https://huggingface.co/datasets/mstz/ozone). Revision: `5d212857f5ff830b714abdeb014532aa34c3e659`.
- Dung lượng nguồn ứng viên theo metadata: **1.27 MB** (cả hai config; mỗi config khoảng 0,637 MB).
- CSV làm việc ước lượng theo shape paper: **0.49–1.48 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Đã đọc CSV: 1hr có 1.848 dòng, 8hr có 1.847 dòng; cả hai 73 cột. Không khớp paper 2.574 × 32. Cần đúng bản xử lý missing/chọn đặc trưng; không tự cắt cột để khớp.

### 7. Sensors

- Hugging Face của nhóm: [pham121/studd-sensors](https://huggingface.co/datasets/pham121/studd-sensors).
- Commit cố định: `6a197a84d50fdb4c758ad7c26484f3aa4b9dbe80`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-sensors/resolve/6a197a84d50fdb4c758ad7c26484f3aa4b9dbe80/SensorStream.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (sha256); dung lượng **88.70 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — SensorStream.csv](https://drive.google.com/file/d/1T8QhOH4z6gV2yvB4gA7dyisvnotKcy6t/view); [link download](https://drive.google.com/uc?export=download&id=1T8QhOH4z6gV2yvB4gA7dyisvnotKcy6t). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Danh mục USP ghi 5 feature, trong khi Bảng 1 paper ghi 5 cột tổng; không tự bỏ một feature để khớp.
- Paper: 2,219,803 mẫu × 5 cột tổng (gồm nhãn), 54 lớp; chạy tối đa 150,000 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **4.50–13.50 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần nhận diện 54 sensor từ dữ liệu môi trường. Không thay bằng Gas hoặc dữ liệu IoT có tên gần giống.

### 8. Powersupply

- Hugging Face của nhóm: [pham121/studd-powersupply](https://huggingface.co/datasets/pham121/studd-powersupply).
- Commit cố định: `72bb16473088cf5c29ab8721ae892db4ffd4af67`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-powersupply/resolve/72bb16473088cf5c29ab8721ae892db4ffd4af67/Powersupply.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **0.40 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — Powersupply.csv](https://drive.google.com/file/d/1YWncempzW_MDMJcRvyh377449ytOSh9g/view); [link download](https://drive.google.com/uc?export=download&id=1YWncempzW_MDMJcRvyh377449ytOSh9g). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 29,928 mẫu × 2 cột tổng (gồm nhãn), 24 lớp; chạy tối đa 29,928 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **0.36–1.08 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần bộ nhận diện giờ, 24 lớp từ công ty điện Italy; dữ liệu robot cắm sạc không liên quan.

### 9. Rialto

- Hugging Face của nhóm: [pham121/studd-rialto](https://huggingface.co/datasets/pham121/studd-rialto).
- Commit cố định: `9a876990a741fded06b2a6ede80302fe83e12eb1`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-rialto/resolve/9a876990a741fded06b2a6ede80302fe83e12eb1/Rialto.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **9.74 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — Rialto.csv](https://drive.google.com/file/d/1TVGYtJ1VnFFhAgrG5oZd1J8Jk6ShRbMi/view); [link download](https://drive.google.com/uc?export=download&id=1TVGYtJ1VnFFhAgrG5oZd1J8Jk6ShRbMi). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 82,250 mẫu × 27 cột tổng (gồm nhãn), 10 lớp; chạy tối đa 82,250 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **13.32–39.97 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần đặc trưng ảnh đã xử lý, 10 lớp; ảnh/corpus có nhắc cầu Rialto không phải benchmark.

### 10. Outdoor

- Hugging Face của nhóm: [pham121/studd-outdoor](https://huggingface.co/datasets/pham121/studd-outdoor).
- Commit cố định: `34d9a05ba9f8b3bf385ab78b66b5eccb0301611f`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-outdoor/resolve/34d9a05ba9f8b3bf385ab78b66b5eccb0301611f/Outdoor.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **0.57 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — Outdoor.csv](https://drive.google.com/file/d/1u-5rDllhrsUKzrHnvouD-gQwqDLIeOjt/view); [link download](https://drive.google.com/uc?export=download&id=1u-5rDllhrsUKzrHnvouD-gQwqDLIeOjt). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 4,000 mẫu × 21 cột tổng (gồm nhãn), 40 lớp; chạy tối đa 4,000 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **0.50–1.51 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần 4.000 mẫu, 40 lớp đối tượng dưới điều kiện ánh sáng thay đổi; không phải ảnh outdoor tổng quát.

### 11. Keystroke

- Hugging Face của nhóm: [pham121/studd-keystroke](https://huggingface.co/datasets/pham121/studd-keystroke).
- Commit cố định: `38da3e5d1a59471dee117d9e7dfe83e3a8e9d0c0`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-keystroke/resolve/38da3e5d1a59471dee117d9e7dfe83e3a8e9d0c0/Keystroke.data).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **0.11 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [incremental-ks — Keystroke.zip](https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Keystroke.zip); nguồn supplemental được paper dẫn. File đã được tải và kiểm tra checksum/shape; chưa xác nhận khớp preprocessing paper.
- Paper: 1,600 mẫu × 10 cột tổng (gồm nhãn), 4 lớp; chạy tối đa 1,600 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **0.10–0.29 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Các nguồn Aalto/136M trên HF khác bộ 1.600 mẫu, 4 lớp của paper.

### 12. NOAA

- Hugging Face của nhóm: [pham121/studd-noaa](https://huggingface.co/datasets/pham121/studd-noaa).
- Commit cố định: `1b6992238d2c8fee692f5b5fd24de691380b7bc2`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-noaa/resolve/1b6992238d2c8fee692f5b5fd24de691380b7bc2/NOAA.csv).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **0.70 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [USP — NOAA.csv](https://drive.google.com/file/d/1lfisx3jZsj7Y1UM5HcYaMr9nOzXsTrEX/view); [link download](https://drive.google.com/uc?export=download&id=1lfisx3jZsj7Y1UM5HcYaMr9nOzXsTrEX). File đã tải vào data/dataset_not_huggingface; chưa đối chiếu preprocessing với paper.
- Paper: 18,159 mẫu × 8 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 18,159 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **0.87–2.61 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Cần luồng rain/no-rain 50 năm. KunhanGao/NOAA-weather chứa forecasting/infill/QA, không phải bản benchmark đã chứng minh tương đương.

### 13. Bike

- Paper: 17,378 mẫu × 5 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 17,378 mẫu.
- Trạng thái: **Đã xác minh với upstream**.
- Hugging Face: [t22000t/bike-sharing-tabular](https://huggingface.co/datasets/t22000t/bike-sharing-tabular). Revision: `6b645e0cedbc5208fcca11ac94544b088478bf61`.
- [File tải cố định](https://huggingface.co/datasets/t22000t/bike-sharing-tabular/resolve/6b645e0cedbc5208fcca11ac94544b088478bf61/hour.csv).
- Dung lượng nguồn đã đo: **1.16 MB**.
- CSV làm việc ước lượng theo shape paper: **0.52–1.56 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Đã tái tạo khớp tất cả 17.379 mẫu nguồn: chọn temp/atemp/hum/windspeed, high nếu cnt ≥ 142. Paper ghi 17.378, giữ nguyên 17.379 và báo chênh lệch.

### 14. Arabic

- Nguồn ngoài HF: [incremental-ks — Arabic_Paper_Fixed.zip](https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Arabic_Paper_Fixed.zip); nguồn supplemental được paper dẫn. Link đã xác nhận trong kho; ZIP này chưa tải ở lượt tải 10 bộ, chưa đối chiếu preprocessing paper.
- Paper: 8,800 mẫu × 28 cột tổng (gồm nhãn), 10 lớp; chạy tối đa 8,800 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [ailuntz/SpokenArabicDigits](https://huggingface.co/datasets/ailuntz/SpokenArabicDigits). Revision: `5f2e9861cfc1dff9caa9e37cd43aa22d38f05ee9`.
- Dung lượng nguồn ứng viên theo metadata: **37.61 MB**.
- CSV làm việc ước lượng theo shape paper: **1.48–4.44 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Có TRAIN/TEST .ts của SpokenArabicDigits. Chưa xác minh đủ mẫu, thống kê MFCC, cột giới tính hoặc thứ tự luồng. Không phải bản tabular 28 cột sẵn dùng.

### 15. ArabicShuffled

- Nguồn ngoài HF: [incremental-ks — ArabicDigit_Shuffled_With_Sex.zip](https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/ArabicDigit_Shuffled_With_Sex.zip); nguồn supplemental được paper dẫn. Link đã xác nhận trong kho; ZIP này chưa tải ở lượt tải 10 bộ, chưa đối chiếu preprocessing paper.
- Paper: 8,800 mẫu × 28 cột tổng (gồm nhãn), 10 lớp; chạy tối đa 8,800 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [ailuntz/SpokenArabicDigits](https://huggingface.co/datasets/ailuntz/SpokenArabicDigits). Revision: `5f2e9861cfc1dff9caa9e37cd43aa22d38f05ee9`.
- Dung lượng nguồn ứng viên theo metadata: **37.61 MB**.
- CSV làm việc ước lượng theo shape paper: **1.48–4.44 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Chỉ là nguồn dữ liệu nền như Arabic, không phải mirror ArabicShuffled. Phải tái tạo thứ tự theo giới tính đúng nguồn dos Reis; không shuffle ngẫu nhiên thay thế.

### 16. Insects

- Hugging Face của nhóm: [pham121/studd-insects](https://huggingface.co/datasets/pham121/studd-insects).
- Commit cố định: `4fa53b650844ef482a59e486a2927e84cf64e09a`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-insects/resolve/4fa53b650844ef482a59e486a2927e84cf64e09a/Insects.data).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **2.49 MB**. License metadata hiển thị: `mit` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [incremental-ks — Insects.zip](https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Insects.zip); nguồn supplemental được paper dẫn. File đã được tải và kiểm tra checksum/shape; chưa xác nhận khớp preprocessing paper.
- Paper: 5,325 mẫu × 50 cột tổng (gồm nhãn), 5 lớp; chạy tối đa 5,325 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **1.60–4.79 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Mirror HF hiện đã có đúng file upstream benchmark 5 lớp, 49 feature. Không dùng bản INSECTS mới 6 lớp/33 feature hoặc InsectSound.

### 17. InsectsAbrupt

- Paper: 5,325 mẫu × 50 cột tổng (gồm nhãn), 5 lớp; chạy tối đa 5,325 mẫu.
- Trạng thái: **Đã xác minh với upstream**.
- Hugging Face: [pham121/studd-insects-abrupt](https://huggingface.co/datasets/pham121/studd-insects-abrupt). Revision: `90238617f8f88f5ae0f99350e65abc1f194bf37a`.
- [File tải cố định](https://huggingface.co/datasets/pham121/studd-insects-abrupt/resolve/90238617f8f88f5ae0f99350e65abc1f194bf37a/AbruptInsects.zip).
- Dung lượng nguồn đã đo: **0.75 MB**.
- CSV làm việc ước lượng theo shape paper: **1.60–4.79 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Mirror nhóm đã đăng, giữ nguyên archive upstream. Đã xác minh 5.325 dòng, 49 feature, nhãn và thứ tự.

### 18. Posture

- Hugging Face của nhóm: [pham121/studd-posture](https://huggingface.co/datasets/pham121/studd-posture).
- Commit cố định: `cfd7b8b678acb60f888901839e2c109e6b5a589f`; [file tải cố định](https://huggingface.co/datasets/pham121/studd-posture/resolve/cfd7b8b678acb60f888901839e2c109e6b5a589f/Posture.data).
- Kiểm tra: **Public, đủ file, hash dữ liệu khớp bản upstream trên máy** (git-blob-sha1); dung lượng **10.04 MB**. License metadata hiển thị: `cc-by-4.0` (không phải xác nhận pháp lý độc lập).
- Nguồn ngoài HF: [incremental-ks — Posture.zip](https://raw.githubusercontent.com/denismr/incremental-ks/c6bc98833db67d9dc90a0f997a11877860966337/datasets/Posture.zip); nguồn supplemental được paper dẫn. File đã được tải và kiểm tra checksum/shape; chưa xác nhận khớp preprocessing paper.
- Paper: 164,859 mẫu × 4 cột tổng (gồm nhãn), 11 lớp; chạy tối đa 150,000 mẫu.
- Trạng thái: **Đã có mirror HF, hash khớp upstream; còn cần đối chiếu preprocessing paper**.
- CSV làm việc ước lượng theo shape paper: **3.60–10.80 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- ronka/postureDataset là ảnh good/bad posture, không phải dữ liệu cảm biến 11 lớp trong paper.

### 19. GMSC

- Paper: 150,000 mẫu × 11 cột tổng (gồm nhãn), 2 lớp; chạy tối đa 150,000 mẫu.
- Trạng thái: **Ứng viên / cần tái tạo và đối chiếu**.
- Hugging Face: [algcache/GiveMeSomeCredit](https://huggingface.co/datasets/algcache/GiveMeSomeCredit). Revision: `4c2dab0dce91172737cf3413e023bd16834f1fa1`.
- Dung lượng nguồn ứng viên theo metadata: **5.41 MB**.
- CSV làm việc ước lượng theo shape paper: **9.90–29.70 MB**; chỉ là mô hình 6–18 byte/ô, không phải size nguồn đã đo.
- Đã giải nén Give Me Some Credit; cs-training.csv có 150.000 dòng và 12 cột gồm ID. Còn cần xác minh missing và thứ tự. Không dùng tập test không có nhãn làm phần đánh giá.

## Ước lượng dung lượng toàn bộ

Theo các shape **được in trong Bảng 1**, tổng trước giới hạn khoảng 4,385,946 dòng; sau giới hạn tối đa 150.000 mỗi luồng còn khoảng **995,262 dòng**. Bike thực tế đang có thêm một dòng; các bản còn lại có thể có sai khác tương tự.

- Tổng CSV làm việc theo 6–18 byte/ô: **104.8–314.5 MB**. Không gồm file thô, cache tải, metadata hoặc model.
- Ma trận float64 gồm feature và một nhãn số, cộng dồn cả 19 shape: khoảng **139.8 MB** thuần số. Đây không phải dự báo RAM process; pandas, chuỗi, bản sao train/validation và model làm RAM cao hơn.
- Ngân sách ổ đĩa thực hành: **1–3 GB** cho dữ liệu dạng bảng, cache và outputs vừa phải. Đây là dự phòng, không phải dung lượng tải đã đo cho đủ 19 bộ; các nguồn thời gian thực/ảnh/audio thô sai phiên bản có thể lớn hơn nhiều.
- Cấu hình khởi đầu đề xuất cho pipeline batch: CPU 4–8 core, RAM 16 GB, chạy tuần tự từng dataset. Chưa benchmark nên không đảm bảo cấu hình này đủ cho mọi thiết lập tuning.

Không cộng size các ứng viên thành “tổng dung lượng đúng paper”: nhiều ứng viên khác shape, hai biến thể Arabic dùng chung nguồn và một số nguồn vẫn thiếu. Ví dụ CoverType nguồn CSV đầy đủ 75,17 MB dù thí nghiệm chỉ lấy 150.000 mẫu; mirror Poker 7,12 MB nhưng thiếu số mẫu so với paper.

## Ước lượng thời gian chạy — chưa benchmark STUDD

**Giả định:** CPU 4–8 core, RAM 16 GB; không chạy nhiều job tranh CPU; dùng teacher/student đã chọn; giới hạn 150.000 dòng; dự đoán theo từng mẫu; giữ nguyên protocol batch/retraining; chưa tính tìm kiếm toàn bộ mô hình và phân tích nhãn trễ. GPU không tự tăng tốc Random Forest/scikit-learn và vòng lặp Python.

Không có code mô hình chạy thực tế ở project này nên không thể đưa giờ chạy đo được. Ước lượng dưới đây là **kịch bản tính toán**, không phải benchmark hay cam kết hoàn thành.

Theo W=500 cho sáu luồng nhỏ và W=2000 cho các bộ khác, tổng số mẫu đánh giá sau batch đầu là khoảng **966,262 mỗi phương pháp/seed**.

Dùng độ trễ hiệu dụng giả định **0,2–5 ms cho mỗi mẫu/phương pháp** (đã gộp các dự đoán cần thiết của phương pháp, nhưng chưa gồm retraining, tuning, mạng/logging):

- STUDD, 19 dataset, 1 seed: 19 run, khoảng **0.05–1.34 giờ** phần xử lý mẫu.
- 7 phương pháp, 19 dataset, 1 seed: 133 run, khoảng **0.38–9.39 giờ** phần xử lý mẫu.
- 10 phương pháp chính của paper, 19 dataset, 1 seed: 190 run, khoảng **0.54–13.42 giờ** phần xử lý mẫu.
- 10 phương pháp, 19 dataset, 3 seed: 570 run, khoảng **1.61–40.26 giờ** phần xử lý mẫu.

Đây chưa phải tổng thời gian. Công thức đầy đủ:

`T = tổng(N_test × thời_gian_xử_lý_mẫu + số_lần_retrain × thời_gian_fit_batch) + chọn_mô_hình + tiền_xử_lý + ghi_log`

Một phương pháp có nhiều cảnh báo có thể dành phần lớn thời gian cho fit batch. Nếu cả original và rebuild chạy toàn bộ cùng ma trận, chi phí gần gấp đôi (không nhất thiết bằng nhau vì implementation khác). Phân tích label access/delay và tuning tạo thêm run, không nằm trong 190/570 run ở trên. RF/XGB, SVM, số cây, số lớp, thread count và số alarm có thể làm thời gian ra ngoài khoảng giả định.

Để lập lịch trước benchmark, nên dành **1–3 ngày cho ma trận 19 × 10 × 3** với mô hình/cấu hình cố định, và dự phòng thêm nếu retraining/tuning nặng. Đây là ngân sách làm việc thô, không phải kết quả đo hay khẳng định có thể tái lập đầy đủ paper trong 3 ngày.

### Cách đo để thay ước lượng bằng con số đáng tin

1. Chạy STUDD và một baseline trên InsectsAbrupt, Electricity; thêm một luồng lớn 150.000 mẫu khi có nguồn đúng.
2. Đo riêng thời gian fit ban đầu, tổng thời gian predict/detect, số lần và thời gian retrain; tắt logging từng mẫu, ghi theo lô để tránh đo chủ yếu độ trễ mạng.
3. Đo throughput sau giai đoạn khởi động và ngoại suy theo số mẫu của từng dataset; cộng retrain, lặp 3 seed. Không ngoại suy một RF binary nhỏ sang SVM hoặc bộ 54 lớp mà không đo thêm.
4. Ghi CPU, RAM, phiên bản thư viện, thread count và cấu hình vào W&B. Chỉ sau bước này mới chốt số giờ chạy và lịch nộp.

## Những nguồn đã loại

- `takara-ai/poker_hands`: lịch sử ván poker, hơn 21 triệu record; không phải bộ UCI Poker classification của paper.
- `ronka/postureDataset`: ảnh good/bad posture, không phải 11 lớp cảm biến.
- `KunhanGao/NOAA-weather`: forecasting/infill/QA, chưa phù hợp benchmark rain/no-rain.
- `Laborator/gas-sensor-profiles`: dữ liệu MOX synthetic; `Debdoot/gas-chamber-sensor-logs`: log năm 2025; khác bộ Gas paper.
- `mohnasgbr/spoken-arabic-digits`: SA2Dataset audio nhỏ; không tự đồng nhất với benchmark 8.800 mẫu.
- `bromleyp/ozone_level`: hourly ozone 2024, khác benchmark paper.

Các nhận định loại trừ dựa vào card/file tree của chính repository Hugging Face được kiểm tra ngày 17/11/2026; bản tóm tắt và ứng viên được lưu trong `hf19_audit.json`.

## Việc cần làm để mở rộng lên đủ 19

Với các mục thiếu, tìm bản upstream mà paper dẫn (kho [incremental-ks](https://github.com/denismr/incremental-ks), [AdaptiveRandomForest](https://github.com/hmgomes/AdaptiveRandomForest), kho benchmark stream của Souza et al.), kiểm tra quyền phân phối và đăng mirror HF nếu cần. Với ứng viên, xác minh toàn bộ giá trị, thứ tự, nhãn, missing và ID; đạt kiểm tra rồi mới đưa vào `data/hosted_sources.json`. Không cập nhật manifest chạy bằng những link chưa được kiểm chứng.

## Cập nhật tải dữ liệu và nguồn ngoài HF

Đã tải nguồn HF cho năm bộ ứng viên: CoverType, Poker, Ozone (cả 1hr/8hr), Arabic, GMSC. ArabicShuffled dùng chung dữ liệu nền Arabic; **chưa có bản shuffled đúng protocol**, không nhân đôi file hay coi là đã tái tạo được biến thể.

- `data/`: Electricity, Bike, InsectsAbrupt đã xác minh; dùng cho thực nghiệm hiện tại.
- `data/dataset_huggingface/CoverType/`: CSV tên `covtype.data`; 581.012 dòng dữ liệu có 55 cột và 7 lớp, nhưng header có 52 cột. Cần loader kiểm soát lỗi header.
- `data/dataset_huggingface/Poker/`: hai file Parquet train/validation; số mẫu theo metadata HF, chưa đối chiếu nội dung từng hàng với upstream.
- `data/dataset_huggingface/Ozone/`: hai config CSV; shape không khớp paper như đã ghi.
- `data/dataset_huggingface/Arabic/`: hai file `.ts`; đếm trực tiếp 6.599 train + 2.199 test = 8.798 dòng chuỗi sau `@data`. Phải đối chiếu UCI/upstream, không tự thêm dòng.
- `data/dataset_huggingface/GMSC/`: đã giải nén ZIP và xóa ZIP; train có 150.000 dòng, 12 cột gồm ID. Chưa xác nhận cách bỏ ID, missing và thứ tự; test có 101.503 dòng và không có nhãn thật để đánh giá.

Tổng file ứng viên đang chiếm **135.642.002 byte (~135,64 MB)**, cộng ba CSV prepared khoảng **6,19 MB**. Chỉ lưu một bản các file dữ liệu, không giữ cả ZIP lẫn bản giải nén. `hf_download_inventory.json` lưu URL pinned và checksum file nguồn tải; `local_candidate_validation.json` lưu checksum của file còn trên máy và shape đã kiểm tra.

Các nguồn ngoài HF đã được cập nhật ngay trong mục từng dataset. Danh mục USP: [trang chính thức](https://sites.google.com/view/uspdsrepository), [bảng thông số](https://docs.google.com/spreadsheets/d/1a0H7yjdbfKAuinp34VceQzqJOZJimHrKEOnNzuPJnpg/edit), [folder file](https://drive.google.com/drive/folders/1J0EHI7TS_pQB8iQVlbxnhMnbKdkWvhr0). Trang chính thức dẫn trực tiếp hai nguồn này. Google Drive có thể hiện màn hình xác nhận khi tải file lớn; link download không bảo đảm trả CSV trực tiếp trong mọi trường hợp.

**Tuân thủ đề:** nguồn ngoài HF giúp chuẩn bị và đối chiếu. Trước run nộp bài, các bộ chỉ có nguồn ngoài vẫn cần mirror trên HF/Kaggle hoặc tìm bản hosted tương đương; link Drive/GitHub trong report chưa thay thế yêu cầu tải dữ liệu từ HF/Kaggle khi chạy. Đã có 10 mirror mới; manifest chạy chính thức cho các ứng viên chưa được mở rộng.

## Gói nguồn ngoài HF đã tải về

Đã tải đủ 10 bộ chưa tìm được nguồn HF phù hợp vào `data/dataset_not_huggingface/`, khoảng 130,24 MB dữ liệu. Mỗi bộ có README và provenance. Đã upload và kiểm tra trên HF; xem link mới trong từng mục. Xem `external_download_validation.json` để biết shape và checksum. Luxembourg có header; Powersupply có dòng 22.203 thừa field cuối; Sensors có 55 target strings. Chưa sửa các bất thường hoặc chứng minh khớp preprocessing paper. Các ghi chú “chưa tải” trong lượt khảo sát ban đầu được thay thế bởi cập nhật này.

## Xác minh mirror HF sau upload

Đã kiểm tra 10 mirror bằng API công khai và đối chiếu hash file dữ liệu ở commit cố định: Git blob SHA-1 cho file Git thường, SHA-256 theo LFS cho file lớn. Tất cả khớp file nguồn local. Kiểm tra license ở đây chỉ xác nhận metadata hiển thị; ghi chú điều khoản nguồn và sự mâu thuẫn ở Gas vẫn giữ nguyên. Không sửa file dữ liệu hoặc manifest chạy chính thức trong bước cập nhật report.
