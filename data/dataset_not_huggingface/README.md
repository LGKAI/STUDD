# Các dataset nguồn ngoài Hugging Face để upload

Đã tải 10 dataset: Gas, Insects, Keystroke, Luxembourg, NOAA, Outdoor, Posture,
Powersupply, Rialto và Sensors. Mỗi thư mục có file dữ liệu, README và provenance.json.
Các bộ từ incremental-ks có thêm LICENSE gốc. Không giữ ZIP sau giải nén.

Có thể tạo một HF dataset chứa cả thư mục này, hoặc tạo một HF dataset cho mỗi thư mục con.
Giữ README/provenance/LICENSE cùng file dữ liệu để bảo toàn nguồn, điều kiện phân phối và checksum.
Chưa upload các bộ này lên HF. Link nguồn nằm trong provenance.json của từng bộ.

Đây là bản nguồn upstream, chưa phải dữ liệu đã chuẩn hóa để tái lập paper:

- Luxembourg có header; 1.901 dòng dữ liệu và 32 cột sau header.
- Powersupply: dòng 22.203 có dấu phẩy cuối thừa; chưa sửa file gốc.
- Sensors: có 55 chuỗi nhãn khác nhau, paper ghi 54 lớp; cần đối chiếu.
- Gas: có 128 feature, paper ghi 17 cột tổng; cần xác định tiền xử lý.
- Keystroke, NOAA, Outdoor, Rialto và Posture cũng có khác biệt shape/số dòng với bảng paper.

Xem chi tiết từng README trước khi chạy. Không tự cắt cột hoặc bỏ dòng đầu để ép khớp paper.
Tổng file dữ liệu: 130.244.938 byte (~130,24 MB), chưa tính README và metadata.

Nguồn: https://sites.google.com/view/uspdsrepository và
https://github.com/denismr/incremental-ks . Giữ attribution tác giả gốc;
trang USP lưu ý các dataset bên thứ ba có thể có điều kiện riêng.
