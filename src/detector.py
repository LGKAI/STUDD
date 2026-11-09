"""Module 1: Thuật toán kiểm định tuần tự Page-Hinkley (Page-Hinkley Test - PHT).

Tài liệu tham khảo:
    1. Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2), 100-115.
    2. Cerqueira et al. (2023). STUDD: a student-teacher method for unsupervised
       concept drift detection. Machine Learning, Mục 4.2 (trang 4358) & Mục 5.7 (trang 4365).
"""

from typing import Tuple


class PageHinkley:
    """Bộ phát hiện điểm thay đổi tuần tự Page-Hinkley (Sequential Change Detector).

    Trong phương pháp STUDD, bộ kiểm định Page-Hinkley không theo dõi lỗi thật của mô hình
    (do không có nhãn y_t), mà theo dõi "sai số bắt chước" (Mimicry Loss) giữa Học trò (Student)
    và Thầy giáo (Teacher):
        L_t = 1.0 nếu y_hat_Student != y_hat_Teacher (Hai mô hình bất đồng)
        L_t = 0.0 nếu y_hat_Student == y_hat_Teacher (Hai mô hình đồng thuận)

    Khi phân phối dữ liệu bị trôi dạt (Concept Drift), tỷ lệ bất đồng L_t tăng lên.
    Thuật toán cộng dồn độ lệch giữa L_t và trung bình lịch sử L_bar_t. Nếu tổng tích lũy
    vượt quá ngưỡng cho phép (threshold), hệ thống sẽ phát tín hiệu báo động Drift.

    Parameters
    ----------
    delta : float, default=0.001
        Tham số dung sai độ lớn thay đổi (magnitude tolerance).
        Theo Mục 5.7 của bài báo STUDD, tác giả cố định delta = 0.001.
    threshold : float, default=50.0
        Ngưỡng kích hoạt báo động lambda (detection threshold).
        Khi thống kê kiểm định (m_t - M_t) > threshold, còi báo động Drift sẽ bật.
    alpha : float, default=0.9999
        Hệ số quên (forgetting factor) giúp giảm nhẹ trọng số của các quan sát quá cũ.
        Giá trị mặc định chuẩn là 1 - 0.0001 = 0.9999 (hoặc 1.0 nếu giữ nguyên toàn bộ lịch sử).
    min_instances : int, default=30
        Số lượng mẫu tối thiểu cần quan sát để trung bình lịch sử ổn định trước khi
        cho phép phát báo động (tránh báo động giả ở những mẫu đầu tiên).
    """

    def __init__(
        self,
        delta: float = 0.001,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        min_instances: int = 30,
    ) -> None:
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances

        # Khởi tạo các biến trạng thái thống kê
        self.sample_count: int = 0
        self.running_mean: float = 0.0
        self.sum_deviation: float = 0.0
        self.min_sum_deviation: float = 0.0
        self.reset()

    def reset(self) -> None:
        """Đặt lại (Reset) toàn bộ bộ đếm thống kê về trạng thái ban đầu.

        Được gọi lúc khởi tạo hoặc ngay sau khi phát hiện một điểm Concept Drift.
        """
        self.sample_count = 0
        self.running_mean = 0.0      # Trung bình lịch sử: L_bar_t
        self.sum_deviation = 0.0     # Tổng độ lệch tích lũy: m_t
        self.min_sum_deviation = 0.0 # Giá trị cực tiểu của tổng độ lệch: M_t = min(m_t)

    def update(self, loss_val: float) -> Tuple[bool, float]:
        """Cập nhật trạng thái bộ kiểm định với một giá trị lỗi mới trên luồng dữ liệu.

        Công thức toán học tại mỗi bước thời gian t:
            1. Cập nhật trung bình cộng dồn:
               L_bar_t = L_bar_{t-1} + (L_t - L_bar_{t-1}) / t
            2. Cập nhật tổng độ lệch tích lũy:
               m_t = alpha * m_{t-1} + (L_t - L_bar_t - delta)
            3. Cập nhật mức đáy lịch sử:
               M_t = min(M_{t-1}, m_t)
            4. Tính đại lượng thống kê Page-Hinkley:
               PH_t = m_t - M_t
            5. Kiểm tra điều kiện báo động:
               Nếu t >= min_instances và PH_t > threshold -> Báo động Drift!

        Parameters
        ----------
        loss_val : float
            Giá trị đầu vào tại thời điểm t (trong STUDD, loss_val thuộc {0.0, 1.0}).

        Returns
        -------
        Tuple[bool, float]
            Trả về bộ 2 giá trị `(drift_detected, ph_statistic)`:
            - `drift_detected` (bool): `True` nếu phát hiện có Concept Drift, ngược lại `False`.
            - `ph_statistic` (float): Giá trị thống kê hiện tại (PH_t = m_t - M_t), dùng để vẽ biểu đồ.
        """
        self.sample_count += 1

        # Bước 1: Cập nhật trung bình lịch sử L_bar_t theo công thức đệ quy
        self.running_mean += (loss_val - self.running_mean) / self.sample_count

        # Bước 2: Cộng dồn độ lệch m_t (có áp dụng hệ số quên alpha và dung sai delta)
        self.sum_deviation = (
            self.alpha * self.sum_deviation
            + (loss_val - self.running_mean - self.delta)
        )

        # Bước 3: Theo dõi mức cực tiểu lịch sử M_t
        if self.sum_deviation < self.min_sum_deviation:
            self.min_sum_deviation = self.sum_deviation

        # Bước 4: Tính điểm thống kê kiểm định PH_t = m_t - M_t
        ph_statistic = self.sum_deviation - self.min_sum_deviation

        # Bước 5: Kiểm tra xem đã đủ mẫu khởi động và vượt ngưỡng báo động chưa
        drift_detected = (
            self.sample_count >= self.min_instances
            and ph_statistic > self.threshold
        )

        # Nếu phát hiện Drift, tự động reset bộ đếm để chuẩn bị cho chu kỳ mới
        if drift_detected:
            self.reset()

        return drift_detected, ph_statistic


# ==============================================================================
# ĐOẠN CODE KIỂM THỬ ĐỘC LẬP (Chạy lệnh: python src/detector.py)
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== KIỂM THỬ MODULE 1: THUẬT TOÁN PAGE-HINKLEY ===")
    detector = PageHinkley(delta=0.001, threshold=10.0, min_instances=30)

    # Giai đoạn 1: Luồng ổn định (100 mẫu đầu tiên, Teacher và Student đồng thuận hoàn toàn)
    print("[1] Đang chạy 100 mẫu ổn định (loss = 0.0)...")
    for step in range(1, 101):
        alarm, stat = detector.update(0.0)
        assert not alarm, f"Lỗi: Báo động giả tại bước {step}!"
    print(f" -> Hoàn tất 100 mẫu ổn định. Điểm thống kê PH hiện tại: {stat:.4f} (Không có báo động)")

    # Giai đoạn 2: Xảy ra Concept Drift (Teacher và Student bắt đầu bất đồng liên tục: loss = 1.0)
    print("\n[2] Giả lập xảy ra Concept Drift từ bước 101 (loss = 1.0)...")
    for step in range(101, 151):
        alarm, stat = detector.update(1.0)
        if alarm:
            print(f" -> [BÁO ĐỘNG DRIFT!] Phát hiện thành công tại bước t = {step} (PH_stat = {stat:.2f} > threshold = 10.0)")
            break
    print("=== KIỂM THỬ HOÀN TẤT THÀNH CÔNG! ===")
