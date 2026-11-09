"""Module 2: Kiến trúc Học trò - Thầy giáo STUDD (Student-Teacher Concept Drift Detection).

Tài liệu tham khảo:
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: a student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351–4378. (Mục 4, trang 4357–4360).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

try:
    from .detector import PageHinkley
except ImportError:
    from detector import PageHinkley


class STUDD:
    """Khung phát hiện Concept Drift không giám sát bằng mô hình Học trò - Thầy giáo (STUDD).

    Nguyên lý hoạt động gồm 3 giai đoạn (bám sát Hình 4, Hình 5 và Hình 6 trong bài báo):
        1. Giai đoạn Offline (`fit_offline`):
           - Huấn luyện mô hình chính Thầy giáo (Teacher T) trên cửa sổ W mẫu có nhãn thật (X_tr, y_tr).
           - Dùng Teacher T dự đoán nhãn giả trên chính tập đó: y_hat_T = T(X_tr).
           - Huấn luyện mô hình phụ Học trò (Student S) trên (X_tr, y_hat_T) để bắt chước hành vi của T.
        2. Giai đoạn Online (`predict_and_monitor`):
           - Với mỗi mẫu dữ liệu mới x_t (không có nhãn thật y_t), cả T và S cùng đưa ra dự đoán.
           - Tính sai số bắt chước (Mimicry Loss): L_t = I(T(x_t) != S(x_t)) thuộc {0.0, 1.0}.
           - Đưa L_t vào bộ kiểm định tuần tự Page-Hinkley để phát hiện sự gia tăng bất đồng.
        3. Giai đoạn Thích nghi (`adapt`):
           - Khi Page-Hinkley phát báo động Drift, hệ thống truy vấn nhãn thật của W mẫu gần nhất
             để huấn luyện lại (retrain) cả Teacher T và Student S.

    Parameters
    ----------
    teacher_model : ClassifierMixin, optional
        Mô hình phân loại làm Teacher (T). Mặc định là RandomForestClassifier(n_estimators=50).
    student_model : ClassifierMixin, optional
        Mô hình phân loại làm Student (S). Mặc định là bản sao cùng cấu trúc với Teacher.
    window_size : int, default=500
        Kích thước cửa sổ huấn luyện ban đầu và tái huấn luyện W (theo Mục 5.7: W = 500 hoặc 2000).
    delta : float, default=0.001
        Tham số dung sai độ lớn thay đổi của kiểm định Page-Hinkley (Mục 5.7: delta = 0.001).
    threshold : float, default=50.0
        Ngưỡng báo động lambda của kiểm định Page-Hinkley.
    random_state : int, default=42
        Hạt giống ngẫu nhiên cố định để đảm bảo tính tái lập 100% (Reproducibility).
    """

    def __init__(
        self,
        teacher_model: Optional[ClassifierMixin] = None,
        student_model: Optional[ClassifierMixin] = None,
        window_size: int = 500,
        delta: float = 0.001,
        threshold: float = 50.0,
        random_state: int = 42,
    ) -> None:
        self.window_size = window_size
        self.delta = delta
        self.threshold = threshold
        self.random_state = random_state

        # Khởi tạo mô hình Teacher (T) và Student (S)
        # Lưu ý quan trọng: Nếu T và S dùng chung thuật toán rừng ngẫu nhiên trên cùng tập X_train,
        # cần đặt random_state khác nhau (hoặc giới hạn độ sâu của S theo nguyên lý Model Compression)
        # để tạo sự đa dạng cấu trúc cây ở vùng ngoại lai (Out-of-distribution), tránh việc S copy 100% T.
        if teacher_model is None:
            self.teacher_model = RandomForestClassifier(
                n_estimators=50, random_state=random_state
            )
        else:
            self.teacher_model = clone(teacher_model)

        if student_model is None:
            self.student_model = RandomForestClassifier(
                n_estimators=30, max_depth=6, random_state=random_state + 100
            )
        else:
            self.student_model = clone(student_model)

        # Khởi tạo bộ phát hiện thay đổi Page-Hinkley (Module 1)
        self.detector = PageHinkley(delta=delta, threshold=threshold)
        self.is_fitted: bool = False
        self._single_class_fallback: Optional[Any] = None

    def fit_offline(self, X_train: np.ndarray, y_train: np.ndarray) -> "STUDD":
        """Giai đoạn 1: Huấn luyện cặp mô hình Teacher (T) và Student (S) trên cửa sổ có nhãn.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Ma trận đặc trưng huấn luyện.
        y_train : np.ndarray of shape (n_samples,)
            Vector nhãn thật tương ứng.

        Returns
        -------
        STUDD
            Đối tượng STUDD đã được huấn luyện.
        """
        # 1. Huấn luyện Teacher T trên nhãn thật (X_train, y_train)
        self.teacher_model.fit(X_train, y_train)

        # 2. Lấy nhãn dự đoán của Teacher trên tập huấn luyện: y_hat_T = T(X_train)
        y_hat_teacher = self.teacher_model.predict(X_train)

        # 3. Huấn luyện Student S học bắt chước nhãn dự đoán của Teacher (X_train, y_hat_T)
        # Xử lý an toàn nếu Teacher chỉ dự đoán ra duy nhất 1 lớp (tránh lỗi với LogisticRegression/SVM)
        unique_classes = np.unique(y_hat_teacher)
        if len(unique_classes) < 2:
            self._single_class_fallback = unique_classes[0]
        else:
            self._single_class_fallback = None
            self.student_model.fit(X_train, y_hat_teacher)

        # 4. Đặt lại bộ kiểm định Page-Hinkley sau mỗi lần huấn luyện mới
        self.detector.reset()
        self.is_fitted = True
        return self

    def _predict_student_batch(self, X: np.ndarray) -> np.ndarray:
        """Hàm nội bộ lấy dự đoán của Student S trên một lô dữ liệu X."""
        if self._single_class_fallback is not None:
            return np.full(shape=(len(X),), fill_value=self._single_class_fallback)
        return self.student_model.predict(X)

    def predict_and_monitor(
        self, x_inst: np.ndarray
    ) -> Tuple[Any, Any, bool, float, float]:
        """Giai đoạn 2: Dự đoán mẫu mới và giám sát Concept Drift không cần nhãn thật.

        Parameters
        ----------
        x_inst : np.ndarray of shape (n_features,) or (1, n_features)
            Một quan sát mới đến từ luồng dữ liệu.

        Returns
        -------
        Tuple[Any, Any, bool, float, float]
            Trả về bộ 5 giá trị `(pred_teacher, pred_student, drift_alarm, mimic_loss, ph_stat)`:
            - `pred_teacher`: Nhãn dự đoán chính thức của hệ thống (do Teacher quyết định).
            - `pred_student`: Nhãn dự đoán của Student.
            - `drift_alarm` (bool): `True` nếu phát hiện Concept Drift tại mẫu này.
            - `mimic_loss` (float): Sai số bắt chước L_t thuộc {0.0, 1.0}.
            - `ph_stat` (float): Điểm thống kê Page-Hinkley hiện tại.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình STUDD chưa được huấn luyện! Hãy gọi fit_offline() trước.")

        x_2d = np.atleast_2d(x_inst)
        pred_teacher = self.teacher_model.predict(x_2d)[0]
        pred_student = self._predict_student_batch(x_2d)[0]

        # Tính sai số bắt chước nhị phân: L_t = I(y_hat_T != y_hat_S)
        mimic_loss = 1.0 if pred_teacher != pred_student else 0.0

        # Cập nhật kiểm định tuần tự Page-Hinkley
        drift_alarm, ph_stat = self.detector.update(mimic_loss)

        return pred_teacher, pred_student, drift_alarm, mimic_loss, ph_stat

    def adapt(self, X_recent: np.ndarray, y_recent: np.ndarray) -> None:
        """Giai đoạn 3: Thích nghi (Adaptation) khi phát hiện Concept Drift.

        Loại bỏ mô hình cũ và huấn luyện lại cả Teacher và Student trên cửa sổ W mẫu gần nhất.

        Parameters
        ----------
        X_recent : np.ndarray of shape (W, n_features)
            Các mẫu dữ liệu gần nhất vừa được truy vấn nhãn.
        y_recent : np.ndarray of shape (W,)
            Nhãn thật của các mẫu dữ liệu gần nhất.
        """
        self.fit_offline(X_recent, y_recent)

    def evaluate_stream(
        self, X_stream: np.ndarray, y_stream: np.ndarray
    ) -> Dict[str, Any]:
        """Chạy thực nghiệm toàn bộ luồng dữ liệu theo đúng quy trình bài báo (Hình 6, Mục 5.3 & 5.4).

        Quy trình:
            - Dùng `W = window_size` mẫu đầu tiên làm tập Offline Training ban đầu.
            - Triển khai trên phần luồng trực tuyến còn lại (từ mẫu W đến hết).
            - Mỗi khi Page-Hinkley phát báo động tại bước t, hệ thống sử dụng cửa sổ W mẫu
              gần nhất `X[t - W + 1 : t + 1]` kèm nhãn thật để tái huấn luyện mô hình (Retraining).
            - Đo lường 2 chỉ số đánh giá chính của bài báo (Mục 5.4, trang 4360):
              1. Cohen's Kappa (và Accuracy) của dự đoán từ Teacher.
              2. Labeling Cost (% tổng số mẫu trực tuyến phải yêu cầu nhãn thật để thích nghi).

        Parameters
        ----------
        X_stream : np.ndarray of shape (N, n_features)
            Toàn bộ luồng đặc trưng đầu vào.
        y_stream : np.ndarray of shape (N,)
            Toàn bộ nhãn thật (chỉ dùng để đánh giá độ chính xác và cấp nhãn khi có báo động).

        Returns
        -------
        Dict[str, Any]
            Từ điển chứa kết quả tổng hợp và lịch sử từng bước để vẽ biểu đồ / log lên W&B.
        """
        n_total = len(X_stream)
        W = self.window_size
        if n_total <= W:
            raise ValueError(f"Độ dài luồng ({n_total}) phải lớn hơn kích thước cửa sổ W ({W}).")

        # Bước 1: Huấn luyện Offline trên W mẫu đầu tiên
        self.fit_offline(X_stream[:W], y_stream[:W])

        # Các danh sách lưu trữ kết quả trên pha Online (từ mẫu thứ W đến n_total)
        n_online = n_total - W
        y_true_online = y_stream[W:]
        y_pred_online = np.zeros_like(y_true_online)
        mimic_losses = np.zeros(n_online, dtype=np.float32)
        ph_statistics = np.zeros(n_online, dtype=np.float32)
        alarm_indices: List[int] = []

        # Để tối ưu tốc độ trên dữ liệu lớn mà vẫn giữ đúng 100% tính tuần tự:
        # Giữa 2 lần báo động, mô hình T và S không đổi nên ta dự đoán theo lô (vectorized)
        # cho đoạn luồng phía trước, sau đó duyệt tuần tự từng mẫu qua bộ lọc Page-Hinkley.
        curr_pos = W
        while curr_pos < n_total:
            # Dự đoán trước bằng T và S từ vị trí hiện tại đến cuối luồng
            preds_T_chunk = self.teacher_model.predict(X_stream[curr_pos:])
            preds_S_chunk = self._predict_student_batch(X_stream[curr_pos:])

            alarm_triggered = False
            for offset in range(len(preds_T_chunk)):
                stream_idx = curr_pos + offset
                online_idx = stream_idx - W

                pred_T = preds_T_chunk[offset]
                pred_S = preds_S_chunk[offset]

                y_pred_online[online_idx] = pred_T
                loss_t = 1.0 if pred_T != pred_S else 0.0
                mimic_losses[online_idx] = loss_t

                drift_alarm, ph_stat = self.detector.update(loss_t)
                ph_statistics[online_idx] = ph_stat

                if drift_alarm:
                    alarm_indices.append(stream_idx)
                    # Thích nghi (Adaptation): Huấn luyện lại trên W mẫu gần nhất
                    start_adapt = max(0, stream_idx - W + 1)
                    self.adapt(
                        X_stream[start_adapt : stream_idx + 1],
                        y_stream[start_adapt : stream_idx + 1],
                    )
                    curr_pos = stream_idx + 1
                    alarm_triggered = True
                    break

            if not alarm_triggered:
                break

        # Tính toán các chỉ số đánh giá chính thức trên luồng Online
        acc = float(accuracy_score(y_true_online, y_pred_online))
        kappa = float(cohen_kappa_score(y_true_online, y_pred_online))

        # Chi phí gán nhãn (Labeling Cost): mỗi lần báo động cần xin W nhãn (tính trên tỷ lệ mẫu online)
        # Giới hạn tối đa bằng 1.0 (100% mẫu online)
        labeling_cost = float(min(1.0, (len(alarm_indices) * W) / n_online))

        return {
            "accuracy": acc,
            "kappa": kappa,
            "labeling_cost": labeling_cost,
            "num_alarms": len(alarm_indices),
            "alarm_indices": alarm_indices,
            "y_true_online": y_true_online,
            "y_pred_online": y_pred_online,
            "mimic_losses": mimic_losses,
            "ph_statistics": ph_statistics,
        }


# ==============================================================================
# ĐOẠN CODE KIỂM THỬ ĐỘC LẬP (Chạy lệnh: python src/studd.py)
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== KIỂM THỬ MODULE 2: KIẾN TRÚC TEACHER-STUDENT (STUDD) ===")
    rng = np.random.default_rng(42)

    # Tạo một luồng dữ liệu nhân tạo 2500 mẫu (6 đặc trưng), có Concept Drift tại mẫu thứ 1200
    n_samples, n_features, drift_point = 2500, 6, 1200
    X_pre = rng.normal(loc=0.0, scale=1.0, size=(drift_point, n_features))
    # Sau bước 1200: Phân phối đầu vào dịch chuyển sang vùng biên có phương sai lớn (Concept Drift)
    X_post = rng.normal(loc=[1.2, -1.2, 1.5, -1.5, 0.0, 0.0], scale=2.0, size=(n_samples - drift_point, n_features))
    X_demo = np.vstack([X_pre, X_post])

    # Quy luật phân lớp phi tuyến thay đổi sau mẫu 1200
    y_pre = (X_pre[:, 0] + X_pre[:, 1] + 0.5 * X_pre[:, 2] > 0).astype(int)
    y_post = (X_post[:, 0] - X_post[:, 1] - 0.8 * X_post[:, 2] > 0).astype(int)
    y_demo = np.concatenate([y_pre, y_post])

    # Khởi tạo mô hình STUDD với cửa sổ W = 400, ngưỡng báo động = 10.0
    studd_model = STUDD(window_size=400, delta=0.001, threshold=10.0, random_state=42)
    results = studd_model.evaluate_stream(X_demo, y_demo)

    print(f" -> Độ chính xác (Accuracy) : {results['accuracy'] * 100:.2f}%")
    print(f" -> Hệ số Cohen's Kappa     : {results['kappa']:.4f}")
    print(f" -> Chi phí nhãn (Cost)     : {results['labeling_cost'] * 100:.2f}%")
    print(f" -> Số lần báo động Drift   : {results['num_alarms']} lần (tại mẫu: {results['alarm_indices']})")
    print("=== KIỂM THỬ MODULE 2 HOÀN TẤT THÀNH CÔNG! ===")
