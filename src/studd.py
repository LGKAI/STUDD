"""Module 2: Kiến trúc Học trò - Thầy giáo STUDD (Student-Teacher Concept Drift Detection).

Tài liệu tham khảo:
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: a student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351–4378. (Mục 4, trang 4357–4360 & Mục 5.6, trang 4363–4365).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    from .detector import PageHinkley
except ImportError:
    from detector import PageHinkley


# Cặp thuật toán Teacher–Student theo Bảng 2 (trang 4364) của bài báo STUDD
PAPER_TABLE2_MODELS: Dict[str, Tuple[str, str]] = {
    "abruptinsects": ("RF", "RF"),
    "insects": ("XGB", "XGB"),
    "posture": ("XGB", "RF"),
    "arabic": ("XGB", "XGB"),
    "bike": ("XGB", "RF"),
    "noaa": ("RF", "RF"),
    "sensors": ("DT", "DT"),
    "sensor": ("DT", "DT"),
    "powersupply": ("SVC", "RF"),
    "poker": ("DT", "DT"),
    "pokerhand": ("DT", "DT"),
    "rialto": ("XGB", "XGB"),
    "ozone": ("XGB", "XGB"),
    "outdoor": ("RF", "RF"),
    "luxembourg": ("RF", "RF"),
    "gas": ("RC", "RC"),
    "keystroke": ("XGB", "RF"),
    "arabicshuffled": ("RF", "RF"),
    "covertype": ("LR", "LR"),
    "covtype": ("LR", "LR"),
    "gmsc": ("XGB", "XGB"),
    "electricity": ("NB", "XGB"),
}

ALGORITHM_POOL: Tuple[str, ...] = ("RF", "SVC", "DT", "LR", "RC", "NB", "XGB")


class SafeXGBClassifier(BaseEstimator, ClassifierMixin):
    """Bộ bọc an toàn cho `XGBClassifier` hỗ trợ cửa sổ luồng thiếu lớp cục bộ.

    Trong dữ liệu luồng đa lớp (ví dụ `Poker` hoặc `Sensors`), một cửa sổ huấn luyện $W$
    có thể không chứa đủ toàn bộ dải nhãn `[0..K-1]`. `XGBoost >= 2.0` yêu cầu nhãn
    liên tiếp từ `0` đến `n_local_classes - 1`; lớp này tự động ánh xạ hai chiều
    cục bộ và xử lý an toàn trường hợp chỉ có 1 lớp trong cửa sổ.
    """

    def __init__(self, random_state: int = 42, n_jobs: int = 4) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model: Optional[XGBClassifier] = None
        self.classes_: np.ndarray = np.array([])
        self._single_class: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafeXGBClassifier":
        """Huấn luyện XGBoost với ánh xạ nhãn cục bộ liên tiếp."""
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        if len(unique_classes) < 2:
            self._single_class = unique_classes[0]
            self._model = None
            return self

        self._single_class = None
        class_to_idx = {c: idx for idx, c in enumerate(unique_classes)}
        y_mapped = np.array([class_to_idx[val] for val in y], dtype=np.int64)
        objective = "binary:logistic" if len(unique_classes) == 2 else "multi:softprob"
        eval_metric = "logloss" if len(unique_classes) == 2 else "mlogloss"

        self._model = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective=objective,
            eval_metric=eval_metric,
            verbosity=0,
        )
        self._model.fit(X, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn theo không gian nhãn gốc."""
        if self._single_class is not None or self._model is None:
            return np.full(shape=(len(X),), fill_value=self._single_class)
        preds_idx = self._model.predict(X).astype(np.int64)
        return self.classes_[preds_idx]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Trả về ma trận xác suất theo thứ tự `self.classes_`."""
        if self._single_class is not None or self._model is None:
            return np.ones((len(X), 1), dtype=np.float64)
        return self._model.predict_proba(X)


def create_learner(algo_code: str, random_state: int = 42, n_jobs: int = 4) -> ClassifierMixin:
    """Khởi tạo mô hình phân lớp chuẩn theo ký hiệu bảng 2 của bài báo STUDD.

    Parameters
    ----------
    algo_code : str
        Mã thuật toán thuộc `{RF, SVC, SVM, DT, LR, RC, NB, XGB}`.
    random_state : int, default=42
        Hạt giống ngẫu nhiên.
    n_jobs : int, default=4
        Số luồng CPU cho các thuật toán hỗ trợ song song (`RF`, `XGB`).

    Returns
    -------
    ClassifierMixin
        Ước lượng tương thích giao diện `scikit-learn`.
    """
    code = algo_code.strip().upper()
    if code == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
    if code == "DT":
        return DecisionTreeClassifier(random_state=random_state)
    if code == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
    if code == "RC":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(random_state=random_state)),
        ])
    if code in {"SVC", "SVM"}:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=random_state)),
        ])
    if code == "NB":
        return GaussianNB()
    if code == "XGB":
        return SafeXGBClassifier(random_state=random_state, n_jobs=n_jobs)
    raise ValueError(f"Mã thuật toán không được hỗ trợ: '{algo_code}'")


def build_paper_models(
    dataset_name: str,
    random_state: int = 42,
    n_jobs: int = 4,
) -> Tuple[ClassifierMixin, ClassifierMixin, str, str]:
    """Khởi tạo cặp mô hình Teacher và Student đúng theo Bảng 2 của bài báo.

    Parameters
    ----------
    dataset_name : str
        Tên bộ dữ liệu (không phân biệt hoa thường).
    random_state : int, default=42
        Hạt giống ngẫu nhiên cho Teacher (`random_state`) và Student (`random_state + 100`).
    n_jobs : int, default=4
        Số luồng CPU cho các mô hình cây/boosting.

    Returns
    -------
    Tuple[ClassifierMixin, ClassifierMixin, str, str]
        `(teacher_model, student_model, teacher_code, student_code)`.
    """
    key = dataset_name.lower().strip()
    teacher_code, student_code = PAPER_TABLE2_MODELS.get(key, ("RF", "RF"))
    teacher = create_learner(teacher_code, random_state=random_state, n_jobs=n_jobs)
    student = create_learner(student_code, random_state=random_state + 100, n_jobs=n_jobs)
    return teacher, student, teacher_code, student_code


def select_best_models(
    X_init: np.ndarray,
    y_init: np.ndarray,
    random_state: int = 42,
    train_ratio: float = 0.7,
    n_jobs: int = 4,
) -> Tuple[str, str, Dict[str, float]]:
    """Chạy quy trình chọn cặp thuật toán Teacher–Student theo Mục 5.6.2 (trang 4364).

    Chia cửa sổ khởi tạo $W$ thành 70% đầu để huấn luyện và 30% sau để kiểm định
    theo Cohen's Kappa; chọn Teacher tốt nhất trước, sau đó chọn Student bắt chước
    nhãn dự đoán của Teacher tốt nhất.

    Parameters
    ----------
    X_init : np.ndarray of shape (W, n_features)
        Đặc trưng của cửa sổ khởi tạo đầu tiên.
    y_init : np.ndarray of shape (W,)
        Nhãn thật của cửa sổ khởi tạo đầu tiên.
    random_state : int, default=42
        Hạt giống ngẫu nhiên.
    train_ratio : float, default=0.7
        Tỷ lệ chia tập huấn luyện ban đầu (mặc định 70%).
    n_jobs : int, default=4
        Số luồng CPU.

    Returns
    -------
    Tuple[str, str, Dict[str, float]]
        `(best_teacher_code, best_student_code, validation_kappas)`.
    """
    split_idx = int(len(X_init) * train_ratio)
    X_tr, X_val = X_init[:split_idx], X_init[split_idx:]
    y_tr, y_val = y_init[:split_idx], y_init[split_idx:]

    teacher_scores: Dict[str, float] = {}
    fitted_teachers: Dict[str, ClassifierMixin] = {}
    for code in ALGORITHM_POOL:
        clf = create_learner(code, random_state=random_state, n_jobs=n_jobs)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        teacher_scores[code] = float(cohen_kappa_score(y_val, preds))
        fitted_teachers[code] = clf

    best_teacher_code = max(teacher_scores, key=lambda k: teacher_scores[k])
    best_teacher = fitted_teachers[best_teacher_code]
    yhat_tr = best_teacher.predict(X_tr)
    yhat_val = best_teacher.predict(X_val)

    best_student_code = best_teacher_code
    best_student_score = -np.inf
    if len(np.unique(yhat_tr)) >= 2:
        for code in ALGORITHM_POOL:
            s_clf = create_learner(code, random_state=random_state + 100, n_jobs=n_jobs)
            s_clf.fit(X_tr, yhat_tr)
            s_preds = s_clf.predict(X_val)
            score = float(cohen_kappa_score(yhat_val, s_preds))
            if score > best_student_score:
                best_student_score = score
                best_student_code = code

    return best_teacher_code, best_student_code, teacher_scores


class STUDD:
    """Khung phát hiện Concept Drift không giám sát bằng mô hình Học trò - Thầy giáo (STUDD).

    Nguyên lý hoạt động gồm 3 giai đoạn (bám sát Hình 4, Hình 5 và Hình 6 trong bài báo):
        1. Giai đoạn Offline (`fit_offline`):
           - Huấn luyện Teacher T trên cửa sổ W mẫu có nhãn thật `(X_tr, y_tr)`.
           - Dùng Teacher T dự đoán nhãn giả trên chính tập đó: `y_hat_T = T(X_tr)`.
           - Huấn luyện Student S trên `(X_tr, y_hat_T)` để bắt chước hành vi của T.
        2. Giai đoạn Online (`predict_and_monitor`):
           - Với mỗi mẫu mới `x_t`, tính sai số bắt chước `L_t = I(T(x_t) != S(x_t))`.
           - Đưa `L_t` vào bộ kiểm định tuần tự Page-Hinkley.
        3. Giai đoạn Thích nghi (`adapt`):
           - Khi Page-Hinkley báo động tại bước `t`, thu thập đúng `W` mẫu quá khứ
             gần nhất `[t - W + 1 : t + 1]` để huấn luyện lại cả Teacher T và Student S.

    Parameters
    ----------
    teacher_model : ClassifierMixin, optional
        Mô hình phân loại làm Teacher (T). Mặc định là `RandomForestClassifier(n_estimators=50)`.
    student_model : ClassifierMixin, optional
        Mô hình phân loại làm Student (S). Mặc định là `RandomForestClassifier(n_estimators=30, max_depth=6)`.
    window_size : int, default=500
        Kích thước cửa sổ huấn luyện và tái huấn luyện W (Mục 5.7: 500 hoặc 2000).
    delta : float, default=0.001
        Tham số dung sai độ lớn thay đổi của Page-Hinkley (Mục 5.7: delta = 0.001).
    threshold : float, default=50.0
        Ngưỡng báo động lambda của kiểm định Page-Hinkley.
    random_state : int, default=42
        Hạt giống ngẫu nhiên cố định để đảm bảo tính tái lập.
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

        self.detector = PageHinkley(delta=delta, threshold=threshold)
        self.is_fitted: bool = False
        self._single_teacher_class_fallback: Optional[Any] = None
        self._single_student_class_fallback: Optional[Any] = None

    def fit_offline(self, X_train: np.ndarray, y_train: np.ndarray) -> "STUDD":
        """Giai đoạn 1: Huấn luyện cặp Teacher (T) và Student (S) trên cửa sổ có nhãn."""
        self.teacher_model = clone(self.teacher_model)
        self.student_model = clone(self.student_model)

        unique_true = np.unique(y_train)
        if len(unique_true) < 2:
            self._single_teacher_class_fallback = unique_true[0]
            self._single_student_class_fallback = unique_true[0]
            self.detector.reset()
            self.is_fitted = True
            return self

        self._single_teacher_class_fallback = None
        self.teacher_model.fit(X_train, y_train)
        y_hat_teacher = self.teacher_model.predict(X_train)

        unique_teacher = np.unique(y_hat_teacher)
        if len(unique_teacher) < 2:
            self._single_student_class_fallback = unique_teacher[0]
        else:
            self._single_student_class_fallback = None
            self.student_model.fit(X_train, y_hat_teacher)

        self.detector.reset()
        self.is_fitted = True
        return self

    def _predict_teacher_batch(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán theo lô bằng Teacher T (có fallback an toàn nếu chỉ có 1 lớp)."""
        if self._single_teacher_class_fallback is not None:
            return np.full(shape=(len(X),), fill_value=self._single_teacher_class_fallback)
        return self.teacher_model.predict(X)

    def _predict_student_batch(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán theo lô bằng Student S (có fallback an toàn nếu chỉ có 1 lớp)."""
        if self._single_student_class_fallback is not None:
            return np.full(shape=(len(X),), fill_value=self._single_student_class_fallback)
        return self.student_model.predict(X)

    def predict_and_monitor(
        self, x_inst: np.ndarray
    ) -> Tuple[Any, Any, bool, float, float]:
        """Giai đoạn 2: Dự đoán mẫu mới và giám sát Concept Drift không cần nhãn thật."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình STUDD chưa được huấn luyện! Hãy gọi fit_offline() trước.")

        x_2d = np.atleast_2d(x_inst)
        pred_teacher = self._predict_teacher_batch(x_2d)[0]
        pred_student = self._predict_student_batch(x_2d)[0]

        mimic_loss = 1.0 if pred_teacher != pred_student else 0.0
        drift_alarm, ph_stat = self.detector.update(mimic_loss)

        return pred_teacher, pred_student, drift_alarm, mimic_loss, ph_stat

    def adapt(self, X_recent: np.ndarray, y_recent: np.ndarray) -> None:
        """Giai đoạn 3: Loại bỏ mô hình cũ và tái huấn luyện trên cửa sổ W mẫu gần nhất."""
        self.fit_offline(X_recent, y_recent)

    def evaluate_stream(
        self, X_stream: np.ndarray, y_stream: np.ndarray
    ) -> Dict[str, Any]:
        """Chạy thực nghiệm toàn bộ luồng dữ liệu theo đúng quy trình bài báo (Hình 6, Mục 5.3 & 5.4).

        Parameters
        ----------
        X_stream : np.ndarray of shape (N, n_features)
            Toàn bộ luồng đặc trưng đầu vào.
        y_stream : np.ndarray of shape (N,)
            Toàn bộ nhãn thật (chỉ dùng để đánh giá và cấp nhãn khi có báo động).

        Returns
        -------
        Dict[str, Any]
            Từ điển kết quả gồm `accuracy`, `kappa`, `labeling_cost` (theo Bảng 4 bài báo),
            `online_labeling_cost`, `num_alarms`, `alarm_indices`, chuỗi dự đoán và thống kê PH.
        """
        n_total = len(X_stream)
        W = self.window_size
        if n_total <= W:
            raise ValueError(f"Độ dài luồng ({n_total}) phải lớn hơn kích thước cửa sổ W ({W}).")

        self.fit_offline(X_stream[:W], y_stream[:W])

        n_online = n_total - W
        y_true_online = y_stream[W:]
        y_pred_online = np.zeros_like(y_true_online)
        mimic_losses = np.zeros(n_online, dtype=np.float32)
        ph_statistics = np.zeros(n_online, dtype=np.float32)
        alarm_indices: List[int] = []

        curr_pos = W
        while curr_pos < n_total:
            preds_T_chunk = self._predict_teacher_batch(X_stream[curr_pos:])
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
                    # Bám sát Hình 6 & Mục 5.3: tái huấn luyện bằng đúng W mẫu quá khứ gần nhất
                    end_adapt = stream_idx + 1
                    start_adapt = max(0, end_adapt - W)
                    self.adapt(
                        X_stream[start_adapt:end_adapt],
                        y_stream[start_adapt:end_adapt],
                    )
                    curr_pos = stream_idx + 1
                    alarm_triggered = True
                    break

            if not alarm_triggered:
                break

        acc = float(accuracy_score(y_true_online, y_pred_online))
        kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
        # Theo Bảng 4 bài báo: tổng tỷ lệ nhãn sử dụng (W ban đầu + các lần retrain) trên toàn bộ luồng
        labeling_cost = float(((1 + len(alarm_indices)) * W) / n_total)
        online_labeling_cost = float((len(alarm_indices) * W) / n_online)

        return {
            "method": "STUDD",
            "accuracy": acc,
            "kappa": kappa,
            "labeling_cost": labeling_cost,
            "online_labeling_cost": online_labeling_cost,
            "num_alarms": len(alarm_indices),
            "num_retrains": len(alarm_indices),
            "alarm_indices": alarm_indices,
            "y_true_online": y_true_online,
            "y_pred_online": y_pred_online,
            "mimic_losses": mimic_losses,
            "ph_statistics": ph_statistics,
        }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("=== KIỂM THỬ MODULE 2: STUDD & BỘ MÔ HÌNH CHUẨN BẢNG 2 ===")
    rng = np.random.default_rng(42)
    n_samples, n_features, drift_point = 2500, 6, 1200
    X_pre = rng.normal(loc=0.0, scale=1.0, size=(drift_point, n_features))
    X_post = rng.normal(loc=[1.2, -1.2, 1.5, -1.5, 0.0, 0.0], scale=2.0, size=(n_samples - drift_point, n_features))
    X_demo = np.vstack([X_pre, X_post])
    y_pre = (X_pre[:, 0] + X_pre[:, 1] + 0.5 * X_pre[:, 2] > 0).astype(int)
    y_post = (X_post[:, 0] - X_post[:, 1] - 0.8 * X_post[:, 2] > 0).astype(int)
    y_demo = np.concatenate([y_pre, y_post])

    # Kiểm thử toàn bộ các mã thuật toán dùng trong 6 bộ dữ liệu của Đạt
    for ds in ("electricity", "powersupply", "noaa", "covertype", "poker", "sensors"):
        t_m, s_m, t_c, s_c = build_paper_models(ds, random_state=42)
        studd_ds = STUDD(teacher_model=t_m, student_model=s_m, window_size=400, delta=0.001, threshold=10.0)
        res = studd_ds.evaluate_stream(X_demo, y_demo)
        print(
            f"[OK] {ds:12s} ({t_c:3s}/{s_c:3s}) | Acc={res['accuracy']*100:.2f}% | "
            f"Kappa={res['kappa']:.4f} | Cost={res['labeling_cost']:.4f} | Alarms={res['num_alarms']}"
        )
    print("=== KIỂM THỬ MODULE 2 HOÀN TẤT THÀNH CÔNG! ===")
