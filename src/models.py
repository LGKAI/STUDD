"""Module hỗ trợ khởi tạo các mô hình học máy theo Bảng 2 và Mục 5.6.2 bài báo STUDD.

Tài liệu tham khảo:
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: a student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351–4378. (Mục 5.6.2 & Bảng 2, trang 4364).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# Bảng 2 (trang 4364): Cặp thuật toán (Teacher, Student) cho 19 bộ dữ liệu trong bài báo
PAPER_TABLE2_MODELS: Dict[str, Tuple[str, str]] = {
    "insectsabrupt": ("RF", "RF"),
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


class SafeStreamClassifier(BaseEstimator, ClassifierMixin):
    """Bộ bọc phân loại an toàn cho luồng dữ liệu (Data Stream Wrapper).

    Giải quyết 4 vấn đề thực tế khi chạy thực nghiệm trên 19 bộ dữ liệu:
        1. Mã hóa nhãn cục bộ về `0 .. K-1` liên tục trong mỗi cửa sổ huấn luyện `W`
           (bắt buộc đối với `XGBClassifier` khi cửa sổ `W` thiếu lớp hoặc nhãn gốc là `1..C`).
        2. Xử lý an toàn trường hợp cửa sổ `W` chỉ xuất hiện duy nhất 1 lớp (ví dụ khi
           Teacher dự đoán toàn bộ mẫu trong cửa sổ thuộc cùng 1 lớp).
        3. Tự động điền giá trị khuyết (`NaN`) bằng trung vị (`median`) tính riêng trên
           cửa sổ huấn luyện hiện tại (không làm rò rỉ dữ liệu tương lai, cần cho `GMSC`).
        4. Tự động chuẩn hóa (`StandardScaler`) riêng trên cửa sổ huấn luyện đối với
           các mô hình tuyến tính / khoảng cách (`RC`, `LR`, `SVC`) và hỗ trợ `predict_proba`
           cho cả `RidgeClassifier` (thông qua hàm sigmoid/softmax trên `decision_function`)
           để phương pháp đối chiếu `OS` (KS-test trên xác suất đầu ra) hoạt động đồng nhất.
    """

    def __init__(
        self,
        base_estimator: ClassifierMixin,
        scale_features: bool = False,
    ) -> None:
        self.base_estimator = base_estimator
        self.scale_features = scale_features

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X_arr).all():
            X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
            self.imputer_: Optional[SimpleImputer] = SimpleImputer(
                strategy="median", keep_empty_features=True
            )
            X_arr = self.imputer_.fit_transform(X_arr).astype(np.float32)
        else:
            self.imputer_ = None

        if self.scale_features:
            self.scaler_: Optional[StandardScaler] = StandardScaler()
            X_arr = self.scaler_.fit_transform(X_arr).astype(np.float32)
        else:
            self.scaler_ = None

        return X_arr

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X_arr).all():
            X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
            if getattr(self, "imputer_", None) is not None:
                X_arr = self.imputer_.transform(X_arr).astype(np.float32)
            else:
                X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        elif getattr(self, "imputer_", None) is not None:
            X_arr = self.imputer_.transform(X_arr).astype(np.float32)

        if getattr(self, "scaler_", None) is not None:
            X_arr = self.scaler_.transform(X_arr).astype(np.float32)

        return X_arr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SafeStreamClassifier":
        X_proc = self._preprocess_fit(X)
        y_arr = np.asarray(y)
        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)

        if len(self.classes_) < 2:
            self.single_class_: Optional[Any] = self.classes_[0]
            self.estimator_: Optional[ClassifierMixin] = None
        else:
            self.single_class_ = None
            self.estimator_ = clone(self.base_estimator)
            self.estimator_.fit(X_proc, y_encoded)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._preprocess_transform(X)
        if self.single_class_ is not None or self.estimator_ is None:
            return np.full(
                shape=(len(X_proc),),
                fill_value=self.single_class_,
                dtype=self.classes_.dtype,
            )
        pred_encoded = np.asarray(self.estimator_.predict(X_proc), dtype=np.int64)
        pred_encoded = np.clip(pred_encoded, 0, len(self.classes_) - 1)
        return self.classes_[pred_encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_proc = self._preprocess_transform(X)
        if self.single_class_ is not None or self.estimator_ is None:
            return np.ones((len(X_proc), 1), dtype=np.float32)

        if hasattr(self.estimator_, "predict_proba"):
            probs = np.asarray(self.estimator_.predict_proba(X_proc), dtype=np.float32)
            return probs

        # Dự phòng cho RidgeClassifier hoặc mô hình chỉ có decision_function
        if hasattr(self.estimator_, "decision_function"):
            scores = np.asarray(self.estimator_.decision_function(X_proc), dtype=np.float64)
            scores = np.clip(scores, -50.0, 50.0)
            if scores.ndim == 1:
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - p1, p1]).astype(np.float32)
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return (exp_s / np.sum(exp_s, axis=1, keepdims=True)).astype(np.float32)

        # Dự phòng cuối cùng nếu mô hình chỉ có predict
        preds = np.asarray(self.estimator_.predict(X_proc), dtype=np.int64)
        out = np.zeros((len(X_proc), len(self.classes_)), dtype=np.float32)
        out[np.arange(len(X_proc)), np.clip(preds, 0, len(self.classes_) - 1)] = 1.0
        return out


def build_model(
    code: str,
    random_state: int = 42,
    n_jobs: int = -1,
) -> SafeStreamClassifier:
    """Khởi tạo bộ phân loại theo ký hiệu trong Bảng 2 bài báo STUDD.

    Parameters
    ----------
    code : str
        Mã thuật toán: `'RF'`, `'XGB'`, `'RC'`, `'DT'`, `'LR'`, `'NB'`, `'SVC'`/`'SVM'`.
    random_state : int, default=42
        Hạt giống ngẫu nhiên.
    n_jobs : int, default=-1
        Số luồng CPU cho RF và XGB.
    """
    key = code.upper().strip()
    if key == "RF":
        base = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=n_jobs
        )
        scale = False
    elif key == "XGB":
        base = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=n_jobs,
        )
        scale = False
    elif key == "RC":
        base = RidgeClassifier(random_state=random_state)
        scale = True
    elif key == "DT":
        base = DecisionTreeClassifier(random_state=random_state)
        scale = False
    elif key == "LR":
        base = LogisticRegression(max_iter=1000, random_state=random_state)
        scale = True
    elif key == "NB":
        base = GaussianNB()
        scale = False
    elif key in ("SVC", "SVM"):
        base = SVC(probability=True, random_state=random_state)
        scale = True
    else:
        raise ValueError(
            f"Mã mô hình '{code}' không hợp lệ! Hỗ trợ: RF, XGB, RC, DT, LR, NB, SVC."
        )

    return SafeStreamClassifier(base_estimator=base, scale_features=scale)


def get_table2_models(
    dataset_name: str, random_state: int = 42
) -> Tuple[SafeStreamClassifier, SafeStreamClassifier, str, str]:
    """Lấy cặp mô hình (Teacher, Student) đúng theo Bảng 2 của bài báo STUDD."""
    key = dataset_name.lower().strip().replace("_", "").replace("-", "")
    if key not in PAPER_TABLE2_MODELS:
        raise KeyError(
            f"Không tìm thấy cấu hình Bảng 2 cho '{dataset_name}'. Hỗ trợ: {list(PAPER_TABLE2_MODELS.keys())}"
        )
    t_code, s_code = PAPER_TABLE2_MODELS[key]
    teacher = build_model(t_code, random_state=random_state)
    # Nếu Teacher và Student cùng là RF, đặt random_state khác nhau để Student không copy hệt cây của Teacher
    s_seed = random_state + 100 if t_code == s_code else random_state
    student = build_model(s_code, random_state=s_seed)
    return teacher, student, t_code, s_code


def select_best_teacher_student(
    X_init: np.ndarray,
    y_init: np.ndarray,
    candidate_codes: Optional[List[str]] = None,
    random_state: int = 42,
) -> Tuple[SafeStreamClassifier, SafeStreamClassifier, str, str, Dict[str, float]]:
    """Tái hiện quy trình chọn thuật toán tự động ở Mục 5.6.2 (trang 4364).

    Chia cửa sổ huấn luyện ban đầu `W` theo tỷ lệ 70% đầu làm tập train, 30% sau làm
    tập validation; chọn thuật toán tối đa hóa Cohen's Kappa cho Teacher, sau đó chọn
    thuật toán tối đa hóa Cohen's Kappa cho Student khi học bắt chước dự đoán của Teacher.
    """
    if candidate_codes is None:
        candidate_codes = ["RF", "XGB", "RC", "DT", "LR", "NB", "SVC"]

    split_idx = int(len(X_init) * 0.7)
    X_tr, X_val = X_init[:split_idx], X_init[split_idx:]
    y_tr, y_val = y_init[:split_idx], y_init[split_idx:]

    best_t_code = candidate_codes[0]
    best_t_kappa = -2.0
    teacher_scores: Dict[str, float] = {}

    for code in candidate_codes:
        model = build_model(code, random_state=random_state)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        k = float(cohen_kappa_score(y_val, preds))
        if np.isnan(k):
            k = 0.0
        teacher_scores[code] = round(k, 4)
        if k > best_t_kappa:
            best_t_kappa = k
            best_t_code = code

    # Huấn luyện lại Teacher tốt nhất trên 70% đầu để tạo nhãn giả cho Student chọn mô hình
    best_teacher_tmp = build_model(best_t_code, random_state=random_state)
    best_teacher_tmp.fit(X_tr, y_tr)
    yhat_tr = best_teacher_tmp.predict(X_tr)
    yhat_val = best_teacher_tmp.predict(X_val)

    best_s_code = candidate_codes[0]
    best_s_kappa = -2.0
    for code in candidate_codes:
        s_seed = random_state + 100 if code == best_t_code else random_state
        s_model = build_model(code, random_state=s_seed)
        s_model.fit(X_tr, yhat_tr)
        s_preds = s_model.predict(X_val)
        k = float(cohen_kappa_score(yhat_val, s_preds))
        if np.isnan(k):
            k = 0.0
        if k > best_s_kappa:
            best_s_kappa = k
            best_s_code = code

    teacher, student, _, _ = (
        build_model(best_t_code, random_state=random_state),
        build_model(
            best_s_code,
            random_state=random_state + 100 if best_s_code == best_t_code else random_state,
        ),
        best_t_code,
        best_s_code,
    )
    return teacher, student, best_t_code, best_s_code, teacher_scores
