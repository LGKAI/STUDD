"""Module 4: Cài đặt đầy đủ 10 phương pháp đối sánh và bộ độ đo đánh giá (Mục 5.5, 5.8.3 & 5.8.4).

Tài liệu tham khảo:
    Cerqueira, V., Gomes, H. M., Bifet, A., & Torgo, L. (2023).
    STUDD: a student–teacher method for unsupervised concept drift detection.
    Machine Learning, 112(11), 4351–4378.

Danh sách 10 phương pháp trong Bảng 3 & Bảng 4:
    - Baselines: `BL-st`, `BL-ret`
    - Proposed: `STUDD`
    - Supervised benchmarks: `SS`, `DSS`, `WS`, `DWS`
    - Unsupervised benchmarks: `OS`, `OF`, `FF`
"""

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from scipy.stats import ks_2samp
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

try:
    from .detector import PageHinkley
    from .studd import STUDD
except ImportError:
    from detector import PageHinkley
    from studd import STUDD


ALL_METHODS: tuple = (
    "BL-st",
    "BL-ret",
    "STUDD",
    "SS",
    "DSS",
    "WS",
    "DWS",
    "OS",
    "OF",
    "FF",
)


def _fit_safe(model_template: ClassifierMixin, X: np.ndarray, y: np.ndarray) -> tuple:
    """Huấn luyện bản sao mô hình một cách an toàn ngay cả khi cửa sổ chỉ có 1 nhãn."""
    model = clone(model_template)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return None, unique_classes[0]
    model.fit(X, y)
    return model, None


def _predict_safe(model: Optional[ClassifierMixin], single_class: Optional[Any], X: np.ndarray) -> np.ndarray:
    """Dự đoán từ mô hình đã huấn luyện hoặc trả về nhãn đơn nhất nếu bị fallback."""
    if single_class is not None or model is None:
        return np.full(shape=(len(X),), fill_value=single_class)
    return model.predict(X)


def _build_result(
    method: str,
    y_true_online: np.ndarray,
    y_pred_online: np.ndarray,
    alarm_indices: List[int],
    window_size: int,
    n_total: int,
    num_retrains: Optional[int] = None,
) -> Dict[str, Any]:
    """Đóng gói kết quả đánh giá chuẩn theo Bảng 3 & Bảng 4 của bài báo."""
    acc = float(accuracy_score(y_true_online, y_pred_online))
    kappa = float(cohen_kappa_score(y_true_online, y_pred_online))
    if np.isnan(kappa):
        kappa = 0.0
    retrain_count = len(alarm_indices) if num_retrains is None else num_retrains
    labeling_cost = float(((1 + retrain_count) * window_size) / n_total)
    return {
        "method": method,
        "accuracy": acc,
        "kappa": kappa,
        "labeling_cost": labeling_cost,
        "num_alarms": len(alarm_indices),
        "num_retrains": retrain_count,
        "alarm_indices": alarm_indices,
        "y_true_online": y_true_online,
        "y_pred_online": y_pred_online,
    }


def _safe_ks_critical_d(window_size: int, ks_alpha: float) -> float:
    """Tính cận dưới an toàn (hệ số 0.90) cho thống kê KS D tới hạn nhằm bỏ qua bước thừa chính xác."""
    return float(0.90 * np.sqrt(-0.5 * np.log(ks_alpha / 2.0) * (2.0 / window_size)))


def evaluate_static_baseline(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    base_model: Optional[ClassifierMixin] = None,
    window_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `BL-st` (Static Baseline): Huấn luyện 1 lần trên W mẫu đầu, không thích nghi."""
    W = window_size
    template = base_model if base_model is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)
    model, single_cls = _fit_safe(template, X_stream[:W], y_stream[:W])

    y_true_online = y_stream[W:]
    y_pred_online = _predict_safe(model, single_cls, X_stream[W:])
    return _build_result("BL-st", y_true_online, y_pred_online, [], W, len(X_stream), num_retrains=0)


def evaluate_periodic_retrain_baseline(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    base_model: Optional[ClassifierMixin] = None,
    window_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `BL-ret` (Periodic Retrain Baseline): Huấn luyện lại định kỳ sau mỗi W mẫu."""
    n_total = len(X_stream)
    W = window_size
    template = base_model if base_model is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)

    model, single_cls = _fit_safe(template, X_stream[:W], y_stream[:W])
    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    retrain_points: List[int] = []

    curr_pos = W
    while curr_pos < n_total:
        next_pos = min(n_total, curr_pos + W)
        y_pred_online[curr_pos - W : next_pos - W] = _predict_safe(
            model, single_cls, X_stream[curr_pos:next_pos]
        )
        if next_pos < n_total:
            retrain_points.append(next_pos)
            model, single_cls = _fit_safe(template, X_stream[next_pos - W : next_pos], y_stream[next_pos - W : next_pos])
        curr_pos = next_pos

    return _build_result(
        "BL-ret",
        y_true_online,
        y_pred_online,
        alarm_indices=[],
        window_size=W,
        n_total=n_total,
        num_retrains=len(retrain_points),
    )


def evaluate_supervised_pht(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    base_model: Optional[ClassifierMixin] = None,
    window_size: int = 2000,
    delta: float = 0.001,
    threshold: float = 50.0,
    l_access: float = 1.0,
    l_delay: int = 0,
    method_name: str = "SS",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Khung chung cho nhóm phương pháp có giám sát bằng Page-Hinkley (`SS`, `DSS`, `WS`, `DWS`).

    Parameters
    ----------
    l_access : float, default=1.0
        Xác suất nhận được nhãn thật sau dự đoán (1.0 cho SS/DSS; 0.5 cho WS/DWS).
    l_delay : int, default=0
        Độ trễ nhãn tính theo số mẫu (0 cho SS/WS; W//2 cho DSS/DWS).
    """
    n_total = len(X_stream)
    W = window_size
    template = base_model if base_model is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)
    rng = np.random.default_rng(random_state)

    model, single_cls = _fit_safe(template, X_stream[:W], y_stream[:W])
    detector = PageHinkley(delta=delta, threshold=threshold)

    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    curr_pos = W
    last_reset_online_idx = 0
    while curr_pos < n_total:
        preds_chunk = _predict_safe(model, single_cls, X_stream[curr_pos:])
        alarm_triggered = False

        for offset in range(len(preds_chunk)):
            stream_idx = curr_pos + offset
            online_idx = stream_idx - W
            y_pred_online[online_idx] = preds_chunk[offset]

            # Nhãn của mẫu `online_idx - l_delay` chỉ khả dụng sau `l_delay` bước và với xác suất `l_access`
            obs_online_idx = online_idx - l_delay
            if obs_online_idx >= last_reset_online_idx:
                label_available = l_access >= 1.0 or (rng.random() < l_access)
                if label_available:
                    err = 1.0 if y_pred_online[obs_online_idx] != y_true_online[obs_online_idx] else 0.0
                    drift_alarm, _ = detector.update(err)
                    if drift_alarm:
                        alarm_indices.append(stream_idx)
                        start_adapt = max(0, stream_idx - W + 1)
                        model, single_cls = _fit_safe(
                            template,
                            X_stream[start_adapt : stream_idx + 1],
                            y_stream[start_adapt : stream_idx + 1],
                        )
                        detector.reset()
                        last_reset_online_idx = online_idx + 1
                        curr_pos = stream_idx + 1
                        alarm_triggered = True
                        break

        if not alarm_triggered:
            break

    return _build_result(method_name, y_true_online, y_pred_online, alarm_indices, W, n_total)


def evaluate_output_ks(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    base_model: Optional[ClassifierMixin] = None,
    window_size: int = 2000,
    ks_alpha: float = 0.001,
    mode: str = "sliding",
    use_prob: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp không giám sát `OS` (Output Sliding) và `OF` (Output Fixed) theo Žliobaitė (2010).

    Sử dụng kiểm định Kolmogorov-Smirnov 2 mẫu trên đầu ra mô hình với chiến lược bỏ qua
    bước thừa có bảo chứng toán học (exact bound-skipping), đảm bảo kết quả trùng khớp 100%
    với việc kiểm tra từng mẫu.
    """
    if mode not in {"sliding", "fixed"}:
        raise ValueError("mode phải là 'sliding' hoặc 'fixed'.")

    n_total = len(X_stream)
    W = window_size
    template = base_model if base_model is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)
    method_name = "OS" if mode == "sliding" else "OF"

    model, single_cls = _fit_safe(template, X_stream[:W], y_stream[:W])
    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    d_crit_safe = _safe_ks_critical_d(W, ks_alpha)
    change_per_step = 2.0 / W if mode == "sliding" else 1.0 / W

    def _signal_from_model(X_batch: np.ndarray, preds_batch: np.ndarray) -> np.ndarray:
        if not use_prob or single_cls is not None or model is None or not hasattr(model, "predict_proba"):
            return preds_batch.astype(np.float64)
        probs = model.predict_proba(X_batch)
        return np.max(probs, axis=1).astype(np.float64)

    ref_signal = _signal_from_model(X_stream[:W], _predict_safe(model, single_cls, X_stream[:W]))

    curr_pos = W
    while curr_pos < n_total:
        preds_chunk = _predict_safe(model, single_cls, X_stream[curr_pos:])
        sig_chunk = _signal_from_model(X_stream[curr_pos:], preds_chunk)
        n_chunk = len(preds_chunk)
        y_pred_online[curr_pos - W : curr_pos - W + n_chunk] = preds_chunk

        min_offset = (2 * W - 1) if mode == "sliding" else (W - 1)
        offset = min_offset
        alarm_triggered = False

        while offset < n_chunk:
            det_win = sig_chunk[offset - W + 1 : offset + 1]
            ref_win = sig_chunk[offset - 2 * W + 1 : offset - W + 1] if mode == "sliding" else ref_signal
            stat, p_val = ks_2samp(ref_win, det_win, method="asymp")
            if p_val < ks_alpha:
                stream_idx = curr_pos + offset
                alarm_indices.append(stream_idx)
                start_adapt = max(0, stream_idx - W + 1)
                X_adapt = X_stream[start_adapt : stream_idx + 1]
                y_adapt = y_stream[start_adapt : stream_idx + 1]
                model, single_cls = _fit_safe(template, X_adapt, y_adapt)
                if mode == "fixed":
                    ref_signal = _signal_from_model(X_adapt, _predict_safe(model, single_cls, X_adapt))
                curr_pos = stream_idx + 1
                alarm_triggered = True
                break

            gap = d_crit_safe - float(stat)
            skip = max(1, int(np.floor(gap / change_per_step))) if gap > 0 else 1
            offset += skip

        if not alarm_triggered:
            break

    return _build_result(method_name, y_true_online, y_pred_online, alarm_indices, W, n_total)


def evaluate_feature_fixed_ks(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    base_model: Optional[ClassifierMixin] = None,
    window_size: int = 2000,
    ks_alpha: float = 0.001,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Phương pháp `FF` (Feature Fixed KS-Test, dos Reis et al. 2016) theo Mục 5.5.

    Giám sát phân phối của từng đặc trưng đầu vào giữa cửa sổ tham chiếu cố định (kích thước W)
    và cửa sổ phát hiện trượt (kích thước W). Sử dụng bound-skipping độc lập trên từng đặc trưng
    để đạt tốc độ cao trên dữ liệu nhiều chiều mà vẫn chính xác tuyệt đối.
    """
    n_total, n_features = X_stream.shape
    W = window_size
    template = base_model if base_model is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)

    model, single_cls = _fit_safe(template, X_stream[:W], y_stream[:W])
    y_true_online = y_stream[W:]
    y_pred_online = np.zeros_like(y_true_online)
    alarm_indices: List[int] = []

    d_crit_safe = _safe_ks_critical_d(W, ks_alpha)
    change_per_step = 1.0 / W

    ref_X = X_stream[:W]
    curr_pos = W

    while curr_pos < n_total:
        preds_chunk = _predict_safe(model, single_cls, X_stream[curr_pos:])
        n_chunk = len(preds_chunk)
        y_pred_online[curr_pos - W : curr_pos - W + n_chunk] = preds_chunk

        if n_chunk < W:
            break

        next_check = np.full(n_features, fill_value=W - 1, dtype=np.int64)
        alarm_triggered = False

        while True:
            min_offset = int(np.min(next_check))
            if min_offset >= n_chunk:
                break

            due_features = np.flatnonzero(next_check == min_offset)
            for feat_idx in due_features:
                det_col = X_stream[curr_pos + min_offset - W + 1 : curr_pos + min_offset + 1, feat_idx]
                stat, p_val = ks_2samp(ref_X[:, feat_idx], det_col, method="asymp")
                if p_val < ks_alpha:
                    stream_idx = curr_pos + min_offset
                    alarm_indices.append(stream_idx)
                    start_adapt = max(0, stream_idx - W + 1)
                    ref_X = X_stream[start_adapt : stream_idx + 1]
                    model, single_cls = _fit_safe(template, ref_X, y_stream[start_adapt : stream_idx + 1])
                    curr_pos = stream_idx + 1
                    alarm_triggered = True
                    break

                gap = d_crit_safe - float(stat)
                skip = max(1, int(np.floor(gap / change_per_step))) if gap > 0 else 1
                next_check[feat_idx] = min_offset + skip

            if alarm_triggered:
                break

        if not alarm_triggered:
            break

    return _build_result("FF", y_true_online, y_pred_online, alarm_indices, W, n_total)


def evaluate_all_10_methods(
    X_stream: np.ndarray,
    y_stream: np.ndarray,
    teacher_model: ClassifierMixin,
    student_model: ClassifierMixin,
    window_size: int = 2000,
    delta: float = 0.001,
    ph_threshold: float = 50.0,
    ks_alpha: float = 0.001,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Chạy đầy đủ cả 10 phương pháp đối sánh trong Mục 5.5 trên cùng một luồng dữ liệu."""
    runners = [
        ("BL-st", lambda: evaluate_static_baseline(X_stream, y_stream, base_model=teacher_model, window_size=window_size, random_state=random_state)),
        ("BL-ret", lambda: evaluate_periodic_retrain_baseline(X_stream, y_stream, base_model=teacher_model, window_size=window_size, random_state=random_state)),
        ("STUDD", lambda: STUDD(teacher_model=teacher_model, student_model=student_model, window_size=window_size, delta=delta, threshold=ph_threshold, random_state=random_state).evaluate_stream(X_stream, y_stream)),
        ("SS", lambda: evaluate_supervised_pht(X_stream, y_stream, base_model=teacher_model, window_size=window_size, delta=delta, threshold=ph_threshold, l_access=1.0, l_delay=0, method_name="SS", random_state=random_state)),
        ("DSS", lambda: evaluate_supervised_pht(X_stream, y_stream, base_model=teacher_model, window_size=window_size, delta=delta, threshold=ph_threshold, l_access=1.0, l_delay=window_size // 2, method_name="DSS", random_state=random_state)),
        ("WS", lambda: evaluate_supervised_pht(X_stream, y_stream, base_model=teacher_model, window_size=window_size, delta=delta, threshold=ph_threshold, l_access=0.5, l_delay=0, method_name="WS", random_state=random_state)),
        ("DWS", lambda: evaluate_supervised_pht(X_stream, y_stream, base_model=teacher_model, window_size=window_size, delta=delta, threshold=ph_threshold, l_access=0.5, l_delay=window_size // 2, method_name="DWS", random_state=random_state)),
        ("OS", lambda: evaluate_output_ks(X_stream, y_stream, base_model=teacher_model, window_size=window_size, ks_alpha=ks_alpha, mode="sliding", random_state=random_state)),
        ("OF", lambda: evaluate_output_ks(X_stream, y_stream, base_model=teacher_model, window_size=window_size, ks_alpha=ks_alpha, mode="fixed", random_state=random_state)),
        ("FF", lambda: evaluate_feature_fixed_ks(X_stream, y_stream, base_model=teacher_model, window_size=window_size, ks_alpha=ks_alpha, random_state=random_state)),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for name, fn in runners:
        res = fn()
        results[name] = res
        if verbose:
            retrains = res.get("num_retrains", res["num_alarms"])
            print(
                f"   -> {name:6s} | Kappa={res['kappa']:.4f} | Acc={res['accuracy']:.4f} | "
                f"Cost={res['labeling_cost']:.4f} | Alarms={res['num_alarms']:3d} | Retrains={retrains:3d}"
            )
    return results


def compute_alarm_metrics(
    results_by_method: Dict[str, Dict[str, Any]],
    window_size: int,
    n_total: int,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Tính các độ đo phân tích báo động (`MDR`, `FAR`, `MTD`, `MTFA`) theo Mục 5.8.3 & Bảng 5.

    Lấy báo động đầu tiên của phương pháp giám sát mạnh `SS` làm mốc thời điểm xảy ra Drift.
    Một báo động được tính là phát hiện đúng nếu xuất hiện từ `t_ss_first - W` trở về sau;
    nếu xuất hiện sớm hơn `t_ss_first - W`, báo động đó bị tính là báo động giả (false alarm).
    """
    ss_alarms = results_by_method["SS"]["alarm_indices"]
    if not ss_alarms:
        return {}

    t_ref = ss_alarms[0]
    grace_start = t_ref - window_size
    metrics_out: Dict[str, Dict[str, Optional[float]]] = {}

    for method_name in ("STUDD", "DSS", "WS", "DWS", "OS", "OF", "FF"):
        if method_name not in results_by_method:
            continue
        alarms: Sequence[int] = results_by_method[method_name]["alarm_indices"]
        true_dets = [t for t in alarms if t >= grace_start]
        false_alarms = [t for t in alarms if t < grace_start]

        mdr = 0.0 if true_dets else 1.0
        far = 1.0 if false_alarms else 0.0
        mtd: Optional[float] = float((true_dets[0] - grace_start) / n_total) if true_dets else None

        if len(false_alarms) >= 2:
            mtfa: Optional[float] = float(np.mean(np.diff(false_alarms)) / n_total)
        elif len(false_alarms) == 1:
            mtfa = float(false_alarms[0] / n_total)
        else:
            mtfa = None

        metrics_out[method_name] = {
            "MDR": mdr,
            "FAR": far,
            "MTD": mtd,
            "MTFA": mtfa,
        }
    return metrics_out


def compute_labeling_efficiency(
    results_by_method: Dict[str, Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Tính hiệu quả sử dụng nhãn (`Labeling_Efficiency`) theo Công thức (1)-(3) Mục 5.8.4."""
    bl_st = results_by_method["BL-st"]
    base_kappa = bl_st["kappa"]
    base_cost = bl_st["labeling_cost"]
    eff: Dict[str, Optional[float]] = {}

    for method_name, res in results_by_method.items():
        if method_name == "BL-st":
            continue
        if base_kappa <= 0.0 or base_cost <= 0.0:
            eff[method_name] = None
            continue
        norm_perf = res["kappa"] / base_kappa
        norm_cost = res["labeling_cost"] / base_cost
        eff[method_name] = float(norm_perf / norm_cost) if norm_cost > 0 else None
    return eff


if __name__ == "__main__":
    import sys
    from data_loader import generate_abrupt_drift_stream
    from studd import create_learner

    sys.stdout.reconfigure(encoding="utf-8")
    print("=== KIỂM THỬ MODULE 4: TOÀN BỘ 10 PHƯƠNG PHÁP ĐỐI SÁNH (MỤC 5.5) ===")
    X_syn, y_syn, meta_syn = generate_abrupt_drift_stream(n_samples=6000, drift_points=[2000, 4000], random_state=42)
    t_model = create_learner("RF", random_state=42)
    s_model = create_learner("RF", random_state=142)

    all_res = evaluate_all_10_methods(
        X_syn,
        y_syn,
        teacher_model=t_model,
        student_model=s_model,
        window_size=500,
        delta=0.001,
        ph_threshold=25.0,
        ks_alpha=0.001,
        random_state=42,
    )
    alarm_stats = compute_alarm_metrics(all_res, window_size=500, n_total=len(X_syn))
    eff_stats = compute_labeling_efficiency(all_res)
    print(f" -> Alarm metrics (SS first alarm={all_res['SS']['alarm_indices'][:1]}): {alarm_stats['STUDD']}")
    print(f" -> Labeling efficiency (STUDD): {eff_stats['STUDD']:.4f}")
    print("=== KIỂM THỬ MODULE 4 HOÀN TẤT THÀNH CÔNG! ===")
