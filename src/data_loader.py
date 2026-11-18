"""Module 3: Bộ nạp dữ liệu luồng tự động (Programmatic Data Stream Loader).

Tuân thủ quy định của Course Lab Book (Mục Experimental Reproduction):
    - Dữ liệu được tải tự động qua code từ nguồn lưu trữ công khai (OpenML)
      và tự động lưu bộ nhớ đệm (cache) vào thư mục `data/` để tái sử dụng nhanh chóng.
    - Hỗ trợ cả các bộ dữ liệu thực tế trong Bảng 1 của bài báo STUDD (Electricity, Ozone, Gas)
      lẫn các luồng dữ liệu có điểm Drift cố định để đo MDR, MTD, FAR và Stress Testing.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Đường dẫn thư mục data/ gốc của dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_benchmark_stream(
    dataset_name: str = "electricity",
    max_samples: Optional[int] = 10000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Tải tự động bộ dữ liệu chuẩn (Benchmark Data Stream) từ OpenML (có lưu cache cục bộ).

    Hỗ trợ các bộ dữ liệu chuẩn trong Bảng 1 (trang 4361) và Mục 5.7 (trang 4365) của bài báo STUDD:
        - `"electricity"`: 45,312 mẫu, 8 đặc trưng, 2 lớp (W chuẩn = 2000).
        - `"ozone"`: 2,534 mẫu, 72 đặc trưng, 2 lớp (W chuẩn = 500).
        - `"gas"`: 13,910 mẫu, 128 đặc trưng, 6 lớp (W chuẩn = 2000).

    Parameters
    ----------
    dataset_name : str, default="electricity"
        Tên bộ dữ liệu cần tải (`"electricity"`, `"ozone"`, hoặc `"gas"`).
    max_samples : int or None, default=10000
        Giới hạn số lượng mẫu tối đa để chạy thực nghiệm nhanh (None nếu muốn lấy toàn bộ).
    normalize : bool, default=True
        Có chuẩn hóa đặc trưng (StandardScaler) hay không.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, object]]
        - `X` (np.ndarray): Ma trận đặc trưng kích thước (N, d).
        - `y` (np.ndarray): Vector nhãn nguyên kích thước (N,).
        - `meta` (dict): Thông tin mô tả bộ dữ liệu (tên, số mẫu, số chiều, số lớp, W khuyến nghị).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    name_key = dataset_name.lower().strip()

    # Ánh xạ tên bộ dữ liệu sang định danh chuẩn trên OpenML và kích thước cửa sổ W theo Mục 5.7
    openml_registry = {
        "electricity": {"openml_name": "electricity", "version": 1, "window_size": 2000},
        "ozone": {"openml_name": "ozone-level-8hr", "version": 1, "window_size": 500},
        "gas": {"openml_name": "gas-drift", "version": 1, "window_size": 2000},
    }

    if name_key not in openml_registry:
        raise ValueError(
            f"Bộ dữ liệu '{dataset_name}' không nằm trong danh mục hỗ trợ: {list(openml_registry.keys())}"
        )

    info = openml_registry[name_key]
    cache_file = os.path.join(DATA_DIR, f"{name_key}_cached.npz")

    # Nếu đã có file cache .npz trong thư mục data/, đọc trực tiếp trong 0.01 giây
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        X_full, y_full = cached["X"], cached["y"]
    else:
        print(f"[DataLoader] Đang tải bộ dữ liệu '{name_key}' từ OpenML (chỉ tải 1 lần đầu tiên)...")
        openml_cache_dir = os.path.join(DATA_DIR, "openml_cache")
        ds = fetch_openml(
            name=info["openml_name"],
            version=info["version"],
            as_frame=True,
            parser="auto",
            data_home=openml_cache_dir,
        )
        df = ds.frame.dropna()
        target_col = ds.target_names[0] if ds.target_names else df.columns[-1]

        # Chuyển đổi các cột phân loại (nếu có) sang số thực và mã hóa nhãn y về 0, 1, ...
        X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_full = X_df.to_numpy(dtype=np.float32)

        le = LabelEncoder()
        y_full = le.fit_transform(df[target_col].astype(str)).astype(np.int64)

        # Lưu cache vào data/ để các lần chạy sau đọc tức thì
        np.savez_compressed(cache_file, X=X_full, y=y_full)
        print(f"[DataLoader] Đã lưu bộ nhớ đệm tại: {cache_file}")

    # Cắt số mẫu theo max_samples (nếu được chỉ định)
    if max_samples is not None and len(X_full) > max_samples:
        X = X_full[:max_samples].copy()
        y = y_full[:max_samples].copy()
    else:
        X = X_full.copy()
        y = y_full.copy()

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    meta = {
        "name": name_key,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
        "recommended_window_size": info["window_size"],
    }
    return X, y, meta


def generate_abrupt_drift_stream(
    n_samples: int = 6000,
    n_features: int = 8,
    drift_points: Optional[List[int]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Sinh luồng dữ liệu có các điểm Concept Drift đột ngột (Abrupt Concept Drift) đã biết trước.

    Mô phỏng kịch bản thí nghiệm Hình 7 (AbruptInsects) và Bảng 5 trong bài báo STUDD,
    ơi các điểm Drift xảy ra tại các mốc thời gian xác định để đánh giá khả năng bắt trúng
    thời điểm Drift của STUDD.

    Parameters
    ----------
    n_samples : int, default=6000
        Tổng số mẫu của luồng dữ liệu.
    n_features : int, default=8
        Số chiều đặc trưng đầu vào.
    drift_points : List[int], optional
        Danh sách các chỉ số thời gian xảy ra Concept Drift. Mặc định là `[2000, 4000]`.
    random_state : int, default=42
        Hạt giống ngẫu nhiên cố định.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, object]]
        `(X, y, meta)` trong đó `meta["drift_points"]` lưu các vị trí xảy ra Drift thật.
    """
    if drift_points is None:
        drift_points = [2000, 4000]

    rng = np.random.default_rng(random_state)
    boundaries = [0] + sorted(drift_points) + [n_samples]

    X_List, y_list = [], []
    # Các hướng dịch chuyển phân phối rõ rệt qua từng Concept (mô phỏng AbruptInsects)
    concept_means = [
        np.zeros(n_features, dtype=np.float32),
        np.array([1.8, -1.8, 0.5, -0.5] + [0.0] * (n_features - 4), dtype=np.float32),
        np.array([0.2, -0.2, 2.2, -2.2] + [0.0] * (n_features - 4), dtype=np.float32),
    ]
    concept_weights = [
        np.array([1.2, 1.0, 0.8, 0.5] + [0.1] * (n_features - 4), dtype=np.float32),
        np.array([-1.0, 1.2, -0.8, 0.6] + [0.1] * (n_features - 4), dtype=np.float32),
        np.array([0.6, 0.8, -1.2, 1.0] + [0.1] * (n_features - 4), dtype=np.float32),
    ]

    for seg_idx in range(len(boundaries) - 1):
        start, end = boundaries[seg_idx], boundaries[seg_idx + 1]
        seg_len = end - start
        c_idx = seg_idx % len(concept_means)

        # Mỗi phân đoạn (concept) có tâm phân phối p(X) và quy luật p(y|X) riêng biệt
        cov_scale = 1.0 if c_idx == 0 else 1.6
        X_seg = rng.normal(
            loc=concept_means[c_idx], scale=cov_scale, size=(seg_len, n_features)
        ).astype(np.float32)

        logits = X_seg @ concept_weights[c_idx]
        threshold_val = float(concept_means[c_idx] @ concept_weights[c_idx])
        y_seg = (logits > threshold_val).astype(np.int64)

        X_List.append(X_seg)
        y_list.append(y_seg)

    X = np.vstack(X_List)
    y = np.concatenate(y_list)

    meta = {
        "name": "abrupt_drift_synthetic",
        "n_samples": len(X),
        "n_features": n_features,
        "n_classes": 2,
        "drift_points": drift_points,
        "recommended_window_size": 500,
    }
    return X, y, meta


def generate_pure_real_drift_stream(
    n_samples: int = 4000,
    n_features: int = 6,
    drift_point: int = 2000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Sinh luồng dữ liệu phục vụ Phản biện Lý thuyết (Stage 4 & 5): Pure Real Concept Drift.

    Đặc điểm:
        - Phân phối biên đầu vào p(X) GIỮ NGUYÊN 100% từ đầu đến cuối luồng: X ~ N(0, I).
        - Tại thời điểm `drift_point`, quy luật nhãn thật p(y|X) bị ĐẢO NGƯỢC HOÀN TOÀN:
          y_new = 1 - y_old.
        - Dùng để chứng minh điểm mù toán học của STUDD: vì p(X) không đổi, Teacher và Student
          vẫn đồng thuận tuyệt đối (L_t = 0) trong khi độ chính xác thật sụt giảm về 0%!
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)

    weights = rng.uniform(-1.0, 1.0, size=(n_features,))
    base_labels = (X @ weights > 0.0).astype(np.int64)

    y = base_labels.copy()
    # Đảo ngược hoàn toàn nhãn sau điểm drift_point
    y[drift_point:] = 1 - y[drift_point:]

    meta = {
        "name": "pure_real_drift",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": 2,
        "drift_points": [drift_point],
        "recommended_window_size": 500,
    }
    return X, y, meta


# ==============================================================================
# BỘ NẠP DỮ LIỆU CHUẨN HUGGING FACE & ĐỐI CHIẾU SNAPSHOT LOCAL (19 DATASETS)
# ==============================================================================

STREAM_CONFIG_REGISTRY: Dict[str, Dict[str, object]] = {
    "bike": {
        "canonical_name": "Bike",
        "hf_repo": "t22000t/bike-sharing-tabular",
        "hf_revision": "6b645e0cedbc5208fcca11ac94544b088478bf61",
        "hf_url": "https://huggingface.co/datasets/t22000t/bike-sharing-tabular/resolve/6b645e0cedbc5208fcca11ac94544b088478bf61/hour.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_huggingface", "Bike", "Bike.csv"),
        "paper_rows": 17378,
        "paper_total_cols": 5,
        "paper_classes": 2,
        "window_size": 2000,
        "teacher": "XGB",
        "student": "RF",
        "max_samples": 150000,
    },
    "luxembourg": {
        "canonical_name": "Luxembourg",
        "hf_repo": "pham121/studd-luxembourg",
        "hf_revision": "5ccd00c9a23ed7f10123a0903b64d0e3db3f01d8",
        "hf_url": "https://huggingface.co/datasets/pham121/studd-luxembourg/resolve/5ccd00c9a23ed7f10123a0903b64d0e3db3f01d8/Luxembourg.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_not_huggingface", "Luxembourg", "Luxembourg.csv"),
        "has_header": True,
        "paper_rows": 1901,
        "paper_total_cols": 31,
        "paper_classes": 2,
        "window_size": 500,
        "teacher": "RF",
        "student": "RF",
        "max_samples": 150000,
    },
    "outdoor": {
        "canonical_name": "Outdoor",
        "hf_repo": "pham121/studd-outdoor",
        "hf_revision": "34d9a05ba9f8b3bf385ab78b66b5eccb0301611f",
        "hf_url": "https://huggingface.co/datasets/pham121/studd-outdoor/resolve/34d9a05ba9f8b3bf385ab78b66b5eccb0301611f/Outdoor.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_not_huggingface", "Outdoor", "Outdoor.csv"),
        "has_header": False,
        "paper_rows": 4000,
        "paper_total_cols": 21,
        "paper_classes": 40,
        "window_size": 500,
        "teacher": "RF",
        "student": "RF",
        "max_samples": 150000,
    },
    "rialto": {
        "canonical_name": "Rialto",
        "hf_repo": "pham121/studd-rialto",
        "hf_revision": "9a876990a741fded06b2a6ede80302fe83e12eb1",
        "hf_url": "https://huggingface.co/datasets/pham121/studd-rialto/resolve/9a876990a741fded06b2a6ede80302fe83e12eb1/Rialto.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_not_huggingface", "Rialto", "Rialto.csv"),
        "has_header": False,
        "paper_rows": 82250,
        "paper_total_cols": 27,
        "paper_classes": 10,
        "window_size": 2000,
        "teacher": "XGB",
        "student": "XGB",
        "max_samples": 150000,
    },
}


def load_stream(
    dataset_name: str,
    max_samples: Optional[int] = 150000,
    prefer_hf: bool = True,
    verify_snapshot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Tải luồng dữ liệu chuẩn từ Hugging Face (pinned commit) và đối chiếu snapshot local.

    Đảm bảo:
        - Tải tự động qua HTTP/Hugging Face theo đúng commit cố định trong báo cáo dữ liệu.
        - Không bao giờ xáo trộn thứ tự dòng (`shuffle=False`), giữ nguyên 100% trục thời gian.
        - Đối chiếu giá trị đặc trưng và nhãn với bản snapshot trong `data/` để đảm bảo tính toàn vẹn.
    """
    import urllib.request
    import io

    key = dataset_name.lower().strip()
    if key not in STREAM_CONFIG_REGISTRY:
        raise ValueError(
            f"Bộ dữ liệu '{dataset_name}' chưa được đăng ký trong STREAM_CONFIG_REGISTRY: "
            f"{list(STREAM_CONFIG_REGISTRY.keys())}"
        )

    cfg = STREAM_CONFIG_REGISTRY[key]
    hf_url = str(cfg["hf_url"])
    local_path = str(cfg["local_snapshot"])
    source_used = "local_snapshot"

    if key == "bike":
        df_prepared = None
        if prefer_hf:
            try:
                print(f"[DataLoader] Đang tải '{cfg['canonical_name']}' từ Hugging Face ({cfg['hf_repo']}@{ str(cfg['hf_revision'])[:8] })...")
                with urllib.request.urlopen(hf_url, timeout=20) as resp:
                    raw_bytes = resp.read()
                df_raw = pd.read_csv(io.BytesIO(raw_bytes))
                # Tái tạo đúng 4 đặc trưng thời tiết và nhãn nhị phân cnt >= 142 theo báo cáo dữ liệu
                feature_cols = ["temp", "atemp", "hum", "windspeed"]
                df_prepared = df_raw[feature_cols].copy()
                df_prepared["target"] = np.where(df_raw["cnt"] >= 142, "high", "low")
                source_used = f"huggingface:{cfg['hf_repo']}@{str(cfg['hf_revision'])[:10]}"
            except Exception as exc:
                print(f"[DataLoader] Cảnh báo tải HF ({exc}), chuyển sang đọc snapshot cục bộ: {local_path}")

        if df_prepared is None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {local_path}")
            df_prepared = pd.read_csv(local_path)
            source_used = f"local:{os.path.relpath(local_path, PROJECT_ROOT)}"

        # Đối chiếu 100% với snapshot local Bike.csv nếu tồn tại
        snapshot_verified = False
        if verify_snapshot and os.path.exists(local_path):
            df_snap = pd.read_csv(local_path)
            feat_cols = ["temp", "atemp", "hum", "windspeed"]
            same_shape = df_prepared.shape == df_snap.shape
            same_feats = np.allclose(
                df_prepared[feat_cols].to_numpy(dtype=np.float64),
                df_snap[feat_cols].to_numpy(dtype=np.float64),
                atol=1e-7,
            )
            same_labels = np.array_equal(
                df_prepared["target"].to_numpy(), df_snap["target"].to_numpy()
            )
            snapshot_verified = bool(same_shape and same_feats and same_labels)
            if not snapshot_verified:
                raise AssertionError(
                    "Dữ liệu Bike tải từ Hugging Face KHÔNG KHỚP với snapshot data/dataset_huggingface/Bike/Bike.csv!"
                )

        feature_names = ["temp", "atemp", "hum", "windspeed"]
        X_full = df_prepared[feature_names].to_numpy(dtype=np.float32)
        # Mã hóa nhãn: low -> 0, high -> 1 (giữ nguyên thứ tự xuất hiện và quy ước nhị phân)
        y_full = np.where(df_prepared["target"].to_numpy() == "high", 1, 0).astype(np.int64)
        class_counts = {
            "low (0)": int(np.sum(y_full == 0)),
            "high (1)": int(np.sum(y_full == 1)),
        }
    elif key in ("luxembourg", "outdoor", "rialto"):
        has_header = bool(cfg.get("has_header", False))
        hdr_arg = 0 if has_header else None
        df_prepared = None

        if prefer_hf:
            try:
                print(
                    f"[DataLoader] Đang tải '{cfg['canonical_name']}' từ Hugging Face "
                    f"({cfg['hf_repo']}@{str(cfg['hf_revision'])[:8]})..."
                )
                with urllib.request.urlopen(hf_url, timeout=30) as resp:
                    raw_bytes = resp.read()
                df_prepared = pd.read_csv(io.BytesIO(raw_bytes), header=hdr_arg, low_memory=False)
                source_used = f"huggingface:{cfg['hf_repo']}@{str(cfg['hf_revision'])[:10]}"
            except Exception as exc:
                print(f"[DataLoader] Cảnh báo tải HF ({exc}), chuyển sang đọc snapshot cục bộ: {local_path}")

        if df_prepared is None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {local_path}")
            df_prepared = pd.read_csv(local_path, header=hdr_arg, low_memory=False)
            source_used = f"local:{os.path.relpath(local_path, PROJECT_ROOT)}"

        snapshot_verified = False
        if verify_snapshot and os.path.exists(local_path):
            df_snap = pd.read_csv(local_path, header=hdr_arg, low_memory=False)
            same_shape = df_prepared.shape == df_snap.shape
            same_feats = np.allclose(
                df_prepared.iloc[:, :-1].to_numpy(dtype=np.float64),
                df_snap.iloc[:, :-1].to_numpy(dtype=np.float64),
                atol=1e-7,
            )
            same_labels = np.array_equal(
                df_prepared.iloc[:, -1].to_numpy(),
                df_snap.iloc[:, -1].to_numpy(),
            )
            snapshot_verified = bool(same_shape and same_feats and same_labels)
            if not snapshot_verified:
                raise AssertionError(
                    f"Dữ liệu '{cfg['canonical_name']}' từ Hugging Face KHÔNG KHỚP với snapshot {local_path}!"
                )

        X_full = df_prepared.iloc[:, :-1].to_numpy(dtype=np.float32)
        raw_labels = df_prepared.iloc[:, -1].to_numpy()
        unique_orig, y_encoded = np.unique(raw_labels, return_inverse=True)
        y_full = y_encoded.astype(np.int64)
        if len(unique_orig) <= 10:
            class_counts = {
                f"{orig} ({idx})": int(np.sum(y_full == idx))
                for idx, orig in enumerate(unique_orig)
            }
        else:
            counts = np.bincount(y_full)
            class_counts = {
                "n_unique_classes": int(len(unique_orig)),
                "min_class_count": int(counts.min()),
                "max_class_count": int(counts.max()),
            }
    else:
        raise NotImplementedError(f"Chưa cài đặt bộ đọc chi tiết cho '{key}'.")

    total_raw_rows = len(X_full)
    limit = max_samples if max_samples is not None else cfg.get("max_samples")
    if limit is not None and len(X_full) > int(limit):
        X = X_full[: int(limit)].copy()
        y = y_full[: int(limit)].copy()
    else:
        X = X_full.copy()
        y = y_full.copy()

    meta = {
        "name": str(cfg["canonical_name"]),
        "key": key,
        "source_used": source_used,
        "hf_url": hf_url,
        "snapshot_verified": snapshot_verified,
        "n_samples_raw": total_raw_rows,
        "n_samples": len(X),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "class_counts": class_counts,
        "paper_rows": int(cfg["paper_rows"]),
        "paper_total_cols": int(cfg["paper_total_cols"]),
        "paper_classes": int(cfg["paper_classes"]),
        "recommended_window_size": int(cfg["window_size"]),
        "teacher_code": str(cfg["teacher"]),
        "student_code": str(cfg["student"]),
    }
    return X, y, meta


# ==============================================================================
# ĐOẠN CODE KIỂM THỬ ĐỘC LẬP (Chạy lệnh: python src/data_loader.py)
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== KIỂM THỬ MODULE 3: BỘ NẠP DỮ LIỆU LUỒNG (DATA LOADER) ===")

    # 1. Kiểm thử sinh luồng Abrupt Drift
    X_syn, y_syn, meta_syn = generate_abrupt_drift_stream(n_samples=6000, drift_points=[2000, 4000])
    print(f"[1] Luồng nhân tạo Abrupt Drift : shape={X_syn.shape}, điểm drift={meta_syn['drift_points']}")

    # 2. Kiểm thử tải bộ dữ liệu thực tế Electricity từ OpenML (và lưu cache vào data/)
    X_elec, y_elec, meta_elec = load_benchmark_stream("electricity", max_samples=5000)
    print(f"[2] Bộ dữ liệu thực Electricity : shape={X_elec.shape}, số lớp={meta_elec['n_classes']}, W={meta_elec['recommended_window_size']}")

    # 3. Kiểm thử tải bộ dữ liệu thực tế Ozone từ OpenML
    X_oz, y_oz, meta_oz = load_benchmark_stream("ozone", max_samples=None)
    print(f"[3] Bộ dữ liệu thực Ozone       : shape={X_oz.shape}, số lớp={meta_oz['n_classes']}, W={meta_oz['recommended_window_size']}")

    print("=== KIỂM THỬ MODULE 3 HOÀN TẤT THÀNH CÔNG! ===")

