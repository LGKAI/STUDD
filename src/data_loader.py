"""Module 3: Bộ nạp dữ liệu luồng tự động (Programmatic Data Stream Loader).

Tuân thủ quy định của Course Lab Book (Mục Experimental Reproduction):
    - Dữ liệu được nạp tự động bằng mã nguồn từ nguồn lưu trữ công khai
      (Hugging Face / hosted manifest trong `data/hosted_sources.json`, dự phòng OpenML)
      và lưu cache cục bộ trong `data/cache/` để tái sử dụng nhanh chóng.
    - Hỗ trợ đầy đủ 6 bộ dữ liệu phân công cho Đạt (`electricity`, `covertype`,
      `poker`, `powersupply`, `noaa`, `sensors`) cùng giới hạn tối đa 150.000 mẫu
      mỗi luồng theo Mục 5.2 của bài báo STUDD (Cerqueira et al., 2023).
    - Giữ tương thích với các bộ dữ liệu cũ (`ozone`, `gas`) và các hàm sinh luồng
      nhân tạo (`generate_abrupt_drift_stream`, `generate_pure_real_drift_stream`).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
HOSTED_MANIFEST_PATH = os.path.join(DATA_DIR, "hosted_sources.json")

PAPER_MAX_SAMPLES = 150_000

DAT_6_DATASETS: Tuple[str, ...] = (
    "electricity",
    "powersupply",
    "noaa",
    "covertype",
    "poker",
    "sensors",
)

# Bản đồ ánh xạ tên chuẩn hóa sang khóa trong data/hosted_sources.json
HOSTED_KEY_MAP: Dict[str, str] = {
    "electricity": "Electricity",
    "covertype": "CoverType",
    "covtype": "CoverType",
    "poker": "Poker",
    "pokerhand": "Poker",
    "powersupply": "Powersupply",
    "noaa": "NOAA",
    "sensors": "Sensors",
    "sensor": "Sensors",
    "insectsabrupt": "InsectsAbrupt",
    "abruptinsects": "InsectsAbrupt",
    "bike": "Bike",
}


def _load_hosted_manifest() -> Dict[str, Dict[str, Any]]:
    """Đọc cấu hình nguồn dữ liệu công khai từ `data/hosted_sources.json`."""
    if not os.path.exists(HOSTED_MANIFEST_PATH):
        return {}
    with open(HOSTED_MANIFEST_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("datasets", {})


def _download_file(url: str, dest_path: str, timeout: int = 120) -> str:
    """Tải tệp dữ liệu theo luồng (streaming) về đường dẫn đích."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp_path = dest_path + ".part"
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    os.replace(tmp_path, dest_path)
    return dest_path


def _resolve_raw_source(
    spec: Dict[str, Any],
    source_mode: str = "auto",
) -> Tuple[str, str]:
    """Xác định đường dẫn tệp dữ liệu nguồn cục bộ hoặc tải tự động từ nguồn hosted.

    Parameters
    ----------
    spec : dict
        Mục cấu hình của bộ dữ liệu trong `data/hosted_sources.json`.
    source_mode : {"auto", "hosted", "local"}, default="auto"
        - `"auto"`: ưu tiên cache trong `data/cache/`, sau đó tới snapshot trong repo,
          cuối cùng tải tự động từ URL trong manifest.
        - `"hosted"`: ưu tiên cache đã tải từ URL, nếu chưa có sẽ tải từ URL hosted.
        - `"local"`: chỉ dùng bản snapshot trong repo hoặc cache đã có sẵn.

    Returns
    -------
    Tuple[str, str]
        `(file_path, origin_label)` cho biết tệp được đọc từ đâu.
    """
    cache_name = spec.get("cache_filename") or os.path.basename(spec["url"])
    cache_path = os.path.join(CACHE_DIR, cache_name)
    local_rel = spec.get("local_path")
    local_path = os.path.join(PROJECT_ROOT, local_rel) if local_rel else None

    if source_mode == "auto":
        if os.path.exists(cache_path):
            return cache_path, f"cache:{os.path.relpath(cache_path, PROJECT_ROOT)}"
        if local_path and os.path.exists(local_path):
            return local_path, f"repo:{local_rel}"
        print(f"[DataLoader] Đang tải tự động '{spec['dataset']}' từ {spec['url']}...")
        _download_file(spec["url"], cache_path)
        return cache_path, f"downloaded:{spec['url']}"

    if source_mode == "hosted":
        if not os.path.exists(cache_path):
            print(f"[DataLoader] Đang tải từ nguồn hosted '{spec['dataset']}': {spec['url']}...")
            _download_file(spec["url"], cache_path)
        return cache_path, f"hosted:{spec['url']}"

    if source_mode == "local":
        if local_path and os.path.exists(local_path):
            return local_path, f"repo:{local_rel}"
        if os.path.exists(cache_path):
            return cache_path, f"cache:{os.path.relpath(cache_path, PROJECT_ROOT)}"
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu cục bộ cho '{spec['dataset']}' khi chạy ở chế độ source_mode='local'."
        )

    raise ValueError(f"source_mode không hợp lệ: '{source_mode}'. Chỉ hỗ trợ 'auto', 'hosted', 'local'.")


def _parse_hosted_dataframe(
    file_path: str,
    fmt: str,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc và chuẩn hóa (X, y) theo đúng đặc thù định dạng của từng bộ dữ liệu."""
    nrows = max_samples if (max_samples is not None and max_samples > 0) else None

    if fmt == "electricity_csv":
        df = pd.read_csv(file_path, nrows=nrows)
        target_col = "target" if "target" in df.columns else df.columns[-1]
        X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_raw = df[target_col].astype(str).str.strip().str.upper()
        mapping = {"DOWN": 0, "UP": 1, "0": 0, "1": 1}
        if set(y_raw.unique()).issubset(mapping.keys()):
            y = y_raw.map(mapping).to_numpy(dtype=np.int64)
        else:
            y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
        return X_df.to_numpy(dtype=np.float32), y

    if fmt == "covertype_broken_header_csv":
        # Dòng 1 của covtype.data chỉ có 52 tên cột trong khi dữ liệu có 55 cột (54 features + label)
        df = pd.read_csv(file_path, skiprows=1, header=None, nrows=nrows)
        X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y_raw = df.iloc[:, -1].astype(str).str.strip()
        y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
        return X, y

    if fmt == "powersupply_csv":
        # Bản ghi 22.203 của Powersupply.csv có dấu phẩy thừa ở cuối (`128.3,93.3,2,`)
        df = pd.read_csv(file_path, header=None, usecols=[0, 1, 2], nrows=nrows)
        X = df.iloc[:, :2].to_numpy(dtype=np.float32)
        y_raw = df.iloc[:, 2].astype(str).str.strip()
        y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
        return X, y

    if fmt == "headerless_csv":
        df = pd.read_csv(file_path, header=None, nrows=nrows)
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_raw = df.iloc[:, -1]
        if pd.api.types.is_numeric_dtype(y_raw):
            # Giữ nguyên thứ tự số học khi nhãn là số nguyên (tránh sắp xếp chuỗi '10' < '2')
            unique_sorted = np.sort(y_raw.unique())
            val_to_idx = {val: idx for idx, val in enumerate(unique_sorted)}
            y = y_raw.map(val_to_idx).to_numpy(dtype=np.int64)
        else:
            y = LabelEncoder().fit_transform(y_raw.astype(str).str.strip()).astype(np.int64)
        return X, y

    if fmt in {"insects_zip", "bike_hourly_csv"}:
        df = pd.read_csv(file_path, nrows=nrows)
        target_col = df.columns[-1]
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y = LabelEncoder().fit_transform(df[target_col].astype(str).str.strip()).astype(np.int64)
        return X, y

    raise ValueError(f"Định dạng dataset chưa được hỗ trợ: '{fmt}'")


def _load_from_openml(
    name_key: str,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Bộ nạp dự phòng qua OpenML cho các bộ chuẩn (`ozone`, `gas`, ...)."""
    openml_registry = {
        "electricity": {"openml_name": "electricity", "version": 1, "window_size": 2000},
        "ozone": {"openml_name": "ozone-level-8hr", "version": 1, "window_size": 500},
        "gas": {"openml_name": "gas-drift", "version": 1, "window_size": 2000},
    }
    if name_key not in openml_registry:
        raise ValueError(
            f"Bộ dữ liệu '{name_key}' chưa có trong manifest hoặc danh mục OpenML."
        )

    info = openml_registry[name_key]
    cache_file = os.path.join(DATA_DIR, f"{name_key}_cached.npz")
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        X_full, y_full = cached["X"], cached["y"]
    else:
        print(f"[DataLoader] Đang tải bộ dữ liệu '{name_key}' từ OpenML...")
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
        X_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_full = X_df.to_numpy(dtype=np.float32)
        y_full = LabelEncoder().fit_transform(df[target_col].astype(str)).astype(np.int64)
        np.savez_compressed(cache_file, X=X_full, y=y_full)

    if max_samples is not None and len(X_full) > max_samples:
        return X_full[:max_samples].copy(), y_full[:max_samples].copy(), info["window_size"], "openml"
    return X_full.copy(), y_full.copy(), info["window_size"], "openml"


def load_benchmark_stream(
    dataset_name: str = "electricity",
    max_samples: Optional[int] = 10000,
    normalize: bool = True,
    source_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Tải tự động bộ dữ liệu luồng chuẩn (từ Hugging Face/hosted manifest hoặc OpenML).

    Hỗ trợ đầy đủ 6 bộ dữ liệu phân công cho Đạt (`electricity`, `covertype`,
    `poker`, `powersupply`, `noaa`, `sensors`) cùng các bộ phụ trợ (`ozone`, `gas`).

    Parameters
    ----------
    dataset_name : str, default="electricity"
        Tên bộ dữ liệu cần nạp.
    max_samples : int or None, default=10000
        Giới hạn số mẫu tối đa tính từ đầu luồng (dùng `150000` theo Mục 5.2 bài báo,
        hoặc `None` để đọc toàn bộ).
    normalize : bool, default=True
        Có chuẩn hóa đặc trưng toàn luồng bằng `StandardScaler` hay không.
    source_mode : {"auto", "hosted", "local"}, default="auto"
        Chế độ chọn nguồn dữ liệu (`auto` ưu tiên cache/local rồi tải tự động khi thiếu).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, object]]
        `(X, y, meta)` gồm ma trận đặc trưng `(N, d)`, vector nhãn `(N,)` và metadata.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    name_key = dataset_name.lower().strip()
    manifest = _load_hosted_manifest()

    if name_key in HOSTED_KEY_MAP and HOSTED_KEY_MAP[name_key] in manifest:
        canonical_key = HOSTED_KEY_MAP[name_key]
        spec = manifest[canonical_key]
        file_path, origin = _resolve_raw_source(spec, source_mode=source_mode)
        X, y = _parse_hosted_dataframe(file_path, spec["format"], max_samples=max_samples)
        window_size = int(spec.get("recommended_window_size", 2000))
    else:
        canonical_key = name_key
        X, y, window_size, origin = _load_from_openml(name_key, max_samples=max_samples)

    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    meta: Dict[str, object] = {
        "name": name_key,
        "canonical_name": canonical_key,
        "source": origin,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "recommended_window_size": window_size,
    }
    return X, y, meta


def load_paper_stream(
    dataset_name: str,
    max_samples: Optional[int] = PAPER_MAX_SAMPLES,
    normalize: bool = False,
    source_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Nạp bộ dữ liệu đúng theo quy chuẩn tái hiện bài báo STUDD.

    Mặc định áp dụng trần `max_samples = 150,000` (Mục 5.2, trang 4360) và giữ nguyên
    thang đo thô (`normalize=False`) để tránh rò rỉ thống kê tương lai; việc chuẩn hóa
    (nếu mô hình cần như `LR` hay `SVC`) sẽ được thực hiện cục bộ trên từng cửa sổ $W$.

    Parameters
    ----------
    dataset_name : str
        Tên bộ dữ liệu (ví dụ một trong `DAT_6_DATASETS`).
    max_samples : int or None, default=150000
        Giới hạn số mẫu tối đa mỗi luồng theo bài báo.
    normalize : bool, default=False
        Có chuẩn hóa đặc trưng toàn luồng trước khi chạy hay không.
    source_mode : {"auto", "hosted", "local"}, default="auto"
        Chế độ chọn nguồn dữ liệu.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, object]]
        `(X, y, meta)`.
    """
    return load_benchmark_stream(
        dataset_name=dataset_name,
        max_samples=max_samples,
        normalize=normalize,
        source_mode=source_mode,
    )


def generate_abrupt_drift_stream(
    n_samples: int = 6000,
    n_features: int = 8,
    drift_points: Optional[List[int]] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Sinh luồng dữ liệu có các điểm Concept Drift đột ngột (Abrupt Concept Drift) đã biết trước.

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

    x_list, y_list = [], []
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

        cov_scale = 1.0 if c_idx == 0 else 1.6
        x_seg = rng.normal(
            loc=concept_means[c_idx], scale=cov_scale, size=(seg_len, n_features)
        ).astype(np.float32)

        logits = x_seg @ concept_weights[c_idx]
        threshold_val = float(concept_means[c_idx] @ concept_weights[c_idx])
        y_seg = (logits > threshold_val).astype(np.int64)

        x_list.append(x_seg)
        y_list.append(y_seg)

    X = np.vstack(x_list)
    y = np.concatenate(y_list)

    meta: Dict[str, object] = {
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
    """Sinh luồng dữ liệu phục vụ Phản biện Lý thuyết: Pure Real Concept Drift.

    Parameters
    ----------
    n_samples : int, default=4000
        Tổng số mẫu của luồng dữ liệu.
    n_features : int, default=6
        Số chiều đặc trưng đầu vào.
    drift_point : int, default=2000
        Chỉ số mẫu nơi quy luật nhãn thật `p(y|X)` bị đảo ngược hoàn toàn.
    random_state : int, default=42
        Hạt giống ngẫu nhiên cố định.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict[str, object]]
        `(X, y, meta)`.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(np.float32)

    weights = rng.uniform(-1.0, 1.0, size=(n_features,))
    base_labels = (X @ weights > 0.0).astype(np.int64)

    y = base_labels.copy()
    y[drift_point:] = 1 - y[drift_point:]

    meta: Dict[str, object] = {
        "name": "pure_real_drift",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": 2,
        "drift_points": [drift_point],
        "recommended_window_size": 500,
    }
    return X, y, meta


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print("=== KIỂM THỬ MODULE 3: BỘ NẠP 6 BỘ DỮ LIỆU CỦA ĐẠT (CAP 150,000 MẪU) ===")
    for ds_name in DAT_6_DATASETS:
        X_ds, y_ds, meta_ds = load_paper_stream(ds_name)
        assert not np.isnan(X_ds).any(), f"Phát hiện NaN trong {ds_name}!"
        print(
            f"[OK] {meta_ds['canonical_name']:12s} | shape={X_ds.shape} | "
            f"classes={meta_ds['n_classes']:2d} | W={meta_ds['recommended_window_size']} | "
            f"source={meta_ds['source']}"
        )
    print("=== KIỂM THỬ MODULE 3 HOÀN TẤT THÀNH CÔNG! ===")
