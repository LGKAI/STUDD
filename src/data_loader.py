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

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
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

KHOA_6_DATASETS: Tuple[str, ...] = (
    "insectsabrupt",
    "insects",
    "keystroke",
    "arabic",
    "arabicshuffled",
    "posture",
)

KHANH_7_DATASETS: Tuple[str, ...] = (
    "gmsc",
    "gas",
    "luxembourg",
    "ozone",
    "outdoor",
    "rialto",
    "bike",
)

# Thứ tự chuẩn 19 bộ dữ liệu theo Bảng 1 & Bảng 3 bài báo STUDD
ALL_19_DATASETS: Tuple[str, ...] = (
    "InsectsAbrupt",
    "Insects",
    "Posture",
    "Arabic",
    "Bike",
    "NOAA",
    "Sensors",
    "Powersupply",
    "Poker",
    "Rialto",
    "Ozone",
    "Outdoor",
    "Luxembourg",
    "Gas",
    "Keystroke",
    "ArabicShuffled",
    "CoverType",
    "GMSC",
    "Electricity",
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
    "insects-abrupt": "InsectsAbrupt",
    "abruptinsects": "InsectsAbrupt",
    "insects": "Insects",
    "keystroke": "Keystroke",
    "posture": "Posture",
    "arabic": "Arabic",
    "arabicshuffled": "ArabicShuffled",
    "arabic-shuffled": "ArabicShuffled",
    "arabic_shuffled": "ArabicShuffled",
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


def _find_kaggle_input_file(filename: str) -> Optional[str]:
    """Tìm tệp dữ liệu trong `/kaggle/input` (dành cho Arabic Tier 2)."""
    kaggle_root = Path("/kaggle/input")
    if not kaggle_root.exists():
        return None
    matches = sorted(kaggle_root.rglob(filename))
    return str(matches[0]) if matches else None


def _parse_ts_mfcc_stats(ts_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Đọc tệp `.ts` đa biến SpokenArabicDigits thành 26 đặc trưng (mean + std mỗi kênh MFCC)."""
    X_rows: List[List[float]] = []
    y_rows: List[str] = []
    in_data = False
    with open(ts_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(":")
            label = parts[-1].strip()
            dims = parts[:-1]
            means: List[float] = []
            stds: List[float] = []
            for dim_str in dims:
                vals = np.fromstring(dim_str, sep=",", dtype=np.float32)
                if len(vals) == 0:
                    vals = np.array([0.0], dtype=np.float32)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            X_rows.append(means + stds)
            y_rows.append(label)
    X = np.asarray(X_rows, dtype=np.float32)
    y = LabelEncoder().fit_transform(np.asarray(y_rows, dtype=str)).astype(np.int64)
    return X, y


def _load_arabic_ts_fallback(
    ts_dir: str,
    shuffled: bool = False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tier 3 fallback cho Arabic / ArabicShuffled từ snapshot `.ts` cục bộ."""
    train_ts = os.path.join(ts_dir, "SpokenArabicDigits_TRAIN.ts")
    test_ts = os.path.join(ts_dir, "SpokenArabicDigits_TEST.ts")
    xs, ys = [], []
    for p in (train_ts, test_ts):
        if os.path.exists(p):
            xi, yi = _parse_ts_mfcc_stats(p)
            xs.append(xi)
            ys.append(yi)
    if not xs:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu SpokenArabicDigits .ts tại {ts_dir}")
    X = np.vstack(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    if shuffled:
        # Bổ sung cột sex nhị phân tất định (4400 male=0, 4400 female=1) và xáo trộn theo seed=42
        half = len(X) // 2
        sex_col = np.r_[np.zeros((half, 1), dtype=np.float32), np.ones((len(X) - half, 1), dtype=np.float32)]
        X = np.hstack([X, sex_col])
        perm = np.random.default_rng(seed).permutation(len(X))
        X = X[perm]
        y = y[perm]
    return X, y


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
          cuối cùng tải tự động từ URL trong manifest (và hỗ trợ 3-tier fallback cho Arabic).
        - `"hosted"`: ưu tiên cache đã tải từ URL, nếu chưa có sẽ tải từ URL hosted.
        - `"local"`: chỉ dùng bản snapshot trong repo, `/kaggle/input` hoặc cache đã có sẵn.

    Returns
    -------
    Tuple[str, str]
        `(file_path, origin_label)` cho biết tệp được đọc từ đâu.
    """
    cache_name = spec.get("cache_filename") or os.path.basename(spec["url"])
    cache_path = os.path.join(CACHE_DIR, cache_name)
    local_rel = spec.get("local_path")
    local_path = os.path.join(PROJECT_ROOT, local_rel) if local_rel else None
    kaggle_rel = spec.get("kaggle_relpath")
    ts_rel = spec.get("local_ts_dir")
    ts_dir = os.path.join(PROJECT_ROOT, ts_rel) if ts_rel else None

    if source_mode == "auto":
        if os.path.exists(cache_path):
            return cache_path, f"cache:{os.path.relpath(cache_path, PROJECT_ROOT)}"
        if local_path and os.path.exists(local_path):
            return local_path, f"repo:{local_rel}"
        if kaggle_rel:
            kpath = _find_kaggle_input_file(kaggle_rel)
            if kpath:
                return kpath, f"kaggle_input:{kpath}"
        try:
            print(f"[DataLoader] Đang tải tự động '{spec['dataset']}' từ {spec['url']}...")
            _download_file(spec["url"], cache_path)
            return cache_path, f"downloaded:{spec['url']}"
        except Exception as exc:
            if ts_dir and os.path.isdir(ts_dir):
                print(
                    f"[DataLoader] Cảnh báo: không tải được '{spec['dataset']}' từ Tier 1/2 ({exc}). "
                    f"Chuyển sang Tier 3 fallback (.ts) tại {ts_dir}."
                )
                return ts_dir, f"fallback_ts:{ts_rel}"
            raise

    if source_mode == "hosted":
        if not os.path.exists(cache_path):
            print(f"[DataLoader] Đang tải từ nguồn hosted '{spec['dataset']}': {spec['url']}...")
            _download_file(spec["url"], cache_path)
        return cache_path, f"hosted:{spec['url']}"

    if source_mode == "local":
        if local_path and os.path.exists(local_path):
            return local_path, f"repo:{local_rel}"
        if kaggle_rel:
            kpath = _find_kaggle_input_file(kaggle_rel)
            if kpath:
                return kpath, f"kaggle_input:{kpath}"
        if os.path.exists(cache_path):
            return cache_path, f"cache:{os.path.relpath(cache_path, PROJECT_ROOT)}"
        if ts_dir and os.path.isdir(ts_dir):
            return ts_dir, f"fallback_ts:{ts_rel}"
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

    if fmt == "insects_zip":
        if str(file_path).lower().endswith(".zip"):
            with zipfile.ZipFile(file_path) as zf:
                member = (
                    "AbruptInsects.data"
                    if "AbruptInsects.data" in zf.namelist()
                    else next(
                        (m for m in zf.namelist() if m.endswith("AbruptInsects.data")),
                        zf.namelist()[0],
                    )
                )
                with zf.open(member) as fp:
                    df = pd.read_csv(fp, header=None, nrows=nrows)
        else:
            df = pd.read_csv(file_path, header=None, nrows=nrows)
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str).str.strip()).astype(np.int64)
        return X, y

    if fmt == "bike_hourly_csv":
        df = pd.read_csv(file_path, nrows=nrows)
        feature_cols = ["temp", "atemp", "hum", "windspeed"]
        if "cnt" in df.columns:
            # Raw hour.csv: chỉ chọn 4 đặc trưng thời tiết, tránh rò rỉ nhãn từ cnt/casual/registered
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y = (pd.to_numeric(df["cnt"], errors="coerce").to_numpy() >= 142).astype(np.int64)
            return X, y
        if "target" in df.columns and all(col in df.columns for col in feature_cols):
            # Bản đã tiền xử lý gồm 4 đặc trưng thời tiết và cột target ('high'/'low')
            X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            y_raw = df["target"].astype(str).str.strip().str.lower()
            mapping = {"low": 0, "high": 1, "0": 0, "1": 1}
            if set(y_raw.unique()).issubset(mapping.keys()):
                y = y_raw.map(mapping).to_numpy(dtype=np.int64)
            else:
                y = LabelEncoder().fit_transform(y_raw).astype(np.int64)
            return X, y
        raise ValueError(
            "Định dạng bike_hourly_csv không hợp lệ: thiếu cột 'cnt' hoặc bộ cột "
            "['temp', 'atemp', 'hum', 'windspeed', 'target']."
        )

    if fmt in ("arabic_at_data_zip", "arabic_sex_csv_zip"):
        if os.path.isdir(file_path):
            X_fb, y_fb = _load_arabic_ts_fallback(
                file_path,
                shuffled=(fmt == "arabic_sex_csv_zip"),
                seed=42,
            )
            if nrows is not None:
                return X_fb[:nrows], y_fb[:nrows]
            return X_fb, y_fb

        if str(file_path).lower().endswith(".zip"):
            with zipfile.ZipFile(file_path) as zf:
                preferred = (
                    "Arabic_Paper_Fixed.data"
                    if fmt == "arabic_at_data_zip"
                    else "ArabicDigit_Shuffled_With_Sex.csv"
                )
                member = preferred if preferred in zf.namelist() else zf.namelist()[0]
                raw_bytes = zf.read(member)
        else:
            raw_bytes = Path(file_path).read_bytes()

        if fmt == "arabic_at_data_zip":
            with io.BytesIO(raw_bytes) as stream:
                first_line = stream.readline().strip().lower()
                if first_line != b"@data":
                    stream.seek(0)
                df = pd.read_csv(stream, header=None, nrows=nrows)
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes), nrows=nrows)
            if "sex" in df.columns and df["sex"].dtype == object:
                df["sex"] = df["sex"].astype(str).str.strip().str.lower().map({"male": 0, "female": 1}).fillna(0)

        X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str).str.strip()).astype(np.int64)
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


def load_all_19_stream(
    dataset_name: str,
    max_samples: Optional[int] = PAPER_MAX_SAMPLES,
    normalize: bool = False,
    source_mode: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Nạp hợp nhất bất kỳ bộ dữ liệu nào trong toàn bộ 19 bộ của bài báo STUDD.

    Điều phối tự động giữa:
        - Manifest `data/hosted_sources.json` (6 bộ của Đạt, 6 bộ của Khoa, Bike của Khánh,
          kèm cơ chế fallback 3 tầng cho `Arabic` / `ArabicShuffled`).
        - Registry `STREAM_CONFIG_REGISTRY` (6 bộ còn lại của Khánh: `Luxembourg`,
          `Outdoor`, `Rialto`, `Gas`, `Ozone`, `GMSC`), ưu tiên snapshot cục bộ khi ở chế độ `auto`.
    """
    name_key = dataset_name.lower().strip().replace("-", "").replace("_", "")
    # Map sanitized keys like arabicshuffled -> arabicshuffled
    if name_key in STREAM_CONFIG_REGISTRY:
        cfg = STREAM_CONFIG_REGISTRY[name_key]
        local_exists = os.path.exists(str(cfg["local_snapshot"]))
        prefer_hf = (source_mode == "hosted") or (source_mode == "auto" and not local_exists)
        verify_snapshot = prefer_hf and local_exists
        X, y, raw_meta = load_stream(
            name_key,
            max_samples=max_samples,
            prefer_hf=prefer_hf,
            verify_snapshot=verify_snapshot,
        )
        if normalize:
            X = StandardScaler().fit_transform(X).astype(np.float32)
        meta: Dict[str, object] = {
            "name": name_key,
            "canonical_name": str(raw_meta["name"]),
            "source": str(raw_meta["source_used"]),
            "n_samples": int(raw_meta["n_samples"]),
            "n_features": int(raw_meta["n_features"]),
            "n_classes": int(raw_meta["n_classes"]),
            "recommended_window_size": int(raw_meta["recommended_window_size"]),
        }
        return X, y, meta

    return load_paper_stream(
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
    "gas": {
        "canonical_name": "Gas",
        "hf_repo": "pham121/studd-gas",
        "hf_revision": "9a73335cbe0368e5ae0a57245b1c9dabcbed5e6b",
        "hf_url": "https://huggingface.co/datasets/pham121/studd-gas/resolve/9a73335cbe0368e5ae0a57245b1c9dabcbed5e6b/GasSensorArray.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_not_huggingface", "Gas", "GasSensorArray.csv"),
        "has_header": False,
        "paper_rows": 13910,
        "paper_total_cols": 17,
        "paper_classes": 6,
        "window_size": 2000,
        "teacher": "RC",
        "student": "RC",
        "max_samples": 150000,
    },
    "ozone": {
        "canonical_name": "Ozone",
        "hf_repo": "mstz/ozone",
        "hf_revision": "5d212857f5ff830b714abdeb014532aa34c3e659",
        "hf_url": "https://huggingface.co/datasets/mstz/ozone/resolve/5d212857f5ff830b714abdeb014532aa34c3e659/8hr/train.csv",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_huggingface", "Ozone", "8hr", "train.csv"),
        "has_header": True,
        "paper_rows": 2574,
        "paper_total_cols": 32,
        "paper_classes": 2,
        "window_size": 500,
        "teacher": "XGB",
        "student": "XGB",
        "max_samples": 150000,
    },
    "gmsc": {
        "canonical_name": "GMSC",
        "hf_repo": "algcache/GiveMeSomeCredit",
        "hf_revision": "4c2dab0dce91172737cf3413e023bd16834f1fa1",
        "hf_url": "https://huggingface.co/datasets/algcache/GiveMeSomeCredit/resolve/4c2dab0dce91172737cf3413e023bd16834f1fa1/Give-Me-Some-Credit.zip",
        "local_snapshot": os.path.join(DATA_DIR, "dataset_huggingface", "GMSC", "cs-training.csv"),
        "has_header": True,
        "paper_rows": 150000,
        "paper_total_cols": 11,
        "paper_classes": 2,
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
    elif key in ("luxembourg", "outdoor", "rialto", "gas", "ozone"):
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
                equal_nan=True,
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
    elif key == "gmsc":
        import zipfile

        df_prepared = None
        if prefer_hf:
            try:
                print(
                    f"[DataLoader] Đang tải '{cfg['canonical_name']}' (ZIP) từ Hugging Face "
                    f"({cfg['hf_repo']}@{str(cfg['hf_revision'])[:8]})..."
                )
                with urllib.request.urlopen(hf_url, timeout=45) as resp:
                    raw_zip_bytes = resp.read()
                with zipfile.ZipFile(io.BytesIO(raw_zip_bytes)) as zf:
                    train_member = [m for m in zf.namelist() if m.endswith("cs-training.csv")][0]
                    with zf.open(train_member) as csv_fp:
                        df_prepared = pd.read_csv(csv_fp, low_memory=False)
                source_used = f"huggingface:{cfg['hf_repo']}@{str(cfg['hf_revision'])[:10]}"
            except Exception as exc:
                print(f"[DataLoader] Cảnh báo tải HF ({exc}), chuyển sang đọc snapshot cục bộ: {local_path}")

        if df_prepared is None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {local_path}")
            df_prepared = pd.read_csv(local_path, low_memory=False)
            source_used = f"local:{os.path.relpath(local_path, PROJECT_ROOT)}"

        snapshot_verified = False
        if verify_snapshot and os.path.exists(local_path):
            df_snap = pd.read_csv(local_path, low_memory=False)
            same_shape = df_prepared.shape == df_snap.shape
            same_vals = np.allclose(
                df_prepared.iloc[:, 1:].to_numpy(dtype=np.float64),
                df_snap.iloc[:, 1:].to_numpy(dtype=np.float64),
                atol=1e-7,
                equal_nan=True,
            )
            snapshot_verified = bool(same_shape and same_vals)
            if not snapshot_verified:
                raise AssertionError(
                    f"Dữ liệu GMSC từ Hugging Face KHÔNG KHỚP với snapshot {local_path}!"
                )

        # Cột 0 là ID ('Unnamed: 0') -> loại bỏ; Cột 1 là nhãn 'SeriousDlqin2yrs'; 10 cột còn lại là đặc trưng
        target_col = "SeriousDlqin2yrs" if "SeriousDlqin2yrs" in df_prepared.columns else df_prepared.columns[1]
        id_col = df_prepared.columns[0]
        feat_cols = [c for c in df_prepared.columns if c not in (id_col, target_col)]

        X_full = df_prepared[feat_cols].to_numpy(dtype=np.float32)
        y_full = df_prepared[target_col].to_numpy(dtype=np.int64)
        class_counts = {
            "0": int(np.sum(y_full == 0)),
            "1": int(np.sum(y_full == 1)),
            "missing_cells_imputed_per_window": int(np.isnan(X_full).sum()),
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
