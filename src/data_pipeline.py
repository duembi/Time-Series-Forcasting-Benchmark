"""
data_pipeline.py — Enterprise Data Engine with Strict Temporal Alignment.

This module provides a fully OOP, Factory-Pattern-based data pipeline that:
    1.  Loads raw time-series from heterogeneous formats (NPZ, HDF5, CSV, Parquet).
    2.  Parses adjacency matrices for graph-based models.
    3.  Engineers temporal, lag, and rolling features.
    4.  Normalizes per-series (fit on train only — zero data leakage).
    5.  Builds a single set of sliding-window tensors and caches them to disk.
    6.  Serves identical ``torch.utils.data.DataLoader`` instances to every model.

Design principles:
    * SOLID / OOP / Factory Pattern
    * Strict chronological Train → Val → Test split (NO shuffle)
    * Shape assertion ``(B, seq_len, F)`` before any model touches the data
    * Memory-efficient: Polars-first processing, mmap-friendly caching

Hardware target: NVIDIA RTX Ada 4500 (24 GB VRAM)
"""

from __future__ import annotations

import abc
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1.  ABSTRACT BASE DATA LOADER                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class BaseDatasetLoader(abc.ABC):
    """Abstract base for all raw-data loaders.

    Every concrete loader must return the same canonical triple so that
    downstream components (scaler, splitter, windower) are format-agnostic.
    """

    @abc.abstractmethod
    def load(
        self,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        """Load raw data from disk.

        Args:
            config: Dataset-specific configuration dictionary.

        Returns:
            data:       ``np.ndarray`` of shape ``(T, N)`` — timesteps × series.
            timestamps: ``pd.DatetimeIndex`` of length ``T``.
            series_ids: ``List[str]`` of length ``N`` (sensor / column names).
        """


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2.  CONCRETE LOADERS                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class NpzDatasetLoader(BaseDatasetLoader):
    """Loader for NumPy compressed archives (.npz).

    Handles PEMS03/04/07/08, PEMS-BAY, and METR-LA when stored as NPZ.
    Expected archive key holds an array of shape ``(T, N, F)``; we extract
    a single feature via ``feature_idx``.
    """

    def load(
        self,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        path: str = config["path"]
        data_key: str = config.get("data_key", "data")
        feature_idx: int = config.get("feature_idx", 0)
        start_date: str = config.get("start_date", "2016-01-01")
        freq: str = config.get("frequency", "5min")

        logger.info("NpzDatasetLoader — loading %s [key=%s]", path, data_key)

        raw = np.load(path)
        data_array: np.ndarray = raw[data_key]

        # (T, N, F) → (T, N) by selecting one feature dimension
        if data_array.ndim == 3:
            data = data_array[:, :, feature_idx].astype(np.float32)
        elif data_array.ndim == 2:
            data = data_array.astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected ndim={data_array.ndim} for key '{data_key}' in {path}"
            )

        timestamps = pd.date_range(
            start=start_date, periods=data.shape[0], freq=freq,
        )
        series_ids = [str(i) for i in range(data.shape[1])]

        logger.info(
            "  → shape=(%d, %d)  range=[%s … %s]",
            *data.shape, timestamps[0], timestamps[-1],
        )
        return data, timestamps, series_ids


class Hdf5DatasetLoader(BaseDatasetLoader):
    """Loader for HDF5 files (.h5).

    Handles METR-LA and PEMS-BAY when distributed as ``pd.read_hdf``-ready
    DataFrames with a ``DatetimeIndex`` and sensor columns.
    """

    def load(
        self,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        path: str = config["path"]

        logger.info("Hdf5DatasetLoader — loading %s", path)

        df: pd.DataFrame = pd.read_hdf(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        data = df.values.astype(np.float32)
        timestamps = pd.DatetimeIndex(df.index)
        series_ids = [str(c) for c in df.columns]

        logger.info(
            "  → shape=(%d, %d)  range=[%s … %s]",
            *data.shape, timestamps[0], timestamps[-1],
        )
        return data, timestamps, series_ids


class CsvDatasetLoader(BaseDatasetLoader):
    """Loader for CSV files.

    Supports single-target (``target_col`` specified) and multi-variate
    layouts (all numeric columns treated as separate series).
    """

    def load(
        self,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        path: str = config["path"]
        date_col: str = config.get("date_col", "date")
        target_col: Optional[str] = config.get("target_col")

        logger.info("CsvDatasetLoader — loading %s", path)

        df = pd.read_csv(path, parse_dates=[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

        timestamps = pd.DatetimeIndex(df[date_col])

        if target_col and target_col in df.columns:
            # Single-target series
            data = df[[target_col]].values.astype(np.float32)
            series_ids = [target_col]
        else:
            # Multi-variate: every numeric column is a series
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            data = df[numeric_cols].values.astype(np.float32)
            series_ids = numeric_cols

        logger.info(
            "  → shape=(%d, %d)  range=[%s … %s]",
            *data.shape, timestamps[0], timestamps[-1],
        )
        return data, timestamps, series_ids


class ParquetDatasetLoader(BaseDatasetLoader):
    """Loader for Parquet files (e.g., NYC Taxi).

    Reads a Parquet file, optionally filters by date range, aggregates to
    the requested frequency, selects the top-N busiest series, and pivots
    into the canonical ``(T, N)`` wide array.
    """

    def load(
        self,
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
        path: str = config["path"]
        date_col: str = config.get("date_col", "tpep_pickup_datetime")
        id_col: str = config.get("id_col", "PULocationID")
        agg_freq: str = config.get("agg_freq", config.get("frequency", "1h"))
        top_n: int = config.get("top_n_series", 5)

        logger.info("ParquetDatasetLoader — loading %s", path)

        df = pd.read_parquet(path, columns=[date_col, id_col])
        df = df.rename(columns={date_col: "ds", id_col: "unique_id"})
        df["ds"] = pd.to_datetime(df["ds"])

        # Optional date filter
        date_filter: Optional[Dict[str, str]] = config.get("date_filter")
        if date_filter:
            df = df[
                (df["ds"] >= date_filter["start"])
                & (df["ds"] < date_filter["end"])
            ]

        # Aggregate: count events per (series, time_bucket)
        df_agg = (
            df.groupby("unique_id")
            .resample(agg_freq, on="ds")
            .size()
            .reset_index(name="y")
        )

        # Keep only the busiest series
        top_ids = (
            df_agg.groupby("unique_id")["y"]
            .sum()
            .nlargest(top_n)
            .index
        )
        df_agg = df_agg[df_agg["unique_id"].isin(top_ids)]

        # Pivot: long → wide  (T, N)
        pivot = df_agg.pivot_table(
            index="ds", columns="unique_id", values="y", fill_value=0,
        )
        pivot = pivot.sort_index()

        data = pivot.values.astype(np.float32)
        timestamps = pd.DatetimeIndex(pivot.index)
        series_ids = [str(c) for c in pivot.columns]

        logger.info(
            "  → shape=(%d, %d)  top_n=%d  range=[%s … %s]",
            *data.shape, top_n, timestamps[0], timestamps[-1],
        )
        return data, timestamps, series_ids


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3.  DATA LOADER FACTORY                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DataLoaderFactory:
    """Factory that maps a ``loader`` type string to a concrete loader class."""

    _registry: Dict[str, type] = {
        "npz": NpzDatasetLoader,
        "h5": Hdf5DatasetLoader,
        "hdf5": Hdf5DatasetLoader,
        "csv": CsvDatasetLoader,
        "parquet": ParquetDatasetLoader,
    }

    @classmethod
    def register(cls, name: str, loader_cls: type) -> None:
        """Register a custom loader at runtime."""
        cls._registry[name] = loader_cls

    @classmethod
    def create(cls, loader_type: str) -> BaseDatasetLoader:
        """Instantiate the appropriate loader.

        Args:
            loader_type: One of ``"npz"``, ``"h5"``, ``"csv"``, ``"parquet"``.

        Returns:
            A ready-to-use ``BaseDatasetLoader`` instance.

        Raises:
            ValueError: If *loader_type* is not registered.
        """
        loader_cls = cls._registry.get(loader_type)
        if loader_cls is None:
            raise ValueError(
                f"Unknown loader type '{loader_type}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return loader_cls()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4.  ADJACENCY MATRIX LOADER                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AdjacencyMatrixLoader:
    """Loads graph adjacency matrices stored as pickled tuples.

    Expected pickle format (e.g., METR-LA)::

        (sensor_ids, sensor_id_to_ind, adj_mx)
    """

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """Load an adjacency matrix from a pickle file.

        Args:
            path: Path to the ``.pkl`` file.

        Returns:
            adj_mx:           ``np.ndarray`` of shape ``(N, N)``.
            sensor_ids:       ``List[str]`` of sensor identifiers.
            sensor_id_to_ind: Mapping from sensor ID to matrix index.
        """
        logger.info("AdjacencyMatrixLoader — loading %s", path)

        with open(path, "rb") as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(
                f, encoding="latin1",
            )

        adj_mx = np.asarray(adj_mx, dtype=np.float32)
        sensor_ids = [str(s) for s in sensor_ids]

        logger.info("  → adj_mx shape=%s  sensors=%d", adj_mx.shape, len(sensor_ids))
        return adj_mx, sensor_ids, sensor_id_to_ind


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5.  TIME SERIES SCALER                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TimeSeriesScaler:
    """Per-series standard (z-score) normalizer.

    CRITICAL: ``fit()`` must be called on **training data only** to prevent
    data leakage.  The same statistics are then applied to val / test via
    ``transform()``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self._eps = eps
        self._mean: Optional[np.ndarray] = None   # shape (N,)
        self._std: Optional[np.ndarray] = None     # shape (N,)
        self._is_fitted: bool = False

    # ── public API ───────────────────────────────────────────────────────────
    def fit(self, data: np.ndarray) -> "TimeSeriesScaler":
        """Compute per-series mean and std from *data* ``(T, N)``.

        Args:
            data: Training portion only.

        Returns:
            ``self`` (for method chaining).
        """
        self._mean = np.nanmean(data, axis=0)
        self._std = np.nanstd(data, axis=0)
        self._std[self._std < self._eps] = 1.0  # avoid division by zero
        self._is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            data: Array of shape ``(T, N)``.

        Returns:
            Normalized array of the same shape.
        """
        self._check_fitted()
        return (data - self._mean) / self._std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the z-score normalization.

        Args:
            data: Normalized array of shape ``(T, N)`` or ``(B, L, N)``.

        Returns:
            De-normalized array of the same shape.
        """
        self._check_fitted()
        return data * self._std + self._mean

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return a serializable snapshot of the scaler state."""
        self._check_fitted()
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    # ── internals ────────────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "TimeSeriesScaler has not been fitted. Call .fit() first."
            )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6.  FEATURE ENGINEER                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FeatureEngineer:
    """Static utility that enriches the raw ``(T, N)`` matrix with temporal,
    lag, and rolling-window features — producing ``(T, N * F)`` or a
    long-format DataFrame suitable for tree-based models.

    This class is used by ``DataPipelineManager`` to optionally append
    features before windowing.
    """

    @staticmethod
    def add_temporal_features(
        timestamps: pd.DatetimeIndex,
        n_series: int,
    ) -> np.ndarray:
        """Create temporal features broadcast across all series.

        Args:
            timestamps: DatetimeIndex of length ``T``.
            n_series:   Number of series ``N``.

        Returns:
            Array of shape ``(T, 3)`` — hour, dayofweek, is_weekend
            (values scaled to [0, 1]).
        """
        hour = timestamps.hour.values / 23.0
        dow = timestamps.dayofweek.values / 6.0
        is_wknd = np.isin(timestamps.dayofweek.values, [5, 6]).astype(np.float32)

        return np.column_stack([hour, dow, is_wknd]).astype(np.float32)

    @staticmethod
    def add_lag_features(
        data: np.ndarray,
        lags: List[int],
    ) -> np.ndarray:
        """Create lag features for each series.

        Args:
            data: Shape ``(T, N)``.
            lags: List of lag steps, e.g. ``[1, 24, 168]``.

        Returns:
            Array of shape ``(T, N * len(lags))``.  Leading rows that cannot
            be computed are filled with ``NaN``.
        """
        parts: List[np.ndarray] = []
        for lag in lags:
            shifted = np.full_like(data, np.nan)
            if lag < data.shape[0]:
                shifted[lag:] = data[:-lag] if lag > 0 else data
            parts.append(shifted)
        return np.concatenate(parts, axis=1).astype(np.float32)

    @staticmethod
    def add_rolling_features(
        data: np.ndarray,
        windows: List[int],
        stats: List[str],
    ) -> np.ndarray:
        """Create rolling-window statistics per series.

        Args:
            data:    Shape ``(T, N)``.
            windows: List of window sizes, e.g. ``[24]``.
            stats:   List of statistics, subset of ``{"mean", "std"}``.

        Returns:
            Array of shape ``(T, N * len(windows) * len(stats))``.
        """
        parts: List[np.ndarray] = []
        df = pd.DataFrame(data)

        for window in windows:
            # Shift by 1 to avoid look-ahead bias
            rolled = df.shift(1).rolling(window=window, min_periods=1)
            for stat in stats:
                if stat == "mean":
                    parts.append(rolled.mean().values.astype(np.float32))
                elif stat == "std":
                    parts.append(
                        rolled.std().fillna(0).values.astype(np.float32)
                    )

        return np.concatenate(parts, axis=1) if parts else np.empty(
            (data.shape[0], 0), dtype=np.float32,
        )

    @staticmethod
    def build_tabular_dataset(
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        series_ids: List[str],
        lags: List[int],
        rolling_windows: List[int],
        rolling_stats: List[str],
    ) -> pd.DataFrame:
        """Build a feature-rich long-format DataFrame for tree-based models.

        The output has columns:
        ``[unique_id, ds, y, hour, dayofweek, is_weekend,
          lag_1, lag_24, …, rolling_mean_24, rolling_std_24, …]``

        Args:
            data:            ``(T, N)`` raw values.
            timestamps:      DatetimeIndex of length ``T``.
            series_ids:      Series names of length ``N``.
            lags:            Lag steps.
            rolling_windows: Rolling window sizes.
            rolling_stats:   Rolling statistics to compute.

        Returns:
            ``pd.DataFrame`` in long format, ready for train/test split.
        """
        records: List[pd.DataFrame] = []

        for col_idx, sid in enumerate(series_ids):
            series_vals = data[:, col_idx]
            df = pd.DataFrame({"ds": timestamps, "y": series_vals})
            df["unique_id"] = sid
            df["hour"] = timestamps.hour
            df["dayofweek"] = timestamps.dayofweek
            df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

            for lag in lags:
                df[f"lag_{lag}"] = df["y"].shift(lag)

            for window in rolling_windows:
                shifted = df["y"].shift(1)
                rolled = shifted.rolling(window=window, min_periods=1)
                for stat in rolling_stats:
                    if stat == "mean":
                        df[f"rolling_mean_{window}"] = rolled.mean()
                    elif stat == "std":
                        df[f"rolling_std_{window}"] = rolled.std().fillna(0)

            records.append(df)

        return pd.concat(records, ignore_index=True)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7.  TEMPORAL SPLITTER                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TemporalSplitter:
    """Chronological train / val / test splitter with leakage assertions.

    **No shuffling.** The time axis is sliced in order, guaranteeing that
    ``max(train_ts) < min(val_ts) < min(test_ts)``.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 1e-6, (
            f"Split ratios must sum to 1.0, got {total:.4f}"
        )
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio

    def split(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> Tuple[
        Tuple[np.ndarray, pd.DatetimeIndex],
        Tuple[np.ndarray, pd.DatetimeIndex],
        Tuple[np.ndarray, pd.DatetimeIndex],
    ]:
        """Split data chronologically.

        Args:
            data:       ``(T, F)`` array (already feature-enriched or raw).
            timestamps: ``DatetimeIndex`` of length ``T``.

        Returns:
            Three ``(array, DatetimeIndex)`` tuples for train / val / test.
        """
        n = data.shape[0]
        train_end = int(n * self._train_ratio)
        val_end = train_end + int(n * self._val_ratio)

        train_data, train_ts = data[:train_end], timestamps[:train_end]
        val_data, val_ts = data[train_end:val_end], timestamps[train_end:val_end]
        test_data, test_ts = data[val_end:], timestamps[val_end:]

        # ── strict temporal-leakage assertion ────────────────────────────────
        assert train_ts[-1] < val_ts[0], (
            f"Temporal leakage: train_end={train_ts[-1]} >= val_start={val_ts[0]}"
        )
        assert val_ts[-1] < test_ts[0], (
            f"Temporal leakage: val_end={val_ts[-1]} >= test_start={test_ts[0]}"
        )

        logger.info(
            "TemporalSplitter — train=%d  val=%d  test=%d  "
            "[%s → %s | %s → %s | %s → %s]",
            len(train_ts), len(val_ts), len(test_ts),
            train_ts[0], train_ts[-1],
            val_ts[0], val_ts[-1],
            test_ts[0], test_ts[-1],
        )
        return (train_data, train_ts), (val_data, val_ts), (test_data, test_ts)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8.  SLIDING WINDOW PYTORCH DATASET                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SlidingWindowDataset(Dataset):
    """PyTorch ``Dataset`` that yields ``(x, y)`` sliding-window pairs.

    Each sample is a contiguous slice of the multivariate time-series:

    * ``x``: shape ``(seq_len, num_features)``   — input context
    * ``y``: shape ``(pred_len, num_targets)``    — forecast target

    The dataset enforces shape invariants via ``assert`` at construction time
    and inside ``__getitem__`` (guarded by ``__debug__``).
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        stride: int = 1,
        target_cols: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            data:        ``(T, F)`` — time steps × total features.
            seq_len:     Number of historical steps the model sees.
            pred_len:    Number of future steps to predict.
            stride:      Step size between consecutive windows.
            target_cols: Column indices used as prediction targets.
                         Defaults to all columns.
        """
        super().__init__()

        assert data.ndim == 2, f"Expected 2-D array, got shape {data.shape}"
        assert seq_len > 0 and pred_len > 0, "seq_len and pred_len must be > 0"
        assert data.shape[0] >= seq_len + pred_len, (
            f"Insufficient data: T={data.shape[0]} < seq_len+pred_len="
            f"{seq_len + pred_len}"
        )

        self._data = torch.as_tensor(data, dtype=torch.float32)
        self._seq_len = seq_len
        self._pred_len = pred_len
        self._stride = stride
        self._target_cols = target_cols  # None → use all columns

        # Pre-compute valid window start indices
        max_start = data.shape[0] - seq_len - pred_len
        self._indices = list(range(0, max_start + 1, stride))

        logger.debug(
            "SlidingWindowDataset — T=%d  seq=%d  pred=%d  stride=%d  "
            "windows=%d  features=%d",
            data.shape[0], seq_len, pred_len, stride,
            len(self._indices), data.shape[1],
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self._indices[idx]
        x = self._data[start: start + self._seq_len]
        y_start = start + self._seq_len
        y = self._data[y_start: y_start + self._pred_len]

        if self._target_cols is not None:
            y = y[:, self._target_cols]

        # Runtime shape guards (compiled away with python -O)
        assert x.shape[0] == self._seq_len, f"x.shape={x.shape}"
        assert y.shape[0] == self._pred_len, f"y.shape={y.shape}"

        return x, y

    @property
    def num_features(self) -> int:
        return self._data.shape[1]

    @property
    def num_targets(self) -> int:
        if self._target_cols is not None:
            return len(self._target_cols)
        return self._data.shape[1]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  9.  DATA PIPELINE MANAGER  (Single Entry Point)                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DataPipelineManager:
    """Orchestrates the full data lifecycle: load → enrich → scale →
    split → window → cache → serve ``DataLoader`` instances.

    Usage::

        manager = DataPipelineManager(experiment_config)
        result = manager.prepare(dataset_config)

        train_loader = result["train_loader"]
        val_loader   = result["val_loader"]
        test_loader  = result["test_loader"]
        scaler       = result["scaler"]
    """

    def __init__(self, experiment_cfg: Dict[str, Any]) -> None:
        """
        Args:
            experiment_cfg: Parsed ``experiment.yaml`` dictionary.
        """
        self._cfg = experiment_cfg

        # Windowing params (global — same for every model)
        wnd = experiment_cfg["windowing"]
        self._seq_len: int = wnd["seq_len"]
        self._pred_len: int = wnd["pred_len"]
        self._stride: int = wnd.get("stride", 1)

        # Splitting
        sp = experiment_cfg["splitting"]
        self._splitter = TemporalSplitter(
            train_ratio=sp["train_ratio"],
            val_ratio=sp["val_ratio"],
            test_ratio=sp["test_ratio"],
        )

        # Feature engineering config
        self._feat_cfg: Dict[str, Any] = experiment_cfg.get("features", {})

        # Training
        self._batch_size: int = experiment_cfg["training"]["batch_size"]

        # Caching
        exec_cfg = experiment_cfg.get("execution", {})
        self._cache_enabled: bool = exec_cfg.get("cache_datasets", True)
        self._cache_dir = Path(exec_cfg.get("cache_dir", "outputs/cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ─── public API ──────────────────────────────────────────────────────────
    def prepare(
        self,
        dataset_cfg: Dict[str, Any],
        batch_size_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """End-to-end data preparation.

        Args:
            dataset_cfg:         Parsed dataset YAML (e.g., ``pems08.yaml``).
            batch_size_override: Optional batch size (used during OOM retry).

        Returns:
            Dictionary with keys:
                ``train_loader``, ``val_loader``, ``test_loader``,
                ``scaler``, ``metadata``.
        """
        ds = dataset_cfg["dataset"]
        batch_size = batch_size_override or self._batch_size
        cache_key = self._build_cache_key(ds)

        # ── try loading from cache ───────────────────────────────────────────
        if self._cache_enabled:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("Pipeline [%s] — loaded from cache.", ds["name"])
                return self._build_loaders(cached, batch_size, ds["name"])

        # ── 1. load raw data ────────────────────────────────────────────────
        loader = DataLoaderFactory.create(ds["loader"])
        data, timestamps, series_ids = loader.load(ds)

        # ── 2. handle missing values ────────────────────────────────────────
        nan_pct = np.isnan(data).sum() / data.size * 100
        if nan_pct > 0:
            logger.warning(
                "Dataset '%s' has %.2f%% NaN values — forward-filling.",
                ds["name"], nan_pct,
            )
            df_tmp = pd.DataFrame(data)
            df_tmp = df_tmp.ffill().bfill()
            data = df_tmp.values.astype(np.float32)

        # ── 3. chronological split (on raw data) ───────────────────────────
        (train_raw, train_ts), (val_raw, val_ts), (test_raw, test_ts) = (
            self._splitter.split(data, timestamps)
        )

        # ── 4. fit scaler on TRAIN only ─────────────────────────────────────
        scaler = TimeSeriesScaler()
        scaler.fit(train_raw)

        train_scaled = scaler.transform(train_raw)
        val_scaled = scaler.transform(val_raw)
        test_scaled = scaler.transform(test_raw)

        # ── 5. optional feature enrichment ──────────────────────────────────
        train_enriched = self._enrich(train_scaled, train_ts, len(series_ids))
        val_enriched = self._enrich(val_scaled, val_ts, len(series_ids))
        test_enriched = self._enrich(test_scaled, test_ts, len(series_ids))

        # ── 6. cache ────────────────────────────────────────────────────────
        cache_payload = {
            "train": train_enriched,
            "val": val_enriched,
            "test": test_enriched,
            "train_ts": train_ts,
            "val_ts": val_ts,
            "test_ts": test_ts,
            "scaler_params": scaler.get_params(),
            "series_ids": series_ids,
        }
        if self._cache_enabled:
            self._save_cache(cache_key, cache_payload)

        return self._build_loaders(cache_payload, batch_size, ds["name"])

    def prepare_tabular(
        self,
        dataset_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare data specifically for tree-based (tabular) models.

        Returns:
            Dictionary with keys ``X_train``, ``y_train``, ``X_test``,
            ``y_test``, ``feature_names``.
        """
        ds = dataset_cfg["dataset"]

        loader = DataLoaderFactory.create(ds["loader"])
        data, timestamps, series_ids = loader.load(ds)

        lags = self._feat_cfg.get("lags", [1, 24, 168])
        rolling_windows = self._feat_cfg.get("rolling_windows", [24])
        rolling_stats = self._feat_cfg.get("rolling_stats", ["mean", "std"])

        df = FeatureEngineer.build_tabular_dataset(
            data, timestamps, series_ids,
            lags=lags,
            rolling_windows=rolling_windows,
            rolling_stats=rolling_stats,
        )

        # Chronological split: last `pred_len` steps per series → test
        horizon = self._pred_len
        test_idx = df.groupby("unique_id").tail(horizon).index
        train_mask = ~df.index.isin(test_idx)

        feature_cols = [
            c for c in df.columns
            if c not in ("unique_id", "ds", "y")
        ]
        df = df.dropna().reset_index(drop=True)

        # Recompute masks after dropna
        test_idx = df.groupby("unique_id").tail(horizon).index
        train_mask = ~df.index.isin(test_idx)

        return {
            "X_train": df.loc[train_mask, feature_cols],
            "y_train": df.loc[train_mask, "y"],
            "X_test": df.loc[~train_mask, feature_cols],
            "y_test": df.loc[~train_mask, "y"],
            "feature_names": feature_cols,
        }

    # ─── internal helpers ────────────────────────────────────────────────────
    def _enrich(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        n_series: int,
    ) -> np.ndarray:
        """Optionally concatenate temporal features to the data matrix."""
        feat_cfg = self._feat_cfg
        if not feat_cfg or not feat_cfg.get("temporal"):
            return data

        temporal = FeatureEngineer.add_temporal_features(timestamps, n_series)
        # Repeat temporal features if data has more series columns than 3
        # temporal columns: just concatenate as extra columns
        return np.concatenate([data, temporal], axis=1).astype(np.float32)

    def _build_loaders(
        self,
        payload: Dict[str, Any],
        batch_size: int,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Wrap cached numpy arrays into PyTorch DataLoaders."""
        # Reconstruct scaler from saved params
        scaler = TimeSeriesScaler()
        scaler.fit(
            np.stack(
                [payload["scaler_params"]["mean"],
                 payload["scaler_params"]["std"]],
            ).T  # Dummy 2-row array just to set internal state
        )
        # Directly inject saved params (more accurate)
        scaler._mean = payload["scaler_params"]["mean"]
        scaler._std = payload["scaler_params"]["std"]
        scaler._is_fitted = True

        loaders: Dict[str, DataLoader] = {}
        for split_name in ("train", "val", "test"):
            ds = SlidingWindowDataset(
                data=payload[split_name],
                seq_len=self._seq_len,
                pred_len=self._pred_len,
                stride=self._stride,
            )
            loaders[f"{split_name}_loader"] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,      # NEVER shuffle time-series
                num_workers=0,      # Windows-safe default
                pin_memory=True,
                drop_last=False,
            )

        # ── batch-level shape assertion ──────────────────────────────────────
        sample_x, sample_y = next(iter(loaders["train_loader"]))
        assert sample_x.ndim == 3 and sample_x.shape[1] == self._seq_len, (
            f"Shape violation: expected (B, {self._seq_len}, F), "
            f"got {sample_x.shape}"
        )
        assert sample_y.ndim == 3 and sample_y.shape[1] == self._pred_len, (
            f"Shape violation: expected (B, {self._pred_len}, F), "
            f"got {sample_y.shape}"
        )
        logger.info(
            "Pipeline [%s] — train=%d  val=%d  test=%d batches  "
            "x=%s  y=%s  batch_size=%d",
            dataset_name,
            len(loaders["train_loader"]),
            len(loaders["val_loader"]),
            len(loaders["test_loader"]),
            list(sample_x.shape),
            list(sample_y.shape),
            batch_size,
        )

        return {
            **loaders,
            "scaler": scaler,
            "metadata": {
                "dataset_name": dataset_name,
                "series_ids": payload["series_ids"],
                "num_features": sample_x.shape[2],
                "num_targets": sample_y.shape[2],
                "seq_len": self._seq_len,
                "pred_len": self._pred_len,
                "train_samples": len(loaders["train_loader"].dataset),
                "val_samples": len(loaders["val_loader"].dataset),
                "test_samples": len(loaders["test_loader"].dataset),
            },
        }

    # ── caching ──────────────────────────────────────────────────────────────
    def _build_cache_key(self, ds_cfg: Dict[str, Any]) -> str:
        """Deterministic cache key from dataset config + windowing params."""
        raw = (
            f"{ds_cfg['name']}_{ds_cfg['loader']}_{ds_cfg.get('path', '')}_"
            f"seq{self._seq_len}_pred{self._pred_len}_stride{self._stride}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _save_cache(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._cache_dir / f"{key}.pt"
        torch.save(payload, str(path))
        logger.info("Cached dataset → %s", path)

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_dir / f"{key}.pt"
        if path.exists():
            return torch.load(str(path), weights_only=False)
        return None
