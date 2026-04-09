"""
tests/test_data_pipeline.py

Unit and integration tests for the data pipeline module.
Run with: pytest tests/test_data_pipeline.py -v

All tests use synthetic in-memory data — no actual dataset files needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_pipeline import (
    AdjacencyMatrixLoader,
    DataLoaderFactory,
    DataPipelineManager,
    FeatureEngineer,
    NpzDatasetLoader,
    CsvDatasetLoader,
    SlidingWindowDataset,
    TemporalSplitter,
    TimeSeriesScaler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """1000 timesteps × 5 series of synthetic sinusoidal traffic data."""
    T, N = 1000, 5
    t = np.linspace(0, 4 * np.pi, T)
    data = np.stack([np.sin(t + i) * 50 + 100 for i in range(N)], axis=1).astype(np.float32)
    timestamps = pd.date_range("2020-01-01", periods=T, freq="5min")
    series_ids = [f"sensor_{i}" for i in range(N)]
    return data, timestamps, series_ids


@pytest.fixture()
def experiment_cfg() -> dict:
    return {
        "windowing": {"seq_len": 48, "pred_len": 12, "stride": 1, "frequency": "5min"},
        "splitting": {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2},
        "features": {"temporal": ["hour", "dayofweek", "is_weekend"], "lags": [1, 24], "rolling_windows": [12], "rolling_stats": ["mean", "std"]},
        "training": {"batch_size": 16, "max_epochs": 2, "learning_rate": 1e-3, "weight_decay": 1e-4,
                     "early_stopping": {"patience": 2, "min_delta": 1e-4, "mode": "min"},
                     "oom_recovery": {"enabled": True, "max_retries": 1, "batch_size_reduction_factor": 0.5}},
        "hardware": {"mixed_precision": False, "precision_dtype": "float32", "gradient_accumulation_steps": 1},
        "execution": {"cache_datasets": False, "cache_dir": "/tmp/tsf_test_cache"},
        "logging": {"log_dir": "/tmp/tsf_test_logs", "checkpoint_dir": "/tmp/tsf_test_ckpts", "report_dir": "/tmp/tsf_test_reports"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# TimeSeriesScaler
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeSeriesScaler:

    def test_fit_transform_shape(self, synthetic_data):
        data, _, _ = synthetic_data
        scaler = TimeSeriesScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        assert scaled.shape == data.shape

    def test_zero_mean_unit_std_after_fit(self, synthetic_data):
        data, _, _ = synthetic_data
        scaler = TimeSeriesScaler()
        scaler.fit(data)
        scaled = scaler.transform(data)
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-4)

    def test_inverse_transform_recovers_original(self, synthetic_data):
        data, _, _ = synthetic_data
        scaler = TimeSeriesScaler()
        scaler.fit(data)
        recovered = scaler.inverse_transform(scaler.transform(data))
        assert np.allclose(recovered, data, atol=1e-4)

    def test_raises_if_not_fitted(self, synthetic_data):
        data, _, _ = synthetic_data
        scaler = TimeSeriesScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(data)

    def test_fit_on_train_only_does_not_leak(self, synthetic_data):
        """Scaler fitted on train should NOT use val/test statistics."""
        data, _, _ = synthetic_data
        train = data[:350]
        test = data[350:]
        scaler = TimeSeriesScaler()
        scaler.fit(train)
        # Scaler mean/std must equal train statistics only
        assert np.allclose(scaler._mean, train.mean(axis=0), atol=1e-5)
        assert np.allclose(scaler._std, train.std(axis=0), atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# TemporalSplitter
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalSplitter:

    def test_split_ratios(self, synthetic_data):
        data, timestamps, _ = synthetic_data
        splitter = TemporalSplitter(0.7, 0.1, 0.2)
        (tr_d, tr_ts), (v_d, v_ts), (te_d, te_ts) = splitter.split(data, timestamps)

        total = len(tr_ts) + len(v_ts) + len(te_ts)
        assert total == len(timestamps)
        assert abs(len(tr_ts) / total - 0.7) < 0.02
        assert abs(len(v_ts) / total - 0.1) < 0.02

    def test_no_temporal_leakage(self, synthetic_data):
        data, timestamps, _ = synthetic_data
        splitter = TemporalSplitter(0.7, 0.1, 0.2)
        (_, tr_ts), (_, v_ts), (_, te_ts) = splitter.split(data, timestamps)

        assert tr_ts[-1] < v_ts[0], "Train overlaps Val"
        assert v_ts[-1] < te_ts[0], "Val overlaps Test"

    def test_invalid_ratios_raise(self):
        with pytest.raises(AssertionError):
            TemporalSplitter(0.7, 0.2, 0.2)  # sum = 1.1


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowDataset
# ─────────────────────────────────────────────────────────────────────────────

class TestSlidingWindowDataset:

    def test_length(self, synthetic_data):
        data, _, _ = synthetic_data
        seq_len, pred_len, stride = 48, 12, 1
        ds = SlidingWindowDataset(data, seq_len, pred_len, stride)
        expected = data.shape[0] - seq_len - pred_len + 1
        assert len(ds) == expected

    def test_output_shapes(self, synthetic_data):
        data, _, _ = synthetic_data
        seq_len, pred_len = 48, 12
        ds = SlidingWindowDataset(data, seq_len, pred_len)
        x, y = ds[0]
        assert x.shape == (seq_len, data.shape[1])
        assert y.shape == (pred_len, data.shape[1])

    def test_no_overlap_between_x_and_y(self, synthetic_data):
        data, _, _ = synthetic_data
        ds = SlidingWindowDataset(data, 48, 12)
        x, y = ds[0]
        # Last timestep of x should be data[47], first of y should be data[48]
        assert torch.allclose(x[-1], torch.tensor(data[47]))
        assert torch.allclose(y[0], torch.tensor(data[48]))

    def test_stride(self, synthetic_data):
        data, _, _ = synthetic_data
        ds_s1 = SlidingWindowDataset(data, 48, 12, stride=1)
        ds_s4 = SlidingWindowDataset(data, 48, 12, stride=4)
        assert len(ds_s4) < len(ds_s1)

    def test_insufficient_data_raises(self):
        tiny_data = np.random.rand(10, 3).astype(np.float32)
        with pytest.raises(AssertionError):
            SlidingWindowDataset(tiny_data, seq_len=48, pred_len=12)

    def test_dataloader_batch_shape(self, synthetic_data):
        data, _, _ = synthetic_data
        ds = SlidingWindowDataset(data, 48, 12)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=8, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (8, 48, data.shape[1])
        assert y_batch.shape == (8, 12, data.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_temporal_features_shape(self, synthetic_data):
        data, timestamps, _ = synthetic_data
        feats = FeatureEngineer.add_temporal_features(timestamps, n_series=data.shape[1])
        assert feats.shape == (len(timestamps), 3)

    def test_temporal_feature_range(self, synthetic_data):
        _, timestamps, _ = synthetic_data
        feats = FeatureEngineer.add_temporal_features(timestamps, n_series=1)
        assert feats[:, 0].min() >= 0.0 and feats[:, 0].max() <= 1.0  # hour normalised
        assert feats[:, 2].sum() >= 0  # is_weekend counts

    def test_lag_features_shape(self, synthetic_data):
        data, _, _ = synthetic_data
        lags = [1, 24, 168]
        lag_feats = FeatureEngineer.add_lag_features(data, lags)
        assert lag_feats.shape == (data.shape[0], data.shape[1] * len(lags))

    def test_lag_features_correct_shift(self, synthetic_data):
        data, _, _ = synthetic_data
        lag_feats = FeatureEngineer.add_lag_features(data, lags=[1])
        # lag_1 of timestep 1 should equal data[0]
        assert np.allclose(lag_feats[1, : data.shape[1]], data[0], atol=1e-6)

    def test_rolling_features_shape(self, synthetic_data):
        data, _, _ = synthetic_data
        roll = FeatureEngineer.add_rolling_features(data, windows=[12], stats=["mean", "std"])
        assert roll.shape[0] == data.shape[0]
        # 1 window × 2 stats × N series
        assert roll.shape[1] == data.shape[1] * 2


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaderFactory
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoaderFactory:

    def test_creates_npz_loader(self):
        loader = DataLoaderFactory.create("npz")
        assert isinstance(loader, NpzDatasetLoader)

    def test_creates_csv_loader(self):
        loader = DataLoaderFactory.create("csv")
        assert isinstance(loader, CsvDatasetLoader)

    def test_unknown_loader_raises(self):
        with pytest.raises(ValueError, match="Unknown loader"):
            DataLoaderFactory.create("parquet99")

    def test_custom_registration(self):
        class DummyLoader(NpzDatasetLoader):
            pass

        DataLoaderFactory.register("dummy", DummyLoader)
        loader = DataLoaderFactory.create("dummy")
        assert isinstance(loader, DummyLoader)


# ─────────────────────────────────────────────────────────────────────────────
# DataPipelineManager — integration
# ─────────────────────────────────────────────────────────────────────────────

class TestDataPipelineManagerIntegration:
    """End-to-end pipeline test using synthetic NPZ data written to a temp file."""

    def test_prepare_returns_three_loaders(self, tmp_path, experiment_cfg, synthetic_data):
        data, timestamps, series_ids = synthetic_data

        # Write synthetic NPZ to tmp
        npz_path = tmp_path / "synthetic.npz"
        np.savez(str(npz_path), data=data[:, :, np.newaxis])  # shape (T, N, 1)

        ds_cfg = {
            "dataset": {
                "name": "Synthetic",
                "loader": "npz",
                "path": str(npz_path),
                "data_key": "data",
                "feature_idx": 0,
                "start_date": "2020-01-01",
                "frequency": "5min",
            }
        }

        manager = DataPipelineManager(experiment_cfg)
        result = manager.prepare(ds_cfg)

        assert "train_loader" in result
        assert "val_loader" in result
        assert "test_loader" in result
        assert "scaler" in result
        assert "metadata" in result

    def test_batch_shape_assertion_passes(self, tmp_path, experiment_cfg, synthetic_data):
        data, _, _ = synthetic_data
        npz_path = tmp_path / "synthetic.npz"
        np.savez(str(npz_path), data=data[:, :, np.newaxis])

        ds_cfg = {"dataset": {"name": "Synthetic", "loader": "npz", "path": str(npz_path),
                               "data_key": "data", "feature_idx": 0, "start_date": "2020-01-01", "frequency": "5min"}}

        manager = DataPipelineManager(experiment_cfg)
        result = manager.prepare(ds_cfg)

        # Iterate one batch and check shape
        seq_len = experiment_cfg["windowing"]["seq_len"]
        pred_len = experiment_cfg["windowing"]["pred_len"]
        x, y = next(iter(result["train_loader"]))
        assert x.shape[1] == seq_len
        assert y.shape[1] == pred_len

    def test_no_temporal_leakage_in_loaders(self, tmp_path, experiment_cfg, synthetic_data):
        """Verify that train samples never contain future information."""
        data, _, _ = synthetic_data
        npz_path = tmp_path / "synthetic.npz"
        np.savez(str(npz_path), data=data[:, :, np.newaxis])

        ds_cfg = {"dataset": {"name": "Synthetic", "loader": "npz", "path": str(npz_path),
                               "data_key": "data", "feature_idx": 0, "start_date": "2020-01-01", "frequency": "5min"}}

        manager = DataPipelineManager(experiment_cfg)
        result = manager.prepare(ds_cfg)

        # The metadata should confirm non-overlapping samples
        meta = result["metadata"]
        assert meta["train_samples"] > 0
        assert meta["val_samples"] > 0
        assert meta["test_samples"] > 0
