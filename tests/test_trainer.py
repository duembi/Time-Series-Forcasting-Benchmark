"""
tests/test_trainer.py

Unit tests for the trainer module.
Run with: pytest tests/test_trainer.py -v

Tests use tiny synthetic models and data — no GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trainer import (
    CheckpointManager,
    EarlyStopper,
    ExperimentLogger,
    HardwareMonitor,
    MetricCalculator,
    TimingContext,
    TrainingResult,
    BaseTrainer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_model() -> nn.Module:
    """A minimal 2-layer MLP for testing the training loop."""

    class TinySeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(48 * 3, 64), nn.ReLU(), nn.Linear(64, 12 * 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b = x.shape[0]
            return self.fc(x.reshape(b, -1)).reshape(b, 12, 3)

    return TinySeq2Seq()


@pytest.fixture()
def tiny_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tiny synthetic DataLoaders: (B=4, seq=48, F=3) → (B=4, pred=12, F=3)."""
    N, seq, pred, F = 40, 48, 12, 3
    x = torch.randn(N, seq, F)
    y = torch.randn(N, pred, F)
    ds = TensorDataset(x, y)
    train_loader = DataLoader(Subset(ds, range(28)), batch_size=4, shuffle=False)
    val_loader   = DataLoader(Subset(ds, range(28, 34)), batch_size=4, shuffle=False)
    test_loader  = DataLoader(Subset(ds, range(34, 40)), batch_size=4, shuffle=False)
    return train_loader, val_loader, test_loader


@pytest.fixture()
def trainer_cfg() -> dict:
    return {
        "windowing": {"seq_len": 48, "pred_len": 12},
        "training": {
            "max_epochs": 3, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 4,
            "early_stopping": {"patience": 2, "min_delta": 1e-4, "mode": "min"},
            "oom_recovery": {"enabled": False, "max_retries": 1, "batch_size_reduction_factor": 0.5},
        },
        "hardware": {"mixed_precision": False, "precision_dtype": "float32", "gradient_accumulation_steps": 1},
        "logging": {"checkpoint_dir": "/tmp/tsf_test_ckpts"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# MetricCalculator
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricCalculator:

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert MetricCalculator.mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert MetricCalculator.mae(y_true, y_pred) == pytest.approx(2.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert MetricCalculator.rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_mape_zero_denominator_returns_nan(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        assert np.isnan(MetricCalculator.mape(y_true, y_pred))

    def test_mape_nonzero(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        # (10/100 + 20/200) / 2 * 100 = (0.1 + 0.1) / 2 * 100 = 10.0
        assert MetricCalculator.mape(y_true, y_pred) == pytest.approx(10.0)

    def test_smape_symmetry(self):
        y_true = np.array([100.0])
        y_pred = np.array([150.0])
        val1 = MetricCalculator.smape(y_true, y_pred)
        val2 = MetricCalculator.smape(y_pred, y_true)
        assert val1 == pytest.approx(val2)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert MetricCalculator.r2_score(y, y) == pytest.approx(1.0)

    def test_r2_constant_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert MetricCalculator.r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_compute_all_returns_all_keys(self):
        y = np.ones(10)
        result = MetricCalculator.compute_all(y, y * 1.01)
        assert set(result.keys()) == {"MAE", "RMSE", "MAPE", "sMAPE", "R2"}


# ─────────────────────────────────────────────────────────────────────────────
# HardwareMonitor
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareMonitor:

    def test_flush_vram_does_not_raise(self):
        HardwareMonitor.flush_vram()

    def test_get_vram_usage_returns_float(self):
        val = HardwareMonitor.get_vram_usage_gb()
        assert isinstance(val, float)
        assert val >= 0.0

    def test_get_gpu_name_returns_string(self):
        name = HardwareMonitor.get_gpu_name()
        assert isinstance(name, str)
        assert len(name) > 0


# ─────────────────────────────────────────────────────────────────────────────
# TimingContext
# ─────────────────────────────────────────────────────────────────────────────

class TestTimingContext:

    def test_measures_elapsed(self):
        import time
        timer = TimingContext()
        with timer:
            time.sleep(0.05)
        assert timer.elapsed >= 0.04

    def test_elapsed_positive(self):
        timer = TimingContext()
        with timer:
            _ = sum(range(10_000))
        assert timer.elapsed > 0


# ─────────────────────────────────────────────────────────────────────────────
# EarlyStopper
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyStopper:

    def test_does_not_stop_before_patience(self):
        stopper = EarlyStopper(patience=3, min_delta=0.0)
        assert not stopper.step(1.0, 1)
        assert not stopper.step(1.0, 2)  # no improvement but counter=1
        assert not stopper.step(1.0, 3)  # counter=2

    def test_stops_after_patience(self):
        stopper = EarlyStopper(patience=2, min_delta=0.0)
        stopper.step(1.0, 1)  # best=1.0
        stopper.step(1.0, 2)  # counter=1
        assert stopper.step(1.0, 3)  # counter=2 → stop

    def test_resets_counter_on_improvement(self):
        stopper = EarlyStopper(patience=3, min_delta=0.0)
        stopper.step(1.0, 1)
        stopper.step(0.9, 2)  # improvement → counter reset
        stopper.step(0.9, 3)  # no improvement counter=1
        assert not stopper.step(0.9, 4)  # counter=2 < 3

    def test_best_epoch_tracked(self):
        stopper = EarlyStopper(patience=5)
        stopper.step(1.0, 1)
        stopper.step(0.8, 3)
        stopper.step(0.9, 5)
        assert stopper.best_epoch == 3
        assert stopper.best_value == pytest.approx(0.8)

    def test_max_mode(self):
        stopper = EarlyStopper(patience=2, mode="max")
        stopper.step(0.5, 1)  # best=0.5
        stopper.step(0.4, 2)  # no improvement, counter=1
        assert stopper.step(0.4, 3)  # counter=2 → stop


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointManager:

    def test_save_and_load(self, tmp_path, tiny_model):
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        optimizer = torch.optim.Adam(tiny_model.parameters())
        metrics = {"MAE": 0.1, "RMSE": 0.2}

        manager.save(tiny_model, optimizer, epoch=5, metrics=metrics, tag="best")

        # Load into a fresh model
        fresh_model = type(tiny_model)()
        ckpt = manager.load(fresh_model, tag="best")

        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["MAE"] == pytest.approx(0.1)

    def test_checkpoint_file_exists(self, tmp_path, tiny_model):
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        optimizer = torch.optim.Adam(tiny_model.parameters())
        manager.save(tiny_model, optimizer, epoch=1, metrics={}, tag="epoch_1")
        assert (tmp_path / "epoch_1.pt").exists()


# ─────────────────────────────────────────────────────────────────────────────
# TrainingResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingResult:

    def test_to_dict_has_required_keys(self):
        result = TrainingResult(
            model_name="PatchTST",
            dataset_name="PeMS08",
            metrics={"MAE": 0.5, "RMSE": 0.7, "MAPE": 5.0, "sMAPE": 4.8, "R2": 0.85},
            train_time_sec=120.0,
            inference_time_sec=0.5,
            vram_peak_gb=6.2,
        )
        d = result.to_dict()
        assert d["Model"] == "PatchTST"
        assert d["Dataset"] == "PeMS08"
        assert "MAE" in d
        assert "VRAM_Peak_GB" in d


# ─────────────────────────────────────────────────────────────────────────────
# BaseTrainer — integration (CPU, no AMP)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseTrainerIntegration:

    def test_train_epoch_returns_float(self, tiny_model, tiny_loaders, trainer_cfg):
        train_loader, val_loader, _ = tiny_loaders
        trainer = BaseTrainer(tiny_model, trainer_cfg, device="cpu")
        loss = trainer.train_epoch(train_loader)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_evaluate_returns_metrics(self, tiny_model, tiny_loaders, trainer_cfg):
        train_loader, val_loader, _ = tiny_loaders
        trainer = BaseTrainer(tiny_model, trainer_cfg, device="cpu")
        metrics = trainer.evaluate(val_loader)
        assert "val_loss" in metrics
        assert "MAE" in metrics

    def test_predict_returns_arrays(self, tiny_model, tiny_loaders, trainer_cfg):
        _, _, test_loader = tiny_loaders
        trainer = BaseTrainer(tiny_model, trainer_cfg, device="cpu")
        y_true, y_pred, latency = trainer.predict(test_loader)
        assert y_true.shape == y_pred.shape
        assert latency >= 0.0

    def test_fit_runs_without_error(self, tiny_model, tiny_loaders, trainer_cfg):
        train_loader, val_loader, _ = tiny_loaders
        trainer = BaseTrainer(tiny_model, trainer_cfg, device="cpu")
        epochs, stopped = trainer.fit(train_loader, val_loader)
        assert epochs > 0
        assert isinstance(stopped, bool)


# ─────────────────────────────────────────────────────────────────────────────
# ExperimentLogger
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentLogger:

    def test_log_and_export_csv(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path), filename="test_results")
        result = TrainingResult(
            model_name="DLinear",
            dataset_name="METR-LA",
            metrics={"MAE": 0.3, "RMSE": 0.5, "MAPE": 4.0, "sMAPE": 3.9, "R2": 0.9},
        )
        logger.log(result)
        csv_path = tmp_path / "test_results.csv"
        assert csv_path.exists()

        import pandas as pd
        df = pd.read_csv(str(csv_path))
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "DLinear"

    def test_multiple_results(self, tmp_path):
        logger = ExperimentLogger(log_dir=str(tmp_path))
        for i in range(5):
            r = TrainingResult(
                model_name=f"Model{i}",
                dataset_name="PeMS08",
                metrics={"MAE": float(i), "RMSE": float(i), "MAPE": float(i), "sMAPE": float(i), "R2": 0.0},
            )
            logger.log(r)
        df = logger.get_dataframe()
        assert len(df) == 5
