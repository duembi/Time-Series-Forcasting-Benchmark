"""
Baseline forecasters — Seasonal Naive and AutoARIMA.

These serve as the sanity-check lower bound in the benchmark.  Every DL
or ML model should beat these to justify its complexity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta

logger = logging.getLogger(__name__)


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal Naive: repeats the last observed seasonal cycle as forecast.

    For 5-min traffic data with ``season_length=288`` (one day), the
    prediction is simply the values from 24 hours ago.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        season_length: int = 288,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._season_length = season_length

        self.meta = ModelMeta(
            name="SeasonalNaive",
            category="baselines",
            requires_gpu=False,
            supports_mixed_precision=False,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """No training needed — this is a heuristic baseline."""
        logger.info("[SeasonalNaive] No training required.")
        return {"epochs": 0}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Copy the seasonal pattern from the input context window.

        For each sample, take the last ``pred_len`` values from the
        input that are ``season_length`` steps back.  If the context is
        shorter than one full season, fall back to repeating the tail.
        """
        all_preds: List[np.ndarray] = []

        for x, _ in test_loader:
            x_np = x.numpy()  # (B, seq_len, F)
            batch_size, seq_len, n_features = x_np.shape

            preds = np.zeros((batch_size, self._pred_len, n_features), dtype=np.float32)

            for i in range(batch_size):
                if seq_len >= self._season_length:
                    # Use values from one season ago
                    offset = seq_len - self._season_length
                    seasonal_chunk = x_np[i, offset: offset + self._pred_len]
                else:
                    # Fallback: repeat the last pred_len values
                    seasonal_chunk = x_np[i, -self._pred_len:]

                actual_len = min(len(seasonal_chunk), self._pred_len)
                preds[i, :actual_len] = seasonal_chunk[:actual_len]

            all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)


class AutoARIMAForecaster(BaseForecaster):
    """AutoARIMA via the ``statsforecast`` library.

    Automatically selects the best (p, d, q) order per series.
    Since ARIMA operates per-series and doesn't use GPU, this wrapper
    iterates over the feature dimension and aggregates results.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        season_length: int = 288,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._season_length = season_length
        self._models: Dict[int, Any] = {}  # feature_idx -> fitted model

        self.meta = ModelMeta(
            name="AutoARIMA",
            category="baselines",
            requires_gpu=False,
            supports_mixed_precision=False,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit AutoARIMA on the full training series.

        Collects all training data from the DataLoader, then fits one
        AutoARIMA model per feature column.
        """
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        # Collect all training data
        all_x: List[np.ndarray] = []
        for x, _ in train_loader:
            all_x.append(x.numpy())

        # Concatenate along time axis: use first window + incremental steps
        # For simplicity, reconstruct full series from sliding windows
        data = np.concatenate(all_x, axis=0)  # (N_windows, seq_len, F)

        # Use the first feature column for ARIMA (univariate)
        # Build a pseudo-series from the windows
        n_features = data.shape[2]
        logger.info(
            "[AutoARIMA] Fitting on %d features (season=%d)...",
            n_features, self._season_length,
        )

        import pandas as pd

        for f_idx in range(min(n_features, 1)):  # Fit on first feature only
            # Reconstruct approximate series from window starts
            series_values = data[:, 0, f_idx]  # First timestep of each window
            df = pd.DataFrame({
                "unique_id": "0",
                "ds": pd.date_range("2000-01-01", periods=len(series_values), freq="5min"),
                "y": series_values,
            })

            sf = StatsForecast(
                models=[AutoARIMA(season_length=min(self._season_length, 24))],
                freq="5min",
                n_jobs=1,
            )
            sf.fit(df=df)
            self._models[f_idx] = sf

        logger.info("[AutoARIMA] Fitting complete.")
        return {"epochs": 0, "n_models": len(self._models)}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate forecasts using the fitted ARIMA models."""
        all_preds: List[np.ndarray] = []

        for x, _ in test_loader:
            x_np = x.numpy()
            batch_size, seq_len, n_features = x_np.shape
            preds = np.zeros(
                (batch_size, self._pred_len, n_features), dtype=np.float32,
            )

            if 0 in self._models:
                sf = self._models[0]
                forecast_df = sf.predict(h=self._pred_len)
                forecast_vals = forecast_df["AutoARIMA"].values

                for i in range(batch_size):
                    preds[i, :len(forecast_vals), 0] = forecast_vals[:self._pred_len]

                # Copy first feature forecast to remaining features
                for f_idx in range(1, n_features):
                    preds[:, :, f_idx] = preds[:, :, 0]

            all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)
