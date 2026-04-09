"""
Probabilistic forecasters — DeepAR.

DeepAR is a recurrent autoregressive model that outputs full predictive
distributions (Gaussian likelihood) rather than point forecasts.  We wrap
it via the NeuralForecast library which provides the PyTorch Lightning
training loop and GPU support out of the box.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta
from .dl_complex import _loader_to_long_df

logger = logging.getLogger(__name__)


class DeepARForecaster(BaseForecaster):
    """DeepAR: probabilistic forecasting via autoregressive RNN.

    Uses the NeuralForecast implementation with Gaussian likelihood.
    Point forecasts are extracted as the predictive mean.

    Paper: "DeepAR: Probabilistic Forecasting with Autoregressive
    Recurrent Networks" (Salinas et al., 2019)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout_prob_theta: float = 0.1,
        max_steps: int = 1000,
        early_stop_patience: int = 5,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout_prob_theta
        self._max_steps = max_steps
        self._patience = early_stop_patience
        self._config = config
        self._nf: Any = None
        self._train_df: Optional[pd.DataFrame] = None

        self.meta = ModelMeta(
            name="DeepAR",
            category="probabilistic",
            requires_gpu=True,
            supports_mixed_precision=True,
            default_batch_size=config.get("training", {}).get("batch_size", 64),
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train DeepAR via NeuralForecast."""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR as NF_DeepAR
        from pytorch_lightning.callbacks import EarlyStopping

        self._train_df = _loader_to_long_df(train_loader, self._seq_len, self._freq)

        model = NF_DeepAR(
            h=self._pred_len,
            input_size=self._seq_len,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout_prob_theta=self._dropout,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=self._patience,
                    mode="min",
                )
            ],
            enable_progress_bar=True,
            alias="DeepAR",
        )

        self._nf = NeuralForecast(models=[model], freq=self._freq)
        self._nf.fit(df=self._train_df, val_size=self._pred_len)

        logger.info("[DeepAR] Training complete.")
        return {"epochs": self._max_steps}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Return point forecasts (predictive mean)."""
        if self._nf is None:
            raise RuntimeError("DeepAR has not been fitted yet.")

        forecast_df = self._nf.predict().reset_index()

        # DeepAR outputs quantiles; use median (0.5) or mean column
        pred_cols = [c for c in forecast_df.columns
                     if c not in ("unique_id", "ds") and "DeepAR" in c]

        if not pred_cols:
            raise RuntimeError(
                f"No DeepAR prediction columns found. Got: {forecast_df.columns.tolist()}"
            )

        # Prefer the median quantile for point forecast
        median_cols = [c for c in pred_cols if "median" in c.lower() or "50" in c]
        target_col = median_cols[0] if median_cols else pred_cols[0]

        y_pred = forecast_df[target_col].values.astype(np.float32)
        n_series = self._train_df["unique_id"].nunique()
        return y_pred.reshape(n_series, self._pred_len, 1)
