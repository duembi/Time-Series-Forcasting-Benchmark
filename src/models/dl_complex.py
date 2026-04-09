"""
Complex Deep Learning forecasters — NeuralForecast-backed wrappers.

These models are loaded from the ``neuralforecast`` library and wrapped
in our ``BaseForecaster`` interface.  NeuralForecast handles the internal
PyTorch Lightning training loop, so the wrapper converts between our
``DataLoader``-based API and NeuralForecast's long-format DataFrame API.

Models: TCN, TimeMixer, NHITS, PatchTST, Informer, TimesNet, iTransformer.
Mamba is wrapped as a custom PyTorch module (not in NeuralForecast).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Helper: DataLoader → NeuralForecast long-format DataFrame                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _loader_to_long_df(
    loader: DataLoader,
    seq_len: int,
    freq: str = "5min",
) -> pd.DataFrame:
    """Convert a sliding-window DataLoader into NeuralForecast long format.

    NeuralForecast expects columns: ``[unique_id, ds, y]``.
    We reconstruct an approximate time series from the window heads.
    """
    all_x: List[np.ndarray] = []
    for x, _ in loader:
        all_x.append(x.numpy())

    windows = np.concatenate(all_x, axis=0)  # (N, seq_len, F)
    # Use first feature as target, reconstruct series from window[0] values
    n_features = windows.shape[2]
    series_vals = windows[:, 0, 0]  # First timestep, first feature

    # Build a single pseudo-series
    timestamps = pd.date_range("2000-01-01", periods=len(series_vals), freq=freq)
    df = pd.DataFrame({
        "unique_id": "0",
        "ds": timestamps,
        "y": series_vals.astype(np.float64),
    })
    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Base class for NeuralForecast-backed models                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class _NeuralForecastForecaster(BaseForecaster):
    """Shared wrapper for models available in the NeuralForecast library.

    Subclasses implement ``_build_nf_model()`` to return the configured
    NeuralForecast model object.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        max_steps: int = 1000,
        early_stop_patience: int = 5,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._max_steps = max_steps
        self._patience = early_stop_patience
        self._config = config
        self._nf: Any = None  # NeuralForecast instance
        self._train_df: Optional[pd.DataFrame] = None

        self.meta = ModelMeta(
            name=name,
            category="dl_complex",
            requires_gpu=True,
            supports_mixed_precision=True,
            default_batch_size=config.get("training", {}).get("batch_size", 64),
        )

    def _build_nf_model(self, n_series: int = 1) -> Any:
        """Return a configured NeuralForecast model instance."""
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from neuralforecast import NeuralForecast

        self._train_df = _loader_to_long_df(train_loader, self._seq_len, self._freq)
        n_series = self._train_df["unique_id"].nunique()

        model_obj = self._build_nf_model(n_series)

        self._nf = NeuralForecast(models=[model_obj], freq=self._freq)
        self._nf.fit(df=self._train_df, val_size=self._pred_len)

        logger.info("[%s] NeuralForecast training complete.", self.meta.name)
        return {"epochs": self._max_steps}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        if self._nf is None:
            raise RuntimeError(f"{self.meta.name} has not been fitted yet.")

        forecast_df = self._nf.predict().reset_index()

        # Extract predicted column (NeuralForecast names it after the model alias)
        pred_col = self.meta.name
        if pred_col not in forecast_df.columns:
            # Fallback: use first numeric column that isn't ds/unique_id
            numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
            pred_col = numeric_cols[0] if len(numeric_cols) > 0 else forecast_df.columns[-1]

        y_pred = forecast_df[pred_col].values.astype(np.float32)

        # Reshape: NF returns (n_series * pred_len,) → (n_series, pred_len, 1)
        n_series = self._train_df["unique_id"].nunique()
        return y_pred.reshape(n_series, self._pred_len, 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Concrete NeuralForecast model wrappers                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TCNForecaster(_NeuralForecastForecaster):
    """Temporal Convolutional Network."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="TCN", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import TCN
        from pytorch_lightning.callbacks import EarlyStopping

        return TCN(
            h=self._pred_len,
            input_size=self._seq_len,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16],
            encoder_hidden_size=64,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="TCN",
        )


class TimeMixerForecaster(_NeuralForecastForecaster):
    """TimeMixer: cross-variate mixing architecture."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="TimeMixer", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import TimeMixer
        from pytorch_lightning.callbacks import EarlyStopping

        return TimeMixer(
            h=self._pred_len,
            input_size=self._seq_len,
            n_series=max(n_series, 1),
            d_model=64,
            dropout=0.1,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="TimeMixer",
        )


class NHITSForecaster(_NeuralForecastForecaster):
    """N-HiTS: Neural Hierarchical Interpolation for Time Series."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="NHITS", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import NHITS
        from pytorch_lightning.callbacks import EarlyStopping

        return NHITS(
            h=self._pred_len,
            input_size=self._seq_len,
            n_blocks=[1, 1, 1],
            mlp_units=[[512, 512], [512, 512], [512, 512]],
            n_pool_kernel_size=[2, 2, 1],
            n_freq_downsample=[4, 2, 1],
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="NHITS",
        )


class PatchTSTForecaster(_NeuralForecastForecaster):
    """PatchTST: Patched Time Series Transformer."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="PatchTST", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import PatchTST
        from pytorch_lightning.callbacks import EarlyStopping

        patch_len = min(24, self._seq_len // 4)

        return PatchTST(
            h=self._pred_len,
            input_size=self._seq_len,
            patch_len=patch_len,
            stride=patch_len,
            revin=True,
            hidden_size=64,
            n_heads=4,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="PatchTST",
        )


class InformerForecaster(_NeuralForecastForecaster):
    """Informer: efficient Transformer with ProbSparse self-attention."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="Informer", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import Informer
        from pytorch_lightning.callbacks import EarlyStopping

        return Informer(
            h=self._pred_len,
            input_size=self._seq_len,
            hidden_size=64,
            n_head=4,
            conv_hidden_size=32,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="Informer",
        )


class TimesNetForecaster(_NeuralForecastForecaster):
    """TimesNet: Temporal 2D-Variation modeling."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="TimesNet", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import TimesNet
        from pytorch_lightning.callbacks import EarlyStopping

        return TimesNet(
            h=self._pred_len,
            input_size=self._seq_len,
            hidden_size=64,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="TimesNet",
        )


class ITransformerForecaster(_NeuralForecastForecaster):
    """iTransformer: inverted Transformer with cross-variate attention."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="iTransformer", **kwargs)

    def _build_nf_model(self, n_series: int = 1) -> Any:
        from neuralforecast.models import iTransformer
        from pytorch_lightning.callbacks import EarlyStopping

        return iTransformer(
            h=self._pred_len,
            input_size=self._seq_len,
            n_series=max(n_series, 1),
            hidden_size=64,
            n_heads=4,
            scaler_type="standard",
            max_steps=self._max_steps,
            accelerator="gpu",
            devices=1,
            callbacks=[EarlyStopping(monitor="train_loss", patience=self._patience, mode="min")],
            enable_progress_bar=True,
            alias="iTransformer",
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Mamba — Custom PyTorch Implementation (not in NeuralForecast)            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class _MambaBlock(nn.Module):
    """Simplified Mamba-style selective state-space block.

    This is a lightweight SSM approximation suitable for time-series
    benchmarking.  For the full Mamba implementation, see the ``mamba-ssm``
    package.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        residual = x
        x_and_gate = self.in_proj(x)
        x_proj, gate = x_and_gate.chunk(2, dim=-1)

        # Conv1d expects (B, D, T)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :x.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = self.act(x_conv)

        # Gate
        out = x_conv * self.act(gate)
        out = self.out_proj(out)

        return self.norm(out + residual)


class MambaModule(nn.Module):
    """Mamba-based sequence model for time-series forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model * seq_len, pred_len * n_features)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, F)
        batch_size = x.shape[0]
        h = self.input_proj(x)  # (B, seq_len, d_model)

        for block in self.blocks:
            h = self.dropout(block(h))

        # Flatten and project to output
        h_flat = h.reshape(batch_size, -1)
        out = self.output_proj(h_flat)
        return out.reshape(batch_size, self.pred_len, self.n_features)


class MambaForecaster(BaseForecaster):
    """BaseForecaster wrapper around the custom Mamba module."""

    def __init__(
        self,
        config: Dict[str, Any],
        d_model: int = 64,
        n_layers: int = 4,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._d_model = d_model
        self._n_layers = n_layers
        self._config = config
        self._module: Optional[MambaModule] = None

        self.meta = ModelMeta(
            name="Mamba",
            category="dl_complex",
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
        from ..trainer import BaseTrainer

        sample_x, _ = next(iter(train_loader))
        n_features = sample_x.shape[2]

        self._module = MambaModule(
            seq_len=self._seq_len,
            pred_len=self._pred_len,
            n_features=n_features,
            d_model=self._d_model,
            n_layers=self._n_layers,
        )

        trainer = BaseTrainer(model=self._module, config=self._config)
        epochs, stopped = trainer.fit(train_loader, val_loader)
        return {"epochs": epochs, "early_stopped": stopped}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        from ..trainer import BaseTrainer

        trainer = BaseTrainer(model=self._module, config=self._config)
        _, y_pred, _ = trainer.predict(test_loader)
        n_samples = len(test_loader.dataset)
        return y_pred.reshape(n_samples, self._pred_len, -1)

    def save(self, path: str) -> None:
        if self._module is not None:
            torch.save(self._module.state_dict(), path)

    def load(self, path: str) -> None:
        if self._module is not None:
            self._module.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
