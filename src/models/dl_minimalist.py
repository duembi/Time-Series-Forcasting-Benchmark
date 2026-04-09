"""
Minimalist / Fast Deep Learning forecasters — DLinear and TiDE.

These lightweight architectures are known as "Transformer Killers" for
their competitive accuracy at a fraction of the compute cost.  They serve
as strong baselines that complex DL models must beat.

Both models are implemented as pure ``torch.nn.Module`` subclasses with
a thin ``BaseForecaster`` wrapper that delegates training to ``BaseTrainer``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DLinear — Series Decomposition + Dual Linear                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class _MovingAvgBlock(nn.Module):
    """Moving-average kernel for trend / seasonal decomposition."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → transpose for AvgPool1d → (B, C, T)
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_t = x_padded.permute(0, 2, 1)
        out = self.avg(x_t).permute(0, 2, 1)
        return out[:, :x.shape[1], :]


class _SeriesDecomp(nn.Module):
    """Decomposes a time series into trend and seasonal components."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_avg = _MovingAvgBlock(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearModule(nn.Module):
    """DLinear: decomposition + independent linear layers for trend & season.

    Paper: "Are Transformers Effective for Time Series Forecasting?" (Zeng et al.)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decomp = _SeriesDecomp(kernel_size)

        # One linear projection per feature channel
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, C)
        seasonal, trend = self.decomp(x)

        # (B, seq_len, C) → permute → (B, C, seq_len) → linear → (B, C, pred_len)
        seasonal_out = self.linear_seasonal(seasonal.permute(0, 2, 1))
        trend_out = self.linear_trend(trend.permute(0, 2, 1))

        # (B, C, pred_len) → (B, pred_len, C)
        return (seasonal_out + trend_out).permute(0, 2, 1)


class DLinearForecaster(BaseForecaster):
    """BaseForecaster wrapper around DLinear."""

    def __init__(
        self,
        config: Dict[str, Any],
        kernel_size: int = 25,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._kernel_size = kernel_size
        self._config = config
        self._module: Optional[DLinearModule] = None

        self.meta = ModelMeta(
            name="DLinear",
            category="dl_minimalist",
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

        # Infer n_features from first batch
        sample_x, _ = next(iter(train_loader))
        n_features = sample_x.shape[2]

        self._module = DLinearModule(
            seq_len=self._seq_len,
            pred_len=self._pred_len,
            n_features=n_features,
            kernel_size=self._kernel_size,
        )

        trainer = BaseTrainer(
            model=self._module,
            config=self._config,
            device="cuda" if self.meta.requires_gpu else "cpu",
        )
        epochs, stopped = trainer.fit(train_loader, val_loader)

        return {"epochs": epochs, "early_stopped": stopped}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        from ..trainer import BaseTrainer

        trainer = BaseTrainer(
            model=self._module,
            config=self._config,
            device="cuda" if self.meta.requires_gpu else "cpu",
        )
        y_true, y_pred, _ = trainer.predict(test_loader)

        # Reshape back to (N, pred_len, F)
        n_samples = len(test_loader.dataset)
        n_features = self._module.pred_len  # approximate
        return y_pred.reshape(n_samples, self._pred_len, -1)

    def save(self, path: str) -> None:
        if self._module is not None:
            torch.save(self._module.state_dict(), path)

    def load(self, path: str) -> None:
        if self._module is not None:
            self._module.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TiDE — Time-series Dense Encoder                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class _ResidualBlock(nn.Module):
    """MLP block with residual connection, LayerNorm, and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.relu(self.fc1(x))
        out = self.dropout(self.fc2(out))
        return self.ln(out + residual)


class TiDEModule(nn.Module):
    """TiDE: Time-series Dense Encoder for long-term forecasting.

    Paper: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
    (Das et al., 2023)

    Architecture: Encoder (flatten→MLP) → Dense core → Decoder (MLP→reshape)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_features: int,
        hidden_dim: int = 256,
        encoder_dim: int = 128,
        decoder_dim: int = 128,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features

        input_dim = seq_len * n_features

        # Encoder stack
        encoder_layers: List[nn.Module] = [
            _ResidualBlock(input_dim, hidden_dim, encoder_dim, dropout)
        ]
        for _ in range(n_encoder_layers - 1):
            encoder_layers.append(
                _ResidualBlock(encoder_dim, hidden_dim, encoder_dim, dropout)
            )
        self.encoder = nn.Sequential(*encoder_layers)

        # Dense core mapping
        self.dense_core = nn.Linear(encoder_dim, pred_len * decoder_dim)

        # Decoder stack (per time-step)
        decoder_layers: List[nn.Module] = []
        for _ in range(n_decoder_layers - 1):
            decoder_layers.append(
                _ResidualBlock(decoder_dim, hidden_dim, decoder_dim, dropout)
            )
        decoder_layers.append(nn.Linear(decoder_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, C)
        batch_size = x.shape[0]

        # Flatten input
        x_flat = x.reshape(batch_size, -1)  # (B, seq_len * C)

        # Encode
        encoded = self.encoder(x_flat)  # (B, encoder_dim)

        # Dense core → (B, pred_len * decoder_dim)
        core_out = self.dense_core(encoded)
        core_out = core_out.reshape(batch_size, self.pred_len, -1)  # (B, pred_len, decoder_dim)

        # Decode each timestep
        output = self.decoder(core_out)  # (B, pred_len, C)
        return output


class TiDEForecaster(BaseForecaster):
    """BaseForecaster wrapper around TiDE."""

    def __init__(
        self,
        config: Dict[str, Any],
        hidden_dim: int = 256,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._config = config
        self._module: Optional[TiDEModule] = None

        self.meta = ModelMeta(
            name="TiDE",
            category="dl_minimalist",
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

        self._module = TiDEModule(
            seq_len=self._seq_len,
            pred_len=self._pred_len,
            n_features=n_features,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
        )

        trainer = BaseTrainer(
            model=self._module,
            config=self._config,
            device="cuda" if self.meta.requires_gpu else "cpu",
        )
        epochs, stopped = trainer.fit(train_loader, val_loader)

        return {"epochs": epochs, "early_stopped": stopped}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        from ..trainer import BaseTrainer

        trainer = BaseTrainer(
            model=self._module,
            config=self._config,
            device="cuda" if self.meta.requires_gpu else "cpu",
        )
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
