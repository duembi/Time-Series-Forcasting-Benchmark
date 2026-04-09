"""
Foundation / Zero-Shot forecasters.

These models require no training on the target dataset (zero-shot) or use
an external API.  They represent the frontier of time-series foundation
models as of 2024-2025.

Models:
    - Chronos   (Amazon)     — zero-shot, HuggingFace weights
    - Moirai    (Salesforce) — zero-shot, HuggingFace weights
    - TimeGPT   (Nixtla)     — API-based
    - TimesFM   (Google)     — zero-shot, HuggingFace weights
    - Lag-Llama (Meta)       — zero-shot, HuggingFace weights
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta
from .dl_complex import _loader_to_long_df

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect context tensors from DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def _collect_context(
    loader: DataLoader,
    context_len: int,
) -> tuple[List[torch.Tensor], List[np.ndarray]]:
    """Extract the last `context_len` steps from each window as context.

    Returns:
        contexts: List of 1-D float32 tensors per sample.
        y_trues:  List of ground-truth numpy arrays per sample.
    """
    contexts: List[torch.Tensor] = []
    y_trues: List[np.ndarray] = []

    for x, y in loader:
        # x: (B, seq_len, F) — use first feature channel
        for i in range(x.shape[0]):
            ctx = x[i, -context_len:, 0].float()
            contexts.append(ctx)
            y_trues.append(y[i, :, 0].numpy())

    return contexts, y_trues


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Chronos (Amazon)                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ChronosForecaster(BaseForecaster):
    """Chronos: zero-shot forecasting via language model pre-training.

    Uses ``amazon/chronos-t5-small`` by default (fits in 24 GB VRAM).
    Runs fully in ``bfloat16`` as recommended by the Chronos paper.

    Paper: "Chronos: Learning the Language of Time Series" (Ansari et al., 2024)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_id: str = "amazon/chronos-t5-small",
        context_len: int = 336,
        num_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._pred_len: int = wnd.get("pred_len", 24)
        self._context_len = min(context_len, wnd.get("seq_len", 96))
        self._model_id = model_id
        self._num_samples = num_samples
        self._pipeline: Any = None

        self.meta = ModelMeta(
            name="Chronos",
            category="foundation",
            requires_gpu=True,
            supports_mixed_precision=True,
            is_zero_shot=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load the pre-trained Chronos pipeline (no fine-tuning)."""
        from chronos import ChronosPipeline

        logger.info("[Chronos] Loading %s ...", self._model_id)
        self._pipeline = ChronosPipeline.from_pretrained(
            self._model_id,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        logger.info("[Chronos] Pipeline loaded (zero-shot).")
        return {"epochs": 0, "zero_shot": True}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate zero-shot forecasts using Chronos."""
        if self._pipeline is None:
            raise RuntimeError("Chronos pipeline not loaded. Call fit() first.")

        contexts, y_trues = _collect_context(test_loader, self._context_len)

        all_preds: List[np.ndarray] = []

        # Batch inference to avoid OOM
        batch_size = 16
        for i in range(0, len(contexts), batch_size):
            batch_ctx = contexts[i: i + batch_size]

            forecast = self._pipeline.predict(
                context=batch_ctx,
                prediction_length=self._pred_len,
                num_samples=self._num_samples,
            )
            # forecast: (batch, num_samples, pred_len) → median
            median = np.median(forecast.numpy(), axis=1)  # (batch, pred_len)
            all_preds.append(median)

        y_pred = np.concatenate(all_preds, axis=0)  # (N, pred_len)
        return y_pred[:, :, np.newaxis]              # (N, pred_len, 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Moirai (Salesforce)                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MoiraiForecaster(BaseForecaster):
    """Moirai: Unified Training of Universal Time Series Forecasting Transformers.

    Uses ``Salesforce/moirai-1.0-R-small`` by default.

    Paper: "Unified Training of Universal Time Series Forecasting Transformers"
    (Woo et al., 2024)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_size: str = "small",   # "small", "base", "large"
        context_len: int = 200,
        num_samples: int = 100,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._context_len = min(context_len, wnd.get("seq_len", 96))
        self._model_size = model_size
        self._num_samples = num_samples
        self._model: Any = None
        self._transform: Any = None

        self.meta = ModelMeta(
            name="Moirai",
            category="foundation",
            requires_gpu=True,
            supports_mixed_precision=True,
            is_zero_shot=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load the pre-trained Moirai model."""
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        model_id = f"Salesforce/moirai-1.0-R-{self._model_size}"
        logger.info("[Moirai] Loading %s ...", model_id)

        self._model = MoiraiForecast.from_pretrained(
            model_id,
            prediction_length=self._pred_len,
            context_length=self._context_len,
            patch_size="auto",
            num_samples=self._num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        logger.info("[Moirai] Model loaded (zero-shot).")
        return {"epochs": 0, "zero_shot": True}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate zero-shot forecasts using Moirai."""
        if self._model is None:
            raise RuntimeError("Moirai model not loaded. Call fit() first.")

        contexts, _ = _collect_context(test_loader, self._context_len)
        all_preds: List[np.ndarray] = []

        self._model.eval()
        with torch.no_grad():
            for ctx in contexts:
                # Moirai expects (1, T, 1)
                ctx_tensor = ctx.unsqueeze(0).unsqueeze(-1).cuda()
                samples = self._model(ctx_tensor)
                # samples: (1, num_samples, pred_len)
                median = samples.median(dim=1).values  # (1, pred_len)
                all_preds.append(median.cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0)  # (N, pred_len)
        return y_pred[:, :, np.newaxis]              # (N, pred_len, 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TimeGPT (Nixtla)                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TimeGPTForecaster(BaseForecaster):
    """TimeGPT: API-based foundation model by Nixtla.

    Requires a valid ``NIXTLA_API_KEY`` environment variable or explicit
    ``api_key`` argument.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        api_key: str = "",
        **kwargs: Any,
    ) -> None:
        import os
        wnd = config.get("windowing", {})
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._api_key = api_key or os.environ.get("NIXTLA_API_KEY", "")
        self._client: Any = None
        self._train_df: Optional[pd.DataFrame] = None

        self.meta = ModelMeta(
            name="TimeGPT",
            category="foundation",
            requires_gpu=False,  # API-based
            supports_mixed_precision=False,
            is_zero_shot=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Authenticate with Nixtla API and cache training series."""
        from nixtla import NixtlaClient

        if not self._api_key:
            raise ValueError(
                "TimeGPT requires a NIXTLA_API_KEY. "
                "Set it as an environment variable or pass api_key=..."
            )

        self._client = NixtlaClient(api_key=self._api_key)
        wnd_seq_len = config.get("windowing", {}).get("seq_len", 96) if config else 96
        self._train_df = _loader_to_long_df(train_loader, wnd_seq_len, self._freq)

        logger.info("[TimeGPT] API client authenticated (zero-shot).")
        return {"epochs": 0, "zero_shot": True}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate forecasts via the TimeGPT API."""
        if self._client is None or self._train_df is None:
            raise RuntimeError("TimeGPT not initialized. Call fit() first.")

        forecast_df = self._client.forecast(
            df=self._train_df,
            h=self._pred_len,
            freq=self._freq,
            time_col="ds",
            target_col="y",
            id_col="unique_id",
        )

        y_pred = forecast_df["TimeGPT"].values.astype(np.float32)
        n_series = self._train_df["unique_id"].nunique()
        return y_pred.reshape(n_series, self._pred_len, 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TimesFM (Google DeepMind)                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TimesFMForecaster(BaseForecaster):
    """TimesFM: Time Series Foundation Model by Google DeepMind.

    Uses ``google/timesfm-1.0-200m`` by default.

    Paper: "A decoder-only foundation model for time-series forecasting"
    (Das et al., 2024)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_id: str = "google/timesfm-1.0-200m-pytorch",
        context_len: int = 128,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._context_len = min(context_len, wnd.get("seq_len", 96))
        self._model_id = model_id
        self._tfm: Any = None

        self.meta = ModelMeta(
            name="TimesFM",
            category="foundation",
            requires_gpu=True,
            supports_mixed_precision=True,
            is_zero_shot=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Load TimesFM from HuggingFace."""
        import timesfm

        logger.info("[TimesFM] Loading %s ...", self._model_id)
        self._tfm = timesfm.TimesFm(
            context_len=self._context_len,
            horizon_len=self._pred_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="gpu",
        )
        self._tfm.load_from_checkpoint(repo_id=self._model_id)
        logger.info("[TimesFM] Model loaded (zero-shot).")
        return {"epochs": 0, "zero_shot": True}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate zero-shot forecasts using TimesFM."""
        if self._tfm is None:
            raise RuntimeError("TimesFM model not loaded. Call fit() first.")

        contexts, _ = _collect_context(test_loader, self._context_len)

        # TimesFM expects List[np.ndarray]
        context_arrays = [c.numpy() for c in contexts]
        forecasts, _ = self._tfm.forecast(
            inputs=context_arrays,
            freq=[0] * len(context_arrays),  # 0 = high-frequency
        )

        # forecasts: (N, pred_len)
        y_pred = np.array(forecasts, dtype=np.float32)
        return y_pred[:, : self._pred_len, np.newaxis]  # (N, pred_len, 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Lag-Llama (Meta / ServiceNow Research)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LagLlamaForecaster(BaseForecaster):
    """Lag-Llama: open-source foundation model for probabilistic forecasting.

    Uses the pre-trained checkpoint from HuggingFace.

    Paper: "Lag-Llama: Towards Foundation Models for Probabilistic
    Time Series Forecasting" (Rasul et al., 2024)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_id: str = "time-series-foundation-models/Lag-Llama",
        num_samples: int = 100,
        context_len: int = 32,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._pred_len: int = wnd.get("pred_len", 24)
        self._freq: str = wnd.get("frequency", "5min")
        self._context_len = min(context_len, wnd.get("seq_len", 96))
        self._model_id = model_id
        self._num_samples = num_samples
        self._predictor: Any = None

        self.meta = ModelMeta(
            name="LagLlama",
            category="foundation",
            requires_gpu=True,
            supports_mixed_precision=True,
            is_zero_shot=True,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Download and prepare Lag-Llama predictor."""
        from huggingface_hub import hf_hub_download
        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt_path = hf_hub_download(
            repo_id=self._model_id,
            filename="lag-llama.ckpt",
        )

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=self._pred_len,
            context_length=self._context_len,
            device=torch.device("cuda"),
            batch_size=16,
            num_parallel_samples=self._num_samples,
        )

        self._predictor = estimator.create_predictor(
            training_data=None,
            num_samples=self._num_samples,
        )
        logger.info("[LagLlama] Predictor created (zero-shot).")
        return {"epochs": 0, "zero_shot": True}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Generate zero-shot probabilistic forecasts (median)."""
        if self._predictor is None:
            raise RuntimeError("LagLlama predictor not ready. Call fit() first.")

        from gluonts.dataset.common import ListDataset

        contexts, y_trues = _collect_context(test_loader, self._context_len)

        gluon_dataset = ListDataset(
            [
                {
                    "start": pd.Timestamp("2000-01-01"),
                    "target": ctx.numpy().tolist(),
                }
                for ctx in contexts
            ],
            freq=self._freq,
        )

        forecasts = list(self._predictor.predict(gluon_dataset))
        all_preds: List[np.ndarray] = []

        for fc in forecasts:
            median = np.median(fc.samples, axis=0)[: self._pred_len]
            all_preds.append(median)

        y_pred = np.stack(all_preds, axis=0).astype(np.float32)  # (N, pred_len)
        return y_pred[:, :, np.newaxis]                           # (N, pred_len, 1)
