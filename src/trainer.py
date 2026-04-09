"""
trainer.py — Enterprise Training Engine with Strict VRAM / OOM Management.

Implements the full model training lifecycle for the benchmark framework:
    1.  Mixed-precision training (bfloat16 / float16) via ``torch.amp``.
    2.  Gradient accumulation to simulate large effective batch sizes.
    3.  Aggressive VRAM flushing after every epoch and every model.
    4.  Automatic OOM recovery: catch → flush → halve batch → retry (3×).
    5.  Early stopping with patience and best-checkpoint restore.
    6.  Comprehensive metric computation (MAE, RMSE, MAPE, sMAPE, R²).
    7.  Hardware telemetry: peak VRAM, training time, inference latency.

Hardware target: NVIDIA RTX Ada 4500 (24 GB VRAM)
"""

from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1.  METRIC CALCULATOR                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class MetricCalculator:
    """Static utility for time-series forecasting metrics.

    All methods accept NumPy arrays and handle edge cases (zero
    denominators, empty arrays) gracefully.
    """

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error (zero-safe)."""
        mask = y_true != 0
        if mask.sum() == 0:
            return float("nan")
        return float(
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        )

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error (zero-safe)."""
        denom = np.abs(y_true) + np.abs(y_pred)
        mask = denom != 0
        if mask.sum() == 0:
            return float("nan")
        return float(
            np.mean(
                2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
            ) * 100
        )

    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of Determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return float("nan")
        return float(1.0 - ss_res / ss_tot)

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute every registered metric at once.

        Returns:
            Dictionary with keys ``MAE``, ``RMSE``, ``MAPE``, ``sMAPE``, ``R2``.
        """
        return {
            "MAE": round(cls.mae(y_true, y_pred), 6),
            "RMSE": round(cls.rmse(y_true, y_pred), 6),
            "MAPE": round(cls.mape(y_true, y_pred), 6),
            "sMAPE": round(cls.smape(y_true, y_pred), 6),
            "R2": round(cls.r2_score(y_true, y_pred), 6),
        }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2.  HARDWARE MONITOR                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class HardwareMonitor:
    """GPU telemetry and VRAM management utilities.

    Every public method is safe to call even when CUDA is unavailable — it
    will log a warning and return sensible defaults.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_gpu_name() -> str:
        """Return the name of the current CUDA device, or ``'cpu'``."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    @staticmethod
    def get_vram_usage_gb() -> float:
        """Current VRAM allocation in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / (1024 ** 3)
        return 0.0

    @staticmethod
    def get_vram_peak_gb() -> float:
        """Peak VRAM allocation since last reset, in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        return 0.0

    @staticmethod
    def reset_peak_stats() -> None:
        """Reset peak VRAM tracking counter."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

    @staticmethod
    def flush_vram() -> None:
        """Aggressively release unused GPU memory.

        Called after every epoch and after every model completes, as
        mandated by the architecture spec.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.debug(
            "VRAM flushed — current=%.2f GB",
            HardwareMonitor.get_vram_usage_gb(),
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3.  TIMING CONTEXT MANAGER                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TimingContext:
    """Lightweight wall-clock timer usable as a context manager.

    Usage::

        timer = TimingContext()
        with timer:
            do_work()
        print(timer.elapsed)   # seconds as float
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4.  EARLY STOPPER                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class EarlyStopper:
    """Monitors a validation metric and signals when training should stop.

    Also tracks the best metric value and the epoch at which it occurred,
    enabling checkpoint-based best-model restore.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ) -> None:
        """
        Args:
            patience:  Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
            mode:      ``"min"`` (loss) or ``"max"`` (accuracy/R²).
        """
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._counter: int = 0
        self._best_epoch: int = 0

        if mode == "min":
            self._best_value: float = float("inf")
            self._is_better: Callable[[float, float], bool] = (
                lambda new, best: new < best - min_delta
            )
        else:
            self._best_value = float("-inf")
            self._is_better = (
                lambda new, best: new > best + min_delta
            )

    def step(self, metric: float, epoch: int) -> bool:
        """Record a new metric observation.

        Args:
            metric: The monitored value (e.g. validation loss).
            epoch:  Current epoch number.

        Returns:
            ``True`` if training should stop (patience exhausted).
        """
        if self._is_better(metric, self._best_value):
            self._best_value = metric
            self._best_epoch = epoch
            self._counter = 0
            return False

        self._counter += 1
        if self._counter >= self._patience:
            logger.info(
                "EarlyStopper — triggered at epoch %d  "
                "(best=%.6f @ epoch %d, patience=%d)",
                epoch, self._best_value, self._best_epoch, self._patience,
            )
            return True
        return False

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5.  CHECKPOINT MANAGER                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CheckpointManager:
    """Save / load model + optimizer state for best-epoch restore."""

    def __init__(self, checkpoint_dir: str = "outputs/checkpoints") -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        tag: str = "best",
    ) -> Path:
        """Persist a checkpoint to disk.

        Args:
            model:     The PyTorch model.
            optimizer: The optimizer state.
            epoch:     Epoch number.
            metrics:   Current metric values.
            tag:       Filename tag (e.g., ``"best"``, ``"epoch_10"``).

        Returns:
            Path to the saved ``.pt`` file.
        """
        path = self._dir / f"{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            str(path),
        )
        logger.debug("Checkpoint saved → %s", path)
        return path

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        tag: str = "best",
    ) -> Dict[str, Any]:
        """Restore a checkpoint from disk.

        Args:
            model:     Model whose ``state_dict`` will be loaded.
            optimizer: Optional optimizer to restore.
            tag:       Filename tag used when saving.

        Returns:
            The full checkpoint dictionary.
        """
        path = self._dir / f"{tag}.pt"
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.debug("Checkpoint loaded ← %s  (epoch %d)", path, ckpt["epoch"])
        return ckpt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6.  TRAINING RESULT                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class TrainingResult:
    """Immutable record of a single model's benchmark run."""

    model_name: str
    dataset_name: str
    metrics: Dict[str, float]             # MAE, RMSE, MAPE, sMAPE, R2
    train_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    inference_latency_ms: float = 0.0     # per-step latency
    vram_peak_gb: float = 0.0
    batch_size_final: int = 0
    epochs_trained: int = 0
    early_stopped: bool = False
    status: str = "success"               # "success", "oom_skip", "error"
    error_message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten into a single-row dictionary (Excel / CSV friendly)."""
        row = {
            "Model": self.model_name,
            "Dataset": self.dataset_name,
            **self.metrics,
            "Train_Time_Sec": round(self.train_time_sec, 2),
            "Inference_Time_Sec": round(self.inference_time_sec, 4),
            "Inference_Latency_ms": round(self.inference_latency_ms, 4),
            "VRAM_Peak_GB": round(self.vram_peak_gb, 3),
            "Batch_Size_Final": self.batch_size_final,
            "Epochs_Trained": self.epochs_trained,
            "Early_Stopped": self.early_stopped,
            "Status": self.status,
        }
        return row


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  7.  BASE TRAINER                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class BaseTrainer:
    """Enterprise training engine with mixed-precision, gradient
    accumulation, VRAM management, OOM recovery, and early stopping.

    Usage::

        trainer = BaseTrainer(model, config, device="cuda")
        result  = trainer.run(
            train_loader, val_loader, test_loader,
            model_name="PatchTST", dataset_name="PeMS08",
        )
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            model:   A ``torch.nn.Module`` to train.
            config:  Parsed ``experiment.yaml`` dictionary.
            device:  ``"cuda"`` or ``"cpu"``.
            loss_fn: Loss function.  Defaults to ``nn.MSELoss()``.
        """
        self._device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self._model = model.to(self._device)
        self._config = config

        # ── loss ─────────────────────────────────────────────────────────────
        self._loss_fn = loss_fn or nn.MSELoss()

        # ── training hyper-params ────────────────────────────────────────────
        train_cfg = config.get("training", {})
        self._max_epochs: int = train_cfg.get("max_epochs", 100)
        self._lr: float = train_cfg.get("learning_rate", 1e-3)
        self._weight_decay: float = train_cfg.get("weight_decay", 1e-4)

        # ── mixed precision ──────────────────────────────────────────────────
        hw_cfg = config.get("hardware", {})
        self._use_amp: bool = hw_cfg.get("mixed_precision", True)
        dtype_str: str = hw_cfg.get("precision_dtype", "bfloat16")
        self._amp_dtype: torch.dtype = (
            torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
        )

        # GradScaler is only needed for float16 (bfloat16 doesn't need it)
        self._use_grad_scaler = (
            self._use_amp and self._amp_dtype == torch.float16
        )
        self._grad_scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler("cuda") if self._use_grad_scaler else None
        )

        # ── gradient accumulation ────────────────────────────────────────────
        self._accum_steps: int = hw_cfg.get("gradient_accumulation_steps", 1)

        # ── early stopping ───────────────────────────────────────────────────
        es_cfg = train_cfg.get("early_stopping", {})
        self._early_stopper = EarlyStopper(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 1e-4),
            mode=es_cfg.get("mode", "min"),
        )

        # ── checkpoint ───────────────────────────────────────────────────────
        log_cfg = config.get("logging", {})
        self._ckpt_manager = CheckpointManager(
            checkpoint_dir=log_cfg.get("checkpoint_dir", "outputs/checkpoints"),
        )

        # ── OOM recovery ─────────────────────────────────────────────────────
        oom_cfg = train_cfg.get("oom_recovery", {})
        self._oom_enabled: bool = oom_cfg.get("enabled", True)
        self._oom_max_retries: int = oom_cfg.get("max_retries", 3)
        self._oom_reduction: float = oom_cfg.get(
            "batch_size_reduction_factor", 0.5,
        )

        # ── optimizer (created fresh so LR / weight_decay are from config) ──
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        logger.info(
            "BaseTrainer initialised — device=%s  amp=%s  dtype=%s  "
            "accum=%d  patience=%d  oom_retries=%d",
            self._device, self._use_amp, dtype_str,
            self._accum_steps,
            es_cfg.get("patience", 10),
            self._oom_max_retries,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch with mixed precision and grad accumulation.

        Args:
            train_loader: Yields ``(x, y)`` batches.

        Returns:
            Average training loss for the epoch.
        """
        self._model.train()
        total_loss = 0.0
        n_batches = 0

        self._optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)

            # ── forward (mixed precision) ────────────────────────────────────
            with torch.amp.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp and self._device.type == "cuda",
            ):
                y_pred = self._model(x)
                loss = self._loss_fn(y_pred, y) / self._accum_steps

            # ── backward ─────────────────────────────────────────────────────
            if self._use_grad_scaler and self._grad_scaler is not None:
                self._grad_scaler.scale(loss).backward()
            else:
                loss.backward()

            # ── optimizer step (every accum_steps batches) ───────────────────
            if (batch_idx + 1) % self._accum_steps == 0:
                if self._use_grad_scaler and self._grad_scaler is not None:
                    self._grad_scaler.step(self._optimizer)
                    self._grad_scaler.update()
                else:
                    self._optimizer.step()
                self._optimizer.zero_grad()

            total_loss += loss.item() * self._accum_steps
            n_batches += 1

        # Flush leftover gradients if total batches not divisible by accum
        if n_batches % self._accum_steps != 0:
            if self._use_grad_scaler and self._grad_scaler is not None:
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                self._optimizer.step()
            self._optimizer.zero_grad()

        # ── VRAM flush after every epoch (architecture spec mandate) ─────────
        HardwareMonitor.flush_vram()

        return total_loss / max(n_batches, 1)

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a validation set.

        Args:
            val_loader: Yields ``(x, y)`` batches.

        Returns:
            Dictionary with ``val_loss`` and all forecasting metrics.
        """
        self._model.eval()
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0

        for x, y in val_loader:
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp and self._device.type == "cuda",
            ):
                y_pred = self._model(x)
                loss = self._loss_fn(y_pred, y)

            total_loss += loss.item()
            n_batches += 1

            all_true.append(y.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

        y_true = np.concatenate(all_true).flatten()
        y_pred = np.concatenate(all_pred).flatten()

        metrics = MetricCalculator.compute_all(y_true, y_pred)
        metrics["val_loss"] = total_loss / max(n_batches, 1)
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        test_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Generate predictions on the test set.

        Args:
            test_loader: Yields ``(x, y)`` batches.

        Returns:
            y_true:       Flattened ground-truth array.
            y_pred:       Flattened prediction array.
            latency_ms:   Average per-sample inference latency in ms.
        """
        self._model.eval()
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        total_time = 0.0
        total_samples = 0

        for x, y in test_loader:
            x = x.to(self._device, non_blocking=True)
            y = y.to(self._device, non_blocking=True)
            batch_size = x.shape[0]

            if self._device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=self._amp_dtype,
                enabled=self._use_amp and self._device.type == "cuda",
            ):
                y_pred = self._model(x)

            if self._device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            total_samples += batch_size

            all_true.append(y.cpu().numpy())
            all_pred.append(y_pred.cpu().numpy())

        y_true = np.concatenate(all_true).flatten()
        y_pred = np.concatenate(all_pred).flatten()
        latency_ms = (total_time / max(total_samples, 1)) * 1000

        return y_true, y_pred, latency_ms

    # ─────────────────────────────────────────────────────────────────────────
    # Full training loop with early stopping
    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[int, bool]:
        """Train until early stopping or max epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.

        Returns:
            epochs_trained: Number of completed epochs.
            early_stopped:  Whether early stopping was triggered.
        """
        logger.info(
            "Training started — max_epochs=%d  lr=%.1e  amp=%s",
            self._max_epochs, self._lr, self._use_amp,
        )

        for epoch in range(1, self._max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["val_loss"]

            logger.info(
                "  Epoch %3d/%d — train_loss=%.6f  val_loss=%.6f  "
                "MAE=%.4f  RMSE=%.4f  VRAM=%.2f GB",
                epoch, self._max_epochs, train_loss, val_loss,
                val_metrics["MAE"], val_metrics["RMSE"],
                HardwareMonitor.get_vram_usage_gb(),
            )

            # ── checkpoint on best ───────────────────────────────────────────
            if val_loss <= self._early_stopper.best_value:
                self._ckpt_manager.save(
                    self._model, self._optimizer, epoch, val_metrics, tag="best",
                )

            # ── early stopping check ─────────────────────────────────────────
            if self._early_stopper.step(val_loss, epoch):
                # Restore best checkpoint
                self._ckpt_manager.load(
                    self._model, self._optimizer, tag="best",
                )
                return epoch, True

        return self._max_epochs, False

    # ─────────────────────────────────────────────────────────────────────────
    # OOM-safe wrapper
    # ─────────────────────────────────────────────────────────────────────────
    def safe_fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        initial_batch_size: int,
        rebuild_loader_fn: Callable[[int], Tuple[DataLoader, DataLoader]],
    ) -> Tuple[int, bool, int]:
        """Train with automatic OOM recovery.

        On ``torch.cuda.OutOfMemoryError``:
            1. Log the failure.
            2. Flush VRAM.
            3. Halve the batch size.
            4. Rebuild DataLoaders via *rebuild_loader_fn*.
            5. Retry (up to ``oom_max_retries``).

        If all retries fail, the model is gracefully skipped.

        Args:
            train_loader:      Initial training DataLoader.
            val_loader:        Initial validation DataLoader.
            initial_batch_size: Starting batch size.
            rebuild_loader_fn: ``fn(new_batch_size) -> (train_loader, val_loader)``

        Returns:
            epochs_trained:   Number of completed epochs (0 if skipped).
            early_stopped:    Whether early stopping fired.
            final_batch_size: The batch size that actually succeeded.
        """
        batch_size = initial_batch_size

        for attempt in range(self._oom_max_retries + 1):
            try:
                epochs, stopped = self.fit(train_loader, val_loader)
                return epochs, stopped, batch_size

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "OOM at batch_size=%d (attempt %d/%d) — flushing VRAM…",
                    batch_size, attempt + 1, self._oom_max_retries,
                )
                HardwareMonitor.flush_vram()

                if attempt >= self._oom_max_retries - 1:
                    logger.error(
                        "OOM recovery exhausted after %d retries — skipping model.",
                        self._oom_max_retries,
                    )
                    return 0, False, batch_size

                # Halve batch size and rebuild
                batch_size = max(1, int(batch_size * self._oom_reduction))
                logger.info("Retrying with batch_size=%d …", batch_size)
                train_loader, val_loader = rebuild_loader_fn(batch_size)

                # Re-initialize model weights for a fresh start
                self._reset_model()

        return 0, False, batch_size

    # ─────────────────────────────────────────────────────────────────────────
    # Full benchmark run
    # ─────────────────────────────────────────────────────────────────────────
    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model_name: str,
        dataset_name: str,
        rebuild_loader_fn: Optional[
            Callable[[int], Tuple[DataLoader, DataLoader]]
        ] = None,
    ) -> TrainingResult:
        """Execute the complete benchmark lifecycle for one model.

        Sequence: ``safe_fit → predict → metrics → hardware stats``

        Args:
            train_loader:      Training DataLoader.
            val_loader:        Validation DataLoader.
            test_loader:       Test DataLoader.
            model_name:        Human-readable model name for logging.
            dataset_name:      Dataset identifier for logging.
            rebuild_loader_fn: OOM batch-size rebuild callback.

        Returns:
            A fully populated ``TrainingResult``.
        """
        logger.info(
            "=" * 70 + "\n  [%s × %s] — benchmark run started\n" + "=" * 70,
            model_name, dataset_name,
        )

        HardwareMonitor.reset_peak_stats()
        batch_size = train_loader.batch_size or self._config["training"]["batch_size"]

        # ── FIT ──────────────────────────────────────────────────────────────
        train_timer = TimingContext()
        try:
            with train_timer:
                if self._oom_enabled and rebuild_loader_fn is not None:
                    epochs, stopped, batch_size = self.safe_fit(
                        train_loader, val_loader, batch_size, rebuild_loader_fn,
                    )
                else:
                    epochs, stopped = self.fit(train_loader, val_loader)
        except Exception as e:
            logger.exception("Unhandled error during training of %s", model_name)
            HardwareMonitor.flush_vram()
            return TrainingResult(
                model_name=model_name,
                dataset_name=dataset_name,
                metrics={},
                status="error",
                error_message=str(e),
            )

        if epochs == 0:
            # OOM skip
            HardwareMonitor.flush_vram()
            return TrainingResult(
                model_name=model_name,
                dataset_name=dataset_name,
                metrics={},
                batch_size_final=batch_size,
                status="oom_skip",
                error_message="All OOM retries exhausted.",
            )

        # ── PREDICT ──────────────────────────────────────────────────────────
        infer_timer = TimingContext()
        with infer_timer:
            y_true, y_pred, latency_ms = self.predict(test_loader)

        # ── METRICS ──────────────────────────────────────────────────────────
        metrics = MetricCalculator.compute_all(y_true, y_pred)
        vram_peak = HardwareMonitor.get_vram_peak_gb()

        # ── VRAM flush after model completes (architecture spec mandate) ─────
        HardwareMonitor.flush_vram()

        result = TrainingResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            train_time_sec=train_timer.elapsed,
            inference_time_sec=infer_timer.elapsed,
            inference_latency_ms=latency_ms,
            vram_peak_gb=vram_peak,
            batch_size_final=batch_size,
            epochs_trained=epochs,
            early_stopped=stopped,
            status="success",
        )

        logger.info(
            "  [%s × %s] DONE — %s  train=%.1fs  infer=%.3fs  "
            "latency=%.2fms  VRAM_peak=%.2fGB  epochs=%d%s",
            model_name, dataset_name,
            {k: f"{v:.4f}" for k, v in metrics.items()},
            train_timer.elapsed, infer_timer.elapsed,
            latency_ms, vram_peak, epochs,
            " (early-stopped)" if stopped else "",
        )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _reset_model(self) -> None:
        """Re-initialize model weights (for OOM retry with smaller batch)."""
        def _init_weights(m: nn.Module) -> None:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        self._model.apply(_init_weights)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        logger.debug("Model weights and optimizer re-initialized for OOM retry.")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  8.  EXPERIMENT LOGGER                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class ExperimentLogger:
    """Collects ``TrainingResult`` objects and exports them.

    Supports CSV / Excel out of the box.  W&B and MLflow can be plugged in
    by subclassing and overriding ``_log_backend()``.
    """

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        filename: str = "benchmark_results",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._filename = filename
        self._results: List[TrainingResult] = []

    def log(self, result: TrainingResult) -> None:
        """Record one benchmark result."""
        self._results.append(result)
        self._log_backend(result)
        # Auto-save after every log to prevent data loss on crash
        self.export_csv()

    def export_csv(self) -> Path:
        """Write all results to a CSV file."""
        import pandas as pd

        path = self._log_dir / f"{self._filename}.csv"
        rows = [r.to_dict() for r in self._results]
        pd.DataFrame(rows).to_csv(str(path), index=False)
        return path

    def export_excel(self) -> Path:
        """Write all results to an Excel file."""
        import pandas as pd

        path = self._log_dir / f"{self._filename}.xlsx"
        rows = [r.to_dict() for r in self._results]
        pd.DataFrame(rows).to_excel(str(path), index=False)
        logger.info("Results exported → %s", path)
        return path

    def get_dataframe(self) -> "pd.DataFrame":
        """Return all results as a ``pd.DataFrame``."""
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self._results])

    def _log_backend(self, result: TrainingResult) -> None:
        """Override in subclasses for W&B / MLflow integration."""
