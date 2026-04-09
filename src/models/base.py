"""
BaseForecaster — Abstract base class for all forecasting model wrappers.

Every model in the benchmark (baselines, ML, DL, foundation) must inherit
from this class and implement the `fit` / `predict` interface so that the
Trainer and Orchestrator can treat them uniformly.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModelMeta:
    """Immutable descriptor attached to every forecaster instance."""

    name: str
    category: str                         # e.g. "dl_complex", "classical_ml"
    requires_gpu: bool = False
    supports_mixed_precision: bool = False
    is_zero_shot: bool = False
    default_batch_size: int = 64
    extra: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────
class BaseForecaster(abc.ABC):
    """Abstract base class that all benchmark model wrappers must implement.

    The contract is intentionally thin so that it can wrap everything from a
    Seasonal Naive baseline to a Chronos foundation model.

    Subclasses MUST:
        1.  Set ``self.meta`` in ``__init__`` (a ``ModelMeta`` instance).
        2.  Implement ``fit()`` and ``predict()``.
    """

    meta: ModelMeta

    # ── lifecycle ────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train the model on the given data.

        Args:
            train_loader: PyTorch DataLoader yielding ``(x, y)`` batches.
            val_loader:   Optional validation DataLoader.
            config:       Runtime overrides (learning rate, epochs, …).

        Returns:
            Dictionary of training artifacts (loss history, best epoch, …).
        """

    @abc.abstractmethod
    def predict(
        self,
        test_loader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Generate forecasts for the given test data.

        Args:
            test_loader: PyTorch DataLoader yielding ``(x, y)`` batches.

        Returns:
            ``np.ndarray`` of shape ``(N, pred_len, features)``.
        """

    # ── optional hooks ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Persist model state to *path*.  Override in subclasses."""

    def load(self, path: str) -> None:
        """Restore model state from *path*.  Override in subclasses."""

    # ── convenience ──────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def category(self) -> str:
        return self.meta.category

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.meta.name!r} "
            f"category={self.meta.category!r}>"
        )
