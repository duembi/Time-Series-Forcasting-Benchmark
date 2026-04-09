"""
Classical ML forecasters — tree-based models wrapped as BaseForecaster.

Each model flattens the sliding-window input ``(seq_len, F)`` into a 1-D
feature vector and trains a standard regressor.  GPU acceleration is
enabled where supported (XGBoost ``gpu_hist``, LightGBM ``gpu``,
CatBoost ``GPU``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseForecaster, ModelMeta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: collect data from DataLoader into flat numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def _collect_flat(
    loader: DataLoader,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate a DataLoader and return flattened X and y arrays.

    X shape: ``(N, seq_len * F)`` — each window flattened into a row.
    y shape: ``(N, pred_len * F_target)`` — targets flattened similarly.
    """
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for x, y in loader:
        # x: (B, seq_len, F) → (B, seq_len*F)
        all_x.append(x.numpy().reshape(x.shape[0], -1))
        # y: (B, pred_len, F_target) → (B, pred_len*F_target)
        all_y.append(y.numpy().reshape(y.shape[0], -1))

    return np.concatenate(all_x), np.concatenate(all_y)


class _TreeBaseForecaster(BaseForecaster):
    """Shared logic for all tree-based forecasters.

    Subclasses only need to implement ``_build_regressor()`` which returns
    an sklearn-compatible estimator.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        wnd = config.get("windowing", {})
        self._seq_len: int = wnd.get("seq_len", 96)
        self._pred_len: int = wnd.get("pred_len", 24)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

        self._regressor: Any = None   # Set in _build_regressor
        self._n_targets: int = 0

        self.meta = ModelMeta(
            name=name,
            category="classical_ml",
            requires_gpu=False,
            supports_mixed_precision=False,
        )

    def _build_regressor(self) -> Any:
        """Return an sklearn-compatible regressor.  Override in subclass."""
        raise NotImplementedError

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        X_train, y_train = _collect_flat(train_loader, self._pred_len)
        self._n_targets = y_train.shape[1]

        self._regressor = self._build_regressor()
        logger.info(
            "[%s] Fitting on X=%s  y=%s ...",
            self.meta.name, X_train.shape, y_train.shape,
        )
        self._regressor.fit(X_train, y_train)
        logger.info("[%s] Fitting complete.", self.meta.name)
        return {"epochs": 0}

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        X_test, _ = _collect_flat(test_loader, self._pred_len)
        y_flat = self._regressor.predict(X_test)

        # y_flat: (N, pred_len * F_target) → (N, pred_len, F_target)
        n_features = self._n_targets // self._pred_len
        return y_flat.reshape(-1, self._pred_len, max(n_features, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Concrete implementations
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostForecaster(_TreeBaseForecaster):
    """XGBoost with GPU histogram acceleration."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="XGBoost", **kwargs)

    def _build_regressor(self) -> Any:
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=42,
            n_jobs=-1,
            tree_method="gpu_hist" if torch.cuda.is_available() else "hist",
            verbosity=0,
        )
        return MultiOutputRegressor(base, n_jobs=1)


class LightGBMForecaster(_TreeBaseForecaster):
    """LightGBM with optional GPU support."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="LightGBM", **kwargs)

    def _build_regressor(self) -> Any:
        from lightgbm import LGBMRegressor
        from sklearn.multioutput import MultiOutputRegressor

        device = "gpu" if torch.cuda.is_available() else "cpu"
        base = LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=42,
            n_jobs=-1,
            device=device,
            verbose=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)


class CatBoostForecaster(_TreeBaseForecaster):
    """CatBoost with GPU task type."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="CatBoost", **kwargs)

    def _build_regressor(self) -> Any:
        from catboost import CatBoostRegressor
        from sklearn.multioutput import MultiOutputRegressor

        task_type = "GPU" if torch.cuda.is_available() else "CPU"
        base = CatBoostRegressor(
            iterations=self._n_estimators,
            depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_seed=42,
            verbose=False,
            task_type=task_type,
        )
        return MultiOutputRegressor(base, n_jobs=1)


class ExtraTreesForecaster(_TreeBaseForecaster):
    """Extra-Trees (CPU only, extremely fast)."""

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, name="ExtraTrees", **kwargs)

    def _build_regressor(self) -> Any:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = ExtraTreesRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=42,
            n_jobs=-1,
        )
        return MultiOutputRegressor(base, n_jobs=1)
