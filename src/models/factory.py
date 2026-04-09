"""
ModelFactory — Dynamic model instantiation via the Factory Pattern.

Reads the ``models`` section of ``experiment.yaml`` and creates the
corresponding ``BaseForecaster`` wrappers.  New models are added by
registering them in the ``_REGISTRY`` — no other code needs to change.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Type

from .base import BaseForecaster

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Registry: maps (category, model_name) → fully-qualified class path
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, Dict[str, str]] = {
    # category -> { model_name -> "module_path.ClassName" }
    "baselines": {
        "SeasonalNaive": "src.models.baselines.SeasonalNaiveForecaster",
        "AutoARIMA":     "src.models.baselines.AutoARIMAForecaster",
    },
    "classical_ml": {
        "LightGBM":   "src.models.classical_ml.LightGBMForecaster",
        "XGBoost":     "src.models.classical_ml.XGBoostForecaster",
        "CatBoost":    "src.models.classical_ml.CatBoostForecaster",
        "ExtraTrees":  "src.models.classical_ml.ExtraTreesForecaster",
    },
    "dl_minimalist": {
        "DLinear": "src.models.dl_minimalist.DLinearForecaster",
        "TiDE":    "src.models.dl_minimalist.TiDEForecaster",
    },
    "dl_complex": {
        "TCN":          "src.models.dl_complex.TCNForecaster",
        "TimeMixer":    "src.models.dl_complex.TimeMixerForecaster",
        "NHITS":        "src.models.dl_complex.NHITSForecaster",
        "PatchTST":     "src.models.dl_complex.PatchTSTForecaster",
        "Informer":     "src.models.dl_complex.InformerForecaster",
        "TimesNet":     "src.models.dl_complex.TimesNetForecaster",
        "iTransformer": "src.models.dl_complex.ITransformerForecaster",
        "Mamba":        "src.models.dl_complex.MambaForecaster",
    },
    "probabilistic": {
        "DeepAR": "src.models.probabilistic.DeepARForecaster",
    },
    "foundation": {
        "TimeGPT":  "src.models.foundation.TimeGPTForecaster",
        "Moirai":   "src.models.foundation.MoiraiForecaster",
        "Chronos":  "src.models.foundation.ChronosForecaster",
        "TimesFM":  "src.models.foundation.TimesFMForecaster",
        "LagLlama": "src.models.foundation.LagLlamaForecaster",
    },
}


def _import_class(dotted_path: str) -> Type[BaseForecaster]:
    """Dynamically import a class from a dotted module path.

    Example::

        _import_class("src.models.baselines.SeasonalNaiveForecaster")
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ModelFactory:
    """Creates ``BaseForecaster`` instances from YAML configuration.

    Usage::

        factory = ModelFactory()
        models  = factory.create_enabled(experiment_cfg)
        # → [SeasonalNaiveForecaster(...), PatchTSTForecaster(...), ...]
    """

    @staticmethod
    def register(category: str, name: str, dotted_path: str) -> None:
        """Register a new model at runtime.

        Args:
            category:    Model category (e.g., ``"dl_complex"``).
            name:        Model name as it appears in YAML.
            dotted_path: ``"module.ClassName"`` for dynamic import.
        """
        _REGISTRY.setdefault(category, {})[name] = dotted_path
        logger.info("Registered model %s/%s → %s", category, name, dotted_path)

    @staticmethod
    def create(
        category: str,
        name: str,
        model_cfg: Dict[str, Any],
        experiment_cfg: Dict[str, Any],
    ) -> BaseForecaster:
        """Instantiate a single model.

        Args:
            category:       Model category key.
            name:           Model name key.
            model_cfg:      Model-specific overrides.
            experiment_cfg: Global experiment configuration.

        Returns:
            A ready-to-use ``BaseForecaster`` instance.

        Raises:
            KeyError:  If category/name not found in registry.
            ImportError: If the module cannot be imported.
        """
        if category not in _REGISTRY:
            raise KeyError(
                f"Unknown category '{category}'. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        if name not in _REGISTRY[category]:
            raise KeyError(
                f"Unknown model '{name}' in category '{category}'. "
                f"Available: {list(_REGISTRY[category].keys())}"
            )

        dotted_path = _REGISTRY[category][name]
        cls = _import_class(dotted_path)
        instance = cls(config=experiment_cfg, **model_cfg)

        logger.info("Created %s/%s → %r", category, name, instance)
        return instance

    @staticmethod
    def create_enabled(
        experiment_cfg: Dict[str, Any],
    ) -> List[BaseForecaster]:
        """Create all models that are enabled in the YAML config.

        Reads ``experiment_cfg["models"]`` and instantiates every model
        whose value is ``True``.

        Args:
            experiment_cfg: Full parsed ``experiment.yaml``.

        Returns:
            Ordered list of ``BaseForecaster`` instances.
        """
        models_cfg: Dict[str, Dict[str, bool]] = experiment_cfg.get("models", {})
        instances: List[BaseForecaster] = []

        for category, model_map in models_cfg.items():
            for model_name, enabled in model_map.items():
                if not enabled:
                    logger.debug("Skipping disabled model %s/%s", category, model_name)
                    continue
                try:
                    instance = ModelFactory.create(
                        category=category,
                        name=model_name,
                        model_cfg={},
                        experiment_cfg=experiment_cfg,
                    )
                    instances.append(instance)
                except (KeyError, ImportError, Exception) as e:
                    logger.warning(
                        "Failed to create %s/%s: %s — skipping.",
                        category, model_name, e,
                    )

        logger.info("ModelFactory — %d models instantiated.", len(instances))
        return instances

    @staticmethod
    def list_available() -> Dict[str, List[str]]:
        """Return all registered models grouped by category."""
        return {cat: list(models.keys()) for cat, models in _REGISTRY.items()}
