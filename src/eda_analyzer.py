"""
eda_analyzer.py — Exploratory Data Analysis & Data Health Module.

Performs comprehensive, automated EDA before any model training begins.
All outputs (plots, statistics, tables) are saved to disk and later
injected into the Ministry report by ``report_generator.py``.

Checks performed:
    1. Temporal balance & missing data (NaN %, duplicates, gaps)
    2. Statistical distribution & outlier density (Z-score & IQR)
    3. Stationarity: Augmented Dickey-Fuller test (log p-value)
    4. Autocorrelation: ACF / PACF plots
    5. STL decomposition: trend, seasonality, residuals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EDAReport:
    """Structured container for all EDA outputs."""

    dataset_name: str

    # Temporal balance
    total_timesteps: int = 0
    n_series: int = 0
    start_date: str = ""
    end_date: str = ""
    frequency: str = ""
    missing_pct: float = 0.0
    duplicate_ts_pct: float = 0.0
    n_temporal_gaps: int = 0

    # Distribution & outliers
    target_mean: float = 0.0
    target_std: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0
    outlier_pct_zscore: float = 0.0
    outlier_pct_iqr: float = 0.0

    # Stationarity
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    adf_is_stationary: bool = False

    # Saved artefact paths (relative to report_dir)
    plot_paths: Dict[str, str] = field(default_factory=dict)
    summary_table_path: str = ""

    def is_statistically_sound(self) -> bool:
        """Return True if data quality meets minimum thresholds."""
        return (
            self.missing_pct < 20.0       # < 20 % NaN
            and self.outlier_pct_zscore < 10.0  # < 10 % outliers
            and self.n_temporal_gaps < 100
        )

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("plot_paths", None)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# EDA Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class EDAAnalyzer:
    """Performs and persists all EDA checks for a given dataset.

    Usage::

        analyzer = EDAAnalyzer(output_dir="outputs/reports/eda")
        report   = analyzer.run(data, timestamps, series_ids, dataset_name)
    """

    def __init__(self, output_dir: str = "outputs/reports/eda") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ─── public API ──────────────────────────────────────────────────────────

    def run(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        series_ids: List[str],
        dataset_name: str,
        freq: str = "5min",
    ) -> EDAReport:
        """Execute the full EDA pipeline and save all artefacts.

        Args:
            data:         ``(T, N)`` raw array.
            timestamps:   DatetimeIndex of length ``T``.
            series_ids:   Series names of length ``N``.
            dataset_name: Used for file naming and report headings.
            freq:         Pandas frequency string.

        Returns:
            Populated ``EDAReport`` instance.
        """
        report = EDAReport(dataset_name=dataset_name)
        ds_dir = self._output_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        logger.info("EDAAnalyzer — starting analysis for '%s' …", dataset_name)

        # Use the first series as the representative series for univariate tests
        primary_series = data[:, 0]

        # 1. Temporal balance
        self._temporal_balance(data, timestamps, freq, report)

        # 2. Distribution & outliers
        self._distribution_outliers(data, report)

        # 3. Stationarity (ADF)
        self._stationarity(primary_series, report)

        # 4. Plots
        self._plot_distribution(primary_series, dataset_name, ds_dir, report)
        self._plot_acf_pacf(primary_series, dataset_name, ds_dir, report)
        self._plot_stl(primary_series, timestamps, freq, dataset_name, ds_dir, report)
        self._plot_temporal_gaps(timestamps, freq, dataset_name, ds_dir, report)

        # 5. Summary table
        summary_path = ds_dir / "summary.csv"
        pd.DataFrame([report.to_dict()]).to_csv(str(summary_path), index=False)
        report.summary_table_path = str(summary_path)

        sound = report.is_statistically_sound()
        logger.info(
            "EDA complete — missing=%.1f%%  outliers(z)=%.1f%%  "
            "ADF_pval=%.4f  stationary=%s  sound=%s",
            report.missing_pct, report.outlier_pct_zscore,
            report.adf_pvalue, report.adf_is_stationary, sound,
        )
        return report

    # ─── private analysis steps ──────────────────────────────────────────────

    def _temporal_balance(
        self,
        data: np.ndarray,
        timestamps: pd.DatetimeIndex,
        freq: str,
        report: EDAReport,
    ) -> None:
        """Compute NaN %, duplicates, and temporal gaps."""
        report.total_timesteps = data.shape[0]
        report.n_series = data.shape[1]
        report.start_date = str(timestamps[0])
        report.end_date = str(timestamps[-1])
        report.frequency = freq

        # Missing values
        report.missing_pct = float(np.isnan(data).sum() / data.size * 100)

        # Duplicate timestamps
        n_dupes = int(timestamps.duplicated().sum())
        report.duplicate_ts_pct = float(n_dupes / len(timestamps) * 100)

        # Temporal gaps: steps where the expected interval is violated
        if len(timestamps) > 1:
            diffs = pd.Series(timestamps).diff().dropna()
            expected = diffs.mode()[0]
            gaps = (diffs > expected * 1.5).sum()
            report.n_temporal_gaps = int(gaps)
        else:
            report.n_temporal_gaps = 0

    def _distribution_outliers(
        self,
        data: np.ndarray,
        report: EDAReport,
    ) -> None:
        """Compute summary stats and outlier percentages."""
        flat = data[~np.isnan(data)].flatten()

        report.target_mean = float(np.mean(flat))
        report.target_std = float(np.std(flat))
        report.target_min = float(np.min(flat))
        report.target_max = float(np.max(flat))

        # Z-score outliers (|z| > 3)
        z_scores = np.abs((flat - report.target_mean) / max(report.target_std, 1e-8))
        report.outlier_pct_zscore = float((z_scores > 3).sum() / len(flat) * 100)

        # IQR outliers
        q1, q3 = np.percentile(flat, [25, 75])
        iqr = q3 - q1
        iqr_mask = (flat < q1 - 1.5 * iqr) | (flat > q3 + 1.5 * iqr)
        report.outlier_pct_iqr = float(iqr_mask.sum() / len(flat) * 100)

    def _stationarity(self, series: np.ndarray, report: EDAReport) -> None:
        """Augmented Dickey-Fuller test for stationarity."""
        from statsmodels.tsa.stattools import adfuller

        clean = series[~np.isnan(series)]
        if len(clean) < 20:
            logger.warning("ADF test skipped: insufficient data (%d points).", len(clean))
            return

        result = adfuller(clean, autolag="AIC")
        report.adf_statistic = float(result[0])
        report.adf_pvalue = float(result[1])
        report.adf_is_stationary = report.adf_pvalue < 0.05

    # ─── plotting ─────────────────────────────────────────────────────────────

    def _plot_distribution(
        self,
        series: np.ndarray,
        dataset_name: str,
        out_dir: Path,
        report: EDAReport,
    ) -> None:
        """Histogram + KDE of the target variable."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        clean = series[~np.isnan(series)]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(clean, kde=True, ax=ax, color="steelblue", bins=50)
        ax.set_title(f"{dataset_name} — Target Distribution", fontsize=13)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.axvline(report.target_mean, color="red", linestyle="--", label=f"Mean={report.target_mean:.2f}")
        ax.legend()
        plt.tight_layout()

        path = out_dir / "distribution.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        report.plot_paths["distribution"] = str(path)

    def _plot_acf_pacf(
        self,
        series: np.ndarray,
        dataset_name: str,
        out_dir: Path,
        report: EDAReport,
    ) -> None:
        """ACF and PACF plots."""
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

        clean = series[~np.isnan(series)][:2000]  # limit for speed

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        plot_acf(clean, lags=48, ax=axes[0], title=f"{dataset_name} — ACF")
        plot_pacf(clean, lags=48, ax=axes[1], title=f"{dataset_name} — PACF")
        plt.tight_layout()

        path = out_dir / "acf_pacf.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        report.plot_paths["acf_pacf"] = str(path)

    def _plot_stl(
        self,
        series: np.ndarray,
        timestamps: pd.DatetimeIndex,
        freq: str,
        dataset_name: str,
        out_dir: Path,
        report: EDAReport,
    ) -> None:
        """STL decomposition: trend, seasonal, residual."""
        import matplotlib.pyplot as plt
        from statsmodels.tsa.seasonal import STL

        clean = series[~np.isnan(series)][:5760]  # 20 days at 5min = 5760 points

        # Determine seasonal period from frequency
        period_map = {
            "5min": 288,   # 1 day
            "5T": 288,
            "1h": 24,
            "1H": 24,
            "15min": 96,
            "10min": 144,
        }
        period = period_map.get(freq, 24)

        try:
            stl = STL(clean, period=period, robust=True)
            result = stl.fit()

            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            axes[0].plot(clean, color="steelblue", linewidth=0.8)
            axes[0].set_title(f"{dataset_name} — STL Decomposition", fontsize=13)
            axes[0].set_ylabel("Observed")

            axes[1].plot(result.trend, color="darkorange", linewidth=1.0)
            axes[1].set_ylabel("Trend")

            axes[2].plot(result.seasonal, color="green", linewidth=0.8)
            axes[2].set_ylabel("Seasonal")

            axes[3].plot(result.resid, color="red", linewidth=0.6, alpha=0.7)
            axes[3].set_ylabel("Residual")

            plt.tight_layout()
            path = out_dir / "stl_decomposition.png"
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            report.plot_paths["stl"] = str(path)

        except Exception as e:
            logger.warning("STL decomposition failed: %s", e)

    def _plot_temporal_gaps(
        self,
        timestamps: pd.DatetimeIndex,
        freq: str,
        dataset_name: str,
        out_dir: Path,
        report: EDAReport,
    ) -> None:
        """Timeline plot highlighting temporal gaps."""
        import matplotlib.pyplot as plt

        diffs = pd.Series(timestamps).diff().dt.total_seconds().dropna()
        expected_sec = diffs.median()

        fig, ax = plt.subplots(figsize=(14, 2))
        gap_mask = diffs > expected_sec * 1.5

        ax.plot(range(len(diffs)), diffs / expected_sec, linewidth=0.5, color="steelblue")
        if gap_mask.any():
            gap_idx = np.where(gap_mask)[0]
            ax.scatter(gap_idx, diffs[gap_mask] / expected_sec, color="red", s=10, label="Gap", zorder=5)

        ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="Expected interval")
        ax.set_title(f"{dataset_name} — Temporal Gap Timeline", fontsize=12)
        ax.set_ylabel("Interval / Expected")
        ax.legend(fontsize=8)
        plt.tight_layout()

        path = out_dir / "temporal_gaps.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        report.plot_paths["temporal_gaps"] = str(path)
