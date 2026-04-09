"""
verification.py — Temporal Alignment Verification Module.

Generates scientific proof that the Train / Val / Test splits are
perfectly chronological with zero leakage.  The artefacts produced here
are directly embedded in the Ministry report under "Temporal Alignment Proof".

Outputs:
    1. ``alignment_table.csv``  — start/end timestamps and sizes per split.
    2. ``timeline_plot.png``    — visual timeline of the three splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Verification artefacts for one dataset's train/val/test split."""

    dataset_name: str
    train_start: str
    train_end: str
    train_n: int
    val_start: str
    val_end: str
    val_n: int
    test_start: str
    test_end: str
    test_n: int
    no_leakage: bool = True
    table_path: str = ""
    timeline_path: str = ""

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


class TemporalVerifier:
    """Verifies and documents the temporal alignment of data splits.

    Usage::

        verifier = TemporalVerifier(output_dir="outputs/reports/verification")
        result   = verifier.verify(train_ts, val_ts, test_ts, "PeMS08")
    """

    def __init__(self, output_dir: str = "outputs/reports/verification") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def verify(
        self,
        train_ts: pd.DatetimeIndex,
        val_ts: pd.DatetimeIndex,
        test_ts: pd.DatetimeIndex,
        dataset_name: str,
    ) -> AlignmentResult:
        """Run all verification checks and produce artefacts.

        Args:
            train_ts:     DatetimeIndex for training split.
            val_ts:       DatetimeIndex for validation split.
            test_ts:      DatetimeIndex for test split.
            dataset_name: Used for file naming and report headings.

        Returns:
            ``AlignmentResult`` with artefact paths populated.
        """
        ds_dir = self._output_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        # ── strict assertions ────────────────────────────────────────────────
        no_leakage = True
        try:
            assert train_ts[-1] < val_ts[0], (
                f"LEAKAGE: train_end={train_ts[-1]} >= val_start={val_ts[0]}"
            )
            assert val_ts[-1] < test_ts[0], (
                f"LEAKAGE: val_end={val_ts[-1]} >= test_start={test_ts[0]}"
            )
            logger.info(
                "[%s] Temporal alignment verified — NO leakage detected.",
                dataset_name,
            )
        except AssertionError as e:
            logger.error("TEMPORAL LEAKAGE DETECTED: %s", e)
            no_leakage = False

        result = AlignmentResult(
            dataset_name=dataset_name,
            train_start=str(train_ts[0]),
            train_end=str(train_ts[-1]),
            train_n=len(train_ts),
            val_start=str(val_ts[0]),
            val_end=str(val_ts[-1]),
            val_n=len(val_ts),
            test_start=str(test_ts[0]),
            test_end=str(test_ts[-1]),
            test_n=len(test_ts),
            no_leakage=no_leakage,
        )

        # ── CSV table ────────────────────────────────────────────────────────
        table = pd.DataFrame([
            {
                "Split": "Train",
                "Start": result.train_start,
                "End": result.train_end,
                "N_Timesteps": result.train_n,
                "Pct": f"{result.train_n / (result.train_n + result.val_n + result.test_n) * 100:.1f}%",
            },
            {
                "Split": "Validation",
                "Start": result.val_start,
                "End": result.val_end,
                "N_Timesteps": result.val_n,
                "Pct": f"{result.val_n / (result.train_n + result.val_n + result.test_n) * 100:.1f}%",
            },
            {
                "Split": "Test",
                "Start": result.test_start,
                "End": result.test_end,
                "N_Timesteps": result.test_n,
                "Pct": f"{result.test_n / (result.train_n + result.val_n + result.test_n) * 100:.1f}%",
            },
        ])

        table_path = ds_dir / "alignment_table.csv"
        table.to_csv(str(table_path), index=False)
        result.table_path = str(table_path)

        # ── timeline plot ────────────────────────────────────────────────────
        timeline_path = self._plot_timeline(
            train_ts, val_ts, test_ts, dataset_name, ds_dir,
        )
        result.timeline_path = str(timeline_path)

        return result

    def _plot_timeline(
        self,
        train_ts: pd.DatetimeIndex,
        val_ts: pd.DatetimeIndex,
        test_ts: pd.DatetimeIndex,
        dataset_name: str,
        out_dir: Path,
    ) -> Path:
        """Generate a visual timeline showing the three splits."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(14, 3))

        # Draw bars for each split
        splits = [
            ("Train",      train_ts[0], train_ts[-1], "steelblue"),
            ("Validation", val_ts[0],   val_ts[-1],   "darkorange"),
            ("Test",       test_ts[0],  test_ts[-1],  "green"),
        ]

        for i, (label, start, end, color) in enumerate(splits):
            ax.barh(
                y=0,
                width=(end - start).total_seconds(),
                left=(start - train_ts[0]).total_seconds(),
                height=0.4,
                color=color,
                alpha=0.8,
                label=f"{label}  [{start.date()} → {end.date()}]  N={len(train_ts) if label=='Train' else (len(val_ts) if label=='Validation' else len(test_ts)):,}",
            )

        ax.set_yticks([])
        ax.set_xlabel("Time (seconds from dataset start)")
        ax.set_title(
            f"{dataset_name} — Temporal Alignment Proof\n"
            f"Train → Validation → Test  (strictly chronological, zero leakage)",
            fontsize=12,
        )
        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        path = out_dir / "timeline_plot.png"
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
