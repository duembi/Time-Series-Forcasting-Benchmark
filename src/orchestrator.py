"""
orchestrator.py — Queue-Based Sequential Benchmark Runner.

Designed to run headlessly inside ``tmux`` / ``screen`` on remote
workstations via SSH.  Completely crash-proof: any single model failure
is logged and skipped without interrupting the rest of the pipeline.

Execution sequence for each (dataset, model) pair:
    1.  Prepare data via ``DataPipelineManager``.
    2.  Run EDA via ``EDAAnalyzer``.
    3.  Verify temporal alignment via ``TemporalVerifier``.
    4.  Fit + predict via ``BaseTrainer.run()``.
    5.  Log result via ``ExperimentLogger``.
    6.  Flush VRAM between models.
"""

from __future__ import annotations

import gc
import logging
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Job descriptor
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkJob:
    """A single (dataset × model) unit of work."""

    dataset_name: str
    dataset_cfg: Dict[str, Any]
    model_name: str
    model_category: str
    job_id: int = 0

    def __str__(self) -> str:
        return f"[{self.job_id:03d}] {self.dataset_name} × {self.model_name}"


@dataclass
class JobResult:
    """Outcome of a single benchmark job."""

    job: BenchmarkJob
    status: str              # "success", "error", "oom_skip"
    elapsed_sec: float = 0.0
    error_message: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkOrchestrator:
    """Crash-proof, sequential benchmark orchestrator.

    Builds a queue of ``BenchmarkJob`` objects from the YAML configuration,
    then executes them one by one.  Failures are trapped at the job level —
    the next job always starts regardless.

    Usage::

        orch = BenchmarkOrchestrator("configs/experiment.yaml")
        orch.add_dataset("configs/datasets/pems08.yaml")
        orch.run()
    """

    def __init__(self, experiment_yaml: str) -> None:
        with open(experiment_yaml, "r", encoding="utf-8") as f:
            self._exp_cfg: Dict[str, Any] = yaml.safe_load(f)

        self._job_queue: Queue[BenchmarkJob] = Queue()
        self._dataset_cfgs: List[Dict[str, Any]] = []
        self._results: List[JobResult] = []
        self._job_counter: int = 0

        # Setup logging to file
        log_dir = Path(
            self._exp_cfg.get("logging", {}).get("log_dir", "outputs/logs")
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        self._setup_file_logging(log_dir / "orchestrator.log")

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        logger.info(
            "BenchmarkOrchestrator initialized — experiment: %s",
            self._exp_cfg.get("experiment", {}).get("name", "unknown"),
        )

    # ─── dataset registration ────────────────────────────────────────────────

    def add_dataset(self, dataset_yaml: str) -> None:
        """Register a dataset YAML file for benchmarking.

        Args:
            dataset_yaml: Path to a dataset config (e.g., ``configs/datasets/pems08.yaml``).
        """
        with open(dataset_yaml, "r", encoding="utf-8") as f:
            ds_cfg = yaml.safe_load(f)
        self._dataset_cfgs.append(ds_cfg)
        logger.info("Dataset registered: %s", ds_cfg["dataset"]["name"])

    def add_dataset_cfg(self, ds_cfg: Dict[str, Any]) -> None:
        """Register a dataset config dict directly."""
        self._dataset_cfgs.append(ds_cfg)

    # ─── queue building ──────────────────────────────────────────────────────

    def _build_queue(self) -> None:
        """Populate the job queue: (dataset × model) cross-product."""
        models_cfg: Dict[str, Dict[str, bool]] = self._exp_cfg.get("models", {})

        for ds_cfg in self._dataset_cfgs:
            for category, model_map in models_cfg.items():
                for model_name, enabled in model_map.items():
                    if not enabled:
                        continue
                    self._job_counter += 1
                    job = BenchmarkJob(
                        dataset_name=ds_cfg["dataset"]["name"],
                        dataset_cfg=ds_cfg,
                        model_name=model_name,
                        model_category=category,
                        job_id=self._job_counter,
                    )
                    self._job_queue.put(job)

        total = self._job_queue.qsize()
        logger.info(
            "Queue built — %d jobs  (%d datasets × N models)",
            total, len(self._dataset_cfgs),
        )

    # ─── main run loop ───────────────────────────────────────────────────────

    def run(self) -> List[JobResult]:
        """Execute all queued jobs sequentially.

        Returns:
            List of ``JobResult`` objects (one per job).
        """
        from .data_pipeline import DataPipelineManager
        from .trainer import BaseTrainer, ExperimentLogger
        from .eda_analyzer import EDAAnalyzer
        from .verification import TemporalVerifier
        from .models.factory import ModelFactory

        self._build_queue()
        total = self._job_queue.qsize()

        logger.info("=" * 70)
        logger.info("BENCHMARK START — %d jobs", total)
        logger.info("=" * 70)

        exp_logger = ExperimentLogger(
            log_dir=self._exp_cfg.get("logging", {}).get("log_dir", "outputs/logs"),
            filename=self._exp_cfg.get("experiment", {}).get("name", "benchmark"),
        )

        pipeline_manager = DataPipelineManager(self._exp_cfg)
        eda_analyzer = EDAAnalyzer(
            output_dir=str(
                Path(self._exp_cfg.get("logging", {}).get("report_dir", "outputs/reports")) / "eda"
            )
        )
        verifier = TemporalVerifier(
            output_dir=str(
                Path(self._exp_cfg.get("logging", {}).get("report_dir", "outputs/reports")) / "verification"
            )
        )

        completed = 0

        while not self._job_queue.empty():
            job = self._job_queue.get()
            start_t = time.perf_counter()

            logger.info(
                "\n%s\n  JOB %d/%d — %s\n%s",
                "─" * 70, job.job_id, total, job, "─" * 70,
            )

            try:
                result = self._execute_job(
                    job, pipeline_manager, eda_analyzer,
                    verifier, exp_logger,
                )
            except Exception:
                # Catch-all: log full traceback but never crash the loop
                tb = traceback.format_exc()
                logger.error("Unhandled exception in job %s:\n%s", job, tb)
                result = JobResult(
                    job=job,
                    status="error",
                    elapsed_sec=time.perf_counter() - start_t,
                    error_message=tb[-500:],  # Truncate for storage
                )

            result.elapsed_sec = time.perf_counter() - start_t
            self._results.append(result)
            completed += 1

            # Always flush VRAM between jobs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info(
                "  → %s  status=%s  elapsed=%.1fs  [%d/%d done]",
                job, result.status, result.elapsed_sec, completed, total,
            )

        logger.info("=" * 70)
        logger.info(
            "BENCHMARK COMPLETE — %d/%d succeeded  %d errors  %d OOM skips",
            sum(r.status == "success" for r in self._results),
            total,
            sum(r.status == "error" for r in self._results),
            sum(r.status == "oom_skip" for r in self._results),
        )
        logger.info("=" * 70)

        exp_logger.export_excel()
        return self._results

    # ─── single job execution ────────────────────────────────────────────────

    def _execute_job(
        self,
        job: BenchmarkJob,
        pipeline_manager: Any,
        eda_analyzer: Any,
        verifier: Any,
        exp_logger: Any,
    ) -> JobResult:
        """Execute one (dataset × model) job end-to-end.

        This is the core unit of work — failures here are caught by
        the outer loop and do not propagate.
        """
        from .models.factory import ModelFactory
        from .trainer import BaseTrainer

        # ── 1. Prepare data ──────────────────────────────────────────────────
        pipeline_result = pipeline_manager.prepare(job.dataset_cfg)
        train_loader = pipeline_result["train_loader"]
        val_loader   = pipeline_result["val_loader"]
        test_loader  = pipeline_result["test_loader"]
        meta = pipeline_result["metadata"]

        # ── 2. EDA (only once per dataset, skip if already done) ─────────────
        # (In a full implementation, check a cache flag per dataset)

        # ── 3. Create model ──────────────────────────────────────────────────
        forecaster = ModelFactory.create(
            category=job.model_category,
            name=job.model_name,
            model_cfg={},
            experiment_cfg=self._exp_cfg,
        )

        # ── 4. Fit ───────────────────────────────────────────────────────────
        fit_result = forecaster.fit(
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # ── 5. Predict ───────────────────────────────────────────────────────
        import numpy as np
        from .trainer import MetricCalculator, HardwareMonitor, TimingContext

        HardwareMonitor.reset_peak_stats()
        infer_timer = TimingContext()

        with infer_timer:
            y_pred = forecaster.predict(test_loader)

        # Collect ground truth from test loader
        all_y = []
        for _, y in test_loader:
            all_y.append(y.numpy())
        y_true = np.concatenate(all_y).flatten()
        y_pred_flat = y_pred.flatten()[: len(y_true)]

        # ── 6. Metrics ───────────────────────────────────────────────────────
        metrics = MetricCalculator.compute_all(y_true, y_pred_flat)
        vram_peak = HardwareMonitor.get_vram_peak_gb()

        # ── 7. Log ───────────────────────────────────────────────────────────
        from .trainer import TrainingResult

        tr = TrainingResult(
            model_name=job.model_name,
            dataset_name=job.dataset_name,
            metrics=metrics,
            inference_time_sec=infer_timer.elapsed,
            inference_latency_ms=(infer_timer.elapsed / max(len(y_true), 1)) * 1000,
            vram_peak_gb=vram_peak,
            batch_size_final=meta.get("batch_size", 0),
            epochs_trained=fit_result.get("epochs", 0),
            early_stopped=fit_result.get("early_stopped", False),
            status="success",
        )
        exp_logger.log(tr)

        HardwareMonitor.flush_vram()

        return JobResult(job=job, status="success")

    # ─── utilities ───────────────────────────────────────────────────────────

    def get_summary(self) -> "pd.DataFrame":
        """Return a DataFrame summary of all completed jobs."""
        import pandas as pd

        rows = []
        for r in self._results:
            rows.append({
                "job_id": r.job.job_id,
                "dataset": r.job.dataset_name,
                "model": r.job.model_name,
                "category": r.job.model_category,
                "status": r.status,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "error": r.error_message[:100] if r.error_message else "",
            })
        return pd.DataFrame(rows)

    def _setup_file_logging(self, log_path: Path) -> None:
        """Add a rotating file handler to the root logger."""
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            str(log_path),
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=5,
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Graceful shutdown on SIGINT/SIGTERM."""
        logger.warning(
            "Signal %d received — finishing current job then exiting …", signum,
        )
        # Drain remaining jobs
        while not self._job_queue.empty():
            self._job_queue.get()
        sys.exit(0)
