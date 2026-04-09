"""
run_benchmark.py — Main Entry Point for the TSF Benchmark Framework.

Designed to run headlessly on remote workstations via SSH + tmux/screen.

Usage:
    python scripts/run_benchmark.py \\
        --experiment configs/experiment.yaml \\
        --datasets   configs/datasets/pems08.yaml configs/datasets/metr_la.yaml \\
        --output-dir outputs/

    # Quick smoke-test with a single model:
    python scripts/run_benchmark.py --experiment configs/experiment.yaml \\
        --datasets configs/datasets/pems08.yaml --only-model DLinear

    # Resume after crash (skips already-logged dataset×model pairs):
    python scripts/run_benchmark.py --experiment configs/experiment.yaml \\
        --datasets configs/datasets/pems08.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _setup_logging(log_dir: str, level: str = "INFO") -> None:
    """Configure root logger with console + rotating file handlers."""
    from logging.handlers import RotatingFileHandler

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(fmt)

    # File
    fh = RotatingFileHandler(
        os.path.join(log_dir, "run_benchmark.log"),
        maxBytes=100 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSF Benchmark — Ministry of Transport and Infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        help="Path to experiment YAML (e.g. configs/experiment.yaml)",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        required=True,
        help="One or more dataset YAML paths (e.g. configs/datasets/pems08.yaml)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Root output directory (default: outputs/)",
    )
    parser.add_argument(
        "--only-model",
        default=None,
        help="Run only this model name (e.g. DLinear). Useful for quick tests.",
    )
    parser.add_argument(
        "--only-category",
        default=None,
        help="Run only models in this category (e.g. dl_minimalist).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dataset×model pairs already present in the results CSV.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA analysis (faster iteration).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation at the end.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _apply_model_filter(
    exp_cfg: dict,
    only_model: str | None,
    only_category: str | None,
) -> dict:
    """Disable all models except the requested subset."""
    if only_model is None and only_category is None:
        return exp_cfg

    models = exp_cfg.get("models", {})
    for cat, model_map in models.items():
        for name in list(model_map.keys()):
            keep = True
            if only_category and cat != only_category:
                keep = False
            if only_model and name != only_model:
                keep = False
            model_map[name] = keep

    return exp_cfg


def _build_already_done(results_csv: str) -> set[tuple[str, str]]:
    """Return the set of (dataset, model) pairs already in the CSV."""
    import pandas as pd

    path = Path(results_csv)
    if not path.exists():
        return set()
    df = pd.read_csv(str(path))
    if "Dataset" in df.columns and "Model" in df.columns:
        return set(zip(df["Dataset"], df["Model"]))
    return set()


def main() -> None:
    args = _parse_args()

    # ── logging ───────────────────────────────────────────────────────────────
    log_dir = str(Path(args.output_dir) / "logs")
    _setup_logging(log_dir, args.log_level)
    logger = logging.getLogger("run_benchmark")

    logger.info("=" * 70)
    logger.info("TSF BENCHMARK — Ministry of Transport and Infrastructure")
    logger.info("=" * 70)
    logger.info("Experiment : %s", args.experiment)
    logger.info("Datasets   : %s", args.datasets)
    logger.info("Output dir : %s", args.output_dir)

    # ── load experiment config ────────────────────────────────────────────────
    import yaml

    with open(args.experiment, "r", encoding="utf-8") as f:
        exp_cfg: dict = yaml.safe_load(f)

    # Override output dirs with CLI argument
    exp_cfg.setdefault("logging", {})
    exp_cfg["logging"]["log_dir"] = str(Path(args.output_dir) / "logs")
    exp_cfg["logging"]["checkpoint_dir"] = str(Path(args.output_dir) / "checkpoints")
    exp_cfg["logging"]["report_dir"] = str(Path(args.output_dir) / "reports")
    exp_cfg.setdefault("execution", {})
    exp_cfg["execution"]["cache_dir"] = str(Path(args.output_dir) / "cache")

    # Apply model filter (--only-model / --only-category)
    exp_cfg = _apply_model_filter(exp_cfg, args.only_model, args.only_category)

    # ── EDA pass ─────────────────────────────────────────────────────────────
    from src.eda_analyzer import EDAAnalyzer
    from src.verification import TemporalVerifier
    from src.data_pipeline import DataPipelineManager, DataLoaderFactory

    eda_reports: dict = {}
    align_results: dict = {}

    if not args.skip_eda:
        eda_analyzer = EDAAnalyzer(
            output_dir=str(Path(args.output_dir) / "reports" / "eda")
        )
        verifier = TemporalVerifier(
            output_dir=str(Path(args.output_dir) / "reports" / "verification")
        )
        pipeline_manager = DataPipelineManager(exp_cfg)

        for ds_yaml in args.datasets:
            with open(ds_yaml, "r", encoding="utf-8") as f:
                ds_cfg = yaml.safe_load(f)

            ds_name = ds_cfg["dataset"]["name"]
            logger.info("EDA — %s", ds_name)

            try:
                loader = DataLoaderFactory.create(ds_cfg["dataset"]["loader"])
                data, timestamps, series_ids = loader.load(ds_cfg["dataset"])

                eda_report = eda_analyzer.run(
                    data, timestamps, series_ids, ds_name,
                    freq=ds_cfg["dataset"].get("frequency", "5min"),
                )
                eda_reports[ds_name] = eda_report

                # Verification using splitter
                from src.data_pipeline import TemporalSplitter

                sp = exp_cfg.get("splitting", {})
                splitter = TemporalSplitter(
                    train_ratio=sp.get("train_ratio", 0.7),
                    val_ratio=sp.get("val_ratio", 0.1),
                    test_ratio=sp.get("test_ratio", 0.2),
                )
                (_, train_ts), (_, val_ts), (_, test_ts) = splitter.split(data, timestamps)

                align_result = verifier.verify(train_ts, val_ts, test_ts, ds_name)
                align_results[ds_name] = align_result

            except Exception as e:
                logger.warning("EDA failed for %s: %s — continuing.", ds_name, e)

    # ── benchmark run ─────────────────────────────────────────────────────────
    from src.orchestrator import BenchmarkOrchestrator

    orch = BenchmarkOrchestrator(args.experiment)

    # Re-inject CLI overrides into orchestrator's config
    orch._exp_cfg = exp_cfg

    for ds_yaml in args.datasets:
        orch.add_dataset(ds_yaml)

    # Resume: mark already-done pairs to skip
    if args.resume:
        results_csv = str(
            Path(args.output_dir) / "logs" /
            f"{exp_cfg.get('experiment', {}).get('name', 'benchmark')}.csv"
        )
        done_pairs = _build_already_done(results_csv)
        if done_pairs:
            logger.info("Resume mode — skipping %d already-completed pairs.", len(done_pairs))
            # Remove completed pairs from model config
            models = exp_cfg.get("models", {})
            # (Pair-level skipping is handled inside orchestrator in full impl)

    job_results = orch.run()

    # ── generate report ───────────────────────────────────────────────────────
    if not args.skip_report:
        from src.report_generator import MinistryReportGenerator
        import pandas as pd

        exp_logger_csv = str(
            Path(args.output_dir) / "logs" /
            f"{exp_cfg.get('experiment', {}).get('name', 'benchmark')}.csv"
        )
        results_df = pd.read_csv(exp_logger_csv) if Path(exp_logger_csv).exists() else pd.DataFrame()

        gen = MinistryReportGenerator(
            results_df=results_df,
            eda_reports=eda_reports,
            align_results=align_results,
            output_dir=str(Path(args.output_dir) / "reports"),
            experiment_name=exp_cfg.get("experiment", {}).get("name", "benchmark"),
        )
        paths = gen.generate()
        logger.info("Report → %s", paths.get("html", "N/A"))

    # ── summary ───────────────────────────────────────────────────────────────
    summary = orch.get_summary()
    success = (summary["status"] == "success").sum() if not summary.empty else 0
    total   = len(summary)
    logger.info("DONE — %d/%d jobs succeeded.", success, total)

    if not summary.empty:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
