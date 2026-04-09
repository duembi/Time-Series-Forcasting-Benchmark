"""
download_data.py — Dataset Download & Preparation Helper.

Provides instructions and automated download for publicly available datasets.

Usage:
    python scripts/download_data.py --dataset metr_la --output data/
    python scripts/download_data.py --dataset all --output data/

Note:
    PEMS03/04/07/08 and PEMS-BAY are obtained from the ASTGNN / STGCN repos.
    METR-LA is distributed by the DCRNN authors.
    T-Drive is published by Microsoft Research.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry — public download sources
# ─────────────────────────────────────────────────────────────────────────────

DATASET_INFO: dict[str, dict] = {
    "metr_la": {
        "description": "METR-LA traffic speed (207 sensors, 34,272 timesteps)",
        "instructions": """
  1. Clone the DCRNN repository:
     git clone https://github.com/liyaguang/DCRNN
  2. Follow the data preparation instructions in DCRNN/README.md
  3. Copy metr_la.npz and adj_mx.pkl to:
     data/metr_la/metr_la.npz
     data/metr_la/adj_mx.pkl
""",
    },
    "pems_bay": {
        "description": "PEMS-BAY traffic speed (325 sensors, Bay Area)",
        "instructions": """
  1. Same DCRNN repository as METR-LA
  2. Copy pems_bay.npz and adj_mx_bay.pkl to:
     data/pems_bay/pems_bay.npz
     data/pems_bay/adj_mx_bay.pkl
""",
    },
    "pems08": {
        "description": "PeMS08 traffic flow (170 sensors, Jul–Aug 2016)",
        "instructions": """
  1. Download from the ASTGNN repository:
     https://github.com/guoshnBJTU/ASTGNN/tree/main/data
  2. Place pems08.npz in: data/pems08/pems08.npz
""",
    },
    "pems04": {
        "description": "PeMS04 traffic flow (307 sensors, Jan–Feb 2018)",
        "instructions": """
  1. Same ASTGNN repository as PeMS08
  2. Place pems04.npz in: data/pems04/pems04.npz
""",
    },
    "pems07": {
        "description": "PeMS07 traffic flow (883 sensors, May–Aug 2017)",
        "instructions": """
  1. Same ASTGNN repository as PeMS08
  2. Place pems07.npz in: data/pems07/pems07.npz
""",
    },
    "pems03": {
        "description": "PeMS03 traffic flow (358 sensors, Sep–Nov 2018)",
        "instructions": """
  1. Same ASTGNN repository as PeMS08
  2. Place pems03.npz in: data/pems03/pems03.npz
""",
    },
    "t_drive": {
        "description": "T-Drive taxi trajectories (10,357 taxis, Feb 2008, Beijing)",
        "instructions": """
  1. Download from Microsoft Research:
     https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
  2. Extract and place raw .txt files in: data/t_drive/raw/
  3. Run preprocessing:
     python scripts/download_data.py --preprocess t_drive --output data/
""",
    },
}


def _print_instructions(dataset: str) -> None:
    info = DATASET_INFO.get(dataset)
    if info is None:
        logger.error("Unknown dataset: %s. Available: %s", dataset, list(DATASET_INFO.keys()))
        return

    print(f"\n{'─' * 60}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Description: {info['description']}")
    print(f"Instructions:{info['instructions']}")
    print(f"{'─' * 60}\n")


def _create_directories(output_dir: str) -> None:
    """Create the expected data subdirectory structure."""
    datasets = list(DATASET_INFO.keys())
    for ds in datasets:
        Path(output_dir, ds).mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure created under: %s", output_dir)


def _generate_synthetic_dataset(output_dir: str, dataset: str) -> None:
    """Generate a small synthetic NPZ for smoke-testing without real data."""
    import numpy as np

    out_dir = Path(output_dir, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    T, N, F = 2016, 10, 3   # 1 week at 5-min intervals, 10 sensors, 3 features
    data = (
        np.sin(np.linspace(0, 4 * np.pi, T))[:, None, None]
        * np.ones((1, N, F)) * 50 + 100
        + np.random.randn(T, N, F) * 5
    ).astype(np.float32)

    npz_name = f"{dataset}.npz" if dataset != "metr_la" else "metr_la.npz"
    out_path = out_dir / npz_name
    np.savez(str(out_path), data=data)
    logger.info("Synthetic dataset saved → %s  shape=%s", out_path, data.shape)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSF Benchmark — Dataset Download Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", "-d", default="all",
                        help="Dataset name or 'all' (default: all)")
    parser.add_argument("--output", "-o", default="data",
                        help="Output root directory (default: data/)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate small synthetic NPZ files for smoke-testing")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    datasets = list(DATASET_INFO.keys()) if args.dataset == "all" else [args.dataset]

    _create_directories(args.output)

    if args.synthetic:
        logger.info("Generating synthetic datasets for smoke-testing …")
        for ds in datasets:
            _generate_synthetic_dataset(args.output, ds)
        logger.info("Done. Update configs/datasets/*.yaml paths to point to data/ directory.")
        return

    print("\n" + "=" * 60)
    print(" TSF BENCHMARK — Dataset Download Instructions")
    print("=" * 60)
    print(
        "\nAll datasets require manual download due to licensing constraints.\n"
        "Follow the instructions below for each dataset.\n"
    )

    for ds in datasets:
        _print_instructions(ds)

    print(
        "\nTip: Use --synthetic to generate small fake NPZ files for quick\n"
        "     pipeline smoke-testing without downloading real datasets.\n"
        "     python scripts/download_data.py --synthetic --output data/\n"
    )


if __name__ == "__main__":
    main()
