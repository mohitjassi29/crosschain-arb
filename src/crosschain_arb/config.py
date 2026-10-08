"""Paths, seeds and constants shared across the package."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CCARB_DATA", REPO_ROOT / "data"))
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42

# Lognormal bridge-latency parameters (log scale), fitted to the Dune export.
LATENCY_LOG_MU = 4.495
LATENCY_LOG_SIGMA = 1.141

# Panel construction
BASIS_WINDOW = 50  # rolling-mean window (ticks) for the Binance basis proxy
DEPTH_WINDOW = 1000  # rolling-volume window (ticks) for the depth proxy
DEPTH_MIN_PERIODS = 300

# Episodes and gas
MIN_EPISODE_SECONDS = 5.0
GAS_SPIKE_QUANTILE = 0.95

# Uniswap v3 WETH/USDC raw prices embed the 10**(18-6) decimals difference.
DECIMALS_FACTOR = 1e12
