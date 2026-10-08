"""Input readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def data_path(name: str, data_dir: Path | None = None) -> Path:
    """Return the path to a data file, raising a helpful error if it is missing."""
    path = Path(data_dir or DATA_DIR) / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/README.md for the required inputs "
            "or set the CCARB_DATA environment variable."
        )
    return path


def _naive_utc(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series)
    return out.dt.tz_localize(None) if out.dt.tz is not None else out


def load_panel(name: str = "eth_gas_with_depth.parquet", data_dir: Path | None = None) -> pd.DataFrame:
    """Load the tick-level panel with canonical column names.

    Returns columns ``time, basis_t, depth, latency_t, gas_gwei`` (and any others
    present), sorted by time with timezone-naive UTC timestamps.
    """
    df = pd.read_parquet(data_path(name, data_dir)).rename(columns={"timestamp": "time", "depth_t": "depth"})
    df["time"] = _naive_utc(df["time"])
    return df.sort_values("time", ignore_index=True)


def load_latency(name: str = "dune_latency_latest.csv", data_dir: Path | None = None) -> pd.Series:
    """Load positive bridge latencies (seconds) from the Dune export."""
    df = pd.read_csv(data_path(name, data_dir))
    df.columns = df.columns.str.lower()
    lat = pd.to_numeric(df["latency_seconds"], errors="coerce").dropna()
    return lat[lat > 0].reset_index(drop=True)
