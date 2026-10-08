"""Tick-level panel construction.

The basis, depth and latency series used for H1/H2 are *proxies* built from
Binance 1 ms bars merged with Ethereum base fees (see README, "Limitations"):

* ``basis_t``  = close - rolling mean(close, 50 ticks)
* ``depth_t``  = rolling mean(volume, 1000 ticks)
* ``latency_t``= iid lognormal draw per tick (simulated, not matched to bridge events)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def build_proxy_panel(
    merged_ms: pd.DataFrame,
    seed: int = C.SEED,
    latency_mu: float = C.LATENCY_LOG_MU,
    latency_sigma: float = C.LATENCY_LOG_SIGMA,
) -> pd.DataFrame:
    """Build the proxy panel from Binance bars merged with gas.

    Parameters
    ----------
    merged_ms
        Columns ``timestamp, close, volume, gas_gwei`` (others are kept).
    seed
        Seed for the simulated latency (NumPy ``RandomState`` stream, so the
        output matches the shipped panel exactly).
    """
    df = merged_ms.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", ignore_index=True)

    df["basis_t"] = df["close"] - df["close"].rolling(C.BASIS_WINDOW).mean()
    df = df.dropna(subset=["basis_t"]).reset_index(drop=True)

    rs = np.random.RandomState(seed)  # same stream as np.random.seed(seed)
    df["latency_t"] = rs.lognormal(mean=latency_mu, sigma=latency_sigma, size=len(df))

    df["depth_t"] = df["volume"].rolling(C.DEPTH_WINDOW, min_periods=C.DEPTH_MIN_PERIODS).mean()
    df["log_depth_t"] = np.log1p(df["depth_t"])
    return df.dropna(subset=["depth_t"]).reset_index(drop=True)


def add_gas_spike(
    df: pd.DataFrame,
    rule: str = "global_p95",
    gas_col: str = "gas_gwei",
    time_col: str = "time",
    quantile: float = C.GAS_SPIKE_QUANTILE,
) -> pd.DataFrame:
    """Add a ``spike_dummy`` column.

    ``global_p95``: gas >= the full-sample ``quantile``.
    ``day_top10`` : gas >= the within-day 90th percentile.
    ``file``      : use the ``SpikeDummy_t`` flag already stored in the shipped panel (about
                    9% of ticks; it was built from a different gas file than the panel's, so it
                    does *not* equal the p95 rule).
    """
    out = df.copy()
    if rule == "file":
        out["spike_dummy"] = out["SpikeDummy_t"].astype(int)
    elif rule == "global_p95":
        out["spike_dummy"] = (out[gas_col] >= out[gas_col].quantile(quantile)).astype(int)
    elif rule == "day_top10":
        day = out[time_col].dt.floor("D")
        thr = out.groupby(day)[gas_col].transform(lambda s: s.quantile(0.90))
        out["spike_dummy"] = (out[gas_col] >= thr).astype(int)
    else:
        raise ValueError("rule must be 'global_p95', 'day_top10' or 'file'")
    return out


def with_depth(
    df: pd.DataFrame,
    window: int = C.DEPTH_WINDOW,
    forward: bool = False,
    min_frac: float = C.DEPTH_MIN_PERIODS / C.DEPTH_WINDOW,
    volume_col: str = "volume",
) -> pd.DataFrame:
    """Return a copy with ``depth`` recomputed as a rolling mean of volume.

    ``forward=False`` is the default trailing window (known at the tick).
    ``forward=True`` averages the *next* ``window`` ticks, which is only useful as a
    falsification check (depth that cannot have been known when the gap opened).
    """
    out = df.copy()
    vol = out[volume_col]
    min_periods = max(1, int(window * min_frac))
    if forward:
        depth = vol.iloc[::-1].rolling(window, min_periods=min_periods).mean().iloc[::-1]
    else:
        depth = vol.rolling(window, min_periods=min_periods).mean()
    out["depth"] = depth
    return out.dropna(subset=["depth"])
