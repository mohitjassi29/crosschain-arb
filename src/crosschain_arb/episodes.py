"""Gap episodes and ROI over the latency window."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .config import MIN_EPISODE_SECONDS

RoiDefinition = Literal["retained", "closure"]


def build_gap_episodes(
    panel: pd.DataFrame,
    sign: Literal["positive", "negative"] = "positive",
    min_seconds: float = MIN_EPISODE_SECONDS,
) -> pd.DataFrame:
    """Maximal contiguous runs where the basis has the requested sign.

    Required panel columns: ``time, basis_t, latency_t, gas_gwei, depth, spike_dummy``.
    Episodes with zero duration, or shorter than ``min_seconds``, are dropped.
    Episodes do not overlap: a new one starts only after a sign reversal.
    """
    cols = ["time", "basis_t", "latency_t", "gas_gwei", "depth", "spike_dummy"]
    d = panel[cols].dropna().sort_values("time").copy()
    d["is_gap"] = d["basis_t"] > 0 if sign == "positive" else d["basis_t"] < 0
    d["gap_id"] = d["is_gap"].ne(d["is_gap"].shift()).cumsum()
    gaps = d[d["is_gap"]]
    latency_p95 = gaps.groupby("gap_id", sort=False)["latency_t"].quantile(0.95).to_numpy()

    ep = (
        gaps.groupby("gap_id", sort=False)
        .agg(
            start_time=("time", "first"),
            end_time=("time", "last"),
            basis_start=("basis_t", "first"),
            latency_med=("latency_t", "median"),
            gas_mean=("gas_gwei", "mean"),
            depth_mean=("depth", "mean"),
            depth_entry=("depth", "first"),
            spike_dummy=("spike_dummy", "max"),
            n_ticks=("time", "size"),
        )
        .reset_index(drop=True)
    )
    ep["latency_p95"] = latency_p95
    ep["duration"] = (ep["end_time"] - ep["start_time"]).dt.total_seconds()
    ep = ep[(ep["duration"] > 0) & (ep["duration"] >= min_seconds) & (ep["depth_mean"] > 0)].copy()
    ep["log_duration"] = np.log(ep["duration"])
    ep["log_depth"] = np.log(ep["depth_mean"])
    ep["day"] = ep["start_time"].dt.floor("D")
    return ep.reset_index(drop=True)


def attach_future_basis(episodes: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``basis_future``: the last observed basis at ``start + latency_med``."""
    ep = episodes.copy()
    ep["future_time"] = ep["start_time"] + pd.to_timedelta(ep["latency_med"], unit="s")
    series = (
        panel[["time", "basis_t"]]
        .dropna()
        .rename(columns={"time": "t_future", "basis_t": "basis_future"})
        .sort_values("t_future")
    )
    out = pd.merge_asof(
        ep.sort_values("future_time"),
        series,
        left_on="future_time",
        right_on="t_future",
        direction="backward",
    )
    return out.dropna(subset=["basis_future"]).sort_values("start_time", ignore_index=True)


def add_roi(
    episodes: pd.DataFrame, definition: RoiDefinition = "retained", cost: float = 0.0
) -> pd.DataFrame:
    """Add ``roi``.

    ``retained``: b_l - cost, the gap still open when the transfer settles (the payoff of
                  buying on the cheap side at entry and selling on the dear side at settlement).
    ``closure`` : b0 - b_l, the part of the gap that has closed over the latency window
                  (so ``retained = b0 - closure`` before costs).
    """
    ep = episodes.copy()
    b0, bl = ep["basis_start"], ep["basis_future"]
    if definition == "retained":
        ep["roi"] = bl - cost
    elif definition == "closure":
        ep["roi"] = b0 - bl
    else:
        raise ValueError("definition must be 'retained' or 'closure'")
    return ep


def add_timing_indicator(episodes: pd.DataFrame, tau_half_s: float) -> pd.DataFrame:
    """Add ``timing_I`` = 1{half-life > median episode latency}."""
    ep = episodes.copy()
    ep["timing_I"] = (tau_half_s > ep["latency_med"]).astype(int)
    return ep
