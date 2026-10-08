"""AR(1) -> Ornstein-Uhlenbeck half-life."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class HalfLife:
    phi: float
    half_life_s: float
    dt_s: float


def ar1_half_life(series: pd.Series, dt_seconds: float = 1.0, resample: bool = True) -> HalfLife:
    """Estimate the OU half-life from an AR(1) fit ``x_t = a + phi x_{t-1} + e``.

    kappa = -ln(phi) / dt and half-life = ln(2) / kappa. If ``resample`` is true
    the series (DatetimeIndex) is first averaged onto a ``dt_seconds`` grid.

    Raises
    ------
    ValueError
        If the fitted ``phi`` is outside (0, 1) (no mean reversion).
    """
    s = series.resample(f"{dt_seconds:g}s").mean() if resample else series
    s = s.dropna()
    if len(s) < 100:
        raise ValueError("not enough observations to estimate an AR(1)")
    phi = float(sm.OLS(s.iloc[1:].to_numpy(), sm.add_constant(s.iloc[:-1].to_numpy())).fit().params[1])
    if not 0.0 < phi < 1.0:
        raise ValueError(f"AR(1) coefficient {phi:.4f} outside (0, 1)")
    return HalfLife(phi=phi, half_life_s=float(np.log(2) / (-np.log(phi)) * dt_seconds), dt_s=dt_seconds)
