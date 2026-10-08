import numpy as np
import pandas as pd
import pytest

from crosschain_arb.halflife import ar1_half_life


def _ar1(phi: float, n: int = 30_000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return pd.Series(x, index=pd.date_range("2023-01-01", periods=n, freq="s"))


def test_recovers_known_half_life():
    hl = ar1_half_life(_ar1(0.9))
    assert hl.half_life_s == pytest.approx(np.log(2) / -np.log(0.9), rel=0.1)
    assert hl.phi == pytest.approx(0.9, abs=0.02)


def test_scales_with_grid_step():
    # sampling every 5 s: phi_5 = 0.9 per step -> half-life in seconds is 5x larger
    s = _ar1(0.9)
    s.index = pd.date_range("2023-01-01", periods=len(s), freq="5s")
    assert ar1_half_life(s, dt_seconds=5.0).half_life_s == pytest.approx(
        5 * np.log(2) / -np.log(0.9), rel=0.1
    )


def test_rejects_non_mean_reverting_fit():
    with pytest.raises(ValueError):
        ar1_half_life(_ar1(-0.5))


def test_rejects_short_series():
    with pytest.raises(ValueError):
        ar1_half_life(_ar1(0.9, n=50))
