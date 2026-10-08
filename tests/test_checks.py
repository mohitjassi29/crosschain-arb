import numpy as np
import pandas as pd
import pytest

from crosschain_arb import checks


def test_latency_fit_recovers_parameters_and_reports_misfit():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.lognormal(4.0, 0.8, 20_000))
    out = checks.latency_fit_checks(good, {"fast": 10.0, "slow": 1_000.0})
    assert out["lognormal_mu"] == pytest.approx(4.0, abs=0.02)
    assert out["lognormal_sigma"] == pytest.approx(0.8, abs=0.02)
    assert out["ks_statistic"] < 0.02
    assert out["share_faster_than_half_life"]["slow"] > out["share_faster_than_half_life"]["fast"]

    bimodal = pd.Series(np.r_[rng.uniform(1, 30, 5_000), rng.uniform(600, 700, 5_000)])
    assert checks.latency_fit_checks(bimodal, {"x": 10.0})["ks_statistic"] > 0.1  # a poor fit is detected


def test_share_faster_than_half_life_is_exact():
    out = checks.latency_fit_checks(pd.Series([1.0, 2.0, 3.0, 50.0]), {"h": 2.5})
    assert out["share_faster_than_half_life"]["h"] == 0.5


def test_panel_checks_confirm_definitions_on_a_constructed_panel():
    rng = np.random.default_rng(1)
    n = 3_000
    close = pd.Series(1800 + rng.normal(0, 1, n).cumsum() * 0.01)
    volume = pd.Series(rng.exponential(0.5, n))
    panel = pd.DataFrame(
        {
            "time": pd.date_range("2023-03-01", periods=n, freq="100ms"),
            "close": close,
            "volume": volume,
            "basis_t": close - close.rolling(50).mean(),
            "depth": volume.rolling(1000, min_periods=300).mean(),
            "latency_t": rng.lognormal(4.5, 1.1, n),
            "gas_gwei": rng.uniform(10, 100, n),
            "SpikeDummy_t": (rng.uniform(size=n) < 0.09).astype(int),
        }
    )
    out = checks.panel_checks(panel)
    assert out["basis_equals_close_minus_rolling50"] is True
    assert out["corr_depth_vs_rolling_volume"] == pytest.approx(1.0)
    assert abs(out["latency_lag1_autocorr"]) < 0.1
    assert out["share_spike_recomputed_p95"] == pytest.approx(0.05, abs=0.005)


def test_unit_checks_on_consistent_prices():
    t = pd.date_range("2023-03-01", periods=500, freq="10s")
    px = 1700 + np.sin(np.arange(500) / 20) * 30
    dex = pd.DataFrame({"timestamp": t, "mid_price_eth": 1e12 / px, "mid_price_arb": px / 1e12})
    panel = pd.DataFrame({"time": t, "close": px})
    out = checks.unit_checks(panel, dex)
    assert out["corr_binance_vs_price_l1"] == pytest.approx(1.0)
    assert out["corr_binance_vs_price_l2"] == pytest.approx(1.0)
