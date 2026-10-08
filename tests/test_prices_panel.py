import numpy as np
import pandas as pd
import pytest

from crosschain_arb.panel import add_gas_spike, build_proxy_panel
from crosschain_arb.prices import to_usd


def test_to_usd_conversion_roundtrip():
    dex = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-03-01 00:00:11"]),
            "mid_price_eth": [1e12 / 1600.0],
            "mid_price_arb": [1605.0 / 1e12],
        }
    )
    out = to_usd(dex)
    assert out["price_l1"].iloc[0] == pytest.approx(1600.0)
    assert out["price_l2"].iloc[0] == pytest.approx(1605.0)
    assert out["basis"].iloc[0] == pytest.approx(5.0)


def _merged(n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-03-01", periods=n, freq="100ms"),
            "close": 1800 + rng.normal(0, 1, n).cumsum() * 0.01,
            "volume": rng.exponential(0.5, n),
            "gas_gwei": rng.uniform(10, 100, n),
        }
    )


def test_build_proxy_panel_is_deterministic_and_drops_warmup_rows():
    a, b = build_proxy_panel(_merged()), build_proxy_panel(_merged())
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 3000 - 49 - 299  # 50-tick basis warm-up, then 300-tick depth minimum
    assert a[["basis_t", "depth_t", "latency_t"]].notna().all().all()


def test_build_proxy_panel_seed_changes_latency_only():
    a, b = build_proxy_panel(_merged(), seed=1), build_proxy_panel(_merged(), seed=2)
    assert not np.allclose(a["latency_t"], b["latency_t"])
    assert np.allclose(a["basis_t"], b["basis_t"])


def test_gas_spike_share_is_about_five_percent():
    df = _merged().rename(columns={"timestamp": "time"})
    assert add_gas_spike(df)["spike_dummy"].mean() == pytest.approx(0.05, abs=0.01)
    assert add_gas_spike(df, rule="day_top10")["spike_dummy"].mean() == pytest.approx(0.10, abs=0.02)
    with pytest.raises(ValueError):
        add_gas_spike(df, rule="bad")
