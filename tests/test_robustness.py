import numpy as np
import pandas as pd
import pytest

from crosschain_arb import robustness as R
from crosschain_arb.panel import add_gas_spike, with_depth


@pytest.fixture(scope="module")
def synth_panel():
    """10-second panel with a persistent AR(1) basis (many multi-second gaps, several days)."""
    rng = np.random.default_rng(3)
    n = 40_000
    basis = np.zeros(n)
    for t in range(1, n):
        basis[t] = 0.9 * basis[t - 1] + rng.normal()
    volume = rng.exponential(0.5, n)
    depth = pd.Series(volume).rolling(1000, min_periods=300).mean().bfill()
    return pd.DataFrame(
        {
            "time": pd.date_range("2023-03-01", periods=n, freq="10s"),  # ~4.6 days, so day clusters exist
            "basis_t": basis,
            "latency_t": rng.lognormal(2.0, 1.0, n),
            "gas_gwei": rng.uniform(10, 80, n),
            "volume": volume,
            "depth": depth,
            "SpikeDummy_t": (rng.uniform(size=n) < 0.09).astype(int),
        }
    )


def test_file_spike_rule_uses_stored_flag(synth_panel):
    out = add_gas_spike(synth_panel, "file")
    assert out["spike_dummy"].tolist() == synth_panel["SpikeDummy_t"].tolist()


def test_with_depth_forward_looks_ahead(synth_panel):
    trailing, forward = with_depth(synth_panel, window=50), with_depth(synth_panel, window=50, forward=True)
    i = 1000
    expected_fwd = synth_panel["volume"].iloc[i : i + 50].mean()
    assert forward.set_index("time").loc[synth_panel["time"].iloc[i], "depth"] == pytest.approx(expected_fwd)
    assert trailing.set_index("time").loc[synth_panel["time"].iloc[i], "depth"] == pytest.approx(
        synth_panel["volume"].iloc[i - 49 : i + 1].mean()
    )


def test_h1_robustness_table_shape_and_labels(synth_panel):
    out = R.h1_robustness(synth_panel)
    assert list(out.columns) == R.H1_COLS
    assert set(out["source"]) == {"core", "extended"}
    assert out["spec"].str.contains("FALSIFICATION").any()
    assert out[["n", "beta", "se_hc1", "se_cluster"]].notna().all().all()
    assert (out["se_cluster"] > 0).all()


def test_h2_robustness_returns_table_margin_and_quantiles(synth_panel):
    out = R.h2_robustness(synth_panel, min_seconds=1.0)
    t = out["table"]
    base = t[t["spec"].str.startswith("baseline")].iloc[0]
    assert base["n_treated"] > 30 and np.isfinite(base["interaction"])
    assert {"beta_margin", "se", "p"} <= set(out["margin"])
    assert set(out["quantile"]) == {"q25", "q75"}


def test_half_life_grows_with_grid_for_discretised_ar1(synth_panel):
    grid = R.half_life_by_grid(synth_panel.reset_index(drop=True), grids=(1.0, 2.0))
    assert grid["half_life_s"].is_monotonic_increasing


@pytest.mark.data
def test_robustness_reference_values():
    from crosschain_arb.io import load_panel

    panel = load_panel()
    h1 = R.h1_robustness(panel).set_index("spec")["beta"]
    assert h1["file spike flag, >= 1s"] == pytest.approx(-0.4288, abs=5e-4)
    assert h1["file spike flag, >= 1s, winsorised depth"] == pytest.approx(-0.4416, abs=5e-4)
    assert h1["file spike flag, >= 5s"] == pytest.approx(-0.2638, abs=5e-4)
    h2 = R.h2_robustness(panel)
    base = h2["table"].iloc[0]
    assert base["interaction"] == pytest.approx(1.189, abs=2e-3)
    assert h2["margin"]["beta_margin"] == pytest.approx(0.00136, abs=2e-5)
