"""Checks against the real data. Skipped automatically when data/ is empty."""

import pandas as pd
import pytest

from crosschain_arb.config import DATA_DIR
from crosschain_arb.episodes import build_gap_episodes
from crosschain_arb.halflife import ar1_half_life
from crosschain_arb.io import load_panel
from crosschain_arb.models import fit_h1
from crosschain_arb.panel import add_gas_spike, build_proxy_panel

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def panel():
    return add_gas_spike(load_panel())


def test_h1_reference_values(panel):
    ep = build_gap_episodes(panel, min_seconds=5.0)
    assert len(build_gap_episodes(panel, min_seconds=1.0)) == 20_074
    assert len(ep) == 17_988
    res = fit_h1(ep)
    assert res.beta_log_depth == pytest.approx(-0.279, abs=0.001)
    assert res.se_hc1 == pytest.approx(0.0276, abs=0.0005)
    assert res.se_cluster == pytest.approx(0.0858, abs=0.001)


def test_half_life_is_about_ten_seconds(panel):
    assert ar1_half_life(panel.set_index("time")["basis_t"]).half_life_s == pytest.approx(9.97, abs=0.02)


@pytest.mark.skipif(not (DATA_DIR / "eth_gas_merged_ms.parquet").exists(), reason="merged ms file missing")
def test_proxy_panel_rebuild_matches_shipped_panel():
    rebuilt = build_proxy_panel(pd.read_parquet(DATA_DIR / "eth_gas_merged_ms.parquet"))
    shipped = (
        pd.read_parquet(DATA_DIR / "eth_gas_with_depth.parquet")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    assert len(rebuilt) == len(shipped)
    for col in ["basis_t", "latency_t", "depth_t"]:
        assert rebuilt[col].to_numpy() == pytest.approx(shipped[col].to_numpy(), rel=1e-9, abs=1e-9)
