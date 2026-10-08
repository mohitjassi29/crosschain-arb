import pandas as pd
import pytest

from crosschain_arb.episodes import (
    add_roi,
    add_timing_indicator,
    attach_future_basis,
    build_gap_episodes,
)


def test_positive_episodes_and_durations(toy_panel):
    ep = build_gap_episodes(toy_panel, sign="positive", min_seconds=1.0)
    # runs of positive sign: 0-9, 15-16, 30-39  -> durations 9, 1, 9 seconds
    assert ep["duration"].tolist() == [9.0, 1.0, 9.0]
    assert ep["n_ticks"].tolist() == [10, 2, 10]


def test_min_seconds_filter(toy_panel):
    ep = build_gap_episodes(toy_panel, sign="positive", min_seconds=5.0)
    assert len(ep) == 2 and (ep["duration"] >= 5).all()


def test_negative_episodes(toy_panel):
    ep = build_gap_episodes(toy_panel, sign="negative", min_seconds=1.0)
    assert ep["duration"].tolist() == [4.0, 12.0]


def test_basis_start_is_first_tick_of_run(toy_panel):
    ep = build_gap_episodes(toy_panel, sign="positive", min_seconds=1.0)
    assert ep["basis_start"].tolist() == [1.0, 16.0, 31.0]


def test_future_basis_uses_last_observation_at_or_before(toy_panel):
    ep = build_gap_episodes(toy_panel, min_seconds=5.0)
    out = attach_future_basis(ep, toy_panel)
    # episode 1 starts at t=0, latency 3 s -> basis at t=3 is +4
    assert out.loc[0, "basis_future"] == 4.0
    # episode 2 starts at t=30, +3 s -> t=33 -> +34
    assert out.loc[1, "basis_future"] == 34.0


def test_roi_definitions():
    ep = pd.DataFrame({"basis_start": [10.0], "basis_future": [4.0]})
    assert (
        add_roi(ep, "retained", cost=1.0)["roi"].iloc[0] == 4 - 1
    )  # gap still open at settlement, net of cost
    assert add_roi(ep, "closure")["roi"].iloc[0] == 6.0
    total = add_roi(ep, "closure")["roi"].iloc[0] + add_roi(ep, "retained")["roi"].iloc[0]
    assert total == ep["basis_start"].iloc[0]  # retained = b0 - closure (zero cost)
    with pytest.raises(ValueError):
        add_roi(ep, "nope")


def test_timing_indicator():
    ep = pd.DataFrame({"latency_med": [5.0, 20.0]})
    assert add_timing_indicator(ep, tau_half_s=10.0)["timing_I"].tolist() == [1, 0]
