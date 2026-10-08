import numpy as np
import pandas as pd
import pytest

from crosschain_arb import lp


def test_robust_scale_fallback_chain():
    assert lp.robust_scale(pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])) == 1.0  # MAD
    mostly_zero = pd.Series([0.0] * 6 + [1.0, 2.0, 3.0, 4.0])
    assert lp.robust_scale(mostly_zero) == pytest.approx(1.75 / 1.349)  # MAD == 0 -> IQR / 1.349
    assert lp.robust_scale(pd.Series([0.0] * 50 + [1.0])) == pytest.approx(
        pd.Series([0.0] * 50 + [1.0]).std()
    )
    assert lp.robust_scale(pd.Series([0.0, 0.0, 0.0])) == 1.0


def test_normalise_lvr_scales_within_blocks():
    df = pd.DataFrame(
        {
            "chain": ["a"] * 4 + ["b"] * 4,
            "exit_ts": pd.to_datetime(["2023-03-06"] * 8),
            "lvr": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    out = lp.normalise_lvr(df)
    assert out.loc[out.chain == "a", "lvr_rel"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])  # MAD 1.0
    assert out.loc[out.chain == "b", "lvr_rel"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])  # MAD 10.0


def test_outcome_diagnostics_warns_on_degenerate_outcome():
    with pytest.warns(UserWarning, match="exactly zero"):
        diag = lp.outcome_diagnostics(pd.Series([0.0] * 90 + [-1.0] * 10))
    assert diag["share_zero"] == 0.9


def test_cohort_tests_detect_a_shift_and_handle_empty_cohort():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"multi_chain": [0] * 500 + [1] * 500, "lvr_rel": np.r_[rng.normal(0, 1, 500), rng.normal(1, 1, 500)]}
    )
    assert lp.cohort_tests(df)["p_mannwhitney"] < 1e-6
    assert np.isnan(lp.cohort_tests(df[df.multi_chain == 0])["p_mannwhitney"])


def test_strict_negative_tail_fisher_table_sums_to_n():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"multi_chain": rng.integers(0, 2, 400), "lvr": -rng.exponential(1.0, 400)})
    res = lp.strict_negative_tail_fisher(df)
    assert np.sum(res["table"]) == 400 and 0 <= res["p"] <= 1


def test_multi_chain_ols_runs_with_clusters():
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame(
        {
            "lp": rng.integers(0, 150, n),
            "multi_chain": rng.integers(0, 2, n),
            "dwell": rng.uniform(1, 100, n),
            "vol": rng.uniform(0, 1, n),
        }
    )
    df["lvr_rel"] = 0.5 * df["multi_chain"] + rng.normal(size=n)
    res = lp.multi_chain_ols(df)
    assert res["beta_multi_chain"] == pytest.approx(0.5, abs=0.3) and res["n"] == n


def test_position_amounts_positive_and_scale_with_liquidity():
    dx, dy = lp.position_amounts(np.array([1.0, 2.0]), np.array([-100, -100]), np.array([100, 100]))
    assert (dx > 0).all() and (dy > 0).all()
    assert dx[1] == pytest.approx(2 * dx[0])


def test_build_lvr_panel_end_to_end_on_toy_data():
    t0 = pd.Timestamp("2023-03-01")
    prices = pd.Series(np.linspace(1500, 1600, 100), index=pd.date_range(t0, periods=100, freq="min"))
    summary = pd.DataFrame(
        {
            "origin": ["w1", "w2"],
            "entry_ts": [t0, t0],
            "exit_ts": [t0 + pd.Timedelta("1h"), t0 + pd.Timedelta("30min")],
            "L0": [1e3, 1e3],
            "lower_tick": [-10, -10],
            "upper_tick": [10, 10],
            "L1": [1e3, 5e2],
            "lower_tick_exit": [-10, -10],
            "upper_tick_exit": [10, 10],
            "gas0": [20.0, 90.0],
            "chain": ["arb", "eth"],
        }
    )
    out = lp.build_lvr_panel(
        summary, prices, multi_chain_ids={"w2"}, gas_spike_threshold=50.0, lvr_abs_max=None
    )
    assert out["multi_chain"].tolist() == [0, 1]
    assert out["gas_spike"].tolist() == [0, 1]
    assert (out["dwell"] > 0).all() and out["vol"].notna().all()


def test_multi_chain_wallets_all_time_and_windowed():
    a = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-01-01", "2023-03-05", "2023-03-06"]),
            "lp_address": ["0xAA", "0xBB", "0xCC"],
        }
    )
    b = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-03-07", "2023-08-01", "2023-03-08"]),
            "lp_address": ["0xaa", "0xbb", "0xdd"],
        }
    )
    assert lp.multi_chain_wallets(a, b) == {"0xaa", "0xbb"}  # case-insensitive, all time
    assert (
        lp.multi_chain_wallets(a, b, "2023-03-01", "2023-07-01") == set()
    )  # 0xaa's other leg is 2022; 0xbb's is August
    a2 = a.assign(timestamp=pd.to_datetime(["2023-03-02", "2023-03-05", "2023-03-06"]))
    assert lp.multi_chain_wallets(a2, b, "2023-03-01", "2023-07-01") == {"0xaa"}


def test_ledger_diagnostics_reports_negative_cumulative_liquidity():
    led = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-03-02", "2023-03-03", "2024-01-01", "2021-01-01"]),
            "lp_address": ["0x1", "0x2", "0x3", "0x1"],
            "cum_liquidity": [5.0, -2.0, 1.0, 0.0],
        }
    )
    d = lp.ledger_diagnostics(led)
    assert d["rows"] == 4 and d["wallets"] == 3
    assert d["share_rows_in_window"] == 0.5 and d["wallets_in_window"] == 2
    assert d["share_negative_cum_liquidity"] == 0.25


def test_position_amounts_match_uniswap_v3_formulas_on_an_asymmetric_band():
    lower, upper = -200, 500
    pl, pu = tick_p(lower), tick_p(upper)
    x, y = lp.position_amounts(np.array([3.0]), np.array([lower]), np.array([upper]))
    assert x[0] == pytest.approx(3.0 * (1 / np.sqrt(pl) - 1 / np.sqrt(pu)))
    assert y[0] == pytest.approx(3.0 * (np.sqrt(pu) - np.sqrt(pl)))
    assert x[0] != pytest.approx(y[0], rel=1e-3)  # the two token amounts differ unless the band is symmetric


def tick_p(tick: int) -> float:
    return 1.0001**tick
