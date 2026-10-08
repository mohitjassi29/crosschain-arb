"""Every number the README states must match the committed results files.

Each assertion matches a full phrase (not a bare number), so a wrong figure cannot hide behind
the same digits appearing elsewhere in the README.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text().split())  # collapse line wraps
R = ROOT / "results"


def load(name):
    return json.loads((R / name).read_text())


def num(x, nd, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "−")


def has(text):
    assert text in README, f"README is missing or disagrees with results: {text!r}"


@pytest.fixture(scope="module")
def res():
    return {
        "h1": load("h1.json"),
        "h2": load("h2.json"),
        "rob": load("robustness.json"),
        "uni": load("uniswap_basis.json"),
        "chk": load("data_checks.json"),
        "r1": pd.read_csv(R / "robustness_h1.csv"),
        "r2": pd.read_csv(R / "robustness_h2.csv"),
        "grid": pd.read_csv(R / "half_life_by_grid.csv"),
    }


def test_h1_numbers(res):
    h1, c, r1 = res["h1"], res["h1"]["canonical"], res["r1"]
    has(f"In {c['n']:,} gap episodes of at least 5 s")
    has(
        f"with respect to depth is **{num(c['beta_log_depth'], 3)}** (HC1 SE {c['se_hc1']:.3f}; "
        f"day-clustered SE {c['se_cluster']:.3f}, p = {c['p_cluster']:.3f})"
    )
    has(f"a duration about {abs(h1['doubling_depth_effect_on_duration_pct']):.0f}% shorter")
    has(
        f"negative in all {len(r1)} specifications ({num(r1['beta'].max(), 2)} to {num(r1['beta'].min(), 2)})"
    )
    assert (r1["beta"] < 0).all()
    fwd = r1[r1["spec"].str.startswith("FALSIFICATION")]["beta"].iloc[0]
    has(f"gives the same coefficient ({num(fwd, 3)})")
    has(f"is only suggestive (p = {h1['logrank_depth_terciles']['p_logrank']:.2f})")


def test_half_life_numbers(res):
    has(f"**{res['h1']['half_life_s']:.2f} s** on a 1 s grid")
    g = res["grid"].set_index("grid_s")["half_life_s"]
    has(f"{g[0.5]:.1f} s at 0.5 s, {g[1.0]:.1f} s at 1 s, {g[2.0]:.1f} s at 2 s and {g[5.0]:.1f} s at 5 s")


def test_h2_numbers(res):
    t, h2 = res["r2"], res["h2"]
    base = t.iloc[0]
    m = res["rob"]["h2_margin"]
    assert m["p"] < 1e-3
    has(f"(+{m['beta_margin']:.4f} per second, p < 0.001)")
    est = t.dropna(subset=["interaction"])
    assert (est["interaction"] > 0).all()
    has(
        f"positive in every specification that can be estimated ({num(est['interaction'].min(), 2, True)} "
        f"to {num(est['interaction'].max(), 2, True)})"
    )
    has(
        f"only {h2['n_treated']} of {h2['n_episodes']:,} episodes ({int(base['n_treated'])} of {int(base['n']):,})"
    )
    five = t[t["spec"].str.startswith("half-life from 5s grid")].iloc[0]
    has(
        f"gives {int(five['n_treated'])} episodes and an interaction of {num(five['interaction'], 2, True)}, "
        f"not significant (p = {five['p_interaction']:.2f})"
    )
    assert five["p_interaction"] > 0.05


def test_extension_and_data_notes(res):
    hl, chk = list(res["uni"]["half_life_by_grid_s"].values()), res["chk"]
    b, p, u = chk["bridge_latency"], chk["panel"], chk["units"]
    has(f"about {round(min(hl), -1):.0f}–{round(max(hl), -1):.0f} s (1 s, 5 s and 30 s grids)")
    sh = b["share_faster_than_half_life"]
    has(
        f"{100 * sh['uniswap_basis_1s']:.0f}% of the {b['n']:,} observed transfers settle faster than "
        f"{hl[0]:.0f} s, against {100 * sh['proxy_basis_1s']:.2f}% faster than the proxy's {res['h1']['half_life_s']:.2f} s"
    )
    has(
        f"(KS statistic {b['ks_statistic']:.2f}): the fitted median is {b['fitted_percentiles_s']['p50']:.0f} s but the "
        f"observed median is {b['empirical_percentiles_s']['p50']:.0f} s, and the fitted 99th percentile is "
        f"{b['fitted_percentiles_s']['p99']:,.0f} s against an observed {b['empirical_percentiles_s']['p99']:.0f} s"
    )
    has(
        f"marks {100 * p['share_spike_stored_flag']:.1f}% of ticks; the recomputed 95th-percentile rule marks "
        f"{100 * p['share_spike_recomputed_p95']:.1f}%"
    )
    has(f"(correlation {u['corr_binance_vs_price_l1']:.3f})")
    assert u["corr_binance_vs_price_l2"] > 0.999 - 1e-3
    assert p["basis_equals_close_minus_rolling50"] and abs(p["latency_lag1_autocorr"]) < 0.01
    has(f"The H1/H2 panel covers {pd.Timestamp(p['first']):%-d %B} – {pd.Timestamp(p['last']):%-d %B %Y}")


def test_lp_numbers(res):
    lp = res["chk"]["lp"]
    arb = lp["ledgers"]["arbitrum"]
    has(
        f"identify {lp['wallets_on_both_chains_all_time']:,} wallets active on both chains "
        f"({lp['wallets_on_both_chains_mar_jun_2023']:,} within March–June 2023); {lp['table_rows_multi_chain']:,} of the "
        f"{lp['table_rows']:,} wallets in the LP table ({100 * lp['table_share_multi_chain']:.1f}%)"
    )
    has(f"median 1.8·10⁶ s versus {lp['median_dwell_s_single']:.0f} s")
    assert round(lp["median_dwell_s_multi"] / 1e6, 1) == 1.8
    has(
        f"the difference in the table's loss column is not significant (p = {lp['controlled_ols_p_multi_chain']:.2f})"
    )
    has(f"median absolute value of {lp['table_median_abs_lvr'] / 1e22:.1f}·10²²")
    has(f"the {lp['table_rows_abs_lvr_below_1e4']:,} rows with |value| < 10⁴ are all exactly zero")
    assert lp["table_rows_abs_lvr_below_1e4"] == lp["table_rows_abs_lvr_below_1e4_exactly_zero"]
    has(f"{100 * arb['share_negative_cum_liquidity']:.0f}% of Arbitrum ledger rows")
    assert lp["table_unique_wallets"] == lp["table_rows"]


def _row(label, coef, se):
    return f"| {label} | {num(coef, 4)} | {num(se, 4)} |"


def test_estimated_equation_tables(res):
    c1 = res["h1"]["canonical"]["coefficients"]
    for label, key in [
        ("log depth, $\\beta$", "log_depth"),
        ("mean gas, $\\gamma_1$", "gas_mean"),
        ("gas-spike dummy, $\\gamma_2$", "spike_dummy"),
        ("intercept, $\\alpha$", "Intercept"),
    ]:
        has(_row(label, c1[key]["coef"], c1[key]["se_hc1"]))
    has(f"*N = {res['h1']['canonical']['n']:,}; adjusted R² = {res['h1']['canonical']['adj_r2']:.3f}.*")

    fit = res["h2"]["fits"]["retained"]
    c2 = fit["coefficients"]
    for label, key in [
        ("initial gap $b_0$, $\\theta$", "basis_start"),
        ("timing $I$, $\\delta$", "timing_I"),
        ("$b_0 \\times I$, $\\eta$", "basis_start:timing_I"),
        ("mean gas, $\\gamma_1$", "gas_mean"),
        ("gas-spike dummy, $\\gamma_2$", "spike_dummy"),
        ("intercept, $\\alpha$", "Intercept"),
    ]:
        has(_row(label, c2[key]["coef"], c2[key]["se_cluster"]))
    has(f"*N = {fit['n']:,}, of which {fit['n_treated']} treated.")


def test_method_section_parameters(res):
    b = res["chk"]["bridge_latency"]
    has(f"$\\hat\\mu = {b['lognormal_mu']:.3f}$, $\\hat\\sigma = {b['lognormal_sigma']:.3f}$")
    g = res["grid"].set_index("grid_s")
    has(
        f"\\hat\\varphi = {g.loc[1.0, 'phi']:.4f}\\ \\Rightarrow\\ \\hat\\tau_{{1/2}} = {g.loc[1.0, 'half_life_s']:.2f}"
    )
