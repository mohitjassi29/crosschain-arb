"""Data checks whose results are cited in the README.

Every number the README states about the data comes from these functions (via
``scripts/run_data_checks.py`` -> ``results/data_checks.json``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .lp import ledger_diagnostics, multi_chain_wallets, normalise_lvr
from .prices import to_usd


def panel_checks(panel: pd.DataFrame, spike_quantile: float = 0.95) -> dict:
    """Time range and what the proxy series are, checked against their definitions."""
    ma = panel["close"].rolling(50).mean()
    ok = ma.notna()
    depth = panel["volume"].rolling(1000, min_periods=300).mean()
    both = depth.notna() & panel["depth"].notna()
    return {
        "rows": int(len(panel)),
        "first": str(panel["time"].min()),
        "last": str(panel["time"].max()),
        "basis_equals_close_minus_rolling50": bool(
            np.allclose((panel["close"] - ma)[ok], panel["basis_t"][ok], atol=1e-6)
        ),
        "corr_depth_vs_rolling_volume": float(np.corrcoef(depth[both], panel["depth"][both])[0, 1]),
        "latency_lag1_autocorr": float(panel["latency_t"].autocorr(1)),
        "panel_latency_percentiles_s": dict(
            zip(
                ["p50", "p90", "p99"],
                np.percentile(panel["latency_t"], [50, 90, 99]).round(1).tolist(),
                strict=True,
            )
        ),
        "share_spike_stored_flag": float(panel["SpikeDummy_t"].mean()),
        "share_spike_recomputed_p95": float(
            (panel["gas_gwei"] >= panel["gas_gwei"].quantile(spike_quantile)).mean()
        ),
    }


def latency_fit_checks(latency: pd.Series, half_lives_s: dict[str, float]) -> dict:
    """Lognormal MLE for observed bridge latency and how well it describes the data.

    ``half_lives_s`` maps a label to a half-life; for each, the share of *observed* transfers that
    settle faster than that half-life is reported (a data-based version of the timing condition).
    """
    x = latency[latency > 0].to_numpy(dtype=float)
    shape, _, scale = stats.lognorm.fit(x, floc=0)
    mu, sigma = float(np.log(scale)), float(shape)
    ks = stats.kstest(x, "lognorm", args=(shape, 0, scale))
    q = [50, 90, 99]
    return {
        "n": int(len(x)),
        "lognormal_mu": mu,
        "lognormal_sigma": sigma,
        "empirical_percentiles_s": dict(
            zip(["p50", "p90", "p99"], np.percentile(x, q).round(1).tolist(), strict=True)
        ),
        "fitted_percentiles_s": dict(
            zip(
                ["p50", "p90", "p99"],
                [float(np.exp(mu + stats.norm.ppf(p / 100) * sigma)) for p in q],
                strict=True,
            )
        ),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "share_faster_than_half_life": {k: float((x < v).mean()) for k, v in half_lives_s.items()},
    }


def unit_checks(panel: pd.DataFrame, dex: pd.DataFrame) -> dict:
    """Correlation of the USD-converted Uniswap prices with the Binance close."""
    usd = to_usd(dex)
    m = pd.merge_asof(
        panel[["time", "close"]], usd, on="time", direction="nearest", tolerance=pd.Timedelta("30s")
    ).dropna()
    return {
        "dex_first": str(usd["time"].min()),
        "dex_last": str(usd["time"].max()),
        "corr_binance_vs_price_l1": float(m["close"].corr(m["price_l1"])),
        "corr_binance_vs_price_l2": float(m["close"].corr(m["price_l2"])),
    }


def lp_data_checks(lvr_table: pd.DataFrame, ledger_a: pd.DataFrame, ledger_b: pd.DataFrame) -> dict:
    """What the LP table and ledgers can and cannot support."""
    lv = lvr_table["LVR"]
    small = lv.abs() < 1e4
    all_time = multi_chain_wallets(ledger_a, ledger_b)
    in_window = multi_chain_wallets(ledger_a, ledger_b, "2023-03-01", "2023-07-01")

    tbl = lvr_table.assign(multi_chain=lvr_table["lp_address"].str.lower().isin(all_time).astype(int))
    d = normalise_lvr(tbl.rename(columns={"LVR": "lvr", "close_ts": "exit_ts"}))
    d["log_dwell"] = np.log1p(d["dwell_s"])
    d["lvr_rel_w"] = d["lvr_rel"].clip(*d["lvr_rel"].quantile([0.01, 0.99]))
    fit = smf.ols("lvr_rel_w ~ multi_chain + log_dwell + C(chain)", d).fit(cov_type="HC1")
    multi, single = d[d.multi_chain == 1], d[d.multi_chain == 0]
    return {
        "table_rows": int(len(lvr_table)),
        "table_unique_wallets": int(lvr_table["lp_address"].nunique()),
        "table_median_abs_lvr": float(lv.abs().median()),
        "table_rows_abs_lvr_below_1e4": int(small.sum()),
        "table_rows_abs_lvr_below_1e4_exactly_zero": int((lv[small] == 0).sum()),
        "wallets_on_both_chains_all_time": len(all_time),
        "wallets_on_both_chains_mar_jun_2023": len(in_window),
        "table_rows_multi_chain": int(tbl["multi_chain"].sum()),
        "table_share_multi_chain": float(tbl["multi_chain"].mean()),
        "median_dwell_s_multi": float(multi["dwell_s"].median()),
        "median_dwell_s_single": float(single["dwell_s"].median()),
        "p_mannwhitney_dwell": float(stats.mannwhitneyu(multi["dwell_s"], single["dwell_s"]).pvalue),
        "controlled_ols_beta_multi_chain": float(fit.params["multi_chain"]),
        "controlled_ols_p_multi_chain": float(fit.pvalues["multi_chain"]),
        "ledgers": {"ethereum": ledger_diagnostics(ledger_a), "arbitrum": ledger_diagnostics(ledger_b)},
    }
