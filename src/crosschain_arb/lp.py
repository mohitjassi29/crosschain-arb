"""H3: liquidity-provider panel, LVR normalisation and cohort tests.

Definition used
---------------
``lvr`` here is ``value_entry - value_exit`` where both token bundles (the full
position at the first mint and at the last burn) are valued at the *exit* price.
This is a change in position value at a fixed price, **not** loss-versus-
rebalancing in the sense of Milionis et al. (2022); a proper LVR needs pool state
and token amounts. The column keeps the name ``lvr`` for convenience.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact, ks_2samp, mannwhitneyu


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def tick_to_price(tick: np.ndarray | pd.Series) -> np.ndarray:
    """Uniswap v3 tick -> raw price (1.0001 ** tick)."""
    return 1.0001 ** np.asarray(tick, dtype=float)


def position_amounts(liquidity: np.ndarray, lower_tick: np.ndarray, upper_tick: np.ndarray):
    """Token amounts (x, y) of a Uniswap v3 position of liquidity L on the band [P_l, P_u].

    x = L (1/sqrt(P_l) - 1/sqrt(P_u))   (token0),    y = L (sqrt(P_u) - sqrt(P_l))   (token1).
    These are the amounts when the price is at or beyond the band edge (the whole band converted);
    the formulas ignore where the current price sits inside the band.
    """
    sl, su = np.sqrt(tick_to_price(lower_tick)), np.sqrt(tick_to_price(upper_tick))
    q = np.asarray(liquidity, dtype=float)
    return q * (1 / sl - 1 / su), q * (su - sl)


def summarise_lp_events(mints: pd.DataFrame, burns: pd.DataFrame, chain: str) -> pd.DataFrame:
    """One row per LP wallet: first mint (entry) and last burn (exit) on a chain."""
    entry = mints.groupby("origin").agg(
        entry_ts=("timestamp", "min"),
        L0=("liquidity", "first"),
        lower_tick=("lower_tick", "first"),
        upper_tick=("upper_tick", "first"),
        gas0=("gas_gwei", "first"),
    )
    exit_ = burns.groupby("origin").agg(
        exit_ts=("timestamp", "max"),
        L1=("liquidity", "last"),
        lower_tick_exit=("lower_tick", "last"),
        upper_tick_exit=("upper_tick", "last"),
    )
    out = entry.join(exit_, how="inner").reset_index()
    out["chain"] = chain
    return out


def realised_volatility(prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """std of simple returns over [start, end] times sqrt(number of price points)."""
    window = prices.loc[start:end]
    return float(window.pct_change().std() * np.sqrt(len(window)))


def build_lvr_panel(
    lp: pd.DataFrame,
    prices: pd.Series,
    multi_chain_ids: set,
    gas_spike_threshold: float,
    lvr_abs_max: float | None = 1e4,
) -> pd.DataFrame:
    """Compute LVR and covariates for LP summaries from :func:`summarise_lp_events`.

    ``prices`` is an ascending time-indexed price series. Rows with
    ``|lvr| >= lvr_abs_max`` are dropped (set to ``None`` to keep everything); see
    :func:`outcome_diagnostics` for why this filter deserves scrutiny.
    """
    df = lp.copy()
    idx = prices.index
    df["entry_price"] = prices.to_numpy()[idx.get_indexer(df["entry_ts"], method="nearest")]
    df["exit_price"] = prices.to_numpy()[idx.get_indexer(df["exit_ts"], method="nearest")]

    dx0, dy0 = position_amounts(df["L0"], df["lower_tick"], df["upper_tick"])
    dx1, dy1 = position_amounts(df["L1"], df["lower_tick_exit"], df["upper_tick_exit"])
    token0_is_usdc = df["chain"].eq("eth").to_numpy()
    amt0, amt1 = np.where(token0_is_usdc, dx0, dy0), np.where(token0_is_usdc, dy0, dx0)
    amt0x, amt1x = np.where(token0_is_usdc, dx1, dy1), np.where(token0_is_usdc, dy1, dx1)

    df["lvr"] = (amt0 + amt1 * df["exit_price"]) - (amt0x + amt1x * df["exit_price"])
    df["dwell"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds()
    spans = zip(df["entry_ts"], df["exit_ts"], strict=True)
    df["vol"] = [realised_volatility(prices, start, end) for start, end in spans]
    df["gas_spike"] = (df["gas0"].fillna(0) >= gas_spike_threshold).astype(int)
    df["multi_chain"] = df["origin"].isin(multi_chain_ids).astype(int)

    df = df.rename(columns={"origin": "lp"}).replace([np.inf, -np.inf], np.nan)
    df = df[df["vol"].notna()]
    if lvr_abs_max is not None:
        df = df[df["lvr"].abs() < lvr_abs_max]
    return df[["lp", "chain", "exit_ts", "lvr", "dwell", "vol", "gas_spike", "multi_chain"]].reset_index(
        drop=True
    )


# --------------------------------------------------------------------------- #
# Normalisation and tests
# --------------------------------------------------------------------------- #
def robust_scale(s: pd.Series) -> float:
    """Dispersion with fallbacks: MAD -> IQR/1.349 -> std -> 1.0.

    The result is a *relative* scale (not calibrated to sigma: MAD would need the
    1.4826 factor). Values are not centred when used for ``lvr_rel``.
    """
    med = s.median()
    mad = (s - med).abs().median()
    if np.isfinite(mad) and mad > 0:
        return float(mad)
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if np.isfinite(iqr) and iqr > 0:
        return float(iqr / 1.349)
    std = s.std()
    return float(std) if np.isfinite(std) and std > 0 else 1.0


def normalise_lvr(df: pd.DataFrame, by: tuple[str, ...] = ("chain", "week")) -> pd.DataFrame:
    """Add ``week`` and ``lvr_rel`` = lvr / robust_scale within ``by`` blocks."""
    out = df.copy()
    out["week"] = pd.to_datetime(out["exit_ts"]).dt.to_period("W").dt.start_time
    out["scale"] = out.groupby(list(by))["lvr"].transform(robust_scale)
    out["lvr_rel"] = out["lvr"] / out["scale"]
    return out


def outcome_diagnostics(lvr: pd.Series) -> dict:
    """Summary of how informative the outcome is (share of exact zeros, sign shares)."""
    s = lvr.dropna()
    diag = {
        "n": int(len(s)),
        "share_zero": float((s == 0).mean()),
        "share_negative": float((s < 0).mean()),
        "share_positive": float((s > 0).mean()),
        "abs_median": float(s.abs().median()),
        "abs_p99": float(s.abs().quantile(0.99)),
    }
    if diag["share_zero"] > 0.5:
        warnings.warn(
            f"{diag['share_zero']:.0%} of the outcome is exactly zero; cohort tests have little to compare",
            stacklevel=2,
        )
    return diag


def cohort_tests(df: pd.DataFrame, col: str = "lvr_rel") -> dict:
    """Mann-Whitney and KS tests of multi-chain vs single-chain distributions."""
    multi = df.loc[df["multi_chain"] == 1, col].dropna()
    single = df.loc[df["multi_chain"] == 0, col].dropna()
    if multi.empty or single.empty:
        return {"n_single": len(single), "n_multi": len(multi), "p_mannwhitney": np.nan, "p_ks": np.nan}
    return {
        "n_single": int(len(single)),
        "n_multi": int(len(multi)),
        "p_mannwhitney": float(mannwhitneyu(multi, single, alternative="two-sided").pvalue),
        "p_ks": float(ks_2samp(multi, single).pvalue),
    }


def multi_chain_ols(df: pd.DataFrame, outcome: str = "lvr_rel") -> dict:
    """``outcome ~ multi_chain + dwell + vol`` with LP-clustered standard errors."""
    reg = df.dropna(subset=[outcome, "multi_chain", "dwell", "vol", "lp"])
    res = smf.ols(f"{outcome} ~ multi_chain + dwell + vol", data=reg).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(reg["lp"])[0]}
    )
    return {
        "n": int(res.nobs),
        "beta_multi_chain": float(res.params["multi_chain"]),
        "se": float(res.bse["multi_chain"]),
        "p": float(res.pvalues["multi_chain"]),
    }


def strict_negative_tail_fisher(df: pd.DataFrame, col: str = "lvr", pct: float = 0.05) -> dict:
    """Fisher exact test on a left tail defined as the ``pct`` quantile of negative values."""
    neg = df.loc[df[col] < 0, col]
    if neg.empty:
        return {"threshold": np.nan, "table": [[0, 0], [0, 0]], "odds_ratio": np.nan, "p": np.nan}
    thr = float(neg.quantile(pct))
    tail = (df[col] <= thr).astype(int)
    table = pd.crosstab(df["multi_chain"], tail).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    odds, p = fisher_exact(table.to_numpy(), alternative="two-sided")
    return {"threshold": thr, "table": table.to_numpy().tolist(), "odds_ratio": float(odds), "p": float(p)}


# --------------------------------------------------------------------------- #
# Cohorts from per-chain liquidity ledgers
# --------------------------------------------------------------------------- #
def multi_chain_wallets(
    ledger_a: pd.DataFrame,
    ledger_b: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> set[str]:
    """Lower-cased wallets with at least one event on *both* chains.

    Ledgers need ``timestamp`` and ``lp_address``. With ``start``/``end`` only events inside the
    window count, which is closer to "active on both chains in the same period" than an
    all-time overlap (the shipped ledgers span 2021-2024).
    """

    def wallets(ledger: pd.DataFrame) -> set[str]:
        d = ledger
        if start is not None:
            d = d[d["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            d = d[d["timestamp"] < pd.Timestamp(end)]
        return set(d["lp_address"].str.lower())

    return wallets(ledger_a) & wallets(ledger_b)


def ledger_diagnostics(ledger: pd.DataFrame, start: str = "2023-03-01", end: str = "2023-07-01") -> dict:
    """Coverage and consistency of a long liquidity ledger.

    A negative cumulative liquidity is impossible for a real position; a large share means burns are
    not matched to the mints that opened them (history before the sample, or ``origin`` being the
    transaction sender rather than the position owner).
    """
    in_window = (ledger["timestamp"] >= pd.Timestamp(start)) & (ledger["timestamp"] < pd.Timestamp(end))
    return {
        "rows": int(len(ledger)),
        "wallets": int(ledger["lp_address"].nunique()),
        "first": str(ledger["timestamp"].min()),
        "last": str(ledger["timestamp"].max()),
        "share_rows_in_window": float(in_window.mean()),
        "wallets_in_window": int(ledger.loc[in_window, "lp_address"].nunique()),
        "share_negative_cum_liquidity": float((ledger["cum_liquidity"] < 0).mean()),
    }
