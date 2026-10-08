"""Uniswap raw-price conversion and the cross-chain basis built from it.

``dex_midprices`` stores tick-derived prices in raw token units:

* ``mid_price_eth`` (Ethereum pool, token0 = USDC) is ETH-per-USDC in raw units,
  so USD/ETH = 1e12 / mid_price_eth.
* ``mid_price_arb`` (Arbitrum pool, token0 = WETH) is USDC-per-ETH in raw units,
  so USD/ETH = 1e12 * mid_price_arb.

Both conversions track Binance ETH/USDC (correlation ~0.999).
"""

from __future__ import annotations

import pandas as pd

from .config import DECIMALS_FACTOR


def to_usd(dex: pd.DataFrame) -> pd.DataFrame:
    """Return ``time, price_l1, price_l2, basis`` (USD; basis = L2 - L1)."""
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(dex["timestamp"]).dt.tz_localize(None),
            "price_l1": DECIMALS_FACTOR / dex["mid_price_eth"],
            "price_l2": DECIMALS_FACTOR * dex["mid_price_arb"],
        }
    )
    out["basis"] = out["price_l2"] - out["price_l1"]
    return out.sort_values("time", ignore_index=True)


def resample_basis(usd: pd.DataFrame, freq: str = "1s") -> pd.Series:
    """Last-observation-carried-forward basis on a uniform grid.

    Forward-filling stale quotes biases AR(1) persistence upward, so half-lives
    estimated from this series are upper bounds; compare several ``freq`` values.
    """
    return usd.set_index("time")["basis"].resample(freq).last().ffill()
