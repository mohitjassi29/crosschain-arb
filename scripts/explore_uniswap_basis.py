"""Exploratory: half-life of the *Uniswap* L2-L1 basis (USD) on several grids.

The ~10 s half-life from ``run_h1`` comes from the Binance proxy series. This script estimates the
half-life of the actual cross-chain basis built from ``dex_midprices.parquet``. Forward-filled quotes
bias phi upward, so estimates are upper bounds; coarser grids reduce (but do not remove) that bias.
"""

from __future__ import annotations

import pandas as pd

from _common import setup_logging, write_json
from crosschain_arb.halflife import ar1_half_life
from crosschain_arb.io import data_path
from crosschain_arb.prices import resample_basis, to_usd


def main() -> None:
    log = setup_logging()
    usd = to_usd(pd.read_parquet(data_path("dex_midprices.parquet")))
    out = {
        "n_quotes": len(usd),
        "median_quote_gap_s": float(usd["time"].diff().dt.total_seconds().median()),
        "basis_usd": usd["basis"].describe(percentiles=[0.01, 0.5, 0.99]).to_dict(),
        "share_basis_positive": float((usd["basis"] > 0).mean()),
        "half_life_by_grid_s": {},
    }
    for freq, dt in [("1s", 1.0), ("5s", 5.0), ("30s", 30.0)]:
        hl = ar1_half_life(resample_basis(usd, freq), dt_seconds=dt, resample=False)
        out["half_life_by_grid_s"][freq] = hl.half_life_s
        log.info("grid %s: half-life %.1fs (phi=%.4f)", freq, hl.half_life_s, hl.phi)
    write_json("uniswap_basis.json", out)


if __name__ == "__main__":
    main()
