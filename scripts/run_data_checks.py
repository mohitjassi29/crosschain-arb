"""Data checks cited in the README  ->  results/data_checks.json."""

from __future__ import annotations

import json

import pandas as pd

from _common import setup_logging, write_json
from crosschain_arb import checks
from crosschain_arb.config import RESULTS_DIR
from crosschain_arb.io import data_path, load_latency, load_panel


def main() -> None:
    log = setup_logging()
    panel = load_panel()
    dex = pd.read_parquet(data_path("dex_midprices.parquet"))

    half_lives = {}
    for name in ("h1.json", "uniswap_basis.json"):
        if not (RESULTS_DIR / name).exists():
            raise SystemExit(
                f"run scripts/run_h1.py and scripts/explore_uniswap_basis.py first ({name} missing)"
            )
    half_lives["proxy_basis_1s"] = json.loads((RESULTS_DIR / "h1.json").read_text())["half_life_s"]
    half_lives["uniswap_basis_1s"] = json.loads((RESULTS_DIR / "uniswap_basis.json").read_text())[
        "half_life_by_grid_s"
    ]["1s"]

    out = {
        "panel": checks.panel_checks(panel),
        "bridge_latency": checks.latency_fit_checks(load_latency(), half_lives),
        "units": checks.unit_checks(panel, dex),
        "lp": checks.lp_data_checks(
            pd.read_parquet(data_path("lvr_table.parquet")),
            pd.read_parquet(data_path("ledger_ethereum_long.parquet")),
            pd.read_parquet(data_path("ledger_arbitrum_long.parquet")),
        ),
    }
    write_json("data_checks.json", out)
    lat = out["bridge_latency"]
    log.info(
        "latency: observed median %.0fs vs fitted %.0fs (KS %.2f)",
        lat["empirical_percentiles_s"]["p50"],
        lat["fitted_percentiles_s"]["p50"],
        lat["ks_statistic"],
    )
    log.info(
        "share of observed transfers faster than the half-life: %s",
        {k: round(v, 3) for k, v in lat["share_faster_than_half_life"].items()},
    )


if __name__ == "__main__":
    main()
