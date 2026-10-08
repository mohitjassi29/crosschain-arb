# Data

Data files are not tracked (several are 20–65 MB, and the Binance and bridge-latency files are third-party data).
Place them here, or point `CCARB_DATA` at another folder.

| File | Used by | Description |
|---|---|---|
| `eth_gas_with_depth.parquet` | `run_h1`, `run_h2`, `run_robustness`, `run_data_checks`, notebooks 01, 02, 04, 06 | Tick-level panel: `basis_t`, `depth_t`, `latency_t` (simulated), `gas_gwei`, `SpikeDummy_t` (proxy series; see README, *Data notes*) |
| `eth_gas_merged_ms.parquet` | integration test | Binance 1 ms bars merged with the Ethereum base fee; input to `panel.build_proxy_panel` |
| `dex_midprices.parquet` | `run_data_checks`, notebooks 04, 05, `explore_uniswap_basis` | Uniswap v3 mid-prices (`mid_price_eth`, `mid_price_arb`) in raw tick units |
| `dune_latency_latest.csv` | `run_data_checks`, notebooks 04, 05 | Bridge send/receive latency export (`latency_seconds`) |
| `lvr_table.parquet` | `run_data_checks`, notebook 03 | One row per LP wallet; `LVR` in raw units |
| `ledger_ethereum_long.parquet`, `ledger_arbitrum_long.parquet` | `run_data_checks`, notebook 03 | Per-chain liquidity ledgers (2021–2024): `timestamp, lp_address, liquidity, cum_liquidity` |
| `lp_panel.parquet` *(optional)* | `run_h3` | Processed LP panel from `lp.build_lvr_panel` (needs raw mint/burn exports, not included) |
