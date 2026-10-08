"""Cross-chain ETH/USDC basis arbitrage: liquidity, latency and mean-reversion.

Modules
-------
config      paths, seeds and constants
io          input readers with helpful errors
panel       tick-level panel construction (Binance-based proxy series)
prices      Uniswap raw-price -> USD conversion and the Uniswap L2-L1 basis
halflife    AR(1) -> Ornstein-Uhlenbeck half-life
episodes    gap-episode construction and ROI over the latency window
models      H1 and H2 regressions
lp          H3: LP panel, LVR normalisation and cohort tests
plots       figures
"""

__version__ = "0.2.0"
