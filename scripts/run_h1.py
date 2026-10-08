"""H1: does deeper liquidity shorten gaps?  ->  results/h1.json + figures."""

from __future__ import annotations

import argparse

from _common import setup_logging, write_json
from crosschain_arb import plots
from crosschain_arb.config import FIGURES_DIR, MIN_EPISODE_SECONDS
from crosschain_arb.episodes import build_gap_episodes
from crosschain_arb.halflife import ar1_half_life
from crosschain_arb.io import load_panel
from crosschain_arb.models import fit_h1, logrank_by_depth_tercile
from crosschain_arb.panel import add_gas_spike


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="eth_gas_with_depth.parquet")
    ap.add_argument("--min-seconds", type=float, default=MIN_EPISODE_SECONDS)
    ap.add_argument("--spike-rule", choices=["global_p95", "day_top10"], default="global_p95")
    args = ap.parse_args()
    log = setup_logging()

    panel = add_gas_spike(load_panel(args.panel), rule=args.spike_rule)
    hl = ar1_half_life(panel.set_index("time")["basis_t"], dt_seconds=1.0)
    log.info("half-life %.3fs (phi=%.4f)", hl.half_life_s, hl.phi)

    ep = build_gap_episodes(panel, min_seconds=args.min_seconds)
    ep_all = build_gap_episodes(panel, min_seconds=1.0)
    out = {
        "half_life_s": hl.half_life_s,
        "n_episodes_ge_1s": len(ep_all),
        "n_episodes": len(ep),
        "canonical": fit_h1(ep).to_dict(),
        "winsorised_depth": fit_h1(ep_all, winsorize_depth=True).to_dict(),
        "logrank_depth_terciles": logrank_by_depth_tercile(ep_all),
        "spike_rule": args.spike_rule,
        "doubling_depth_effect_on_duration_pct": 100 * (2 ** fit_h1(ep).beta_log_depth - 1),
    }
    log.info(
        "H1 beta(log depth) = %.4f (HC1 SE %.4f)",
        out["canonical"]["beta_log_depth"],
        out["canonical"]["se_hc1"],
    )
    write_json("h1.json", out)
    plots.plot_h1_scatter(ep, FIGURES_DIR / "h1_scatter.png")
    plots.plot_survival_by_depth(ep_all, FIGURES_DIR / "h1_survival.png")


if __name__ == "__main__":
    main()
