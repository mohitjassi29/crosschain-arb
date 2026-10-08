"""H2: does timing (half-life > latency) change how much of the gap is retained?  ->  results/h2.json."""

from __future__ import annotations

import argparse
import warnings

from _common import setup_logging, write_json
from crosschain_arb import plots
from crosschain_arb.config import FIGURES_DIR, MIN_EPISODE_SECONDS
from crosschain_arb.episodes import add_roi, add_timing_indicator, attach_future_basis, build_gap_episodes
from crosschain_arb.halflife import ar1_half_life
from crosschain_arb.io import load_panel
from crosschain_arb.models import fit_h2
from crosschain_arb.panel import add_gas_spike


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", default="eth_gas_with_depth.parquet")
    ap.add_argument("--min-seconds", type=float, default=MIN_EPISODE_SECONDS)
    args = ap.parse_args()
    log = setup_logging()

    panel = add_gas_spike(load_panel(args.panel))
    tau = ar1_half_life(panel.set_index("time")["basis_t"], dt_seconds=1.0).half_life_s
    log.info("half-life %.3fs", tau)

    base = add_timing_indicator(
        attach_future_basis(build_gap_episodes(panel, min_seconds=args.min_seconds), panel), tau
    )
    results: dict = {
        "half_life_s": tau,
        "n_episodes": len(base),
        "n_treated": int(base["timing_I"].sum()),
        "median_episode_latency_s": float(base["latency_med"].median()),
        "fits": {},
    }
    for definition in ("retained", "closure"):
        ep = add_roi(base, definition)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_h2(ep)
        results["fits"][definition] = {**fit.to_dict(), "warnings": [str(w.message) for w in caught]}
        log.info(
            "%s outcome: interaction %+.3f (p=%.3f), treated=%d",
            definition,
            fit.interaction,
            fit.p_interaction,
            fit.n_treated,
        )
    results["mean_retained_gap"] = float(add_roi(base, "retained")["roi"].mean())
    write_json("h2.json", results)

    plots.plot_ou_decay(tau, FIGURES_DIR / "h2_ou_decay.png")
    plots.plot_latency_cdf(panel["latency_t"], FIGURES_DIR / "h2_latency_cdf.png")
    plots.plot_h2_roi_vs_gap(add_roi(base, "retained"), FIGURES_DIR / "h2_roi_vs_gap.png")


if __name__ == "__main__":
    main()
