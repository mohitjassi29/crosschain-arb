"""Robustness checks for H1 and H2  ->  results/robustness_h1.csv, robustness_h2.csv, robustness.json."""

from __future__ import annotations

from _common import setup_logging, write_json
from crosschain_arb import robustness as R
from crosschain_arb.config import RESULTS_DIR
from crosschain_arb.io import load_panel


def main() -> None:
    log = setup_logging()
    panel = load_panel()

    h1 = R.h1_robustness(panel)
    h2 = R.h2_robustness(panel)
    grid = R.half_life_by_grid(panel)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    h1.to_csv(RESULTS_DIR / "robustness_h1.csv", index=False)
    h2["table"].to_csv(RESULTS_DIR / "robustness_h2.csv", index=False)
    grid.to_csv(RESULTS_DIR / "half_life_by_grid.csv", index=False)
    write_json(
        "robustness.json",
        {"h2_margin": h2["margin"], "h2_quantile": h2["quantile"], "half_life_s": h2["half_life_s"]},
    )
    log.info("H1: %d specs; all betas negative: %s", len(h1), bool((h1["beta"] < 0).all()))
    log.info(
        "half-life by grid (s): %s", dict(zip(grid["grid_s"], grid["half_life_s"].round(1), strict=True))
    )


if __name__ == "__main__":
    main()
