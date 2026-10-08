"""H3: do multi-chain LPs have different outcomes?  ->  results/h3.json.

Input is the processed LP panel (columns ``lp, chain, exit_ts, lvr, dwell, vol, gas_spike,
multi_chain``), e.g. produced by ``crosschain_arb.lp.build_lvr_panel`` from raw mint/burn exports.
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from _common import setup_logging, write_json
from crosschain_arb.io import data_path
from crosschain_arb.lp import (
    cohort_tests,
    multi_chain_ols,
    normalise_lvr,
    outcome_diagnostics,
    strict_negative_tail_fisher,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lp-panel", default="lp_panel.parquet")
    args = ap.parse_args()
    log = setup_logging()

    df = pd.read_parquet(data_path(args.lp_panel))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diag = outcome_diagnostics(df["lvr"])
    df = normalise_lvr(df)
    out = {
        "outcome_diagnostics": diag,
        "warnings": [str(w.message) for w in caught],
        "cohort_tests": cohort_tests(df),
        "ols": multi_chain_ols(df),
        "tail_fisher": strict_negative_tail_fisher(df),
    }
    log.info("share of exactly-zero outcomes: %.1f%%", 100 * diag["share_zero"])
    write_json("h3.json", out)


if __name__ == "__main__":
    main()
