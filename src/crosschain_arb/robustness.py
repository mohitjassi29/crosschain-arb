"""Robustness checks for H1 and H2.

Each row of the returned tables is tagged with a group:

* ``core``: the baseline specification and its direct variants (sample, spike flag, winsorising,
  latency trimming);
* ``extended``: additional checks (alternative spike rule, gap sign, regime splits, fixed effects,
  depth timing and proxy window, forward-depth falsification, half-life grids).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg

from .episodes import add_roi, add_timing_indicator, attach_future_basis, build_gap_episodes
from .halflife import ar1_half_life
from .models import H1_FORMULA, H2_FORMULA, winsorize
from .panel import add_gas_spike, with_depth

H1_COLS = ["spec", "source", "n", "beta", "se_hc1", "se_cluster", "p_cluster", "adj_r2"]


def _fit(df: pd.DataFrame, formula: str, term: str) -> dict:
    """OLS with HC1 and day-clustered SEs for one coefficient."""
    model = smf.ols(formula, data=df)
    hc1 = model.fit(cov_type="HC1")
    if df["day"].nunique() > 1:
        cl = model.fit(cov_type="cluster", cov_kwds={"groups": df["day"]})
        se_cl, p_cl = float(cl.bse[term]), float(cl.pvalues[term])
    else:  # a single cluster has no between-cluster variation
        se_cl = p_cl = float("nan")
    return {
        "n": int(hc1.nobs),
        "beta": float(hc1.params[term]),
        "se_hc1": float(hc1.bse[term]),
        "se_cluster": se_cl,
        "p_cluster": p_cl,
        "adj_r2": float(hc1.rsquared_adj),
    }


def h1_robustness(panel: pd.DataFrame) -> pd.DataFrame:
    """H1 depth elasticity under alternative samples, flags, proxies and controls.

    ``panel`` needs ``time, basis_t, latency_t, gas_gwei, depth, volume, SpikeDummy_t``.
    """
    rows: list[dict] = []

    def add(spec: str, source: str, **kw):
        rows.append({"spec": spec, "source": source, **kw})

    p95 = add_gas_spike(panel, "global_p95")
    fil = add_gas_spike(panel, "file")
    day = add_gas_spike(panel, "day_top10")

    # --- core: spike flag x minimum duration x winsorising ----------------------------------
    for label, pan in [("file spike flag", fil), ("recomputed p95 flag", p95)]:
        for mind in (1.0, 5.0):
            ep = build_gap_episodes(pan, min_seconds=mind)
            add(f"{label}, >= {mind:g}s", "core", **_fit(ep, H1_FORMULA, "log_depth"))
            ep_w = ep.assign(log_depth=np.log(winsorize(ep["depth_mean"])))
            add(f"{label}, >= {mind:g}s, winsorised depth", "core", **_fit(ep_w, H1_FORMULA, "log_depth"))

    base = build_gap_episodes(p95, min_seconds=5.0)

    # --- extended: spike rule, gap sign, gas regimes, day fixed effects ----------------------
    add(
        "within-day top-decile spike rule, >= 5s",
        "extended",
        **_fit(build_gap_episodes(day), H1_FORMULA, "log_depth"),
    )
    neg = build_gap_episodes(p95, sign="negative", min_seconds=5.0)
    add("negative-basis gaps, >= 5s", "extended", **_fit(neg, H1_FORMULA, "log_depth"))
    no_spike = "log_duration ~ log_depth + gas_mean"
    add("gas-spike episodes only", "extended", **_fit(base[base.spike_dummy == 1], no_spike, "log_depth"))
    add("non-spike episodes only", "extended", **_fit(base[base.spike_dummy == 0], no_spike, "log_depth"))
    add("day fixed effects", "extended", **_fit(base, H1_FORMULA + " + C(day)", "log_depth"))

    # --- extended: depth timing and proxy window ---------------------------------------------
    entry = base.assign(log_depth=np.log(base["depth_entry"].clip(lower=1e-12)))
    add(
        "depth at episode entry (trailing window, predetermined)",
        "extended",
        **_fit(entry, H1_FORMULA, "log_depth"),
    )
    fwd = build_gap_episodes(add_gas_spike(with_depth(panel, forward=True), "global_p95"), min_seconds=5.0)
    fwd_entry = fwd.assign(log_depth=np.log(fwd["depth_entry"].clip(lower=1e-12)))
    add(
        "FALSIFICATION: forward-looking depth at entry",
        "extended",
        **_fit(fwd_entry, H1_FORMULA, "log_depth"),
    )
    for w in (500, 2000):
        ep_w = build_gap_episodes(add_gas_spike(with_depth(panel, window=w), "global_p95"), min_seconds=5.0)
        add(f"depth proxy window = {w} ticks", "extended", **_fit(ep_w, H1_FORMULA, "log_depth"))

    return pd.DataFrame(rows)[H1_COLS]


def half_life_by_grid(panel: pd.DataFrame, grids: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """OU half-life of the basis proxy on several resampling grids."""
    s = panel.set_index("time")["basis_t"]
    rows = []
    for dt in grids:
        try:
            hl = ar1_half_life(s, dt_seconds=dt)
            rows.append({"grid_s": dt, "phi": hl.phi, "half_life_s": hl.half_life_s})
        except ValueError as exc:  # pragma: no cover - data dependent
            rows.append({"grid_s": dt, "phi": np.nan, "half_life_s": np.nan, "note": str(exc)})
    return pd.DataFrame(rows)


H2_COLS = ["spec", "source", "n", "n_treated", "level", "slope_untreated", "interaction", "p_interaction"]


def _h2_row(ep: pd.DataFrame, roi: str = "retained") -> dict:
    ep = add_roi(ep, roi)
    res = smf.ols(H2_FORMULA, data=ep).fit(cov_type="cluster", cov_kwds={"groups": ep["day"]})
    return {
        "n": int(res.nobs),
        "n_treated": int(ep["timing_I"].sum()),
        "level": float(res.params["timing_I"]),
        "slope_untreated": float(res.params["basis_start"]),
        "interaction": float(res.params["basis_start:timing_I"]),
        "p_interaction": float(res.pvalues["basis_start:timing_I"]),
    }


def h2_robustness(panel: pd.DataFrame, min_seconds: float = 1.0, min_treated: int = 2) -> dict:
    """H2 timing interaction under alternative definitions of treatment and sample.

    Uses the file spike flag and ``min_seconds = 1`` by default. ROI is the retained gap
    ``b_l - cost`` (cost = 0), i.e. the basis still open at settlement; the continuous-margin
    regression and the quantile regressions use the same outcome. Because ``retained = b0 - closure``,
    the level, interaction and margin coefficients under the ``closure`` outcome are the negatives
    of those reported here.
    Returns ``{"table", "margin", "quantile"}``.
    """
    pan = add_gas_spike(panel, "file")
    tau = ar1_half_life(pan.set_index("time")["basis_t"], dt_seconds=1.0).half_life_s
    ep0 = attach_future_basis(build_gap_episodes(pan, min_seconds=min_seconds), pan)
    rows: list[dict] = []

    def add(spec: str, source: str, ep: pd.DataFrame):
        if ep["timing_I"].sum() < min_treated or ep["timing_I"].sum() == len(ep):
            rows.append(
                {"spec": spec, "source": source, "n": len(ep), "n_treated": int(ep["timing_I"].sum())}
            )
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append({"spec": spec, "source": source, **_h2_row(ep)})

    base = add_timing_indicator(ep0, tau)
    add("baseline: I = 1{half-life > median latency}", "core", base)

    cap = base["latency_med"].quantile(0.99)
    add("trim latency at p99", "core", add_timing_indicator(ep0[ep0.latency_med <= cap], tau))

    p95 = ep0.assign(latency_med=ep0["latency_p95"])
    add("within-episode p95 latency in I", "extended", add_timing_indicator(p95, tau))

    for dt in (0.5, 2.0, 5.0):
        tau_g = ar1_half_life(pan.set_index("time")["basis_t"], dt_seconds=dt).half_life_s
        add(f"half-life from {dt:g}s grid ({tau_g:.1f}s)", "extended", add_timing_indicator(ep0, tau_g))

    add("episodes >= 5s only", "extended", add_timing_indicator(ep0[ep0.duration >= 5.0], tau))

    # no minimum duration at all (every gap with at least two distinct timestamps)
    ep_all = attach_future_basis(build_gap_episodes(pan, min_seconds=0.0), pan)
    add("all episodes, no minimum duration", "extended", add_timing_indicator(ep_all, tau))

    # continuous margin = half-life - latency, and quantile regressions
    marg = ep0.assign(margin=tau - ep0["latency_med"], roi=ep0["basis_future"])
    res = smf.ols("roi ~ margin + basis_start + gas_mean + spike_dummy", data=marg).fit(
        cov_type="cluster", cov_kwds={"groups": marg["day"]}
    )
    margin = {
        "beta_margin": float(res.params["margin"]),
        "se": float(res.bse["margin"]),
        "p": float(res.pvalues["margin"]),
    }

    q = add_timing_indicator(ep0, tau).assign(roi=ep0["basis_future"])
    X = pd.concat([q[["basis_start", "timing_I", "gas_mean", "spike_dummy"]]], axis=1)
    X.insert(0, "const", 1.0)
    quantile = {}
    for level in (0.25, 0.75):
        fit = QuantReg(q["roi"], X).fit(q=level)
        quantile[f"q{int(level * 100)}"] = {k: float(v) for k, v in fit.params.items()}

    return {
        "half_life_s": tau,
        "table": pd.DataFrame(rows),
        "margin": margin,
        "quantile": quantile,
    }
