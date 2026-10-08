"""H1 (depth -> duration) and H2 (timing x gap -> ROI) regressions."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

H1_FORMULA = "log_duration ~ log_depth + gas_mean + spike_dummy"
H2_FORMULA = "roi ~ basis_start * timing_I + gas_mean + spike_dummy"


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    """Clip a series at its ``p`` and ``1-p`` quantiles."""
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


@dataclass(frozen=True)
class H1Result:
    n: int
    beta_log_depth: float
    se_hc1: float
    se_cluster: float
    p_hc1: float
    p_cluster: float
    adj_r2: float
    coefficients: dict  # name -> {"coef", "se_hc1"} for every regressor

    def to_dict(self) -> dict:
        return asdict(self)


def fit_h1(episodes: pd.DataFrame, winsorize_depth: bool = False, p: float = 0.01) -> H1Result:
    """OLS of log(duration) on log(depth), gas and spike dummy.

    Reports HC1 and day-clustered standard errors for the depth elasticity.
    """
    ep = episodes.copy()
    if winsorize_depth:
        ep["log_depth"] = np.log(winsorize(ep["depth_mean"], p))
    model = smf.ols(H1_FORMULA, data=ep)
    hc1 = model.fit(cov_type="HC1")
    cl = model.fit(cov_type="cluster", cov_kwds={"groups": ep["day"]})
    return H1Result(
        n=int(hc1.nobs),
        beta_log_depth=float(hc1.params["log_depth"]),
        se_hc1=float(hc1.bse["log_depth"]),
        se_cluster=float(cl.bse["log_depth"]),
        p_hc1=float(hc1.pvalues["log_depth"]),
        p_cluster=float(cl.pvalues["log_depth"]),
        adj_r2=float(hc1.rsquared_adj),
        coefficients={
            k: {"coef": float(hc1.params[k]), "se_hc1": float(hc1.bse[k])} for k in hc1.params.index
        },
    )


def logrank_by_depth_tercile(episodes: pd.DataFrame) -> dict:
    """Log-rank test of duration between the lowest and highest depth terciles."""
    from lifelines.statistics import logrank_test

    q = episodes["depth_mean"].quantile([1 / 3, 2 / 3]).to_numpy()
    low = episodes.loc[episodes["depth_mean"] <= q[0], "duration"]
    high = episodes.loc[episodes["depth_mean"] >= q[1], "duration"]
    return {
        "median_duration_low": float(low.median()),
        "median_duration_high": float(high.median()),
        "p_logrank": float(logrank_test(low, high).p_value),
    }


@dataclass(frozen=True)
class H2Result:
    n: int
    n_treated: int
    level: float
    slope_untreated: float
    interaction: float
    se_interaction: float
    p_interaction: float
    coefficients: dict  # name -> {"coef", "se_cluster"} for every regressor

    @property
    def slope_treated(self) -> float:
        return self.slope_untreated + self.interaction

    def to_dict(self) -> dict:
        return {**asdict(self), "slope_treated": self.slope_treated}


def fit_h2(episodes: pd.DataFrame, min_treated: int = 30) -> H2Result:
    """Interaction model ``roi ~ basis_start * timing_I + controls`` (day-clustered).

    Warns when fewer than ``min_treated`` episodes have ``timing_I == 1``: the
    interaction is then identified from a handful of observations.
    """
    n_treated = int(episodes["timing_I"].sum())
    if n_treated == 0 or n_treated == len(episodes):
        raise ValueError("timing_I has no variation; the interaction is not identified")
    if n_treated < min_treated:
        warnings.warn(
            f"only {n_treated} treated episodes (timing_I == 1) out of {len(episodes)}; "
            "the interaction estimate is not reliable",
            stacklevel=2,
        )
    res = smf.ols(H2_FORMULA, data=episodes).fit(cov_type="cluster", cov_kwds={"groups": episodes["day"]})
    key = "basis_start:timing_I"
    return H2Result(
        n=int(res.nobs),
        n_treated=n_treated,
        level=float(res.params["timing_I"]),
        slope_untreated=float(res.params["basis_start"]),
        interaction=float(res.params[key]),
        se_interaction=float(res.bse[key]),
        p_interaction=float(res.pvalues[key]),
        coefficients={
            k: {"coef": float(res.params[k]), "se_cluster": float(res.bse[k])} for k in res.params.index
        },
    )
