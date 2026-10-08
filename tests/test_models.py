import numpy as np
import pandas as pd
import pytest

from crosschain_arb.models import fit_h1, fit_h2, winsorize


def _synthetic_episodes(n: int = 4000, beta: float = -0.3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    depth = np.exp(rng.normal(-1, 0.5, n))
    ep = pd.DataFrame(
        {
            "depth_mean": depth,
            "log_depth": np.log(depth),
            "gas_mean": rng.uniform(10, 50, n),
            "spike_dummy": rng.integers(0, 2, n),
            "day": pd.to_datetime("2023-03-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        }
    )
    ep["log_duration"] = 5 + beta * ep["log_depth"] + rng.normal(0, 1, n)
    return ep


def test_fit_h1_recovers_elasticity():
    res = fit_h1(_synthetic_episodes())
    assert res.beta_log_depth == pytest.approx(-0.3, abs=0.06)
    assert res.se_hc1 > 0 and res.se_cluster > 0 and res.n == 4000


def test_winsorize_clips_tails():
    s = pd.Series(np.arange(1000.0))
    w = winsorize(s, 0.01)
    assert w.min() == pytest.approx(s.quantile(0.01)) and w.max() == pytest.approx(s.quantile(0.99))


def _h2_episodes(n_treated: int, n: int = 2000, slope_shift: float = 0.5, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ep = pd.DataFrame(
        {
            "basis_start": rng.exponential(1.0, n),
            "gas_mean": rng.uniform(10, 50, n),
            "spike_dummy": rng.integers(0, 2, n),
            "timing_I": 0,
            "day": pd.to_datetime("2023-03-01") + pd.to_timedelta(rng.integers(0, 60, n), unit="D"),
        }
    )
    ep.loc[ep.index[:n_treated], "timing_I"] = 1
    ep["roi"] = (
        1.0 * ep["basis_start"] + slope_shift * ep["basis_start"] * ep["timing_I"] + rng.normal(0, 0.3, n)
    )
    return ep


def test_fit_h2_recovers_interaction_with_enough_treated():
    res = fit_h2(_h2_episodes(n_treated=600))
    assert res.interaction == pytest.approx(0.5, abs=0.1)
    assert res.slope_treated == pytest.approx(res.slope_untreated + res.interaction)


def test_fit_h2_warns_when_few_treated():
    with pytest.warns(UserWarning, match="treated"):
        fit_h2(_h2_episodes(n_treated=5))


def test_fit_h2_requires_variation():
    with pytest.raises(ValueError):
        fit_h2(_h2_episodes(n_treated=0))
