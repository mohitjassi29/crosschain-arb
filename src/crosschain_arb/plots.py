"""Figures (matplotlib, headless-safe)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from .models import H1_FORMULA, H2_FORMULA  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_h1_scatter(episodes: pd.DataFrame, path: Path, sample: int = 5000, seed: int = 42) -> Path:
    """log(duration) vs log(depth) with the fitted regression line."""
    fit = smf.ols(H1_FORMULA, data=episodes).fit(cov_type="HC1")
    s = episodes.sample(min(sample, len(episodes)), random_state=seed)
    x = np.linspace(s["log_depth"].min(), s["log_depth"].max(), 100)
    mean_gas, mean_spike = episodes["gas_mean"].mean(), episodes["spike_dummy"].mean()
    y = (
        fit.params["Intercept"]
        + fit.params["log_depth"] * x
        + fit.params["gas_mean"] * mean_gas
        + fit.params["spike_dummy"] * mean_spike
    )
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(s["log_depth"], s["log_duration"], s=4, alpha=0.3)
    ax.plot(x, y, color="C3", label=f"slope = {fit.params['log_depth']:.3f}")
    ax.set(xlabel="log depth (proxy)", ylabel="log duration (s)", title="H1: gap duration vs depth")
    ax.legend()
    return _save(fig, path)


def plot_survival_by_depth(episodes: pd.DataFrame, path: Path) -> Path:
    """Empirical survival curves of gap duration for the lowest and highest depth terciles."""
    q = episodes["depth_mean"].quantile([1 / 3, 2 / 3]).to_numpy()
    groups = {
        "Low depth": episodes.loc[episodes["depth_mean"] <= q[0], "duration"],
        "High depth": episodes.loc[episodes["depth_mean"] >= q[1], "duration"],
    }
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for label, d in groups.items():
        x = np.sort(d.to_numpy())
        ax.step(x, 1 - np.arange(1, len(x) + 1) / len(x), where="post", label=f"{label} (n={len(x):,})")
    ax.set(
        xscale="log",
        xlabel="duration (s)",
        ylabel="P(gap persists)",
        title="H1: gap persistence by depth tercile",
    )
    ax.legend()
    return _save(fig, path)


def plot_latency_cdf(latency: pd.Series, path: Path) -> Path:
    """Empirical CDF of latency with median / p90 / p99 markers."""
    x = np.sort(latency.dropna().to_numpy())
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(x, np.arange(1, len(x) + 1) / len(x))
    for p, name in [(50, "median"), (90, "p90"), (99, "p99")]:
        v = np.percentile(x, p)
        ax.axvline(v, ls="--", lw=1)
        ax.text(v, 0.02, f"{name} ≈ {v:.0f}s", rotation=90, ha="right", va="bottom")
    ax.set(xlim=(0, np.percentile(x, 99.9)), xlabel="latency (s)", ylabel="empirical CDF")
    return _save(fig, path)


def plot_ou_decay(half_life_s: float, path: Path, t_max: float = 60.0) -> Path:
    """Fraction of the initial gap remaining under exponential decay with the given half-life."""
    t = np.linspace(0, t_max, 300)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(t, 2.0 ** (-t / half_life_s))
    ax.axhline(0.5, ls="--", lw=1)
    ax.axvline(half_life_s, ls="--", lw=1)
    ax.set(
        xlabel="time (s)", ylabel="fraction of gap remaining", title=f"OU decay, half-life {half_life_s:.2f}s"
    )
    return _save(fig, path)


def plot_h2_roi_vs_gap(episodes: pd.DataFrame, path: Path, bins: int = 12) -> Path:
    """Binned mean ROI against the initial gap, split by timing, with model-implied lines."""
    fit = smf.ols(H2_FORMULA, data=episodes).fit(cov_type="cluster", cov_kwds={"groups": episodes["day"]})
    g_bar, s_bar = episodes["gas_mean"].mean(), episodes["spike_dummy"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for flag, color in [(0, "C0"), (1, "C1")]:
        sub = episodes[episodes["timing_I"] == flag]
        if sub.empty:
            continue
        label = f"I={flag} (n={len(sub):,})"
        if sub["basis_start"].nunique() > bins:
            b = sub.assign(bin=pd.qcut(sub["basis_start"], bins, duplicates="drop")).groupby(
                "bin", observed=True
            )
            ax.scatter(b["basis_start"].mean(), b["roi"].mean(), s=18, color=color, label=label)
        else:
            ax.scatter(sub["basis_start"], sub["roi"], s=18, color=color, label=label)
        grid = pd.DataFrame(
            {
                "basis_start": np.linspace(sub["basis_start"].min(), sub["basis_start"].max(), 100),
                "timing_I": flag,
                "gas_mean": g_bar,
                "spike_dummy": s_bar,
            }
        )
        ax.plot(grid["basis_start"], fit.predict(grid), color=color)
    ax.axhline(0, ls="--", lw=1, color="grey")
    ax.set_xlim(
        0, episodes["basis_start"].quantile(0.995)
    )  # a few extreme gaps would otherwise flatten the plot
    ax.set(
        xlabel="initial gap (basis_start)",
        ylabel="ROI",
        title="H2: ROI vs initial gap (fitted lines from the model)",
    )
    ax.legend()
    return _save(fig, path)
