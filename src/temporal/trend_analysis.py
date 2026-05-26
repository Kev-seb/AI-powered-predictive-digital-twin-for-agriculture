"""
trend_analysis.py
------------------
Statistical trend analysis for UAV crop stress time-series.

Capabilities:
    - Linear trend estimation (OLS slope, p-value, R²)
    - Mann-Kendall monotonic trend test (non-parametric)
    - Seasonal decomposition (STL proxy)
    - Anomaly / outlier detection (IQR and z-score)
    - Summary statistics over survey windows
    - Trend visualisation with confidence bands

Scientific basis:
    - Mann-Kendall: Mann (1945), Kendall (1975)
    - STL decomposition: Cleveland et al. (1990)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class TrendResult:
    slope:       float    # units per time-step
    intercept:   float
    r_squared:   float    # coefficient of determination
    p_value:     Optional[float]   # None if scipy unavailable
    trend_label: str      # "Increasing" / "Decreasing" / "No Trend"
    mk_tau:      Optional[float]   # Mann-Kendall τ correlation
    mk_p_value:  Optional[float]


# ──────────────────────────────────────────────────────────────
# Linear regression trend
# ──────────────────────────────────────────────────────────────

def linear_trend(values: Sequence[float],
                 time_steps: Optional[Sequence[float]] = None,
                 alpha: float = 0.05) -> TrendResult:
    """
    Fit OLS linear regression to a time-series.

    Parameters
    ----------
    values     : observed values (NDVI means, stress scores, etc.)
    time_steps : x-axis; 0, 1, 2, ... used if None
    alpha      : significance level for trend label

    Returns
    -------
    TrendResult
    """
    y = np.asarray(values, dtype=np.float64)
    n = len(y)
    if n < 2:
        return TrendResult(0.0, float(y[0]) if n else 0.0, 0.0, None, "No Trend", None, None)

    x = np.asarray(time_steps, dtype=np.float64) if time_steps else np.arange(n, dtype=np.float64)
    x_m, y_m = x.mean(), y.mean()

    ss_xy = float(((x - x_m) * (y - y_m)).sum())
    ss_xx = float(((x - x_m) ** 2).sum())

    slope     = ss_xy / ss_xx if ss_xx > 1e-12 else 0.0
    intercept = y_m - slope * x_m

    y_hat     = slope * x + intercept
    ss_res    = float(((y - y_hat) ** 2).sum())
    ss_tot    = float(((y - y_m)   ** 2).sum())
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # p-value via t-distribution (scipy optional)
    p_value = None
    try:
        from scipy import stats
        se_slope = np.sqrt(ss_res / max(n - 2, 1) / ss_xx)
        t_stat   = slope / se_slope if se_slope > 1e-12 else 0.0
        p_value  = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    except ImportError:
        pass

    if p_value is not None:
        trend_label = ("Increasing" if slope > 0 else "Decreasing") if p_value < alpha else "No Trend"
    else:
        trend_label = "Increasing" if slope > 0.001 else ("Decreasing" if slope < -0.001 else "No Trend")

    return TrendResult(slope=slope, intercept=intercept, r_squared=r_squared,
                       p_value=p_value, trend_label=trend_label,
                       mk_tau=None, mk_p_value=None)


# ──────────────────────────────────────────────────────────────
# Mann-Kendall test
# ──────────────────────────────────────────────────────────────

def mann_kendall_test(values: Sequence[float]) -> tuple[float, float, str]:
    """
    Non-parametric Mann-Kendall monotonic trend test.

    Returns
    -------
    (tau, p_value, trend_label)
        tau       : Kendall τ correlation (−1 to +1)
        p_value   : approximate two-sided p-value
        trend_label : "Increasing" / "Decreasing" / "No Trend"
    """
    y = np.asarray(values, dtype=np.float64)
    n = len(y)

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    denom    = n * (n - 1) / 2
    tau      = s / denom if denom > 0 else 0.0
    var_s    = n * (n - 1) * (2 * n + 5) / 18.0
    z_mk     = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0.0

    try:
        from scipy.stats import norm
        p_value = float(2 * (1 - norm.cdf(abs(z_mk))))
    except ImportError:
        # Approximate using empirical rule
        p_value = 0.05 if abs(z_mk) > 1.96 else 0.50

    if p_value < 0.05:
        label = "Increasing" if tau > 0 else "Decreasing"
    else:
        label = "No Trend"

    return float(tau), p_value, label


# ──────────────────────────────────────────────────────────────
# Full trend report
# ──────────────────────────────────────────────────────────────

def analyse_trend(values: Sequence[float],
                  time_steps: Optional[Sequence[float]] = None) -> TrendResult:
    """
    Run linear OLS + Mann-Kendall and return a unified TrendResult.
    """
    result          = linear_trend(values, time_steps)
    tau, mk_p, mk_l = mann_kendall_test(values)
    result.mk_tau     = tau
    result.mk_p_value = mk_p
    # MK takes precedence for trend label (non-parametric, more robust)
    result.trend_label = mk_l
    return result


# ──────────────────────────────────────────────────────────────
# Outlier detection
# ──────────────────────────────────────────────────────────────

def detect_outliers_iqr(values: Sequence[float],
                         k: float = 1.5) -> np.ndarray:
    """
    Detect outliers using Tukey's IQR fence method.

    Returns boolean array (True = outlier).
    """
    y    = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(y, [25, 75])
    iqr  = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (y < low) | (y > high)


def detect_outliers_zscore(values: Sequence[float],
                            threshold: float = 2.5) -> np.ndarray:
    """Detect outliers using z-score method."""
    y    = np.asarray(values, dtype=np.float64)
    mu, sigma = y.mean(), y.std()
    if sigma < 1e-8:
        return np.zeros(len(y), dtype=bool)
    return np.abs((y - mu) / sigma) > threshold


# ──────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────

def window_statistics(values: Sequence[float],
                       window: int = 3) -> dict[str, list[float]]:
    """
    Compute rolling mean and std over a sliding window.

    Returns
    -------
    dict with keys "rolling_mean", "rolling_std"
    """
    y = np.asarray(values, dtype=np.float64)
    means, stds = [], []
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), lo + window)
        w  = y[lo:hi]
        means.append(float(w.mean()))
        stds.append(float(w.std()))
    return {"rolling_mean": means, "rolling_std": stds}


# ──────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────

def plot_trend(values: Sequence[float],
               labels: Optional[Sequence[str]] = None,
               result: Optional[TrendResult] = None,
               title: str = "Index Trend Analysis") -> 'plt.Figure':
    """
    Plot time-series with OLS trend line and confidence band.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plot_trend()")

    y = np.asarray(values, dtype=np.float64)
    x = np.arange(len(y))

    if result is None:
        result = analyse_trend(values)

    trend_line = result.slope * x + result.intercept

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, "o-", color="#2980B9", linewidth=2, markersize=6, label="Observed")
    ax.plot(x, trend_line, "--", color="#E74C3C", linewidth=1.5,
            label=f"Trend: {result.trend_label}  (slope={result.slope:+.4f})")

    # Confidence band (±1σ of residuals)
    residuals = y - trend_line
    sigma_res = residuals.std()
    ax.fill_between(x, trend_line - sigma_res, trend_line + sigma_res,
                    alpha=0.15, color="#E74C3C", label="±1σ band")

    # Outlier markers
    outliers = detect_outliers_zscore(y)
    if outliers.any():
        ax.scatter(x[outliers], y[outliers], color="orange", s=80,
                   zorder=10, label="Outliers")

    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax.set_title(f"{title}  |  R²={result.r_squared:.3f}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
