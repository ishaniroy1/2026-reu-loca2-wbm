"""
trends.py
=========
Per-HUC8 trend analysis on the water-year S/P (snow/total-precip) ratio.

  * Mann-Kendall test + Theil-Sen slope for monotonic trend detection
    (robust to non-normal, autocorrelated hydro-climate series)
  * Flags watersheds with a statistically significant *decline*
  * Flags watersheds projected to approach/reach an effectively
    snow-free state (S/P ratio below NEAR_ZERO_SP_THRESHOLD)

Install: pip install pymannkendall
"""

import numpy as np
import pandas as pd
import pymannkendall as mk

import config


def mk_trend_for_series(years, values):
    """
    Run Mann-Kendall + Sen's slope on one HUC8's S/P ratio time series.
    Returns a dict of trend statistics; NaNs if too few valid points.
    """
    mask = ~np.isnan(values)
    years, values = np.asarray(years)[mask], np.asarray(values)[mask]
    if len(values) < 8:
        return dict(n=len(values), trend="insufficient_data", p=np.nan,
                     slope=np.nan, slope_per_decade=np.nan)

    result = mk.original_test(values, alpha=config.MK_ALPHA)
    return dict(
        n=len(values),
        trend=result.trend,               # 'increasing' / 'decreasing' / 'no trend'
        p=result.p,
        significant=bool(result.h),
        slope=result.slope,               # units per year
        slope_per_decade=result.slope * 10,
        tau=result.Tau,
    )


def huc8_sp_trends(sp_df, huc_col="huc8", year_col="water_year", val_col="sp_ratio"):
    """
    sp_df: tidy dataframe with one row per (huc8, water_year) containing
           the area-weighted S/P ratio for that watershed/year.

    Returns a per-HUC8 summary dataframe with trend stats and flags for:
      - significant_decline: MK-significant negative trend
      - reaches_near_zero: minimum (or late-period mean) S/P ratio below
        NEAR_ZERO_SP_THRESHOLD at any point in the series
    """
    records = []
    for huc, g in sp_df.groupby(huc_col):
        g = g.sort_values(year_col)
        stats = mk_trend_for_series(g[year_col].values, g[val_col].values)

        late_period = g[g[year_col] >= g[year_col].max() - 9]  # last 10 yrs
        late_mean = late_period[val_col].mean()
        min_val = g[val_col].min()

        records.append(dict(
            huc8=huc,
            n_years=stats["n"],
            trend=stats["trend"],
            p_value=stats["p"],
            significant=stats.get("significant", False),
            slope_per_decade=stats["slope_per_decade"],
            mean_sp_ratio=g[val_col].mean(),
            late_period_mean_sp_ratio=late_mean,
            min_sp_ratio=min_val,
            significant_decline=bool(
                stats.get("significant", False) and stats["slope_per_decade"] < 0
            ),
            reaches_near_zero=bool(min_val <= config.NEAR_ZERO_SP_THRESHOLD),
        ))

    return pd.DataFrame.from_records(records).sort_values(
        "slope_per_decade", ascending=True
    )


def combine_hist_future_trends(hist_trend_df, fut_trend_df, scenario):
    """
    Merge historical and future (per-scenario) trend tables on huc8,
    flagging watersheds that show significant decline in BOTH periods --
    the core "declining S/P ratio, historically and in the future" filter
    requested in the analysis plan.
    """
    merged = hist_trend_df.merge(
        fut_trend_df, on="huc8", suffixes=("_hist", f"_{scenario}")
    )
    merged["declining_both_periods"] = (
        merged["significant_decline_hist"] & merged[f"significant_decline_{scenario}"]
    )
    merged["scenario"] = scenario
    return merged
