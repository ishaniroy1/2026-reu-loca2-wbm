"""
runoff_metrics.py
==================
Derive spring snowmelt-driven runoff timing and magnitude metrics per
HUC8 watershed and water year, from a daily snowmelt flux (either the
observed SNODAS `snowMelt` variable historically, or the simulated
`snowmelt_sim` from snowpack_model.py for future periods).

Metrics computed (standard snow-hydrology definitions, e.g. Stewart et
al. 2005; Regonda et al. 2005):

  * center_of_timing (CT): day-of-water-year by which 50% of the total
    Mar-Jul cumulative snowmelt has occurred. Earlier CT = earlier spring
    runoff pulse.
  * spring_melt_total: total Mar-Jul melt volume (mm), a magnitude proxy
    for the spring runoff pulse.
  * peak_melt_rate: maximum daily melt rate (mm/day) within the spring
    window, and the day-of-water-year on which it occurs.
"""

import numpy as np
import pandas as pd

import config
from io_utils import water_year


def _day_of_water_year(dates):
    """Day-of-water-year (1 = Oct 1) for a DatetimeIndex."""
    wy_start = pd.to_datetime(
        np.where(dates.month >= 10,
                 [f"{y}-10-01" for y in dates.year],
                 [f"{y-1}-10-01" for y in dates.year])
    )
    return (dates - wy_start).days + 1


def compute_spring_runoff_metrics(melt_series_df, huc_col="huc8", date_col="time",
                                   melt_col="melt"):
    """
    melt_series_df: tidy dataframe with columns [huc_col, date_col, melt_col]
                     giving the area-weighted HUC8 daily melt flux (mm/day).

    Returns a per-HUC8, per-water-year dataframe with center_of_timing,
    spring_melt_total, peak_melt_rate, and peak_melt_doy.
    """
    df = melt_series_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["water_year"] = water_year(pd.DatetimeIndex(df[date_col]))
    df["month"] = df[date_col].dt.month
    df["doy_wy"] = _day_of_water_year(pd.DatetimeIndex(df[date_col]))

    spring = df[df["month"].isin(config.SPRING_MONTHS)].copy()

    records = []
    for (huc, wy), g in spring.groupby([huc_col, "water_year"]):
        g = g.sort_values(date_col)
        cum = g[melt_col].cumsum()
        total = cum.iloc[-1] if len(cum) else np.nan

        if total is not None and total > 0:
            half = total / 2.0
            ct_idx = np.searchsorted(cum.values, half)
            ct_idx = min(ct_idx, len(g) - 1)
            center_of_timing = g["doy_wy"].iloc[ct_idx]
        else:
            center_of_timing = np.nan

        if len(g):
            peak_idx = g[melt_col].values.argmax()
            peak_rate = g[melt_col].iloc[peak_idx]
            peak_doy = g["doy_wy"].iloc[peak_idx]
        else:
            peak_rate, peak_doy = np.nan, np.nan

        records.append(dict(
            huc8=huc,
            water_year=wy,
            center_of_timing_doy=center_of_timing,
            spring_melt_total_mm=total,
            peak_melt_rate_mm_day=peak_rate,
            peak_melt_doy=peak_doy,
        ))

    return pd.DataFrame.from_records(records).sort_values(["huc8", "water_year"])


def summarize_shift(metrics_df, period_a_years, period_b_years, huc_col="huc8"):
    """
    Compare mean CT and spring melt magnitude between two periods (e.g.
    historical baseline vs. end-of-century) per HUC8.

    period_a_years / period_b_years: (start_year, end_year) tuples of
    water years to average over.
    """
    def _period_mean(df, yrs):
        sub = df[(df["water_year"] >= yrs[0]) & (df["water_year"] <= yrs[1])]
        return sub.groupby(huc_col)[
            ["center_of_timing_doy", "spring_melt_total_mm", "peak_melt_rate_mm_day"]
        ].mean()

    a = _period_mean(metrics_df, period_a_years).add_suffix("_periodA")
    b = _period_mean(metrics_df, period_b_years).add_suffix("_periodB")
    out = a.join(b, how="outer")

    out["shift_center_of_timing_days"] = (
        out["center_of_timing_doy_periodB"] - out["center_of_timing_doy_periodA"]
    )
    out["shift_spring_melt_total_pct"] = (
        100 * (out["spring_melt_total_mm_periodB"] - out["spring_melt_total_mm_periodA"])
        / out["spring_melt_total_mm_periodA"]
    )
    return out.reset_index()
