"""
phase_partition.py
===================
Partition daily total precipitation into snow and rain fractions using a
linear ramp between T_ALL_SNOW and T_ALL_RAIN (default -1C / +1C), then
build water-year S/P (snow-to-total-precipitation) ratios.

This mirrors the standard SNOW-17 / VIC-style partitioning approach:

    snow_frac(T) = 1                              if T <= T_ALL_SNOW
                 = 0                              if T >= T_ALL_RAIN
                 = (T_ALL_RAIN - T) / (T_ALL_RAIN - T_ALL_SNOW)   otherwise

    snow  = pr * snow_frac(T)
    rain  = pr * (1 - snow_frac(T))
"""

import xarray as xr
import numpy as np

import config
from io_utils import water_year


def snow_fraction(tas, t_snow=config.T_ALL_SNOW, t_rain=config.T_ALL_RAIN):
    """Elementwise snow fraction of precipitation given daily mean temp (degC)."""
    frac = (t_rain - tas) / (t_rain - t_snow)
    return frac.clip(min=0.0, max=1.0)


def partition(ds, tas_var="tas", pr_var="pr"):
    """
    Add 'snow', 'rain', and 'snow_frac' variables to a dataset that already
    has daily mean temperature (degC) and total precipitation (mm/day).
    """
    frac = snow_fraction(ds[tas_var])
    ds = ds.copy()
    ds["snow_frac"] = frac
    ds["snow"] = ds[pr_var] * frac
    ds["rain"] = ds[pr_var] * (1 - frac)
    return ds


def water_year_sp_ratio(ds, pr_var="pr", snow_var="snow"):
    """
    Aggregate daily snow/total-precip to a water-year S/P ratio per grid cell.

    Returns an xr.DataArray of shape (water_year, lat, lon).
    """
    wy = water_year(ds.indexes["time"])
    ds = ds.assign_coords(water_year=("time", wy))

    annual_pr = ds[pr_var].groupby("water_year").sum("time", skipna=True)
    annual_snow = ds[snow_var].groupby("water_year").sum("time", skipna=True)

    sp_ratio = (annual_snow / annual_pr.where(annual_pr > 0)).rename("sp_ratio")
    sp_ratio.attrs["description"] = (
        "Water-year (Oct-Sep) ratio of snow-water-equivalent precipitation "
        "to total precipitation, from -1C/+1C linear-ramp phase partitioning."
    )
    return sp_ratio


def process_source_to_sp_ratio(ds, tas_var="tas", pr_var="pr"):
    """Convenience wrapper: partition daily data then collapse to annual S/P ratio."""
    partitioned = partition(ds, tas_var=tas_var, pr_var=pr_var)
    return water_year_sp_ratio(partitioned, pr_var=pr_var, snow_var="snow")


# ---------------------------------------------------------------------------
# Generalized seasonal S/P ratio (DJF winter, Nov-Apr cold season), built on
# the same "assign each timestamp to a season-year, handling the months that
# cross the calendar-year boundary" logic as water_year_sp_ratio above, but
# parameterized so it isn't limited to the Oct-Sep water year.
#
# NOTE on "cold season" naming: config.COLD_SEASON_MONTHS is currently
# Oct-Apr (7 months) and is what the existing cdo_cache/*_coldseason_sum.nc
# files were built from elsewhere in this pipeline. The Nov-Apr (6 month)
# definition below is a DIFFERENT window, matching the convention used for
# the bias-map / S-P-ratio comparison work. It's named cold_season_nov_apr_*
# rather than reusing "cold_season" alone specifically so it doesn't get
# confused with the Oct-Apr quantity already cached on disk. If the project
# should standardize on one "cold season" definition going forward, update
# config.COLD_SEASON_MONTHS to match and regenerate the cdo_cache outputs.
# ---------------------------------------------------------------------------

WINTER_MONTHS = [12, 1, 2]
WINTER_SHIFT_MONTHS = [12]          # Dec rolls into the FOLLOWING winter-year

COLD_SEASON_NOV_APR_MONTHS = [11, 12, 1, 2, 3, 4]
COLD_SEASON_NOV_APR_SHIFT_MONTHS = [11, 12]   # Nov+Dec roll into the FOLLOWING season-year


def _assign_season_year(time_index, shift_months):
    """Return a season-year label for every entry in time_index (which
    should already be filtered to just the season's months): entries whose
    month is in shift_months belong to the FOLLOWING calendar year's
    instance of the season, everything else to the current one. Works with
    either a pandas DatetimeIndex or an xarray/cftime time index, since both
    expose .month and .year."""
    months = time_index.month
    years = time_index.year
    shift_mask = np.isin(months, shift_months)
    return np.where(shift_mask, years + 1, years)


def seasonal_sp_ratio(ds, months, shift_months, season_name, pr_var="pr", snow_var="snow"):
    """
    Aggregate daily snow/total-precip to a seasonal S/P ratio per grid cell,
    for an arbitrary (possibly calendar-year-crossing) season.

    months:       the calendar months (1-12) making up the season, e.g.
                  [12, 1, 2] for DJF.
    shift_months: the subset of `months` that belong to the FOLLOWING
                  season-year (e.g. [12] for DJF -- Dec 1999 groups with
                  Jan/Feb 2000 as "winter 2000").

    Any season-year missing one or more of the season's expected months
    (always true at the very start/end of the record, since the season
    reaches into the adjacent calendar year) is dropped rather than
    averaged in as a partial season.

    Returns an xr.DataArray of shape (season_year, lat, lon).
    """
    time_index = ds.indexes["time"]
    month_mask = np.isin(time_index.month, months)
    ds_season = ds.isel(time=month_mask)

    season_time_index = ds_season.indexes["time"]
    season_year = _assign_season_year(season_time_index, shift_months)
    ds_season = ds_season.assign_coords(season_year=("time", season_year))

    # drop any season-year missing one or more of the season's expected
    # calendar months (pure numpy, no pandas dependency needed here)
    months_arr = season_time_index.month
    unique_years = np.unique(season_year)
    complete_years = [
        yr for yr in unique_years
        if len(np.unique(months_arr[season_year == yr])) == len(months)
    ]
    ds_season = ds_season.where(ds_season["season_year"].isin(complete_years), drop=True)

    annual_pr = ds_season[pr_var].groupby("season_year").sum("time", skipna=True)
    annual_snow = ds_season[snow_var].groupby("season_year").sum("time", skipna=True)

    sp_ratio = (annual_snow / annual_pr.where(annual_pr > 0)).rename("sp_ratio")
    sp_ratio.attrs["description"] = (
        f"{season_name} ratio of snow-water-equivalent precipitation to "
        f"total precipitation, from {config.T_ALL_SNOW}C/{config.T_ALL_RAIN}C "
        f"linear-ramp phase partitioning."
    )
    sp_ratio.attrs["season_months"] = months
    return sp_ratio


def winter_sp_ratio(ds, pr_var="pr", snow_var="snow"):
    """DJF (Dec-Feb) S/P ratio. December belongs to the FOLLOWING winter
    (e.g. Dec 1999 + Jan/Feb 2000 = "winter 2000"), the same
    calendar-year-crossing convention as water_year_sp_ratio's Oct-Dec
    shift above."""
    return seasonal_sp_ratio(ds, WINTER_MONTHS, WINTER_SHIFT_MONTHS,
                              season_name="Winter (Dec-Feb)",
                              pr_var=pr_var, snow_var=snow_var)


def cold_season_nov_apr_sp_ratio(ds, pr_var="pr", snow_var="snow"):
    """Nov-Apr cold-season S/P ratio -- see the module-level NOTE above on
    why this is distinct from config.COLD_SEASON_MONTHS (Oct-Apr)."""
    return seasonal_sp_ratio(ds, COLD_SEASON_NOV_APR_MONTHS, COLD_SEASON_NOV_APR_SHIFT_MONTHS,
                              season_name="Cold Season (Nov-Apr)",
                              pr_var=pr_var, snow_var=snow_var)


def process_source_to_seasonal_sp_ratio(ds, season_fn, tas_var="tas", pr_var="pr"):
    """Convenience wrapper: partition daily data then collapse to the
    seasonal S/P ratio produced by `season_fn`. `season_fn` can be any of
    water_year_sp_ratio, winter_sp_ratio, or cold_season_nov_apr_sp_ratio
    (they all share the (ds, pr_var, snow_var) signature)."""
    partitioned = partition(ds, tas_var=tas_var, pr_var=pr_var)
    return season_fn(partitioned, pr_var=pr_var, snow_var="snow")


# Convenience lookup for scripts that want to loop over all three windows,
# e.g.: for key, fn in SP_RATIO_FUNCTIONS.items(): sp = fn(partitioned_ds)
SP_RATIO_FUNCTIONS = {
    "water_year": water_year_sp_ratio,
    "winter": winter_sp_ratio,
    "cold_season_nov_apr": cold_season_nov_apr_sp_ratio,
}
