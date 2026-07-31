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
