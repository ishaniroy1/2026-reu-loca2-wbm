"""
io_utils.py
===========
Loading and basic regridding/clipping helpers for the three data sources:

  * LOCA2-downscaled CMIP6 (tasmax, tasmin, pr) -- historical + SSP futures
  * Livneh gridded observations (tmax, tmin, pr) -- historical baseline
  * SNODAS (SWE, snowmelt) -- historical snowpack state

All datasets are standardized to daily temporal resolution, a shared
lon/lat grid convention (-180..180, ascending lat), and clipped to the
Northeast US region defined in config.py.
"""

import os
import glob
from pathlib import Path
import xarray as xr
import numpy as np
import geopandas as gpd
import rioxarray  # noqa: F401 -- registers the .rio accessor on xarray objects

import config

# Cached, dissolved Northeast-states geometry (EPSG:4326). Loaded once on
# first use so every load_* call doesn't re-read the shapefile from disk.
_NORTHEAST_GEOM = None


def _standardize_lon(ds):
    """Convert 0-360 longitudes to -180..180 if needed."""
    if ds["lon"].max() > 180:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))
        ds = ds.sortby("lon")
    return ds


def load_northeast_boundary(shapefile_path=None, force_reload=False):
    """
    Load the US states shapefile, keep only the Northeast states defined
    in config.NE_STATES, and dissolve them into a single boundary in
    EPSG:4326. Result is cached in-module after the first call.
    """
    global _NORTHEAST_GEOM
    if _NORTHEAST_GEOM is not None and not force_reload:
        return _NORTHEAST_GEOM

    shapefile_path = shapefile_path or config.STATES_SHAPEFILE
    states = gpd.read_file(shapefile_path)
    northeast_states = states[states[config.STATE_ID_FIELD].isin(config.NE_STATES)]

    if northeast_states.empty:
        raise ValueError(
            f"No features matched config.NE_STATES={config.NE_STATES} using "
            f"field '{config.STATE_ID_FIELD}' in {shapefile_path}. Check the "
            f"field name and state codes against the shapefile's attributes."
        )

    northeast_states = northeast_states.to_crs("EPSG:4326")
    _NORTHEAST_GEOM = northeast_states.dissolve().geometry
    return _NORTHEAST_GEOM


def clip_to_region(ds, lat_name="lat", lon_name="lon", all_touched=True):
    """
    Spatial subset to the actual dissolved Northeast-states boundary
    (not a bounding box). Assumes lon/lat are 1-D coordinates; converts
    0-360 longitudes to -180..180 first if needed.
    """
    ds = _standardize_lon(ds)
    if ds["lat"][0] > ds["lat"][-1]:
        ds = ds.sortby("lat")

    northeast_geom = load_northeast_boundary()

    obj = ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    return obj.rio.clip(northeast_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def discover_model_runs(base_dir, scenario=None):
    """
    Find GCM-run subfolders under a LOCA2-WBM period directory (historical
    or future), matching config.MODEL_FOLDER_GLOB (default "*_newprcp"),
    same discovery approach as the Taylor-diagram validation script.

    If `scenario` is given and config.SCENARIO_SUBDIR_TEMPLATE is
    non-empty, looks under base_dir/<scenario-subdir> instead of base_dir
    directly -- see the NOTE in config.py about confirming how SSPs map
    onto the future directory layout before relying on this for futures.

    Returns a dict of {gcm_name: folder_path}, where gcm_name is the
    folder-name prefix before the first underscore (e.g. "ACCESS-CM2"
    from "ACCESS-CM2_historical_1980-2014_newprcp").
    """
    search_dir = base_dir
    if scenario and config.SCENARIO_SUBDIR_TEMPLATE:
        search_dir = base_dir / config.SCENARIO_SUBDIR_TEMPLATE.format(scenario=scenario)

    folders = sorted(glob.glob(str(search_dir / config.MODEL_FOLDER_GLOB)))
    if not folders:
        raise FileNotFoundError(
            f"No GCM-run folders found matching '{config.MODEL_FOLDER_GLOB}' "
            f"under {search_dir}"
        )
    return {os.path.basename(f).split("_")[0]: f for f in folders}


def load_wbm_daily(gcm_run_dir, variables=None, start=None, end=None):
    """
    Load one GCM run's daily LOCA2-WBM output: every wbm_{year}.nc file
    under {gcm_run_dir}/{DAILY_SUBDIR}/, which already bundles all
    variables for that year together (temps, precip, and the model's own
    rain/snow/snowmelt/SWE partitioning).

    `variables`: iterable of variable names to keep (defaults to the full
    set defined in config.py: tasmax/tasmin/precip/snow/rain/SWE/melt).
    Returns the clipped-to-Northeast dataset, temp standardized to degC.
    """
    daily_pattern = str(Path(gcm_run_dir) / config.DAILY_SUBDIR / config.DAILY_FILE_GLOB)
    files = sorted(glob.glob(daily_pattern))
    if not files:
        raise FileNotFoundError(f"No daily WBM files found at {daily_pattern}")

    ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all",
                            chunks={"time": 365})

    if variables is None:
        variables = [v for v in (
            config.VAR_TAS, config.VAR_TASMAX, config.VAR_TASMIN, config.VAR_PRECIP_TOTAL,
            config.VAR_SNOW, config.VAR_SWE, config.VAR_SWE_CHANGE, config.VAR_SNOWMELT,
        ) if v in ds.data_vars]

    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(
            f"Requested variable(s) {missing} not found in {files[0]}. "
            f"Variables actually present: {sorted(ds.data_vars)}. Update the "
            f"VAR_* names in config.py to match."
        )

    ds = ds[variables]
    ds = clip_to_region(ds)

    # Safety net only -- WBM output is already degC per the file header, so
    # this shouldn't trigger, but guards against a differently-sourced
    # dataset slipping in with Kelvin units.
    for tvar in (config.VAR_TAS, config.VAR_TASMAX, config.VAR_TASMIN):
        if tvar in ds and float(ds[tvar].isel(time=0).mean()) > 100:
            ds[tvar] = ds[tvar] - 273.15

    # tas: use the model's own airT directly when present, else fall back
    # to averaging tasmax/tasmin
    if config.VAR_TAS in ds:
        ds["tas"] = ds[config.VAR_TAS]
    elif config.VAR_TASMAX in ds and config.VAR_TASMIN in ds:
        ds["tas"] = (ds[config.VAR_TASMAX] + ds[config.VAR_TASMIN]) / 2.0

    # rain: not output directly by the WBM -- derive as total precip minus
    # snowfall (both mm/day), consistent with the model's own phase split
    if config.VAR_PRECIP_TOTAL in ds and config.VAR_SNOW in ds:
        ds["rain"] = ds[config.VAR_PRECIP_TOTAL] - ds[config.VAR_SNOW]

    if start or end:
        ds = ds.sel(time=slice(start, end))

    ds.attrs["source"] = f"LOCA2-WBM {Path(gcm_run_dir).name}"
    return ds


def load_loca2wbm(gcm, period, scenario=None, variables=None, start=None, end=None):
    """
    High-level convenience wrapper: resolve which GCM-run folder to use
    for a given GCM name + period ("historical" or "future" [+ scenario]),
    then load and clip its daily output.
    """
    base_dir = config.LOCA2WBM_HIST_DIR if period == "historical" else config.LOCA2WBM_FUT_DIR
    runs = discover_model_runs(base_dir, scenario=scenario if period == "future" else None)
    if gcm not in runs:
        raise FileNotFoundError(
            f"GCM '{gcm}' not found under {base_dir} "
            f"(scenario={scenario}). Available: {sorted(runs)}"
        )
    return load_wbm_daily(runs[gcm], variables=variables, start=start, end=end)


def load_livneh_monthly(variables=(config.VAR_TASMAX, config.VAR_TASMIN,
                                    config.VAR_PRECIP_TOTAL), start=None, end=None):
    """
    Load Livneh MONTHLY observations for MODEL VALIDATION purposes only
    (e.g. Taylor diagrams) -- see the note in config.py. Not used as the
    primary source for this pipeline's historical S/P-ratio or snowmelt
    analysis, which runs on the LOCA2-WBM_historical model output instead.

    Expects one subfolder per variable, matching the Taylor-diagram
    validation script: {LIVNEH_MONTHLY_DIR}/{var}/wbm_*.nc
    """
    das = {}
    for var in variables:
        pattern = str(config.LIVNEH_MONTHLY_DIR / var / config.DAILY_FILE_GLOB)
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No Livneh monthly files found for {var} at {pattern}")
        da = xr.open_mfdataset(files, combine="by_coords", data_vars="all")[var]
        das[var] = clip_to_region(da.to_dataset(name=var))[var]

    ds = xr.Dataset(das)
    start = start or config.HIST_START
    end = end or config.HIST_END
    ds = ds.sel(time=slice(start, end))
    ds.attrs["source"] = "Livneh (monthly, validation only)"
    return ds


def load_snodas(variables=("SWE", "snowMelt"), start=None, end=None):
    """
    Load SNODAS daily SWE (mm) and snowMelt (mm/day, model direct-output
    melt flux) for the historical period, clipped to the NE region.
    """
    das = {}
    for var in variables:
        pattern = str(config.SNODAS_DIR / f"{var}_*.nc")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No SNODAS files found for {var} at {pattern}")
        da = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 365})[var]
        das[var] = clip_to_region(da.to_dataset(name=var))[var]

    ds = xr.Dataset(das)
    start = start or config.HIST_START
    end = end or config.HIST_END
    ds = ds.sel(time=slice(start, end))
    ds.attrs["source"] = "SNODAS"
    return ds


def water_year(time_index):
    """Return the water year (Oct 1 - Sep 30, labeled by ending calendar year)."""
    years = time_index.year
    months = time_index.month
    return xr.where(months >= 10, years + 1, years)
