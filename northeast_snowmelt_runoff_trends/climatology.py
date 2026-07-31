"""
climatology.py
===============
Grid-cell (not HUC8) climatology of total snowfall, total precipitation,
and the snow-to-precip ratio, built from the CDO-cached yearly sums in
cdo_reduce.py. Implements the analysis-meeting notes directly:

  * snow and precip are summed per water-year or cold-season FIRST, at
    daily resolution (via CDO) -- not from monthly data
  * a period's "climatology" is the multi-year AVERAGE of those yearly
    sums (not an average of yearly ratios) -- so ratio_clim =
    mean(snow_sum) / mean(precip_sum), computed from the two
    climatological sums
  * averaging ACROSS MODELS happens LAST: per-model climatologies are
    computed and kept individually, and only combined into an ensemble
    mean as the final step (see multi_model_climatology)
"""

import xarray as xr

import config
import io_utils
import cdo_reduce


def climatology_for_run(gcm_run_dir, period_years, window="water_year"):
    """
    One GCM run's climatology (2-D lat/lon grids) for a given
    (start_year, end_year) window and season definition.

    Returns an xr.Dataset with snow_clim, precip_clim, sp_ratio_clim,
    clipped to the Northeast states boundary.
    """
    snow_path = cdo_reduce.get_or_build(gcm_run_dir, config.VAR_SNOW, window=window)
    precip_path = cdo_reduce.get_or_build(gcm_run_dir, config.VAR_PRECIP_TOTAL, window=window)

    snow = xr.open_dataset(snow_path)[config.VAR_SNOW]
    precip = xr.open_dataset(precip_path)[config.VAR_PRECIP_TOTAL]

    start_y, end_y = period_years
    snow = snow.sel(time=slice(f"{start_y}-01-01", f"{end_y}-12-31"))
    precip = precip.sel(time=slice(f"{start_y}-01-01", f"{end_y}-12-31"))

    if snow.sizes.get("time", 0) == 0 or precip.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No years of data fall within {start_y}-{end_y} for {gcm_run_dir} "
            f"(window={window}) -- check that this GCM run actually covers that period."
        )

    snow_clim = snow.mean("time", skipna=True).rename("snow_clim")
    precip_clim = precip.mean("time", skipna=True).rename("precip_clim")
    ratio_clim = (snow_clim / precip_clim.where(precip_clim > 0)).rename("sp_ratio_clim")

    ds = xr.Dataset({"snow_clim": snow_clim, "precip_clim": precip_clim,
                      "sp_ratio_clim": ratio_clim})
    ds = io_utils.clip_to_region(ds)
    ds.attrs.update(window=window, period=f"{start_y}-{end_y}", source=str(gcm_run_dir))
    return ds


def multi_model_climatology(gcm_runs, period_years, window="water_year"):
    """
    gcm_runs: dict {gcm_name: run_dir}, as returned by
    io_utils.discover_model_runs. Computes and KEEPS each model's
    climatology individually, then averages across models LAST.

    Returns (per_model, ensemble):
      per_model -- dict {gcm_name: xr.Dataset}
      ensemble  -- xr.Dataset, the across-model mean of snow_clim/
                   precip_clim/sp_ratio_clim
    """
    per_model = {}
    for gcm, run_dir in gcm_runs.items():
        try:
            per_model[gcm] = climatology_for_run(run_dir, period_years, window=window)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"  Skipping {gcm}: {e}")

    if not per_model:
        raise RuntimeError(
            f"No models produced a valid climatology for period={period_years}, window={window}."
        )

    stacked = xr.concat(list(per_model.values()), dim="model")
    ensemble = stacked.mean("model", skipna=True)
    ensemble.attrs.update(period=f"{period_years[0]}-{period_years[1]}", window=window,
                           source=f"ensemble mean of {len(per_model)} models "
                                  f"({', '.join(sorted(per_model))})")
    return per_model, ensemble
