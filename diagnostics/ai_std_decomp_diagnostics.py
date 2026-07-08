"""
Decompose the std dev going into the Taylor diagrams to see how much of it
is 'real' (spatial + seasonal heterogeneity across CONUS) vs. potentially
still a data/masking problem.

Focuses on airTmax vs Livneh Tmax, CONUS only, first available model.
"""

import os
import glob
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa: F401

LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2013.nc")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"
SHAPEFILE_PATH = os.path.expanduser("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

VAR = "airTmax"
LIVNEH_VAR = "Tmax"

NON_CONUS = {
    "Alaska", "Hawaii", "Puerto Rico", "Guam", "American Samoa",
    "Commonwealth of the Northern Mariana Islands", "United States Virgin Islands",
}


def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    states = gpd.read_file(shapefile_path)
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry


def clip_to_conus(da_or_ds, conus_geom, lat_name="lat", lon_name="lon"):
    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    return obj.rio.clip(conus_geom, crs="EPSG:4326", drop=True, all_touched=True)


# first model folder with airTmax data
folders = sorted(glob.glob(os.path.join(MODEL_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split("_")[0] for f in folders]

first_folder, first_model, model_pattern = None, None, None
for folder, name in zip(folders, model_names):
    pattern = os.path.join(folder, "monthly", VAR, "wbm_*.nc")
    if glob.glob(pattern):
        first_folder, first_model, model_pattern = folder, name, pattern
        break

if first_folder is None:
    raise RuntimeError(f"No model folder has data for {VAR}")

print(f"Using model: {first_model}\n")

conus_geom = load_conus_boundary()

with xr.open_dataset(LIVNEH_REF) as obs_ds, \
     xr.open_mfdataset(model_pattern, combine="by_coords", data_vars="all") as model_ds:

    obs_da = clip_to_conus(obs_ds[LIVNEH_VAR], conus_geom)
    model_da = clip_to_conus(model_ds[VAR].sel(time=slice("1980-01-01", "2013-12-31")), conus_geom)

    obs_regrid = obs_da.interp_like(model_da, method="nearest")

    # =========================================================
    # STEP 1: outlier / sanity check on the values actually feeding the stats
    # =========================================================
    print("=== STEP 1: Value range check (looking for leftover bad data) ===")
    for label, da in [("Livneh", obs_regrid), (first_model, model_da)]:
        vals = da.values.flatten()
        vals = vals[~np.isnan(vals) & (vals != -9999.0) & (vals < 1e10)]
        pct = np.percentile(vals, [0, 1, 50, 99, 100])
        print(f"{label:<15} min={pct[0]:.2f}  p1={pct[1]:.2f}  median={pct[2]:.2f}  "
              f"p99={pct[3]:.2f}  max={pct[4]:.2f}  n={len(vals)}")
    print("If min/max are far outside a plausible Tmax range (roughly -40 to 50 degC),")
    print("or if p1/p99 are way off from the median, there's likely still bad data")
    print("sneaking past the current masks.\n")

    # =========================================================
    # STEP 2: variance decomposition -- spatial vs seasonal vs residual
    # =========================================================
    print("=== STEP 2: Variance decomposition ===")

    for label, da in [("Livneh", obs_regrid), (first_model, model_da)]:
        raw_std = float(da.std(skipna=True))

        # spatial-only: std of the all-time mean map (removes time entirely)
        time_mean_map = da.mean(dim="time", skipna=True)
        spatial_std = float(time_mean_map.std(skipna=True))

        # seasonal-only: std of the domain-mean monthly climatology (removes space entirely)
        domain_mean_ts = da.mean(dim=[d for d in da.dims if d != "time"], skipna=True)
        seasonal_climatology = domain_mean_ts.groupby("time.month").mean()
        seasonal_std = float(seasonal_climatology.std(skipna=True))

        # anomaly std: remove each grid cell's own monthly climatology, what's left over
        clim = da.groupby("time.month").mean(dim="time", skipna=True)
        anomaly = da.groupby("time.month") - clim
        anomaly_std = float(anomaly.std(skipna=True))

        print(f"\n{label}:")
        print(f"  raw std (time+space combined, what the Taylor diagram currently uses): {raw_std:.2f}")
        print(f"  spatial-only std (map of time-means):                                  {spatial_std:.2f}")
        print(f"  seasonal-only std (domain-mean climatology by month):                  {seasonal_std:.2f}")
        print(f"  anomaly std (seasonal cycle removed, per grid cell):                   {anomaly_std:.2f}")

    print("\nIf 'anomaly std' is much smaller than 'raw std' for both datasets, most of your")
    print("Taylor-diagram std dev is coming from real spatial/seasonal climate variability,")
    print("not a bug -- consider rebuilding the Taylor diagram on anomalies instead of raw values.")
