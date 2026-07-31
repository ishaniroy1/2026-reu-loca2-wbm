import os
import sys

proj_path = "/net/home/wsag/ir1101/miniconda3/share/proj"
os.environ["PROJ_DATA"] = proj_path
os.environ["PROJ_LIB"] = proj_path

import pyproj
pyproj.datadir.set_data_dir(proj_path)

import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skill_metrics as sm
import geopandas as gpd
import rioxarray

# filepaths
LIVNEH_BASE_DIR = "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
MODEL_BASE_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

# output location for taylor diagrams
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/taylor_diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# state boundary shapefile used to clip datasets to CONUS
SHAPEFILE_PATH = os.path.expanduser("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

# states/territories in the shapefile that are not part of CONUS
NON_CONUS = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}

# map variables to aggregation operations
variables_config = {
    "airTmax": "mean",
    "airTmin": "mean",
    "precip": "mean"
}

# known "no data" sentinels that show up
FILL_VALUE = -9999.0
FILL_ABS_THRESHOLD = 1e10

model_folders = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in model_folders]

# colormap and symbols for plotting
cmap = plt.get_cmap('tab20')
symbols = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '<', '>', 'p', '8', 'd', 'H']

MODEL_STYLE = {
    m: {
        'faceColor': 'w',
        'edgeColor': cmap(i % 20),
        'symbol': symbols[i % len(symbols)],
        'size': 9,
        'labelColor': 'black',
    }
    for i, m in enumerate(model_names)
}


def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    """Load the state shapefile, keep only the 48 contiguous states + DC, and
    dissolve them into a single boundary in EPSG:4326 (plain lat/lon)."""
    states = gpd.read_file(shapefile_path)
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry


def mask_fill_values(da):
    """Replace known no-data sentinels with real NaN."""
    return da.where((da != FILL_VALUE) & (np.abs(da) < FILL_ABS_THRESHOLD))


def clip_to_conus(da_or_ds, conus_geom, lat_name=None, lon_name=None, all_touched=True):
    # Clip an xarray DataArray/Dataset to the CONUS boundary
    if lat_name is None:
        lat_name = 'lat' if 'lat' in da_or_ds.coords else ('latitude' if 'latitude' in da_or_ds.coords else None)
    if lon_name is None:
        lon_name = 'lon' if 'lon' in da_or_ds.coords else ('longitude' if 'longitude' in da_or_ds.coords else None)
    if not lat_name or not lon_name:
        raise KeyError(f"Could not find lat/lon coordinates in dataset. Found: {list(da_or_ds.coords.keys())}")

    lons = da_or_ds[lon_name].values
    if np.any(lons > 180):
        da_or_ds = da_or_ds.assign_coords(
            {lon_name: np.where(da_or_ds[lon_name] > 180, da_or_ds[lon_name] - 360, da_or_ds[lon_name])}
        )
        da_or_ds = da_or_ds.sortby(lon_name)

    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)

    # Force the nodata value used for out-of-geometry pixels to be NaN
    if isinstance(obj, xr.Dataset):
        for v in obj.data_vars:
            obj[v] = obj[v].rio.write_nodata(np.nan)
    else:
        obj = obj.rio.write_nodata(np.nan, inplace=False)

    return obj.rio.clip(conus_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def normalize_precip_units(da, label):
    # Express precipitation as a daily rate (mm/day), matching the
    units = da.attrs.get('units')
    sample_mean = float(da.isel(time=slice(0, 1)).mean(skipna=True))
    print(f"    [units check] {label}: units='{units}', sample monthly mean={sample_mean:.4f}")

    flux_units = {'kg m-2 s-1', 'kg/m2/s', 'kg m^-2 s^-1', 'mm s-1', 'mm/s'}
    per_day_units = {'mm/day', 'mm day-1', 'mm d-1'}
    per_month_units = {'mm/month', 'mm month-1'}

    if units in flux_units:
        print(f"    [units check] {label}: converting flux -> mm/day")
        return da * 86400
    if units in per_day_units:
        print(f"    [units check] {label}: already mm/day, no conversion needed")
        return da
    if units in per_month_units:
        print(f"    [units check] {label}: converting mm/month -> mm/day")
        return da / da.time.dt.days_in_month
    return da


def get_area_weights(da, lat_name=None):
    # cos(latitude) area weights, broadcast to da's full shape.
    if lat_name is None:
        lat_name = 'lat' if 'lat' in da.coords else 'latitude'
    w = np.cos(np.deg2rad(da[lat_name]))
    w_full, _ = xr.broadcast(w, da)
    return w_full


def weighted_taylor_stats(t, r, w):
    # Area-weighted centered pattern statistics
    wsum = np.sum(w)
    tmean = np.sum(w * t) / wsum
    rmean = np.sum(w * r) / wsum

    tdiff = t - tmean
    rdiff = r - rmean

    tvar = np.sum(w * tdiff ** 2) / wsum
    rvar = np.sum(w * rdiff ** 2) / wsum

    sdev_t = np.sqrt(tvar)
    sdev_r = np.sqrt(rvar)

    denom = np.sqrt(np.sum(w * tdiff ** 2) * np.sum(w * rdiff ** 2))
    pc = float(np.sum(w * tdiff * rdiff) / denom) if denom > 0 else np.nan

    crmsd = float(np.sqrt(max(sdev_t ** 2 + sdev_r ** 2 - 2 * sdev_t * sdev_r * pc, 0.0)))

    return float(sdev_t), float(sdev_r), pc, crmsd


def aggregate_yearly(da, operation):
    # Slice to the 1980-2014 period and resample to annual resolution
    da_sliced = da.sel(time=slice('1980-01-01', '2014-12-31'))

    if operation == "mean":
        def _weighted_annual_mean(group):
            weights = group.time.dt.days_in_month
            return group.weighted(weights).mean(dim='time', skipna=True)
        return da_sliced.resample(time='YS').map(_weighted_annual_mean)
    elif operation == "sum":
        return da_sliced.resample(time='YS').sum(dim='time', skipna=True)
    else:
        raise ValueError(f"Unknown aggregation operation: {operation}")


print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom = load_conus_boundary()

for var, operation in variables_config.items():
    print(f"\n--- Processing variable: {var} (yearly {operation}) ---")

    livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
    livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
    livneh_files = glob.glob(livneh_pattern)

    if not livneh_files:
        print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
        continue

    print(f"Loading Livneh monthly files for [{var}]...")
    with xr.open_mfdataset(livneh_pattern, combine='by_coords', data_vars='all') as livneh_ds:
        livneh_da = livneh_ds[var]
        # Scrub fill values before agg
        livneh_da = mask_fill_values(livneh_da)

        if var == 'precip':
            livneh_da = normalize_precip_units(livneh_da, "Livneh")

        livneh_yearly = aggregate_yearly(livneh_da, operation)

        print(f"Clipping Livneh [{var}] to CONUS")
        livneh_yearly = clip_to_conus(livneh_yearly, conus_geom)
        obs_da = livneh_yearly.load()

    print(f"Yearly Livneh [{var}] dims: {dict(obs_da.sizes)}")

    raw_obs_data = obs_da.values.flatten()
    obs_weights_full = get_area_weights(obs_da)
    raw_obs_weights = obs_weights_full.values.flatten()
    obs_mask = ~np.isnan(raw_obs_data)
    obs_clean = raw_obs_data[obs_mask]
    obs_weights_clean = raw_obs_weights[obs_mask]

    if obs_clean.size == 0:
        print(f"No valid Livneh data for {var} after CONUS clip; skipping Taylor diagram.")
        continue

    # area-weighted baseline standard deviation (obs on its own native grid)
    obs_wsum = np.sum(obs_weights_clean)
    obs_wmean = np.sum(obs_weights_clean * obs_clean) / obs_wsum
    obs_wvar = np.sum(obs_weights_clean * (obs_clean - obs_wmean) ** 2) / obs_wsum
    baseline_sdev = float(np.sqrt(obs_wvar))

    # initialize SkillMetrics tracking lists w/ baseline perfect score
    sdev_list = [baseline_sdev]
    crmsd_list = [0.0]
    cc_list = [1.0]

    active_models = []
    best_model = None
    lowest_error = float('inf')

    for folder, name in zip(model_folders, model_names):
        model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
        if not glob.glob(model_pattern):
            print(f"  Skipping {name} [{var}]: no files found at {model_pattern}")
            continue

        try:
            with xr.open_mfdataset(model_pattern, combine='by_coords', data_vars='all') as model_ds:
                model_da_raw = model_ds[var]
                model_da_raw = mask_fill_values(model_da_raw)

                if var == 'precip':
                    model_da_raw = normalize_precip_units(model_da_raw, name)

                model_yearly = aggregate_yearly(model_da_raw, operation)
                # clip the model grid to CONUS before regridding
                model_yearly = clip_to_conus(model_yearly, conus_geom)
                model_da = model_yearly.load()

            obs_regrid = obs_da.interp_like(model_da, method='nearest')

            model_weights_full = get_area_weights(model_da)

            model_data = model_da.values.flatten()
            obs_regrid_data = obs_regrid.values.flatten()
            weights_flat = model_weights_full.values.flatten()

            valid_mask = ~np.isnan(model_data) & ~np.isnan(obs_regrid_data)

            m_clean = model_data[valid_mask]
            o_clean = obs_regrid_data[valid_mask]
            w_clean = weights_flat[valid_mask]

            # size/dimension checking
            total_cells = len(model_data)
            valid_cells = len(m_clean)
            pct_valid = (valid_cells / total_cells) * 100 if total_cells > 0 else 0
            print(f"    [size check] {name:<15} | Raw 1D: {total_cells:,} | Masked 1D: {valid_cells:,} ({pct_valid:.1f}% valid)")

            if len(m_clean) == 0 or len(o_clean) == 0:
                print(f"  Skipping {name} [{var}]: no overlapping valid data")
                continue

            # area-weighted std devs, pattern correlation, and CRMSD
            # (NCL taylor_stats.ncl equivalent)
            sdev, o_std, corr, crmsd = weighted_taylor_stats(m_clean, o_clean, w_clean)

            sdev_list.append(round(sdev, 3))
            crmsd_list.append(round(crmsd, 3))
            cc_list.append(round(corr, 3))
            active_models.append(name)

            print(f"  Model: {name:<15} | Corr: {corr:.3f} | CRMSE: {crmsd:.3f}")

            if crmsd < lowest_error:
                lowest_error = crmsd
                best_model = name

        except Exception as e:
            print(f"  Skipping model {name} [{var}] due to calculation mismatch: {e}")
            continue

    # generating taylor diagrams
    if active_models:
        sdevs = np.array(sdev_list)
        crmsds = np.array(crmsd_list)
        ccs = np.array(cc_list)

        fig = plt.figure(figsize=(12, 10))

        markers_dict = {m: MODEL_STYLE[m] for m in active_models}

        sm.taylor_diagram(sdevs, crmsds, ccs,
            markerLegend='on',
            markers=markers_dict,
            markerObs='o',
            colObs='red',
            titleOBS='Reference',
            axisMax=float(np.max(sdevs) * 1.2),
            colCOR='black',
            colRMS='RoyalBlue',
            colSTD='SlateGray',
            styleRMS=':',
            styleSTD='--')

        ax = plt.gca()
        leg = ax.get_legend()
        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.get_texts()]
        leg.remove()
        fig.subplots_adjust(right=0.9)
        ax.legend(handles, labels, loc='upper right', fontsize=7,
                  ncol=2, framealpha=0.9, title='Models', title_fontsize=8)

        plt.title(f'LOCA2 CONUS Yearly Historical Validation (1980-2014): {var}',
                  y=1.08, fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_plot = os.path.join(OUTPUT_DIR, f"conus_{var}_yearly_taylor.png")
        plt.savefig(output_plot, dpi=300)
        plt.close()

        print(f"Plot saved to {output_plot}")
        print(f"{best_model} runs closest to observed data")
    else:
        print(f"No models produced valid results for {var}; skipping Taylor diagram.")
