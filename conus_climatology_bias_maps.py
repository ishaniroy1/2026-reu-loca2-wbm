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
from matplotlib.ticker import FormatStrFormatter
import geopandas as gpd
import rioxarray

# filepaths (same layout as the Taylor diagram script)
LIVNEH_BASE_DIR = "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
MODEL_BASE_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/climatology_maps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SHAPEFILE_PATH = os.path.expanduser("~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

NON_CONUS = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}

# annual mean for all three variables (see the Taylor diagram script for why
# precip switched from an annual sum to a days_in_month-weighted mean)
variables_config = {
    "airTmax": "mean",
    "airTmin": "mean",
    "precip": "mean",
}

VAR_LABELS = {
    "airTmax": "Tmax (\u00b0C)",
    "airTmin": "Tmin (\u00b0C)",
    "precip": "Precip (mm/day)",
}

# temperature runs blue (cold) -> red (warm)
# precip runs pale (dry) -> dark blue (wet)
CLIMATOLOGY_CMAP = {
    "airTmax": "RdYlBu_r",
    "airTmin": "RdYlBu_r",
    "precip": "YlGnBu",
}

FILL_VALUE = -9999.0
FILL_ABS_THRESHOLD = 1e10

model_folders = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in model_folders]

# Helpers duplicated from the Taylor diagram script

def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    """Load the state shapefile, keep only the 48 contiguous states + DC.

    Returns both the dissolved boundary (used for clipping) and the
    individual state geometries (used to draw state outlines on the maps).
    """
    states = gpd.read_file(shapefile_path)
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry, conus_states


def mask_fill_values(da):
    """Replace known no-data sentinels with real NaN, before any
    aggregation/clipping happens."""
    return da.where((da != FILL_VALUE) & (np.abs(da) < FILL_ABS_THRESHOLD))


def clip_to_conus(da_or_ds, conus_geom, lat_name=None, lon_name=None, all_touched=True):
    """Clip an xarray DataArray/Dataset to the CONUS boundary."""
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

    if isinstance(obj, xr.Dataset):
        for v in obj.data_vars:
            obj[v] = obj[v].rio.write_nodata(np.nan)
    else:
        obj = obj.rio.write_nodata(np.nan, inplace=False)

    return obj.rio.clip(conus_geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def normalize_precip_units(da, label):
    """Express precipitation as a daily rate (mm/day)."""
    units = da.attrs.get('units')
    flux_units = {'kg m-2 s-1', 'kg/m2/s', 'kg m^-2 s^-1', 'mm s-1', 'mm/s'}
    per_day_units = {'mm/day', 'mm day-1', 'mm d-1'}
    per_month_units = {'mm/month', 'mm month-1'}

    if units in flux_units:
        print(f"    [units check] {label}: converting flux -> mm/day")
        return da * 86400
    if units in per_day_units:
        return da
    if units in per_month_units:
        print(f"    [units check] {label}: converting mm/month -> mm/day")
        return da / da.time.dt.days_in_month
    return da


def normalize_temp_units(da, label):
    """Express temperature in degrees Celsius."""
    units = da.attrs.get('units')
    sample_mean = float(da.isel(time=slice(0, 1)).mean(skipna=True))

    kelvin_units = {'K', 'Kelvin', 'kelvin', 'degK', 'deg_K'}
    celsius_units = {'degC', 'C', 'Celsius', 'celsius', 'deg_C', 'degree_Celsius'}
    fahrenheit_units = {'degF', 'F', 'Fahrenheit', 'fahrenheit'}

    if units in kelvin_units:
        print(f"    [units check] {label}: converting Kelvin -> degC")
        return da - 273.15
    if units in fahrenheit_units:
        print(f"    [units check] {label}: converting Fahrenheit -> degC")
        return (da - 32) * (5.0 / 9.0)
    if units in celsius_units:
        return da
    if sample_mean > 100:
        print(f"    [units check] {label}: units unlabeled but magnitude suggests Kelvin -> converting to degC")
        return da - 273.15
    return da


def aggregate_yearly(da, operation):
    """Slice to 1980-2014 and resample to annual resolution. "mean" is a
    days_in_month-weighted annual mean (calendar-aware, correct for a rate
    variable). See the Taylor diagram script for the full rationale."""
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


def compute_climatology(var, operation, label, pattern, conus_geom):
    """Load monthly files for one dataset (Livneh or a single model),
    process to an annual time series, clip to CONUS, then collapse (average)
    across years to a single 2D (lat, lon) climatology field."""
    with xr.open_mfdataset(pattern, combine='by_coords', data_vars='all') as ds:
        da = ds[var]
        da = mask_fill_values(da)

        if var == 'precip':
            da = normalize_precip_units(da, label)
        elif var in ('airTmax', 'airTmin'):
            da = normalize_temp_units(da, label)

        yearly = aggregate_yearly(da, operation)
        yearly = clip_to_conus(yearly, conus_geom)
        yearly = yearly.load()

    # collapse the lat/lon grid's time (year) dimension
    return yearly.mean(dim='time', skipna=True)


def grid_dims(n):
    """Rows/cols for an approximately-square panel grid holding n panels."""
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def add_state_outlines(ax, state_borders):
    try:
        state_borders.boundary.plot(ax=ax, linewidth=0.3, color='black', alpha=0.5)
    except Exception:
        pass  # cosmetic only -- never let this break the figure


def plot_climatology_grid(var, obs_clim, model_clims, state_borders):
    """One big figure: Livneh + every model, each panel a CONUS map of the
    1980-2014 annual climatology, all on a shared color scale."""
    panels = [("Livneh (obs)", obs_clim)] + list(model_clims.items())
    n = len(panels)
    nrows, ncols = grid_dims(n)

    all_vals = np.concatenate([p[1].values.flatten() for p in panels])
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        print(f"No valid climatology data for {var}; skipping climatology figure.")
        return
    vmin, vmax = np.nanpercentile(all_vals, [2, 98])

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    mesh = None
    for ax, (label, da) in zip(axes, panels):
        mesh = ax.pcolormesh(da['lon'], da['lat'], da.values, cmap=CLIMATOLOGY_CMAP[var],
                              vmin=vmin, vmax=vmax, shading='auto')
        add_state_outlines(ax, state_borders)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis('off')

    cbar = fig.colorbar(mesh, ax=axes[:n].tolist(), shrink=0.75, label=VAR_LABELS[var])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'CONUS Annual Climatology (1980-2014): {VAR_LABELS[var]}',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, f"conus_{var}_climatology_allmodels.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bias_grid(var, model_bias, state_borders):
    """One big figure: every model's bias (model - obs), each panel a CONUS
    map, shared red-white-blue diverging scale centered on zero.
    Red = warmer/wetter (positive bias), blue = cooler/drier (negative bias),
    white = zero bias."""
    panels = list(model_bias.items())
    n = len(panels)
    if n == 0:
        print(f"No model bias data for {var}; skipping bias figure.")
        return
    nrows, ncols = grid_dims(n)

    all_vals = np.concatenate([p[1].values.flatten() for p in panels])
    all_vals = all_vals[~np.isnan(all_vals)]
    if all_vals.size == 0:
        print(f"No valid bias data for {var}; skipping bias figure.")
        return
    abs_max = float(np.nanpercentile(np.abs(all_vals), 98))
    if abs_max <= 0:
        abs_max = float(np.nanmax(np.abs(all_vals))) or 1.0
    vmin, vmax = -abs_max, abs_max

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    mesh = None
    for ax, (label, da) in zip(axes, panels):
        # RdBu_r: low (negative, cooler/drier) -> blue, high (positive,
        # warmer/wetter) -> red, midpoint (0, since vmin/vmax are symmetric)
        # -> white
        mesh = ax.pcolormesh(da['lon'], da['lat'], da.values, cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, shading='auto')
        add_state_outlines(ax, state_borders)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n:]:
        ax.axis('off')

    cbar = fig.colorbar(mesh, ax=axes[:n].tolist(), shrink=0.75,
                         label=f'Bias ({VAR_LABELS[var]})')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=8)

    warm_word = "wetter" if var == "precip" else "warmer"
    cool_word = "drier" if var == "precip" else "cooler"
    fig.suptitle(
        f'CONUS Annual Bias, Model \u2212 Obs (1980-2014): {VAR_LABELS[var]}\n'
        f'Red = {warm_word}, Blue = {cool_word}, White = 0',
        fontsize=13, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, f"conus_{var}_bias_allmodels.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom, state_borders = load_conus_boundary()

for var, operation in variables_config.items():
    print(f"\n--- Processing variable: {var} (annual {operation} climatology) ---")

    livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
    livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
    if not glob.glob(livneh_pattern):
        print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
        continue

    print(f"Computing Livneh [{var}] climatology...")
    obs_clim = compute_climatology(var, operation, "Livneh", livneh_pattern, conus_geom)

    model_clims = {}
    model_bias = {}

    for folder, name in zip(model_folders, model_names):
        model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
        if not glob.glob(model_pattern):
            print(f"  Skipping {name} [{var}]: no files found at {model_pattern}")
            continue

        try:
            print(f"  Computing {name} [{var}] climatology...")
            m_clim = compute_climatology(var, operation, name, model_pattern, conus_geom)
            model_clims[name] = m_clim

            obs_regrid = obs_clim.interp_like(m_clim, method='nearest')
            model_bias[name] = m_clim - obs_regrid

        except Exception as e:
            print(f"  Skipping model {name} [{var}] due to calculation mismatch: {e}")
            continue

    if not model_clims:
        print(f"No models produced valid climatology results for {var}; skipping figures.")
        continue

    plot_climatology_grid(var, obs_clim, model_clims, state_borders)
    plot_bias_grid(var, model_bias, state_borders)

print("\nProcess complete")
