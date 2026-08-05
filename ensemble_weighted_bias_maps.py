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

# filepaths (same layout as the Taylor diagram / climatology scripts)
LIVNEH_BASE_DIR = "/net/nfs/echo/ankaa/LivnehPierceLusu_output/LivnehPierceLusu_historical/monthly"
MODEL_BASE_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"

OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/ensemble_bias_maps")
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

# Northeast states, by USPS postal code (matches the STUSPS field in the
# Census cb_2025_us_state shapefile)
NE_STATES = ["ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA", "DE", "MD", "WV"]

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
    "precip": "Precip (%)",
}

FILL_VALUE = -9999.0
FILL_ABS_THRESHOLD = 1e10

# Fixed bias-map color scale AND hardcoded mask threshold (values beyond
# this are set to NaN rather than shown), applied per variable. airTmax/
# airTmin are unchanged (degC, +/-4). precip is now expressed as PERCENT
# bias rather than mm/day (see the percent-conversion step in the main
# loop below), so its limit is in percent, not mm/day -- adjust if you
# want a tighter/looser cutoff than +/-100%.
BIAS_SCALE_LIMIT = {
    "airTmax": 4.0,
    "airTmin": 4.0,
    "precip": 100.0,
}

# precip cells with a historical (obs) climatology below this are excluded
# from the percent-bias calculation -- dividing by a near-zero denominator
# is what produces wild, physically meaningless percentages
MIN_OBS_PRECIP_FOR_PCT = 0.1  # mm/day

# ---------------------------------------------------------------------------
# Ensemble weights (e.g. climate-sensitivity-informed weighting -- models
# that run overly/under sensitive get downweighted). Ensemble average is:
#     sum(w_i * model_i) / sum(w_i)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = {
    "ACCESS-CM2": 4.72,
    "ACCESS-ESM1-5": 3.87,
    "BCC-CSM2-MR": 3.04,
    "CanESM5": 5.62,
    "EC-Earth3": 4.3,
    "EC-Earth3-Veg": 3.34,
    "FGOALS-g3": 2.88,
    "GFDL-ESM4": 3.9,
    "INM-CM4-8": 1.83,
    "INM-CM5-0": 1.92,
    "IPSL-CM6A-LR": 4.56,
    "MIROC6": 2.61,
    "MPI-ESM1-2-HR": 2.98,
    "MPI-ESM1-2-LR": 3.0,
    "MRI-ESM2-0": 3.15,
    "NorESM2-LM": 2.54,
    "NorESM2-MM": 2.5,
}

model_folders = sorted(glob.glob(os.path.join(MODEL_BASE_DIR, "*_newprcp")))
model_names = [os.path.basename(f).split('_')[0] for f in model_folders]

_unweighted = sorted(set(model_names) - set(MODEL_WEIGHTS))
if _unweighted:
    print(f"NOTE: no weight defined for {_unweighted} -- these will be "
          f"excluded from the weighted ensemble if found on disk.")

# --- helpers (mostly duplicated from the climatology/bias script) ---


def load_conus_boundary(shapefile_path=SHAPEFILE_PATH):
    """Dissolved CONUS boundary + individual state geometries (for outlines)."""
    states = gpd.read_file(shapefile_path)
    name_col = "NAME" if "NAME" in states.columns else "STUSPS"
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry, conus_states


def load_northeast_boundary(shapefile_path=SHAPEFILE_PATH):
    """Dissolved Northeast boundary (NE_STATES) + individual state geometries."""
    states = gpd.read_file(shapefile_path)
    ne_states = states[states["STUSPS"].isin(NE_STATES)].copy()
    ne_states = ne_states.to_crs("EPSG:4326")
    if ne_states.empty:
        raise ValueError(f"No features matched NE_STATES={NE_STATES} in {shapefile_path} "
                          f"(check the STUSPS field/values).")
    return ne_states.dissolve().geometry, ne_states


def mask_fill_values(da):
    return da.where((da != FILL_VALUE) & (np.abs(da) < FILL_ABS_THRESHOLD))


def clip_to_boundary(da_or_ds, geom, lat_name=None, lon_name=None, all_touched=True):
    """Clip an xarray DataArray/Dataset to an arbitrary dissolved boundary
    geometry (CONUS or Northeast -- whichever `geom` is passed in)."""
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

    return obj.rio.clip(geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def normalize_precip_units(da, label):
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
    variable)."""
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


def compute_climatology(var, operation, label, pattern, clip_geom):
    """Load monthly files for one dataset (Livneh or a single model),
    process to an annual time series, clip, then collapse (average) across
    years to a single 2D (lat, lon) climatology field."""
    with xr.open_mfdataset(pattern, combine='by_coords', data_vars='all') as ds:
        da = ds[var]
        da = mask_fill_values(da)

        if var == 'precip':
            da = normalize_precip_units(da, label)
        elif var in ('airTmax', 'airTmin'):
            da = normalize_temp_units(da, label)

        yearly = aggregate_yearly(da, operation)
        yearly = clip_to_boundary(yearly, clip_geom)
        yearly = yearly.load()

    return yearly.mean(dim='time', skipna=True)


def compute_weighted_ensemble(data_dict, weights, label=""):
    """
    Combine per-model 2D grids (already on a COMMON grid -- see main loop)
    into a single weighted-ensemble grid:
        ensemble = sum(w_i * grid_i) / sum(w_i)
    Uses xarray's weighted().mean(skipna=True), which is NaN-aware per grid
    cell: if a model is NaN at a given cell (e.g. right at a clip edge),
    it's excluded from both the numerator and the weight sum at that cell
    rather than corrupting the whole average.
    """
    used = {name: da for name, da in data_dict.items() if name in weights}
    missing = sorted(set(data_dict) - set(weights))
    if missing:
        tag = f" [{label}]" if label else ""
        print(f"    [ensemble{tag}] excluding (no weight defined): {missing}")
    if not used:
        raise ValueError("No models with defined weights available to build the ensemble.")

    stacked = xr.concat(list(used.values()), dim="model")
    wgt_da = xr.DataArray([weights[name] for name in used], dims="model")
    return stacked.weighted(wgt_da).mean(dim="model", skipna=True)


def mask_beyond_limit(bias_da, limit, label=""):
    """Set any cell with |bias| > limit to NaN (hardcoded out, per the
    fixed +/-limit color scale used for these plots)."""
    n_before = int((~np.isnan(bias_da)).sum())
    masked = bias_da.where(np.abs(bias_da) <= limit)
    n_after = int((~np.isnan(masked)).sum())
    if n_after < n_before:
        tag = f" [{label}]" if label else ""
        print(f"    [mask{tag}] excluded {n_before - n_after} cell(s) with |bias| > {limit}")
    return masked


def plot_bias_map(var, bias_da, region_label, state_borders, out_name):
    """Single-panel diverging bias map (model ensemble - obs), using the
    fixed +/-BIAS_SCALE_LIMIT[var] color scale (not auto-computed from
    this map's own data), so every plot for a given variable is directly
    comparable regardless of run-to-run outlier count."""
    vals = bias_da.values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"No valid data for {var} ({region_label}); skipping.")
        return
    abs_max = BIAS_SCALE_LIMIT[var]
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(bias_da['lon'], bias_da['lat'], bias_da.values,
                          cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    try:
        state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])

    warm_word = "wetter" if var == "precip" else "warmer"
    cool_word = "drier" if var == "precip" else "cooler"
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, label=f'Ensemble bias ({VAR_LABELS[var]})')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title(
        f'{region_label} Weighted-Ensemble Bias, Model \u2212 Obs (1980-2014): {VAR_LABELS[var]}\n'
        f'Red = {warm_word}, Blue = {cool_word}, White = 0',
        fontsize=11, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_bias_map(region_label, bias_dict, state_borders, out_name):
    """One figure, one panel per variable (airTmax, airTmin, precip), each
    using its own fixed +/-BIAS_SCALE_LIMIT[var] color scale."""
    var_order = [v for v in variables_config if v in bias_dict]
    fig, axes = plt.subplots(1, len(var_order), figsize=(6 * len(var_order), 6),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, var in zip(axes, var_order):
        da = bias_dict[var]
        vals = da.values
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            ax.axis('off')
            continue
        abs_max = BIAS_SCALE_LIMIT[var]
        vmin, vmax = -abs_max, abs_max

        mesh = ax.pcolormesh(da['lon'], da['lat'], da.values, cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, shading='auto')
        try:
            state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
        except Exception:
            pass
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(VAR_LABELS[var], fontsize=11)

        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.suptitle(f'{region_label} Weighted-Ensemble Bias, Model \u2212 Obs (1980-2014)',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom, conus_borders = load_conus_boundary()
print(f"Loading Northeast boundary from {SHAPEFILE_PATH}")
ne_geom, ne_borders = load_northeast_boundary()

ensemble_bias_conus_all = {}
ensemble_bias_ne_all = {}

for var, operation in variables_config.items():
    print(f"\n--- Processing variable: {var} (weighted ensemble bias) ---")

    livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
    livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
    if not glob.glob(livneh_pattern):
        print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
        continue

    print(f"Computing Livneh [{var}] climatology (CONUS-clipped, common reference grid)...")
    obs_clim = compute_climatology(var, operation, "Livneh", livneh_pattern, conus_geom)

    model_bias_common = {}
    for folder, name in zip(model_folders, model_names):
        if name not in MODEL_WEIGHTS:
            continue  # no point computing a model that'll be excluded anyway
        model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
        if not glob.glob(model_pattern):
            print(f"  Skipping {name} [{var}]: no files found at {model_pattern}")
            continue

        try:
            print(f"  Computing {name} [{var}] climatology...")
            m_clim = compute_climatology(var, operation, name, model_pattern, conus_geom)
            # regrid onto Livneh's grid so every model's bias lives on the
            # SAME common grid before combining -- required since each
            # model's own climatology may be on its own native grid
            m_clim_common = m_clim.interp_like(obs_clim, method='nearest')
            model_bias_common[name] = m_clim_common - obs_clim
        except Exception as e:
            print(f"  Skipping model {name} [{var}] due to calculation mismatch: {e}")
            continue

    if not model_bias_common:
        print(f"No models produced valid bias results for {var}; skipping ensemble maps.")
        continue

    print(f"  Building weighted ensemble ({len(model_bias_common)} models)...")
    ensemble_bias_conus = compute_weighted_ensemble(model_bias_common, MODEL_WEIGHTS, label=var)

    if var == 'precip':
        # convert absolute bias (mm/day) to percent bias relative to obs;
        # mathematically identical to averaging per-model percent biases
        # first, since obs is the same fixed denominator for every model
        # at a given cell (the weighted mean above is linear). Cells where
        # obs is too close to zero are masked rather than left to blow up
        # to a meaningless percentage.
        safe_obs = obs_clim.where(obs_clim >= MIN_OBS_PRECIP_FOR_PCT)
        ensemble_bias_conus = (ensemble_bias_conus / safe_obs) * 100

    ensemble_bias_conus = mask_beyond_limit(ensemble_bias_conus, BIAS_SCALE_LIMIT[var], label=var)

    plot_bias_map(var, ensemble_bias_conus, "CONUS", conus_borders,
                  f"conus_{var}_bias_ensemble_weighted.png")

    print(f"  Clipping ensemble bias to Northeast...")
    ensemble_bias_ne = clip_to_boundary(ensemble_bias_conus, ne_geom)
    plot_bias_map(var, ensemble_bias_ne, "Northeast", ne_borders,
                  f"northeast_{var}_bias_ensemble_weighted.png")

    ensemble_bias_conus_all[var] = ensemble_bias_conus
    ensemble_bias_ne_all[var] = ensemble_bias_ne

if ensemble_bias_conus_all:
    plot_combined_bias_map("CONUS", ensemble_bias_conus_all, conus_borders,
                            "conus_allvars_bias_ensemble_weighted.png")
if ensemble_bias_ne_all:
    plot_combined_bias_map("Northeast", ensemble_bias_ne_all, ne_borders,
                            "northeast_allvars_bias_ensemble_weighted.png")

print("\nProcess complete")
