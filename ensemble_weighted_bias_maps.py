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
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap
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
    "airTmax": "Tmax (°C)",
    "airTmin": "Tmin (°C)",
    "precip": "Precip (%)",
}

FILL_VALUE = -9999.0
FILL_ABS_THRESHOLD = 1e10

# ---------------------------------------------------------------------------
# Poster-scale font sizes (these figures are going on a 36x44 scientific
# poster, so default/journal-article-scale fonts are much too small to read
# from a few feet away). figsize/dpi are left unchanged, so bumping these
# font sizes increases the TEXT-TO-FIGURE ratio, which is what actually
# matters once an image is blown up to poster size.
# ---------------------------------------------------------------------------
POSTER_FONT = {
    "suptitle": 26,
    "title": 20,
    "subplot_title": 22,
    "cbar_label": 20,
    "cbar_ticks": 18,
}
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": POSTER_FONT["title"],
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

# Hard sanity-bound bias-map cutoff: values beyond this are set to NaN as
# physically implausible (mask_beyond_limit). This is intentionally generous
# -- real Tmin/Tmax bias routinely exceeds +/-1 degC in places, so this
# cutoff must stay well above the DISPLAY range below, or mask_beyond_limit
# starts NaN-ing out real data, which fill_isolated_nan_gaps then has to
# patch with large blocky neighbor-averages (exactly the "pixelated" look
# that shows up if this is set too tight).
BIAS_SCALE_LIMIT = {
    "airTmax": 4.0,
    "airTmin": 4.0,
    "precip": 100.0,
}

# DISPLAY color-scale range (vmin/vmax passed to pcolormesh) -- separate
# from BIAS_SCALE_LIMIT above. This just clips/saturates the colormap at
# its ends; it does NOT delete or alter any data, unlike BIAS_SCALE_LIMIT.
# airTmax/airTmin share the same +/-1 degC scale, as requested.
DISPLAY_LIMIT = {
    "airTmax": 1.0,
    "airTmin": 1.0,
    "precip": 100.0,  # precip's actual display range is computed from the
                       # data itself (see get_display_abs_max) -- this value
                       # is only the fallback if a map has no valid data at all
}

TEMP_VARS = {"airTmax", "airTmin"}


def get_display_abs_max(var, vals):
    """The +/- color-scale bound for this map. Temp vars use the fixed
    DISPLAY_LIMIT (+/-1 degC). precip instead uses the actual max |bias|
    PRESENT IN THIS MAP'S OWN DATA, symmetric around zero (same distance
    above/below 0) -- a flat +/-100% made real variation (which rarely
    exceeds +/-20-30%) nearly invisible."""
    if var == "precip":
        return float(np.max(np.abs(vals))) if vals.size else DISPLAY_LIMIT[var]
    return DISPLAY_LIMIT[var]


def apply_colorbar_ticks(cbar, var, abs_max):
    """Temp vars (airTmax/airTmin): explicit half-step ticks at
    [-abs_max, -abs_max/2, 0, abs_max/2, abs_max] with 1-decimal labels,
    e.g. -1, -0.5, 0, 0.5, 1. precip: integer-only tick LOCATIONS (not just
    integer-rounded labels on whatever ticks happened to land, which
    produced duplicate "1"/"1" labels before) -- needed since precip's
    scale is now data-driven and can land on any range."""
    if var in TEMP_VARS:
        cbar.set_ticks([-abs_max, -abs_max / 2, 0, abs_max / 2, abs_max])
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    else:
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

# Diverging colormap for each variable. airTmax/airTmin keep the standard
# red(warm)-white-blue(cool) RdBu_r. precip uses a custom brown-yellow
# (drier) -> white -> blue (wetter) diverging map instead of RdBu_r, so dry
# years/regions read as earthy brown/tan rather than red (which otherwise
# reads as "warmer," not "drier," and is easy to confuse with the temp maps).
PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "brown_yellow_to_blue",
    ["#5c3a0a", "#a9701f", "#d9a441", "#f3e6c8", "#f5f5f5",
     "#cfe6f0", "#6baed6", "#2b6a9e", "#08306b"]
)

VAR_CMAP = {
    "airTmax": "RdBu_r",
    "airTmin": "RdBu_r",
    "precip": PRECIP_CMAP,
}

# precip cells with a historical (obs) climatology below this are excluded
# from the percent-bias calculation -- dividing by a near-zero denominator
# is what produces wild, physically meaningless percentages
MIN_OBS_PRECIP_FOR_PCT = 0.1  # mm/day

# ---------------------------------------------------------------------------
# Despeckle filter: flags isolated single-cell outliers (e.g. coastline/
# island cells where nearest-neighbor regridding between the model's native
# grid and Livneh's grid mismatches, or where an all_touched=True coastal
# clip keeps a mostly-ocean cell with poor underlying data support). A true
# regional bias signal is spatially smooth; a cell that's wildly different
# from every one of its immediate neighbors is very unlikely to be real.
#
# Method: for each cell, compare its value to the median of its (up to 24)
# immediate neighbors. Flag as an outlier if the deviation exceeds
# DESPECKLE_FACTOR times the local median absolute deviation (MAD) of those
# neighbors. Flagged cells are set to NaN (same treatment as any other
# masked-out cell -- shows as blank/white on the map, until fill_isolated_
# nan_gaps below fills them back in from their neighbors).
# ---------------------------------------------------------------------------
DESPECKLE_WINDOW = 5          # 5x5 neighborhood (24 surrounding cells) -- wider
                               # than 3x3 so small islands (Nantucket, Martha's
                               # Vineyard, etc.) still have enough candidate
                               # neighbors even though most of a 3x3 window
                               # around them is ocean/NaN
DESPECKLE_FACTOR = 3.0        # flag if |value - local median| > factor * local MAD
                               # (tightened from 3.5 -- coastline/island clusters
                               # a couple of cells wide were surviving a single
                               # pass; see DESPECKLE_PASSES below)
DESPECKLE_MIN_NEIGHBORS = 2   # lowered from 4 -- a strict minimum was silently
                               # skipping small islands/sparse coastal cells
                               # (exactly the cells most likely to need this
                               # check), since they rarely have >=4 valid
                               # neighbors in any window
DESPECKLE_PASSES = 2          # a cell inside a 2-3-cell-wide bad cluster can
                               # look locally "consistent" with its equally-bad
                               # neighbors on the first pass (their shared MAD
                               # looks small), so it isn't flagged; a second
                               # pass, run after the first pass has already
                               # NaN'd the easy single-cell outliers, catches
                               # most of what's left since the cluster now has
                               # fewer bad neighbors propping up its local MAD


def despeckle_isolated_outliers(da, window=DESPECKLE_WINDOW, factor=DESPECKLE_FACTOR,
                                 min_valid_neighbors=DESPECKLE_MIN_NEIGHBORS, label=""):
    """Flag (set to NaN) grid cells whose value deviates sharply from their
    immediate neighbors, targeting isolated single-cell artifacts (coastline/
    island regrid mismatches) without touching genuinely smooth, spatially
    coherent bias patterns elsewhere on the map."""
    vals = da.values.astype(float)
    ny, nx = vals.shape
    pad = window // 2
    padded = np.pad(vals, pad, mode='edge')

    neighbor_stack = []
    for dy in range(window):
        for dx in range(window):
            if dy == pad and dx == pad:
                continue  # skip the center cell itself
            neighbor_stack.append(padded[dy:dy + ny, dx:dx + nx])
    neighbor_stack = np.stack(neighbor_stack, axis=0)

    with np.errstate(invalid='ignore'):
        n_valid = np.sum(~np.isnan(neighbor_stack), axis=0)
        local_median = np.nanmedian(neighbor_stack, axis=0)
        local_mad = np.nanmedian(np.abs(neighbor_stack - local_median), axis=0)

    local_mad_safe = np.where(local_mad < 1e-6, 1e-6, local_mad)
    deviation = np.abs(vals - local_median) / local_mad_safe

    is_outlier = (deviation > factor) & (n_valid >= min_valid_neighbors) & ~np.isnan(vals)
    n_flagged = int(np.sum(is_outlier))
    if n_flagged > 0:
        tag = f" [{label}]" if label else ""
        print(f"    [despeckle{tag}] flagged {n_flagged} isolated outlier cell(s) as NaN")

    cleaned = np.where(is_outlier, np.nan, vals)
    return xr.DataArray(cleaned, coords=da.coords, dims=da.dims, attrs=da.attrs)


def despeckle_multi_pass(da, passes=DESPECKLE_PASSES, label="", **kwargs):
    """Run despeckle_isolated_outliers repeatedly. A single pass can miss a
    cluster of 2-3 adjacent bad cells (their shared local MAD looks "normal"
    to each other), but once the first pass removes whatever easy, truly
    isolated outliers surrounded that cluster, the cluster's remaining cells
    often stand out clearly against their now-cleaner neighbors on the next
    pass. Returns unchanged data once a pass flags nothing new."""
    for i in range(passes):
        pass_label = f"{label}/pass{i + 1}" if label else f"pass{i + 1}"
        before_nan = int(np.sum(np.isnan(da.values)))
        da = despeckle_isolated_outliers(da, label=pass_label, **kwargs)
        after_nan = int(np.sum(np.isnan(da.values)))
        if after_nan == before_nan:
            break  # nothing new flagged this pass -- later passes won't help either
    return da


# ---------------------------------------------------------------------------
# Gap-fill: fills NaN cells (left behind by despeckle above, or by small
# clip-edge/regrid gaps) using the mean of their valid immediate neighbors --
# but ONLY when enough neighbors actually have data. This never changes a
# cell that already has a value (unlike despeckle, which can only remove
# values, this can only add them back), and it never fabricates a value in
# a large contiguous NaN region (e.g. open ocean) where there isn't enough
# local support to justify a fill. Same approach already used for the S/P
# ratio ensemble maps' blank-spot fix.
# ---------------------------------------------------------------------------
GAPFILL_WINDOW = 3
GAPFILL_MIN_VALID_NEIGHBORS = 4


def fill_isolated_nan_gaps(da, window=GAPFILL_WINDOW, min_valid_neighbors=GAPFILL_MIN_VALID_NEIGHBORS, label=""):
    """Fill NaN cells with the mean of their valid immediate neighbors,
    provided at least min_valid_neighbors of them have real data."""
    vals = da.values.astype(float)
    ny, nx = vals.shape
    pad = window // 2
    padded = np.pad(vals, pad, mode='edge')

    neighbor_stack = []
    for dy in range(window):
        for dx in range(window):
            if dy == pad and dx == pad:
                continue  # skip the center cell itself
            neighbor_stack.append(padded[dy:dy + ny, dx:dx + nx])
    neighbor_stack = np.stack(neighbor_stack, axis=0)

    with np.errstate(invalid='ignore'):
        n_valid = np.sum(~np.isnan(neighbor_stack), axis=0)
        neighbor_mean = np.nanmean(neighbor_stack, axis=0)

    is_nan = np.isnan(vals)
    can_fill = is_nan & (n_valid >= min_valid_neighbors)
    n_filled = int(np.sum(can_fill))
    if n_filled > 0:
        tag = f" [{label}]" if label else ""
        print(f"    [gapfill{tag}] filled {n_filled} NaN cell(s) from neighbor mean")

    filled = np.where(can_fill, neighbor_mean, vals)
    return xr.DataArray(filled, coords=da.coords, dims=da.dims, attrs=da.attrs)


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


HIST_START, HIST_END = '1980-01-01', '2014-12-31'

# ---------------------------------------------------------------------------
# Season definitions for the bias maps. Each spans a set of calendar months;
# "shift_months" lists which of those months belong to the PRECEDING
# calendar year's instance of the season (e.g. December 1999 is part of
# "winter 2000" alongside Jan/Feb 2000, same convention used throughout the
# rest of this project for cold-season/water-year grouping).
# ---------------------------------------------------------------------------
SEASON_DEFINITIONS = {
    "winter": {
        "months": [12, 1, 2],
        "shift_months": [12],
        "label": "Winter (Dec-Feb)",
    },
    "cold_season": {
        "months": [11, 12, 1, 2, 3, 4],
        "shift_months": [11, 12],
        "label": "Cold Season (Nov-Apr)",
    },
    "water_year": {
        "months": [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "shift_months": [10, 11, 12],
        "label": "Water Year (Oct-Sep)",
    },
}


def aggregate_seasonal(da, operation, season_key):
    """Slice to HIST_START-HIST_END, select only the season's months, group
    them into season-years (handling the calendar-year-crossing months via
    SEASON_DEFINITIONS[season_key]['shift_months']), and return one
    days_in_month-weighted seasonal mean per season-year. Any season-year
    missing one or more of its expected months (always true at the very
    start/end of the historical record, since these seasons reach into the
    adjacent calendar year) is dropped rather than averaged in as a partial
    season."""
    season = SEASON_DEFINITIONS[season_key]
    da_sliced = da.sel(time=slice(HIST_START, HIST_END))
    da_season = da_sliced.sel(time=da_sliced['time'].dt.month.isin(season['months']))

    month = da_season['time'].dt.month
    year = da_season['time'].dt.year
    season_year = xr.where(month.isin(season['shift_months']), year + 1, year)
    da_season = da_season.assign_coords(season_year=season_year)

    # count how many of the season's months actually landed in each
    # season-year group, so incomplete edge seasons can be dropped
    month_flag = xr.ones_like(month, dtype=int)
    month_flag = month_flag.assign_coords(season_year=season_year)
    months_present = month_flag.groupby('season_year').sum()
    complete_years = months_present['season_year'].where(
        months_present == len(season['months']), drop=True)

    if operation == "mean":
        def _weighted_seasonal_mean(group):
            weights = group.time.dt.days_in_month
            return group.weighted(weights).mean(dim='time', skipna=True)
        seasonal = da_season.groupby('season_year').map(_weighted_seasonal_mean)
    elif operation == "sum":
        seasonal = da_season.groupby('season_year').sum(dim='time', skipna=True)
    else:
        raise ValueError(f"Unknown aggregation operation: {operation}")

    return seasonal.sel(season_year=complete_years.values)


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


def compute_climatology(var, operation, label, pattern, clip_geom, season_key=None):
    """Load monthly files for one dataset (Livneh or a single model),
    process to an annual OR seasonal time series (season_key=None keeps the
    original full-year behavior; otherwise one of SEASON_DEFINITIONS), clip,
    then collapse (average) across years to a single 2D (lat, lon)
    climatology field."""
    with xr.open_mfdataset(pattern, combine='by_coords', data_vars='all') as ds:
        da = ds[var]
        da = mask_fill_values(da)

        if var == 'precip':
            da = normalize_precip_units(da, label)
        elif var in ('airTmax', 'airTmin'):
            da = normalize_temp_units(da, label)

        if season_key is None:
            period = aggregate_yearly(da, operation)
            group_dim = 'time'
        else:
            period = aggregate_seasonal(da, operation, season_key)
            group_dim = 'season_year'

        period = clip_to_boundary(period, clip_geom)
        period = period.load()

    return period.mean(dim=group_dim, skipna=True)


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
    """Set any cell with |bias| > limit to NaN (hard sanity-bound cutoff,
    independent of the DISPLAY color scale computed below)."""
    n_before = int((~np.isnan(bias_da)).sum())
    masked = bias_da.where(np.abs(bias_da) <= limit)
    n_after = int((~np.isnan(masked)).sum())
    if n_after < n_before:
        tag = f" [{label}]" if label else ""
        print(f"    [mask{tag}] excluded {n_before - n_after} cell(s) with |bias| > {limit}")
    return masked


def plot_bias_map(var, bias_da, region_label, state_borders, out_name, period_label="1980-2014",
                   season_key=""):
    """Single-panel diverging bias map (model ensemble - obs), using the
    fixed DISPLAY_LIMIT[var] +/- color scale (clips/saturates colors at the
    ends -- does not delete any data, unlike BIAS_SCALE_LIMIT)."""
    vals = bias_da.values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"No valid data for {var} ({region_label}); skipping.")
        return
    abs_max = get_display_abs_max(var, vals)
    vmin, vmax = -abs_max, abs_max

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(bias_da['lon'], bias_da['lat'], bias_da.values,
                          cmap=VAR_CMAP[var], vmin=vmin, vmax=vmax, shading='auto')
    try:
        state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])

    warm_word = "wetter" if var == "precip" else "warmer"
    cool_word = "drier" if var == "precip" else "cooler"
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
    cbar.set_label(f'Ensemble bias ({VAR_LABELS[var]})', fontsize=POSTER_FONT["cbar_label"])
    cbar.ax.tick_params(labelsize=POSTER_FONT["cbar_ticks"])
    apply_colorbar_ticks(cbar, var, abs_max)

    ax.set_title(
        f'{region_label} Weighted-Ensemble Bias, Model − Obs, {period_label}: {VAR_LABELS[var]}\n'
        f'Red = {warm_word}, Blue = {cool_word}, White = 0',
        fontsize=POSTER_FONT["title"], fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_bias_map(region_label, bias_dict, state_borders, out_name, period_label="1980-2014",
                            season_key=""):
    """One figure, one panel per variable (airTmax, airTmin, precip), each
    using its own fixed DISPLAY_LIMIT[var] +/- color scale (clips/saturates
    colors at the ends -- does not delete any data, unlike BIAS_SCALE_LIMIT)."""
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
        abs_max = get_display_abs_max(var, vals)
        vmin, vmax = -abs_max, abs_max

        mesh = ax.pcolormesh(da['lon'], da['lat'], da.values, cmap=VAR_CMAP[var],
                              vmin=vmin, vmax=vmax, shading='auto')
        try:
            state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
        except Exception:
            pass
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(VAR_LABELS[var], fontsize=POSTER_FONT["subplot_title"])

        cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=POSTER_FONT["cbar_ticks"])
        apply_colorbar_ticks(cbar, var, abs_max)

    fig.suptitle(f'{region_label} Weighted-Ensemble Bias, Model − Obs, {period_label}',
                 fontsize=POSTER_FONT["suptitle"], fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


print(f"Loading CONUS boundary from {SHAPEFILE_PATH}")
conus_geom, conus_borders = load_conus_boundary()
print(f"Loading Northeast boundary from {SHAPEFILE_PATH}")
ne_geom, ne_borders = load_northeast_boundary()

for season_key, season_info in SEASON_DEFINITIONS.items():
    period_label = f"{season_info['label']}, {HIST_START[:4]}-{HIST_END[:4]}"
    print(f"\n{'=' * 70}\nSEASON: {season_info['label']}\n{'=' * 70}")

    ensemble_bias_conus_all = {}
    ensemble_bias_ne_all = {}

    for var, operation in variables_config.items():
        print(f"\n--- Processing variable: {var} ({season_key}, weighted ensemble bias) ---")

        livneh_var_dir = os.path.join(LIVNEH_BASE_DIR, var)
        livneh_pattern = os.path.join(livneh_var_dir, "wbm_*.nc")
        if not glob.glob(livneh_pattern):
            print(f"No Livneh files found for variable: {var} at {livneh_var_dir}")
            continue

        print(f"Computing Livneh [{var}] {season_key} climatology (CONUS-clipped, common reference grid)...")
        obs_clim = compute_climatology(var, operation, "Livneh", livneh_pattern, conus_geom,
                                        season_key=season_key)

        model_bias_common = {}
        for folder, name in zip(model_folders, model_names):
            if name not in MODEL_WEIGHTS:
                continue  # no point computing a model that'll be excluded anyway
            model_pattern = os.path.join(folder, "monthly", var, "wbm_*.nc")
            if not glob.glob(model_pattern):
                print(f"  Skipping {name} [{var}]: no files found at {model_pattern}")
                continue

            try:
                print(f"  Computing {name} [{var}] {season_key} climatology...")
                m_clim = compute_climatology(var, operation, name, model_pattern, conus_geom,
                                              season_key=season_key)
                # regrid onto Livneh's grid so every model's bias lives on the
                # SAME common grid before combining -- required since each
                # model's own climatology may be on its own native grid
                # bilinear (not nearest-neighbor) regridding -- nearest-neighbor
                # can map several adjacent target cells to the SAME source cell
                # wherever the two grids' resolution/alignment don't line up
                # (common at complex coastlines/islands), producing small blocks
                # of identical, often poorly-matched values. Linear interpolation
                # blends between surrounding source cells instead, so it can't
                # produce that repeated-block artifact. Cells outside the native
                # grid's convex hull become NaN (no data) rather than a wrong
                # extrapolated value, which is the correct trade-off here.
                m_clim_common = m_clim.interp_like(obs_clim, method='linear')
                model_bias = m_clim_common - obs_clim
                # despeckle BEFORE adding to the ensemble -- a coastline/island
                # regrid artifact in one model shouldn't get baked into the
                # weighted ensemble at that cell (see DESPECKLE_* config above)
                model_bias = despeckle_multi_pass(model_bias, label=f"{name}/{var}/{season_key}")
                model_bias_common[name] = model_bias
            except Exception as e:
                print(f"  Skipping model {name} [{var}] due to calculation mismatch: {e}")
                continue

        if not model_bias_common:
            print(f"No models produced valid bias results for {var} ({season_key}); skipping ensemble maps.")
            continue

        print(f"  Building weighted ensemble ({len(model_bias_common)} models)...")
        ensemble_bias_conus = compute_weighted_ensemble(model_bias_common, MODEL_WEIGHTS, label=f"{var}/{season_key}")

        # despeckle the ensemble itself (ALL variables, not just precip) --
        # the weighted average can still produce a residual coastline/island
        # artifact that wasn't quite strong enough to get flagged in any one
        # model, but becomes visible once several models' edge effects line
        # up at the same cell
        ensemble_bias_conus = despeckle_multi_pass(ensemble_bias_conus, label=f"ensemble/{var}/{season_key}")

        if var == 'precip':
            # convert absolute bias (mm/day) to percent bias relative to obs;
            # mathematically identical to averaging per-model percent biases
            # first, since obs is the same fixed denominator for every model
            # at a given cell (the weighted mean above is linear). Cells where
            # obs is too close to zero are masked rather than left to blow up
            # to a meaningless percentage.
            safe_obs = obs_clim.where(obs_clim >= MIN_OBS_PRECIP_FOR_PCT)
            ensemble_bias_conus = (ensemble_bias_conus / safe_obs) * 100
            # re-despeckle after the percent-bias conversion -- dividing by a
            # small-but-not-excluded obs value can still amplify any residual
            # single-cell noise into a large percentage
            ensemble_bias_conus = despeckle_multi_pass(ensemble_bias_conus, label=f"ensemble/{var}/{season_key}/pct")

        # gap-fill NaN cells left by despeckle (and any small clip-edge/
        # regrid gaps) using their valid neighbors' mean, BEFORE the hard
        # sanity-bound mask below -- this is what removes the white/blank
        # speckles from the final maps
        ensemble_bias_conus = fill_isolated_nan_gaps(ensemble_bias_conus, label=f"ensemble/{var}/{season_key}")

        ensemble_bias_conus = mask_beyond_limit(ensemble_bias_conus, BIAS_SCALE_LIMIT[var], label=f"{var}/{season_key}")

        plot_bias_map(var, ensemble_bias_conus, "CONUS", conus_borders,
                      f"conus_{var}_bias_ensemble_weighted_{season_key}.png",
                      period_label=period_label, season_key=season_key)

        print(f"  Clipping ensemble bias to Northeast...")
        ensemble_bias_ne = clip_to_boundary(ensemble_bias_conus, ne_geom)
        plot_bias_map(var, ensemble_bias_ne, "Northeast", ne_borders,
                      f"northeast_{var}_bias_ensemble_weighted_{season_key}.png",
                      period_label=period_label, season_key=season_key)

        ensemble_bias_conus_all[var] = ensemble_bias_conus
        ensemble_bias_ne_all[var] = ensemble_bias_ne

    if ensemble_bias_conus_all:
        plot_combined_bias_map("CONUS", ensemble_bias_conus_all, conus_borders,
                                f"conus_allvars_bias_ensemble_weighted_{season_key}.png",
                                period_label=period_label, season_key=season_key)
    if ensemble_bias_ne_all:
        plot_combined_bias_map("Northeast", ensemble_bias_ne_all, ne_borders,
                                f"northeast_allvars_bias_ensemble_weighted_{season_key}.png",
                                period_label=period_label, season_key=season_key)

print("\nProcess complete")
