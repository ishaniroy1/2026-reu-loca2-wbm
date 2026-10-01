"""
sp_ratio_ensemble_maps.py
==========================
Weighted-ensemble S/P (snow-to-total-precipitation) ratio maps for the
Northeast snowmelt/runoff pipeline, covering the three seasonal windows
defined in phase_partition.py: Winter (Dec-Feb), Cold Season (Nov-Apr), and
Water Year (Oct-Sep).

Unlike the temp/precip bias maps in LOCA2-WBM_code/ensemble_weighted_bias_maps.py,
there is no independent observational S/P-ratio dataset to compare against
(see config.py's note on LIVNEH_MONTHLY_DIR being validation-only, not the
source for this pipeline's S/P-ratio analysis). So this script produces the
weighted-ensemble S/P ratio itself -- across the 16 LOCA2-WBM historical
model runs -- rather than a model-minus-obs bias.

Each model's daily output already contains the WBM's own snowFall variable
(computed with the same -1C/+1C linear ramp as phase_partition.py, applied
internally by WBM), so this script calls water_year_sp_ratio / winter_sp_ratio
/ cold_season_nov_apr_sp_ratio directly on (precip, snowFall) -- it does NOT
call partition()/process_source_to_sp_ratio(), which exist for the case of
re-deriving snow/rain from temperature when a native snowFall variable isn't
already available.

Reuses the same methodology fixes established for the bias maps: bilinear
(not nearest-neighbor) regridding onto a common grid before combining
models, and the despeckle filter for isolated single-cell/small-cluster
regrid artifacts at coastlines and islands.
"""

import glob
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter

import config
from phase_partition import SP_RATIO_FUNCTIONS

OUTPUT_DIR = config.PLOTS_DIR / "sp_ratio_ensemble"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_LABELS = {
    "water_year": "Water Year (Oct-Sep)",
    "winter": "Winter (Dec-Feb)",
    "cold_season_nov_apr": "Cold Season (Nov-Apr)",
}

NON_CONUS = {
    "Alaska",
    "Hawaii",
    "Puerto Rico",
    "Guam",
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "United States Virgin Islands",
}

FILL_VALUE = -9999.0
FILL_ABS_THRESHOLD = 1e10

# ---------------------------------------------------------------------------
# Same weighted-ensemble weights as LOCA2-WBM_code/ensemble_weighted_bias_maps.py.
# Keep these two lists in sync if the weighting scheme ever changes.
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

# ---------------------------------------------------------------------------
# Despeckle filter -- same tuned settings as the bias maps (5x5 window, low
# min-neighbor requirement so sparse island/coastal cells still get
# evaluated; see the chat history for why these specific values were chosen).
# ---------------------------------------------------------------------------
DESPECKLE_WINDOW = 5
DESPECKLE_FACTOR = 3.5
DESPECKLE_MIN_NEIGHBORS = 2


def despeckle_isolated_outliers(da, window=DESPECKLE_WINDOW, factor=DESPECKLE_FACTOR,
                                 min_valid_neighbors=DESPECKLE_MIN_NEIGHBORS, label=""):
    """Flag (set to NaN) grid cells whose value deviates sharply from their
    immediate neighbors, targeting isolated single-cell/small-cluster
    artifacts (coastline/island regrid mismatches) without touching
    genuinely smooth, spatially coherent S/P ratio patterns elsewhere."""
    vals = da.values.astype(float)
    ny, nx = vals.shape
    pad = window // 2
    padded = np.pad(vals, pad, mode='edge')

    neighbor_stack = []
    for dy in range(window):
        for dx in range(window):
            if dy == pad and dx == pad:
                continue
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


# --- boundary / clipping helpers (same pattern as the bias-map scripts) ---

def load_conus_boundary():
    states = gpd.read_file(config.STATES_SHAPEFILE)
    name_col = "NAME" if "NAME" in states.columns else config.STATE_ID_FIELD
    conus_states = states[~states[name_col].isin(NON_CONUS)].copy()
    conus_states = conus_states.to_crs("EPSG:4326")
    return conus_states.dissolve().geometry, conus_states


def load_northeast_boundary():
    states = gpd.read_file(config.STATES_SHAPEFILE)
    ne_states = states[states[config.STATE_ID_FIELD].isin(config.NE_STATES)].copy()
    ne_states = ne_states.to_crs("EPSG:4326")
    if ne_states.empty:
        raise ValueError(f"No features matched NE_STATES={config.NE_STATES} "
                          f"in {config.STATES_SHAPEFILE}.")
    return ne_states.dissolve().geometry, ne_states


def clip_to_boundary(da_or_ds, geom, lat_name=None, lon_name=None, all_touched=True):
    if lat_name is None:
        lat_name = 'lat' if 'lat' in da_or_ds.coords else ('latitude' if 'latitude' in da_or_ds.coords else None)
    if lon_name is None:
        lon_name = 'lon' if 'lon' in da_or_ds.coords else ('longitude' if 'longitude' in da_or_ds.coords else None)
    if not lat_name or not lon_name:
        raise KeyError(f"Could not find lat/lon coordinates. Found: {list(da_or_ds.coords.keys())}")

    lons = da_or_ds[lon_name].values
    if np.any(lons > 180):
        da_or_ds = da_or_ds.assign_coords(
            {lon_name: np.where(da_or_ds[lon_name] > 180, da_or_ds[lon_name] - 360, da_or_ds[lon_name])}
        )
        da_or_ds = da_or_ds.sortby(lon_name)

    obj = da_or_ds.rio.write_crs("EPSG:4326", inplace=False)
    obj = obj.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name, inplace=False)
    obj = obj.rio.write_nodata(np.nan, inplace=False)
    return obj.rio.clip(geom, crs="EPSG:4326", drop=True, all_touched=all_touched)


def mask_fill_values(da):
    return da.where((da != FILL_VALUE) & (np.abs(da) < FILL_ABS_THRESHOLD))


def compute_weighted_ensemble(data_dict, weights, label=""):
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


def collapse_to_climatology(sp_da):
    """water_year_sp_ratio's output dim is 'water_year'; winter_sp_ratio and
    cold_season_nov_apr_sp_ratio's is 'season_year' -- find whichever
    non-spatial dim is present and average over it to get a single 2D
    climatological S/P ratio field."""
    year_dim = [d for d in sp_da.dims if d not in ('lat', 'lon')]
    if len(year_dim) != 1:
        raise ValueError(f"Expected exactly one non-spatial dim on sp_da, found: {sp_da.dims}")
    return sp_da.mean(dim=year_dim[0], skipna=True)


def get_reference_grid():
    """Use the Livneh monthly grid as the common regrid target -- not for
    its values (there's no obs S/P ratio), just for a fixed, consistent
    CONUS grid that matches the bias-map figures elsewhere in this project,
    so maps across the two script outputs line up visually."""
    pattern = os.path.join(str(config.LIVNEH_MONTHLY_DIR), "airTmax", "wbm_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Livneh reference files found at {pattern} "
                                 f"(needed only for its grid, not its values).")
    with xr.open_dataset(files[0]) as ds:
        template = xr.zeros_like(ds['airTmax'].isel(time=0), dtype=float).load()
    return template


def plot_sp_map(sp_da, region_label, season_key, state_borders, out_name):
    vals = sp_da.values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"No valid data for {season_key} ({region_label}); skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    mesh = ax.pcolormesh(sp_da['lon'], sp_da['lat'], sp_da.values,
                          cmap='Blues', vmin=0, vmax=1, shading='auto')
    try:
        state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
    except Exception:
        pass
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.85, label='S/P Ratio')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_title(
        f'{region_label} Weighted-Ensemble S/P Ratio, Historical '
        f'({config.HIST_START[:4]}-{config.HIST_END[:4]})\n{SEASON_LABELS[season_key]}',
        fontsize=11, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_sp_map(region_label, sp_dict, state_borders, out_name):
    """One figure, one panel per season, shared 0-1 color scale."""
    season_order = [s for s in SEASON_LABELS if s in sp_dict]
    fig, axes = plt.subplots(1, len(season_order), figsize=(6 * len(season_order), 6),
                              constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    mesh = None
    for ax, season_key in zip(axes, season_order):
        da = sp_dict[season_key]
        mesh = ax.pcolormesh(da['lon'], da['lat'], da.values, cmap='Blues',
                              vmin=0, vmax=1, shading='auto')
        try:
            state_borders.boundary.plot(ax=ax, linewidth=0.5, color='black', alpha=0.6)
        except Exception:
            pass
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(SEASON_LABELS[season_key], fontsize=11)

    cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.8, label='S/P Ratio')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.suptitle(f'{region_label} Weighted-Ensemble S/P Ratio, Historical '
                 f'({config.HIST_START[:4]}-{config.HIST_END[:4]})',
                 fontsize=14, fontweight='bold')

    out_path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print(f"Loading CONUS boundary from {config.STATES_SHAPEFILE}")
conus_geom, conus_borders = load_conus_boundary()
print(f"Loading Northeast boundary from {config.STATES_SHAPEFILE}")
ne_geom, ne_borders = load_northeast_boundary()

print("Loading reference grid (Livneh, for regridding only -- not used as obs)...")
ref_grid = get_reference_grid()

model_folders = sorted(glob.glob(os.path.join(str(config.LOCA2WBM_HIST_DIR), config.MODEL_FOLDER_GLOB)))
model_names = [os.path.basename(f).split('_')[0] for f in model_folders]
if config.GCM_LIST is not None:
    keep = set(config.GCM_LIST)
    model_folders = [f for f, n in zip(model_folders, model_names) if n in keep]
    model_names = [n for n in model_names if n in keep]

_unweighted = sorted(set(model_names) - set(MODEL_WEIGHTS))
if _unweighted:
    print(f"NOTE: no weight defined for {_unweighted} -- these will be excluded.")

sp_conus_all = {}
sp_ne_all = {}

for season_key, sp_fn in SP_RATIO_FUNCTIONS.items():
    print(f"\n{'=' * 70}\nSEASON: {SEASON_LABELS[season_key]}\n{'=' * 70}")

    model_sp_common = {}
    for folder, name in zip(model_folders, model_names):
        if name not in MODEL_WEIGHTS:
            continue
        daily_pattern = os.path.join(folder, config.DAILY_SUBDIR, config.DAILY_FILE_GLOB)
        if not glob.glob(daily_pattern):
            print(f"  Skipping {name}: no daily files found at {daily_pattern}")
            continue

        try:
            print(f"  Computing {name} {season_key} S/P ratio...")
            with xr.open_mfdataset(daily_pattern, combine='by_coords', data_vars='all') as ds:
                ds = ds[[config.VAR_PRECIP_TOTAL, config.VAR_SNOW]]
                ds[config.VAR_PRECIP_TOTAL] = mask_fill_values(ds[config.VAR_PRECIP_TOTAL])
                ds[config.VAR_SNOW] = mask_fill_values(ds[config.VAR_SNOW])
                ds = ds.sel(time=slice(config.HIST_START, config.HIST_END))

                sp_by_year = sp_fn(ds, pr_var=config.VAR_PRECIP_TOTAL, snow_var=config.VAR_SNOW)
                sp_clim = collapse_to_climatology(sp_by_year)
                sp_clim = clip_to_boundary(sp_clim, conus_geom)
                sp_clim = sp_clim.load()

            # bilinear regrid onto the common reference grid -- see the
            # bias-map scripts for why nearest-neighbor produces blocky
            # artifacts at coastlines/islands
            sp_clim_common = sp_clim.interp_like(ref_grid, method='linear')
            sp_clim_common = despeckle_isolated_outliers(sp_clim_common, label=f"{name}/{season_key}")
            model_sp_common[name] = sp_clim_common
        except Exception as e:
            print(f"  Skipping model {name} [{season_key}] due to calculation mismatch: {e}")
            continue

    if not model_sp_common:
        print(f"No models produced valid S/P ratio results for {season_key}; skipping maps.")
        continue

    print(f"  Building weighted ensemble ({len(model_sp_common)} models)...")
    ensemble_sp_conus = compute_weighted_ensemble(model_sp_common, MODEL_WEIGHTS, label=season_key)
    ensemble_sp_conus = ensemble_sp_conus.clip(min=0, max=1)

    plot_sp_map(ensemble_sp_conus, "CONUS", season_key, conus_borders,
                f"conus_sp_ratio_ensemble_weighted_{season_key}.png")

    print("  Clipping ensemble S/P ratio to Northeast...")
    ensemble_sp_ne = clip_to_boundary(ensemble_sp_conus, ne_geom)
    plot_sp_map(ensemble_sp_ne, "Northeast", season_key, ne_borders,
                f"northeast_sp_ratio_ensemble_weighted_{season_key}.png")

    sp_conus_all[season_key] = ensemble_sp_conus
    sp_ne_all[season_key] = ensemble_sp_ne

if sp_conus_all:
    plot_combined_sp_map("CONUS", sp_conus_all, conus_borders,
                          "conus_sp_ratio_ensemble_weighted_allseasons.png")
if sp_ne_all:
    plot_combined_sp_map("Northeast", sp_ne_all, ne_borders,
                          "northeast_sp_ratio_ensemble_weighted_allseasons.png")

print("\nProcess complete")
