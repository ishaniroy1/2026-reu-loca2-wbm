"""
sp_ratio_future_timeseries.py
==============================
Northeast-region S/P ratio (snow / total precip) timeseries, 2015-2100,
under all three SSP scenarios -- structured to match the existing
future_timeseries_plot.py (temp/precip) script: thin individual-model
trendlines per scenario, a bolder multi-model ensemble mean line per
scenario, and era shading (early/mid/late century).

IMPORTANT DIFFERENCE FROM future_timeseries_plot.py: that script bridges
from Livneh OBSERVATIONS into the future ensembles, because airTmax/airTmin/
precip all have an independent observational record to bias-check against.
S/P ratio does NOT -- there's no independent observed snow/precip-ratio
dataset here, only what WBM produces from each GCM's own output. So instead
of an obs line, this script bridges from the GCM-DRIVEN HISTORICAL ENSEMBLE
(1980-2014, same models/weighting convention as the future ensembles, shown
with its own thin per-model lines + bold mean, explicitly labeled as model-
driven rather than "observations").

S/P ratio is computed water-year by water-year (Oct-Sep, labeled by the
ending/Sep year -- same convention as the rest of the project):
  1. spatial-mean monthly precip and snowFall over the Northeast, per model
  2. summed into water-year totals (sum first, then ratio -- NOT an average
     of monthly ratios, which would be dominated by near-zero-precip months)
  3. ratio = water-year snow sum / water-year precip sum
A water-year bin is kept only if all 12 of its months are present (drops
the always-partial year at each end of a model's record).

Run with: python sp_ratio_future_timeseries.py
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask

# paths (same layout/convention as future_timeseries_plot.py and the bias
# map scripts -- historical and future runs live in sibling directories)
MODEL_DIR_HIST = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_historical"
MODEL_DIR_FUT = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_future"
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots")
SHAPEFILE_PATH = os.path.expanduser(
    "~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# region setup
nca_ne_states = ['ME', 'NH', 'VT', 'MA', 'RI',
                 'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'WV']
states_gdf = gpd.read_file(SHAPEFILE_PATH)
northeast_gdf = states_gdf[states_gdf['STUSPS'].isin(nca_ne_states)]

VAR_PRECIP = "precip"
VAR_SNOW = "snowFall"

# set up 3 scenarios (same labels/colors as future_timeseries_plot.py)
scenarios = {
    'ssp245': {'label': 'SSP2-4.5 (Middle of the Road)', 'color': 'darkorange'},
    'ssp370': {'label': 'SSP3-7.0 (Regional Rivalry)', 'color': 'purple'},
    'ssp585': {'label': 'SSP5-8.5 (Fossil-fueled Development)', 'color': 'crimson'}
}

HIST_COLOR = 'black'


def model_water_year_sp_ratio(folder, ne_gdf):
    """
    One model run's Northeast-averaged water-year S/P ratio series.
    Returns (years, ratio_values) as numpy arrays, years labeled by the
    ending (Sep) calendar year of each water year -- or None if this run
    doesn't have both monthly precip and snowFall output.
    """
    precip_pattern = os.path.join(folder, "monthly", VAR_PRECIP, "wbm_*.nc")
    snow_pattern = os.path.join(folder, "monthly", VAR_SNOW, "wbm_*.nc")

    if not glob.glob(precip_pattern) or not glob.glob(snow_pattern):
        return None

    with xr.open_mfdataset(precip_pattern, combine='by_coords', data_vars='all') as precip_ds, \
         xr.open_mfdataset(snow_pattern, combine='by_coords', data_vars='all') as snow_ds:

        p_mask = regionmask.mask_3D_geopandas(
            ne_gdf, precip_ds['lon'], precip_ds['lat']).any(dim='region')
        s_mask = regionmask.mask_3D_geopandas(
            ne_gdf, snow_ds['lon'], snow_ds['lat']).any(dim='region')

        # spatial mean (mm/day rate) -> approximate monthly total, same
        # *30.5 convention used in future_timeseries_plot.py
        precip_spatial = precip_ds[VAR_PRECIP].where(p_mask).mean(dim=['lat', 'lon']) * 30.5
        snow_spatial = snow_ds[VAR_SNOW].where(s_mask).mean(dim=['lat', 'lon']) * 30.5

        # water-year (Oct-Sep) sums: a YS-OCT-anchored annual bin starting
        # October of calendar year X covers Oct(X)-Sep(X+1) -- i.e. water
        # year X+1 under the ending-year labeling convention used
        # throughout this project. (YS-OCT is the modern pandas/xarray
        # alias; older versions call this AS-OCT -- same meaning.)
        # .load() precip_counts eagerly BEFORE building the boolean mask --
        # dask refuses boolean/drop-indexing with a still-lazy indexer, so
        # `complete` has to be a plain in-memory array before it's used
        # below in .where(complete, drop=True)
        precip_counts = precip_spatial.resample(time='YS-OCT').count().load()
        precip_wy = precip_spatial.resample(time='YS-OCT').sum()
        snow_wy = snow_spatial.resample(time='YS-OCT').sum()

        complete = (precip_counts == 12)
        precip_wy = precip_wy.where(complete, drop=True).load()
        snow_wy = snow_wy.where(complete, drop=True).load()

        if precip_wy.sizes.get('time', 0) == 0:
            return None

        ratio = snow_wy / precip_wy.where(precip_wy > 0)
        years = precip_wy['time'].dt.year.values + 1
        return years, ratio.values


def ensemble_series(folders, ne_gdf, tag=""):
    """Compute each model's water-year S/P ratio series, align them on
    year (outer join -- a model missing a given year just contributes NaN
    there, which .mean(skipna=True) ignores), and return
    (per_model_dict[name] -> pd.Series, ensemble pd.Series)."""
    per_model = {}
    for folder in folders:
        name = os.path.basename(folder).split('_')[0]
        try:
            result = model_water_year_sp_ratio(folder, ne_gdf)
            if result is None:
                print(f"  Skipping {name} ({tag}): no precip/snowFall monthly output found")
                continue
            years, ratio = result
            per_model[name] = pd.Series(ratio, index=years)
        except Exception as e:
            print(f"  Skipping {name} ({tag}): {e}")

    if not per_model:
        return per_model, pd.Series(dtype=float)

    df = pd.DataFrame(per_model)
    ensemble = df.mean(axis=1, skipna=True)
    return per_model, ensemble


print("Computing Northeast historical (GCM-driven) ensemble S/P ratio, 1980-2014...")
hist_folders = sorted(glob.glob(os.path.join(MODEL_DIR_HIST, "*_newprcp")))
hist_per_model, hist_ensemble = ensemble_series(hist_folders, northeast_gdf, tag="historical")

fig, ax = plt.subplots(figsize=(14, 6.5))

if not hist_ensemble.empty:
    for name, series in hist_per_model.items():
        ax.plot(series.index, series.values, color=HIST_COLOR,
                linewidth=0.6, alpha=0.25, zorder=1, label='_nolegend_')
    ax.plot(hist_ensemble.index, hist_ensemble.values, color=HIST_COLOR,
            linewidth=2.5, zorder=3,
            label=f'Historical Ensemble, GCM-driven ({len(hist_per_model)} models, 1980-2014)')
    bridge_year = int(hist_ensemble.index[-1])
    bridge_value = float(hist_ensemble.values[-1])
else:
    print("  WARNING: no historical ensemble available -- future lines will not be bridged.")
    bridge_year, bridge_value = None, None

for ssp_key, ssp_info in scenarios.items():
    print(f"Computing Northeast future ensemble S/P ratio for {ssp_key}, 2015-2100...")
    ssp_folders = sorted(glob.glob(os.path.join(MODEL_DIR_FUT, f"*{ssp_key}*_newprcp")))
    if not ssp_folders:
        continue

    per_model, ensemble = ensemble_series(ssp_folders, northeast_gdf, tag=ssp_key)
    if ensemble.empty:
        continue

    for name, series in per_model.items():
        ax.plot(series.index, series.values, color=ssp_info['color'],
                linewidth=0.6, alpha=0.35, zorder=1, label='_nolegend_')

    years_connected = ensemble.index.values
    values_connected = ensemble.values
    if bridge_year is not None:
        years_connected = np.concatenate([[bridge_year], years_connected])
        values_connected = np.concatenate([[bridge_value], values_connected])

    ax.plot(years_connected, values_connected, color=ssp_info['color'],
            linewidth=2, zorder=3,
            label=f"{ssp_info['label']} ({len(per_model)} models)")

# era shading (same breakpoints as future_timeseries_plot.py)
ax.axvspan(2015, 2040, color='royalblue', alpha=0.03)
ax.axvspan(2041, 2070, color='darkorange', alpha=0.03)
ax.axvspan(2071, 2100, color='crimson', alpha=0.03)

ax.autoscale_view()
y_lo, y_hi = ax.get_ylim()
y_text_pos = y_lo + (y_hi - y_lo) * 0.92

ax.text(2027, y_text_pos, 'Early-Century\n(2015-2040)', ha='center',
        va='top', fontsize=9, style='italic', color='navy')
ax.text(2055, y_text_pos, 'Mid-Century\n(2041-2070)', ha='center',
        va='top', fontsize=9, style='italic', color='darkgoldenrod')
ax.text(2085, y_text_pos, 'Late-Century\n(2071-2100)', ha='center',
        va='top', fontsize=9, style='italic', color='darkred')

ax.set_ylabel("S/P Ratio (snow / total precip)", fontsize=12)
ax.set_xlabel("Water Year (Oct-Sep, labeled by ending year)", fontsize=12)
ax.set_title(
    "Northeast Region S/P Ratio: Historical Ensemble → Future Projections (1980-2100)\n"
    "No independent S/P-ratio observations exist -- historical line is GCM/WBM-driven, not observed",
    fontsize=13, fontweight='bold')

ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
plt.tight_layout()

output_plot = os.path.join(OUTPUT_DIR, "sp_ratio_future_timeseries.png")
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"Completed plot written to {output_plot}")
