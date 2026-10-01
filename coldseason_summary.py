"""
Cold-Season Warming & Rain/Snow Partitioning Summary
------------------------------------------------------
Produces figures + a text summary answering:
  1. Historical cold-season (Nov-Apr) warming in the NE US: how much, over what period?
  2. Historical shift in the snow-to-precipitation (S/P) ratio.
  3. A written summary of the LOCA2-WBM simulation set used (models, scenarios,
     time periods, timestep, variables) and the rain/snow partitioning method.
  4. Historical + future trends in cold-season Tmin, Tmax, and precipitation.
  5. Historical vs. end-of-century S/P ratio, incl. rough % of the region
     that is snow-dominated (S/P > 0.5) now vs. late-century under each SSP.

Outputs are written to OUTPUT_DIR (a new subfolder under plots/), separate
from the future-timeseries plots produced by new_timeseries.py.
"""

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask

# ============================================================
# CONFIGURATION - adjust to match your project setup
# ============================================================
LIVNEH_REF = os.path.expanduser("~/LOCA2-WBM_code/livneh_monthly_1980-2013.nc")
MODEL_DIR = "/net/nfs/echo/ankaa/LOCA2-WBM_output/LOCA2-WBM_future"
OUTPUT_DIR = os.path.expanduser("~/LOCA2-WBM_code/plots/coldseason_summary")
SHAPEFILE_PATH = os.path.expanduser(
    "~/LOCA2-WBM_code/shapefiles/states/cb_2025_us_state_5m.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cold season definition (months). Nov/Dec are assigned to the FOLLOWING
# winter year, so e.g. Nov-Dec 2020 + Jan-Apr 2021 = "winter 2021".
COLD_SEASON_MONTHS = [11, 12, 1, 2, 3, 4]

# Rain/snow linear-ramp thresholds (deg C), applied to (Tmax+Tmin)/2, matching WBM.
# Below T_SNOW -> 100% snow. Above T_RAIN -> 100% rain. Linear in between
# (e.g. 0 C = 50% rain / 50% snow).
T_SNOW = -1.0
T_RAIN = 1.0

# Threshold defining "snow-dominated" for the % of region summary
SNOW_DOMINANT_THRESHOLD = 0.5

# Time period definitions
HIST_START, HIST_END = "1980-01-01", "2013-12-31"
EARLY_START, EARLY_END = "2015-01-01", "2040-12-31"
MID_START, MID_END = "2041-01-01", "2070-12-31"
LATE_START, LATE_END = "2071-01-01", "2100-12-31"

nca_ne_states = ['ME', 'NH', 'VT', 'MA', 'RI',
                  'CT', 'NY', 'NJ', 'PA', 'DE', 'MD', 'WV']

scenarios = {
    'ssp245': {'label': 'SSP2-4.5 (Middle of the Road)', 'color': 'darkorange'},
    'ssp370': {'label': 'SSP3-7.0 (Regional Rivalry)', 'color': 'purple'},
    'ssp585': {'label': 'SSP5-8.5 (Fossil-fueled Development)', 'color': 'crimson'}
}

var_mapping = {
    'airTmax': 'Tmax',
    'airTmin': 'Tmin',
    'precip': 'Prec',
}

summary_lines = []


def log(msg):
    print(msg)
    summary_lines.append(msg)


# ============================================================
# HELPERS
# ============================================================

def snow_fraction(tmean):
    frac = (T_RAIN - tmean) / (T_RAIN - T_SNOW)
    return np.clip(frac, 0.0, 1.0)


def select_cold_season(da):
    return da.sel(time=da['time'].dt.month.isin(COLD_SEASON_MONTHS))


def assign_winter_year(da):
    month = da['time'].dt.month
    year = da['time'].dt.year
    winter_year = xr.where(month >= 11, year + 1, year)
    return da.assign_coords(winter_year=winter_year)


def cold_season_annual_mean(da):
    """Cold-season-selected DataArray -> one value per winter year."""
    da = select_cold_season(da)
    da = assign_winter_year(da)
    return da.groupby('winter_year').mean('time')


def linear_trend(x, y):
    """Returns (slope_per_year, intercept, total_change_over_period)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    total_change = slope * (x.max() - x.min())
    return slope, intercept, total_change


def region_mask(lat, lon, gdf):
    return regionmask.mask_3D_geopandas(gdf, lon, lat).any(dim='region')


def area_weighted_pct_above(field_2d, mask, threshold):
    """field_2d: DataArray(lat, lon). Returns % of masked area where field > threshold,
    weighted by cos(latitude) to roughly account for grid-cell area."""
    weights = np.cos(np.deg2rad(field_2d['lat']))
    weights_2d = xr.ones_like(field_2d) * weights
    masked_weights = weights_2d.where(mask)
    total_weight = float(masked_weights.sum())
    above_weight = float(masked_weights.where(field_2d > threshold).sum())
    return 100.0 * above_weight / total_weight


def compute_grid_sp_ratio(tmax, tmin, precip):
    """tmax/tmin/precip: DataArray(time, lat, lon), cold-season months only.
    Returns a (lat, lon) map of sum(snow)/sum(precip) over all time steps."""
    tmean = (tmax + tmin) / 2.0
    frac = snow_fraction(tmean)
    snow = precip * frac
    return snow.sum(dim='time') / precip.sum(dim='time')


def load_scenario_variable(ssp_key, var_key, start, end):
    """Returns list of (model_name, DataArray) for cold-season-selected,
    time-sliced monthly output across all models available for this scenario."""
    ssp_folders = sorted(glob.glob(os.path.join(MODEL_DIR, f"*{ssp_key}*_newprcp")))
    series = []
    for folder in ssp_folders:
        model_name = os.path.basename(folder).split('_')[0]
        pattern = os.path.join(folder, "monthly", var_key, "wbm_*.nc")
        if not glob.glob(pattern):
            continue
        try:
            with xr.open_mfdataset(pattern, combine='by_coords', data_vars='all') as ds:
                da = ds[var_key].sel(time=slice(start, end))
                da = select_cold_season(da)
                if var_key == 'precip':
                    da = da * 30.5  # mm/day rate -> approx monthly total
                series.append((model_name, da.load()))
        except Exception as e:
            print(f"  Skipping {model_name} ({ssp_key}, {var_key}): {e}")
            continue
    return series


def ensemble_scenario_sp_map(ssp_key, start, end):
    """Per-model S/P ratio map, averaged across models (ensemble mean)."""
    tmax = dict(load_scenario_variable(ssp_key, 'airTmax', start, end))
    tmin = dict(load_scenario_variable(ssp_key, 'airTmin', start, end))
    precip = dict(load_scenario_variable(ssp_key, 'precip', start, end))
    common_models = sorted(set(tmax) & set(tmin) & set(precip))
    if not common_models:
        return None, []
    sp_maps = []
    for m in common_models:
        sp = compute_grid_sp_ratio(tmax[m], tmin[m], precip[m])
        sp_maps.append(sp)
    sp_ensemble = xr.concat(sp_maps, dim='model').mean(dim='model')
    return sp_ensemble, common_models


def ensemble_scenario_regional_series(ssp_key, var_key, mask_gdf):
    """Regional cold-season annual mean, ensemble-averaged across models."""
    series = load_scenario_variable(ssp_key, var_key, "2015-01-01", "2100-12-31")
    if not series:
        return None
    per_model = []
    for model_name, da in series:
        mask = region_mask(da['lat'], da['lon'], mask_gdf)
        regional = da.where(mask).mean(dim=['lat', 'lon'])
        annual = cold_season_annual_mean(regional)
        per_model.append(annual)
    ensemble = xr.concat(per_model, dim='model').mean(dim='model')
    return ensemble


# ============================================================
# SETUP
# ============================================================
log("=" * 60)
log("COLD-SEASON WARMING & S/P RATIO SUMMARY")
log("=" * 60)

states_gdf = gpd.read_file(SHAPEFILE_PATH)
northeast_gdf = states_gdf[states_gdf['STUSPS'].isin(nca_ne_states)]

with xr.open_dataset(LIVNEH_REF) as obs_ds:
    obs_mask = region_mask(obs_ds['lat'], obs_ds['lon'], northeast_gdf)
    obs_tmax = obs_ds[var_mapping['airTmax']].sel(time=slice(HIST_START, HIST_END))
    obs_tmin = obs_ds[var_mapping['airTmin']].sel(time=slice(HIST_START, HIST_END))
    obs_precip = obs_ds[var_mapping['precip']].sel(time=slice(HIST_START, HIST_END))

    # regional cold-season annual series (spatial mean first)
    obs_tmax_regional = obs_tmax.where(obs_mask).mean(dim=['lat', 'lon'])
    obs_tmin_regional = obs_tmin.where(obs_mask).mean(dim=['lat', 'lon'])

    obs_tmax_annual = cold_season_annual_mean(obs_tmax_regional).load()
    obs_tmin_annual = cold_season_annual_mean(obs_tmin_regional).load()

    # grid-level S/P ratio map for the full historical period (for maps + % area)
    obs_tmax_cs = select_cold_season(obs_tmax).load()
    obs_tmin_cs = select_cold_season(obs_tmin).load()
    obs_precip_cs = select_cold_season(obs_precip).load()
    hist_sp_map = compute_grid_sp_ratio(obs_tmax_cs, obs_tmin_cs, obs_precip_cs)

    # regional S/P ratio annual series (for the historical trend figure)
    hist_tmean_regional = ((obs_tmax_cs + obs_tmin_cs) / 2.0).where(obs_mask).mean(dim=['lat', 'lon'])
    hist_precip_regional = obs_precip_cs.where(obs_mask).mean(dim=['lat', 'lon'])
    hist_snowfrac_regional = snow_fraction(hist_tmean_regional)
    hist_snow_regional = hist_precip_regional * hist_snowfrac_regional

    hist_snow_annual = assign_winter_year(hist_snow_regional).groupby('winter_year').sum('time').load()
    hist_precip_annual = assign_winter_year(hist_precip_regional).groupby('winter_year').sum('time').load()
    hist_sp_annual = hist_snow_annual / hist_precip_annual


# ============================================================
# 1. HISTORICAL COLD-SEASON WARMING
# ============================================================
log("\n--- 1. Historical Cold-Season Warming (Nov-Apr) ---")
try:
    years = obs_tmax_annual['winter_year'].values

    tmax_slope, _, tmax_change = linear_trend(years, obs_tmax_annual.values)
    tmin_slope, _, tmin_change = linear_trend(years, obs_tmin_annual.values)

    log(f"Period: {years.min()}-{years.max()} ({years.max() - years.min()} years)")
    log(f"Cold-season Tmax trend: {tmax_slope*10:.3f} C/decade "
        f"(total change: {tmax_change:+.2f} C)")
    log(f"Cold-season Tmin trend: {tmin_slope*10:.3f} C/decade "
        f"(total change: {tmin_change:+.2f} C)")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(years, obs_tmax_annual.values, 'o-', color='firebrick',
            label=f'Cold-Season Tmax ({tmax_slope*10:+.2f} C/decade)')
    ax.plot(years, np.polyval(np.polyfit(years, obs_tmax_annual.values, 1), years),
            '--', color='firebrick', alpha=0.6)
    ax.plot(years, obs_tmin_annual.values, 'o-', color='steelblue',
            label=f'Cold-Season Tmin ({tmin_slope*10:+.2f} C/decade)')
    ax.plot(years, np.polyval(np.polyfit(years, obs_tmin_annual.values, 1), years),
            '--', color='steelblue', alpha=0.6)

    ax.set_xlabel("Winter Year")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(f"Historical Cold-Season (Nov-Apr) Warming, Northeast US\n"
                 f"Livneh Observations, {years.min()}-{years.max()}")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_historical_coldseason_warming.png"), dpi=300)
    plt.close()
    log("Saved: 1_historical_coldseason_warming.png")
except Exception as e:
    log(f"[ERROR] Section 1 failed: {e}")


# ============================================================
# 2. HISTORICAL S/P RATIO TREND
# ============================================================
log("\n--- 2. Historical S/P Ratio Trend ---")
try:
    sp_years = hist_sp_annual['winter_year'].values
    sp_slope, _, sp_change = linear_trend(sp_years, hist_sp_annual.values)

    log(f"Regional mean cold-season S/P ratio trend: {sp_slope*10:+.4f} per decade "
        f"(total change: {sp_change:+.3f} over {sp_years.max()-sp_years.min()} years)")
    log(f"S/P ratio, first 5 winters mean: {float(hist_sp_annual.values[:5].mean()):.3f}")
    log(f"S/P ratio, last 5 winters mean: {float(hist_sp_annual.values[-5:].mean()):.3f}")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sp_years, hist_sp_annual.values, 'o-', color='navy', label='Regional S/P Ratio')
    ax.plot(sp_years, np.polyval(np.polyfit(sp_years, hist_sp_annual.values, 1), sp_years),
            '--', color='navy', alpha=0.6, label=f'Trend ({sp_slope*10:+.4f}/decade)')
    ax.set_ylim(0, 1)
    ax.set_xlabel("Winter Year")
    ax.set_ylabel("Snow-to-Precipitation (S/P) Ratio")
    ax.set_title("Historical Cold-Season S/P Ratio Trend, Northeast US")
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "2_historical_sp_ratio_trend.png"), dpi=300)
    plt.close()
    log("Saved: 2_historical_sp_ratio_trend.png")
except Exception as e:
    log(f"[ERROR] Section 2 failed: {e}")


# ============================================================
# 3. SIMULATION SUMMARY
# ============================================================
log("\n--- 3. LOCA2-WBM Simulation Summary ---")
try:
    all_models = set()
    for ssp_key, ssp_info in scenarios.items():
        ssp_folders = sorted(glob.glob(os.path.join(MODEL_DIR, f"*{ssp_key}*_newprcp")))
        model_names = sorted(set(os.path.basename(f).split('_')[0] for f in ssp_folders))
        all_models.update(model_names)
        log(f"{ssp_info['label']} ({ssp_key}): {len(model_names)} models -> {', '.join(model_names)}")

    log(f"\nTotal unique GCMs used: {len(all_models)}")
    log(f"Scenarios: {', '.join(scenarios.keys())}")
    log(f"Variables: {', '.join(var_mapping.keys())} "
        f"(mapped to Livneh {', '.join(var_mapping.values())})")
    log("Temporal resolution: monthly")
    log(f"Historical baseline (Livneh obs): {HIST_START} to {HIST_END}")
    log(f"Future period simulated: {EARLY_START} to {LATE_END}")
    log(f"  Early-century: {EARLY_START} to {EARLY_END}")
    log(f"  Mid-century:   {MID_START} to {MID_END}")
    log(f"  Late-century:  {LATE_START} to {LATE_END}")
    log(f"Rain/snow partitioning: linear ramp on (Tmax+Tmin)/2, "
        f"100% snow below {T_SNOW} C, 100% rain above {T_RAIN} C")
except Exception as e:
    log(f"[ERROR] Section 3 failed: {e}")


# ============================================================
# 4. HISTORICAL + FUTURE TRENDS: Tmin, Tmax, Precip
# ============================================================
log("\n--- 4. Historical + Future Cold-Season Trends ---")
for var_key in ['airTmax', 'airTmin', 'precip']:
    try:
        log(f"Processing cold-season trend figure for {var_key}...")
        fig, ax = plt.subplots(figsize=(13, 6.5))

        if var_key == 'airTmax':
            obs_annual = obs_tmax_annual
        elif var_key == 'airTmin':
            obs_annual = obs_tmin_annual
        else:
            obs_annual = assign_winter_year(hist_precip_regional).groupby('winter_year').sum('time').load()

        obs_years = obs_annual['winter_year'].values
        ax.plot(obs_years, obs_annual.values, color='black', linewidth=2.5,
                label='Livneh Observations (1980-2013)', zorder=4)

        bridge_year = int(obs_years.max())
        bridge_value = float(obs_annual.values[-1])

        for ssp_key, ssp_info in scenarios.items():
            ensemble = ensemble_scenario_regional_series(ssp_key, var_key, northeast_gdf)
            if ensemble is None:
                continue
            fut_years = ensemble['winter_year'].values
            years_connected = np.concatenate([[bridge_year], fut_years])
            values_connected = np.concatenate([[bridge_value], ensemble.values])
            ax.plot(years_connected, values_connected, color=ssp_info['color'],
                    linewidth=2.5, label=ssp_info['label'], zorder=3)

        for start_yr, end_yr, color in [(2015, 2040, 'royalblue'),
                                          (2041, 2070, 'darkorange'),
                                          (2071, 2100, 'crimson')]:
            ax.axvspan(start_yr, end_yr, color=color, alpha=0.03)

        unit_label = "Precipitation (mm/cold season)" if var_key == 'precip' else "Temperature (deg C)"
        ax.set_ylabel(unit_label, fontsize=12)
        ax.set_xlabel("Winter Year", fontsize=12)
        ax.set_title(f"Cold-Season (Nov-Apr) {var_key}: Historical + Future, Northeast US",
                     fontsize=13, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        plt.tight_layout()

        outname = f"4_coldseason_trend_{var_key}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, outname), dpi=300)
        plt.close()
        log(f"Saved: {outname}")
    except Exception as e:
        log(f"[ERROR] Section 4 ({var_key}) failed: {e}")


# ============================================================
# 5. HISTORICAL vs. LATE-CENTURY S/P RATIO
# ============================================================
log("\n--- 5. Historical vs. Late-Century S/P Ratio ---")
try:
    hist_pct_snow = area_weighted_pct_above(hist_sp_map, obs_mask, SNOW_DOMINANT_THRESHOLD)
    log(f"Historical (1980-2013): {hist_pct_snow:.1f}% of region is snow-dominated "
        f"(S/P > {SNOW_DOMINANT_THRESHOLD})")

    late_maps = {}
    pct_snow_by_scenario = {}
    for ssp_key, ssp_info in scenarios.items():
        sp_map, models_used = ensemble_scenario_sp_map(ssp_key, LATE_START, LATE_END)
        if sp_map is None:
            log(f"  No data available for {ssp_key} late-century S/P map.")
            continue
        model_mask = region_mask(sp_map['lat'], sp_map['lon'], northeast_gdf)
        pct = area_weighted_pct_above(sp_map, model_mask, SNOW_DOMINANT_THRESHOLD)
        late_maps[ssp_key] = sp_map
        pct_snow_by_scenario[ssp_key] = pct
        log(f"Late-century (2071-2100) {ssp_info['label']}: {pct:.1f}% snow-dominated "
            f"({len(models_used)} models)")

    # --- spatial maps: historical + each scenario late-century ---
    n_panels = 1 + len(late_maps)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    vmin, vmax = 0, 1
    im = None
    ax = axes[0]
    im = ax.pcolormesh(hist_sp_map['lon'], hist_sp_map['lat'], hist_sp_map.values,
                        cmap='Blues', vmin=vmin, vmax=vmax, shading='auto')
    northeast_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6)
    ax.set_title(f"Historical\n(1980-2013)\n{hist_pct_snow:.0f}% snow-dominated")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    for i, (ssp_key, sp_map) in enumerate(late_maps.items(), start=1):
        ax = axes[i]
        ax.pcolormesh(sp_map['lon'], sp_map['lat'], sp_map.values,
                      cmap='Blues', vmin=vmin, vmax=vmax, shading='auto')
        northeast_gdf.boundary.plot(ax=ax, color='black', linewidth=0.6)
        ax.set_title(f"{scenarios[ssp_key]['label']}\nLate-Century (2071-2100)\n"
                     f"{pct_snow_by_scenario[ssp_key]:.0f}% snow-dominated")
        ax.set_xlabel("Longitude")

    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.02,
                 label='Snow-to-Precipitation (S/P) Ratio')
    fig.suptitle("Cold-Season S/P Ratio: Historical vs. Late-Century", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "5_sp_ratio_maps_historical_vs_late.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    log("Saved: 5_sp_ratio_maps_historical_vs_late.png")

    # --- bar chart summary ---
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ['Historical\n(1980-2013)'] + [scenarios[k]['label'].split(' (')[0]
                                             for k in pct_snow_by_scenario]
    values = [hist_pct_snow] + list(pct_snow_by_scenario.values())
    colors = ['black'] + [scenarios[k]['color'] for k in pct_snow_by_scenario]
    bars = ax.bar(labels, values, color=colors, alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}%",
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel("% of Region Snow-Dominated (S/P > 0.5)")
    ax.set_title("Northeast US: % of Region Snow-Dominated,\nHistorical vs. Late-Century")
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_pct_snow_dominated_summary.png"), dpi=300)
    plt.close()
    log("Saved: 5_pct_snow_dominated_summary.png")

except Exception as e:
    log(f"[ERROR] Section 5 failed: {e}")


# ============================================================
# WRITE TEXT SUMMARY
# ============================================================
summary_path = os.path.join(OUTPUT_DIR, "summary_stats.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll figures + summary_stats.txt written to: {OUTPUT_DIR}")
